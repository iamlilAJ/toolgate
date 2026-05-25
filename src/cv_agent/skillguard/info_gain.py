"""Information gain measurement via logprob probing.

Provides side-channel logprob collection for measuring tool usefulness
without modifying the agent's conversation.
"""

import json
import logging
import math
import re

import requests

logger = logging.getLogger(__name__)

FORCED_ANSWER_MSG = (
    "Based on all information available to you so far, what is your best answer "
    "to the original question? You MUST choose exactly one option letter. "
    "Respond with ONLY the letter (A, B, C, or D), nothing else."
)


def get_answer_logprobs(
    messages: list[dict],
    endpoint: str,
    model: str,
    api_key: str = "empty",
    timeout: int = 30,
) -> dict[str, float]:
    """Get logprobs for A/B/C/D by appending a forced-answer message."""
    import os as _os
    if api_key == "empty":
        api_key = _os.environ.get("PG_INFOGAIN_KEY", "empty")
    probe_messages = list(messages) + [
        {"role": "user", "content": FORCED_ANSWER_MSG}
    ]

    try:
        resp = requests.post(
            f"{endpoint}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": probe_messages,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": True,
                "top_logprobs": 5,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Logprob probe failed: %s", e)
        return {"A": -10.0, "B": -10.0, "C": -10.0, "D": -10.0}

    logprobs_data = data.get("choices", [{}])[0].get("logprobs", {})
    if not logprobs_data or not logprobs_data.get("content"):
        return {"A": -10.0, "B": -10.0, "C": -10.0, "D": -10.0}

    top = logprobs_data["content"][0]["top_logprobs"]

    answer_lp: dict[str, float] = {}
    for t in top:
        tok = t["token"].strip().lstrip("(").rstrip(")")
        if tok in "ABCD" and tok not in answer_lp:
            answer_lp[tok] = t["logprob"]

    for letter in "ABCD":
        if letter not in answer_lp:
            answer_lp[letter] = -10.0

    return answer_lp


def softmax(logprobs: dict[str, float]) -> dict[str, float]:
    """Convert logprobs dict to probability dict."""
    max_lp = max(logprobs.values())
    exps = {k: math.exp(v - max_lp) for k, v in logprobs.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def compute_label(
    logprobs_before: dict,
    logprobs_after: dict,
    ground_truth: str,
    conf_gain_threshold: float = 0.1,
) -> dict:
    """Compute information gain label from before/after logprobs."""
    probs_before = softmax(logprobs_before)
    probs_after = softmax(logprobs_after)

    answer_before = max(probs_before, key=probs_before.get)
    answer_after = max(probs_after, key=probs_after.get)
    conf_before = probs_before[answer_before]
    conf_after = probs_after[answer_after]

    gt = re.match(r"\(?([A-Da-d])\)?", ground_truth.strip())
    gt_letter = gt.group(1).upper() if gt else ground_truth.strip().upper()

    correct_before = answer_before == gt_letter
    correct_after = answer_after == gt_letter
    conf_gain = conf_after - conf_before

    kl_div = 0.0
    for k in "ABCD":
        if probs_after[k] > 1e-10 and probs_before[k] > 1e-10:
            kl_div += probs_after[k] * math.log(probs_after[k] / probs_before[k])

    if not correct_before and correct_after:
        tool_useful = True
    elif correct_before and not correct_after:
        tool_useful = False
    elif correct_before and correct_after and conf_gain > conf_gain_threshold:
        tool_useful = True
    else:
        tool_useful = False

    return {
        "probs_before": {k: round(v, 4) for k, v in probs_before.items()},
        "probs_after": {k: round(v, 4) for k, v in probs_after.items()},
        "answer_before": answer_before,
        "answer_after": answer_after,
        "conf_before": round(conf_before, 4),
        "conf_after": round(conf_after, 4),
        "correct_before": correct_before,
        "correct_after": correct_after,
        "kl_divergence": round(kl_div, 4),
        "conf_gain": round(conf_gain, 4),
        "tool_useful": tool_useful,
    }


def build_trajectory_prefix(
    question: str,
    trajectory_steps: list[dict],
    pending_thinking: str,
    pending_tool_name: str,
    pending_tool_params: dict,
    max_chars: int = 1500,
) -> str:
    """Build trajectory prefix string for probe input.

    Always keeps [Q] and [PENDING] blocks. Truncates middle history
    steps if the total exceeds max_chars.
    """
    # Fixed parts that are always included
    q_part = f"[Q] {question[:300]}"
    pending_parts = []
    if pending_thinking:
        pending_parts.append(f"[T{len(trajectory_steps)+1}] {pending_thinking[:200]}")
    pending_parts.append(f"[PENDING] {pending_tool_name}({json.dumps(pending_tool_params)[:80]})")
    tail = "\n".join(pending_parts)

    # Budget for middle steps
    fixed_len = len(q_part) + 1 + len(tail) + 1  # +1 for newlines
    budget = max_chars - fixed_len

    # Build middle parts (reverse order, keep most recent)
    mid_parts = []
    for i, step in enumerate(trajectory_steps):
        thinking = step.get("thinking", "")
        tool = step.get("tool_name", "")
        output = step.get("tool_output", {})

        step_parts = []
        if thinking:
            step_parts.append(f"[T{i+1}] {thinking[:200]}")
        if tool and tool != "__final_answer__":
            params = step.get("tool_params", {})
            out_str = json.dumps(output, ensure_ascii=False)[:150] if isinstance(output, dict) else str(output)[:150]
            step_parts.append(f"[TOOL] {tool}({json.dumps(params)[:80]}) -> {out_str}")
        mid_parts.append("\n".join(step_parts))

    # Keep as many recent steps as fit in budget
    kept = []
    used = 0
    for part in reversed(mid_parts):
        if used + len(part) + 1 <= budget:
            kept.append(part)
            used += len(part) + 1
        else:
            break
    kept.reverse()

    if len(kept) < len(mid_parts):
        # Some steps were dropped
        mid_text = "...\n" + "\n".join(kept) if kept else "..."
    else:
        mid_text = "\n".join(kept)

    return f"{q_part}\n{mid_text}\n{tail}" if mid_text else f"{q_part}\n{tail}"


def messages_to_api_format(messages) -> list[dict]:
    """Convert LangChain messages to API-compatible dicts."""
    from langchain_core.messages import AIMessage

    api_msgs = []
    for m in messages:
        if not hasattr(m, "content"):
            continue
        if isinstance(m.content, list):
            api_msgs.append({"role": "user", "content": m.content})
        elif isinstance(m.content, str):
            if isinstance(m, AIMessage):
                api_msgs.append({"role": "assistant", "content": m.content})
            elif hasattr(m, "type") and m.type == "system":
                api_msgs.append({"role": "system", "content": m.content})
            elif hasattr(m, "type") and m.type == "tool":
                api_msgs.append({"role": "assistant", "content": m.content})
            else:
                api_msgs.append({"role": "user", "content": m.content})
    return api_msgs
