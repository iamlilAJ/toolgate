#!/usr/bin/env python3
"""Collect information gain labels by running agent with ORIGINAL prompt
and measuring logprobs before/after each tool call.

The agent runs normally (no belief prompt). At each tool call boundary,
we make side-channel VLM calls with forced-answer suffix to measure
P(A), P(B), P(C), P(D) before and after the tool executes.

Usage:
    # On server, from /workspace/cv-agent:
    uv run python scripts/probing/collect_info_gain.py \
        --dataset probe-max-p1 \
        --output data/probe_info_gain/labels_p1.jsonl \
        --num-processes 4 \
        --vlm-endpoint http://localhost:52308/v1
"""

import argparse
import asyncio
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# Forced answer suffix — appended as a user message in a side-channel VLM call
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
    """Get logprobs for A/B/C/D by appending a forced-answer message.

    Args:
        messages: The conversation so far (system + user + assistant + tool msgs).
        endpoint: VLM API endpoint (e.g., http://....:52308/v1).
        model: Model name.

    Returns:
        Dict mapping letter -> logprob, e.g. {"A": -0.5, "B": -2.1, "C": -0.3, "D": -4.0}
    """
    # Append forced answer as a new user message (side-channel, not added to real conversation)
    probe_messages = list(messages) + [
        {"role": "user", "content": FORCED_ANSWER_MSG}
    ]

    resp = requests.post(
        f"{endpoint}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": probe_messages,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 20,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    logprobs_data = data["choices"][0].get("logprobs", {})
    if not logprobs_data or not logprobs_data.get("content"):
        return {"A": -10.0, "B": -10.0, "C": -10.0, "D": -10.0}

    top = logprobs_data["content"][0]["top_logprobs"]

    # Extract logprobs for A/B/C/D tokens
    answer_lp: dict[str, float] = {}
    for t in top:
        # Handle tokenizations: "A", "(A", "(A)", " A", etc.
        tok = t["token"].strip().lstrip("(").rstrip(")")
        if tok in "ABCD" and tok not in answer_lp:
            answer_lp[tok] = t["logprob"]

    # Fill missing letters with very low prob
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
    """Compute information gain label from before/after logprobs.

    Returns dict with all derived metrics and the binary tool_useful label.
    """
    probs_before = softmax(logprobs_before)
    probs_after = softmax(logprobs_after)

    answer_before = max(probs_before, key=probs_before.get)
    answer_after = max(probs_after, key=probs_after.get)
    conf_before = probs_before[answer_before]
    conf_after = probs_after[answer_after]

    # Normalize ground truth to single letter
    gt = re.match(r"\(?([A-Da-d])\)?", ground_truth.strip())
    gt_letter = gt.group(1).upper() if gt else ground_truth.strip().upper()

    correct_before = answer_before == gt_letter
    correct_after = answer_after == gt_letter
    conf_gain = conf_after - conf_before

    # KL divergence: D_KL(after || before)
    kl_div = 0.0
    for k in "ABCD":
        if probs_after[k] > 1e-10 and probs_before[k] > 1e-10:
            kl_div += probs_after[k] * math.log(probs_after[k] / probs_before[k])

    # Binary label
    if not correct_before and correct_after:
        tool_useful = True  # fixed wrong answer
    elif correct_before and not correct_after:
        tool_useful = False  # broke correct answer
    elif correct_before and correct_after and conf_gain > conf_gain_threshold:
        tool_useful = True  # reinforced correct answer significantly
    else:
        tool_useful = False  # no meaningful change

    return {
        "logprobs_before": logprobs_before,
        "logprobs_after": logprobs_after,
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

    Format:
        [Q] question text
        [T1] thinking text
        [TOOL] tool_name(params) -> output summary
        [T2] thinking text
        [PENDING] tool_name(params)
    """
    parts = [f"[Q] {question}"]

    for i, step in enumerate(trajectory_steps):
        thinking = step.get("thinking", "")
        tool = step.get("tool_name", "")
        output = step.get("tool_output", {})

        if thinking:
            parts.append(f"[T{i+1}] {thinking[:200]}")

        if tool and tool != "__final_answer__":
            params = step.get("tool_params", {})
            # Summarize output
            if isinstance(output, dict):
                out_str = json.dumps(output, ensure_ascii=False)[:150]
            else:
                out_str = str(output)[:150]
            parts.append(f"[TOOL] {tool}({json.dumps(params)[:80]}) -> {out_str}")

    if pending_thinking:
        parts.append(f"[T{len(trajectory_steps)+1}] {pending_thinking[:200]}")

    parts.append(
        f"[PENDING] {pending_tool_name}({json.dumps(pending_tool_params)[:80]})"
    )

    full_text = "\n".join(parts)
    if len(full_text) > max_chars:
        full_text = "..." + full_text[-(max_chars - 3) :]

    return full_text


def main():
    parser = argparse.ArgumentParser(description="Collect information gain labels")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--vlm-endpoint", type=str, default="http://localhost:52308/v1")
    parser.add_argument("--vlm-model", type=str, default="Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print(f"Info gain collection: dataset={args.dataset}, output={args.output}")
    print(f"VLM: {args.vlm_endpoint} model={args.vlm_model}")
    print(f"This script modifies the agent pipeline to collect logprobs.")
    print(f"Run via: uv run python src/cv_agent/main_benchmark.py ...")
    print(f"with --collect-info-gain flag (to be added)")
    print()
    print("NOTE: This script provides helper functions.")
    print("The actual collection is integrated into main_benchmark.py")
    print("via the --collect-info-gain flag.")


if __name__ == "__main__":
    main()
