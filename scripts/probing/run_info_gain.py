#!/usr/bin/env python3
"""Standalone info gain collection: run agent + logprob probes.

Runs the agent with ORIGINAL prompt (no belief), and at each tool call
boundary makes side-channel VLM calls to measure logprobs before/after.

Usage:
    cd /workspace/cv-agent
    uv run python scripts/probing/run_info_gain.py \
        --dataset vstar --limit 10 \
        --output data/probe_info_gain/vstar_test.jsonl
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run_single_episode(
    agent_executor,
    question: str,
    image,
    sample_id: str,
    correct_answer: str,
    invoke_config: dict,
    info_gain_endpoint: str,
    info_gain_model: str,
):
    """Run one episode and return trajectory with info_gain labels."""
    from cv_agent.utils.storage import upload_pil_image_to_minio

    public_url, w, h = await upload_pil_image_to_minio(image, sample_id)
    if not public_url:
        logger.error("Failed to upload image for %s", sample_id)
        return None

    initial_state = {
        "question": question,
        "original_figure_url": public_url,
        "current_turn": 1,
        "max_turns": 10,
        "messages": [],
        "image_dimensions": {public_url: (w, h)},
        "prefix": sample_id,
        "url_map": {public_url: public_url},
        "direct_reasoning_result": "",
        "tool_usage": {},
        "trajectory_steps": [],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "pending_thinking": "",
        "pending_belief": {},
        "episode_context": {},
        "enable_belief_annotation": False,
        "gate_log": [],
        "skillguard_log": [],
        # Ground truth for info gain labeling
        "_ground_truth": correct_answer,
    }

    try:
        final_state = await agent_executor.ainvoke(initial_state, config=invoke_config)
    except Exception as e:
        logger.error("Agent failed for %s: %s", sample_id, e)
        return None

    # Extract final answer
    import re
    final_answer = ""
    for msg in reversed(final_state.get("messages", [])):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            m = re.search(r"<answer>(.*?)</answer>", msg.content, re.DOTALL)
            if m:
                final_answer = m.group(1).strip()
                break

    # Build record
    record = {
        "task_id": sample_id,
        "question": question,
        "ground_truth": correct_answer,
        "final_answer": final_answer,
        "trajectory": final_state.get("trajectory_steps", []),
        "total_input_tokens": final_state.get("total_input_tokens", 0),
        "total_output_tokens": final_state.get("total_output_tokens", 0),
    }

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--vlm-endpoint", default="http://localhost:10010/v1")
    parser.add_argument("--vlm-model", default="Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--info-gain-endpoint", default="http://localhost:52308/v1")
    parser.add_argument("--info-gain-model", default="Qwen3-VL-30B-A3B-Instruct")
    args = parser.parse_args()

    # Load dataset
    from cv_agent.benchmark_loaders import get_dataset_loader
    loader = get_dataset_loader(args.dataset)
    samples = list(loader)
    if args.limit:
        samples = samples[:args.limit]
    logger.info("Loaded %d samples from %s", len(samples), args.dataset)

    # Build agent executor (no belief, no gate)
    from omegaconf import OmegaConf
    from cv_agent.core.builder import GraphBuilder

    # Assume running from project root (e.g., /workspace/cv-agent)
    cfg_path = os.path.join(os.getcwd(), "configs", "qwen30b_info_gain.yaml")
    cfg = OmegaConf.load(cfg_path)
    builder = GraphBuilder(cfg)
    agent_executor = builder.build()
    logger.info("Agent executor built")

    invoke_config = {"recursion_limit": 60}

    # Output file
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_path, "w")

    # Run episodes
    for i, sample in enumerate(samples):
        sid = sample.get("task_id", f"sample_{i}")
        question = sample["question"]
        correct = sample["correct_answer"]
        image = sample["image"]

        logger.info("[%d/%d] Processing %s", i + 1, len(samples), sid)

        record = asyncio.run(run_single_episode(
            agent_executor=agent_executor,
            question=question,
            image=image,
            sample_id=sid,
            correct_answer=correct,
            invoke_config=invoke_config,
            info_gain_endpoint=args.info_gain_endpoint,
            info_gain_model=args.info_gain_model,
        ))

        if record:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            # Count info gain labels
            ig_count = sum(1 for s in record.get("trajectory", []) if s.get("info_gain"))
            logger.info("[%d/%d] %s: answer=%s gt=%s steps=%d info_gain_labels=%d",
                        i + 1, len(samples), sid,
                        record["final_answer"][:10], correct,
                        len(record["trajectory"]), ig_count)

    out_f.close()
    logger.info("Done. Output: %s", args.output)


if __name__ == "__main__":
    main()
