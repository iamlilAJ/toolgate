"""Trajectory logging utilities for the CV Agent.

Writes one JSONL line per task, containing the full step-by-step trajectory.
Each step captures: thinking text, tool name/params, input image, tool output, output image.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TrajectoryLogger:
    """Appends complete agent trajectories to a JSONL file.

    Usage::

        logger = TrajectoryLogger("results/trajectories/worker_1.jsonl")
        record  = TrajectoryLogger.build_task_record(
            task_id="cvbench_42",
            image_url="http://minio/.../image.jpg",
            image_metadata={"width": 1280, "height": 720},
            query="What colour is the car?",
            ground_truth="red",
            final_answer="red",
            is_correct=True,
            trajectory_steps=[...],
            total_input_tokens=1234,
            total_output_tokens=567,
        )
        logger.write(record)
    """

    def __init__(self, output_path: str) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        """Append *record* as a single JSON line (no trailing newline issues)."""
        with open(self.output_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def build_task_record(
        task_id: str,
        image_url: str,
        image_metadata: dict[str, Any],
        query: str,
        ground_truth: str,
        final_answer: str,
        is_correct: bool,
        trajectory_steps: list[dict],
        total_input_tokens: int,
        total_output_tokens: int,
        **extra: Any,
    ) -> dict:
        """Return a dict ready to be serialised by :meth:`write`."""
        return {
            "task_id": task_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_url": image_url,
            "image_metadata": image_metadata,
            "query": query,
            "ground_truth": ground_truth,
            "final_answer": final_answer,
            "is_correct": is_correct,
            "total_steps": len(trajectory_steps),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "trajectory": trajectory_steps,
            **extra,
        }
