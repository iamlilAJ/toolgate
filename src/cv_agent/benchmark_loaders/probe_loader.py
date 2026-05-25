"""Probing dataset loader for VoI alpha table estimation.

Loads MCQ-formatted probing data from JSONL files.
Supports individual sources (gqa, textvqa, rsvlmqa) or merged (all).
"""

import json
import logging
import os
from typing import Any

from PIL import Image

from .base import BaseDatasetLoader

logger = logging.getLogger(__name__)

# Default paths — detect project root from known landmarks, not __file__ or cwd (may be Ray temp)
def _find_data_dir() -> str:
    """Find the data/ directory by checking common locations."""
    candidates = [
        os.environ.get("CV_AGENT_DATA_DIR", ""),
        "/workspace/cv-agent/data",  # Server
        os.path.join(os.getcwd(), "data"),  # Local dev
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"),  # Relative
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return os.path.join(os.getcwd(), "data")  # Fallback

_DATA_DIR = _find_data_dir()


class ProbeDatasetLoader(BaseDatasetLoader):
    """Load probing dataset from JSONL files."""

    def __init__(self, source: str = "all") -> None:
        """
        Args:
            source: "all", "gqa", "textvqa", or "rsvlmqa"
        """
        self.source = source

        if source == "all":
            data_path = os.path.join(_DATA_DIR, "probe_all.jsonl")
        elif source == "all_v2":
            data_path = os.path.join(_DATA_DIR, "probe_v2", "probe_all_v2.jsonl")
        elif source == "all_v3":
            data_path = os.path.join(_DATA_DIR, "probe_v3", "all_samples.jsonl")
        elif source == "cub":
            data_path = os.path.join(_DATA_DIR, "probe_v3", "cub", "samples_verified.jsonl")
        elif source == "max":
            data_path = os.path.join(_DATA_DIR, "probe_max", "probe_all_max.jsonl")
        elif source == "max-p1":
            data_path = os.path.join(_DATA_DIR, "probe_max", "probe_all_max_part1.jsonl")
        elif source == "max-p2":
            data_path = os.path.join(_DATA_DIR, "probe_max", "probe_all_max_part2.jsonl")
        else:
            data_path = os.path.join(_DATA_DIR, f"probe_{source}", "annotated.jsonl")
            # Fallback to verified if annotated doesn't exist
            if not os.path.exists(data_path):
                data_path = os.path.join(_DATA_DIR, f"probe_{source}", "raw.jsonl")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Probing dataset not found: {data_path}")

        self.records: list[dict] = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        logger.info("ProbeDatasetLoader(%s): loaded %d samples from %s", source, len(self.records), data_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]

        # Load image
        image_path = record["image_path"]
        image = Image.open(image_path).convert("RGB")

        return {
            "image": image,
            "question": record["question"],
            "correct_answer": record["correct_answer"],
            "task_name": record.get("query_type", record.get("source", "probe")),
            "sample_id": record["task_id"],
        }
