import json
import logging
import os
from typing import Any

from PIL import Image

from .base import BaseDatasetLoader

logger = logging.getLogger(__name__)


class VStarLoader(BaseDatasetLoader):
    """
    Loads the V*STAR benchmark dataset.

    Priority:
      1. HuggingFace hub (``lmms-lab/vstar-bench``) — no local files needed.
      2. Local .jsonl + image directory — used when ``data_path`` is explicitly
         provided or the server path ``/data/datasets/vstar_bench`` exists.
    """

    def __init__(
        self,
        data_path: str = "/data/datasets/vstar_bench/test_questions.jsonl",
        image_dir: str = "/data/datasets/vstar_bench",
    ):
        self.data_path = data_path
        self.image_dir = image_dir

        # ── Try local path first (server environment) ────────────────────────
        if os.path.exists(data_path):
            try:
                logger.info("Loading V*STAR from local path: %s", data_path)
                self.dataset = []
                self._mode = "local"
                with open(data_path) as f:
                    for line in f:
                        self.dataset.append(json.loads(line))
                logger.info("Loaded %d V*STAR samples from local files.", len(self.dataset))
                return
            except Exception as e:
                logger.warning("Local V*STAR load failed (%s), falling back to HF.", e)

        # ── Fall back to Hugging Face hub ────────────────────────────────────
        try:
            from datasets import load_dataset
            logger.info("Loading V*STAR from HuggingFace: lmms-lab/vstar-bench ...")
            self._hf_dataset = load_dataset(
                "lmms-lab/vstar-bench", split="test", trust_remote_code=True
            )
            self._mode = "hf"
            logger.info("Loaded %d V*STAR samples from HuggingFace.", len(self._hf_dataset))
        except Exception as e:
            logger.error("Failed to load V*STAR dataset from HF: %s", e)
            raise

    def __len__(self) -> int:
        if self._mode == "hf":
            return len(self._hf_dataset)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Fetches and maps a sample from the V*STAR dataset."""
        if self._mode == "hf":
            example = self._hf_dataset[idx]
            image = example["image"]  # PIL Image, already decoded by HF
        else:
            example = self.dataset[idx]
            image_path = os.path.join(self.image_dir, example["image"])
            image = Image.open(image_path)

        return {
            "image": image,
            "question": example["text"],
            "correct_answer": example["label"],
            "task_name": example["category"],
            "sample_id": f"vstar_{example['question_id']}",
        }
