import logging
from typing import Any

from datasets import load_dataset, concatenate_datasets

from .base import BaseDatasetLoader

logger = logging.getLogger(__name__)


class CVBenchLoader(BaseDatasetLoader):
    """Loads the CV-Bench dataset from the Hugging Face hub."""

    def __init__(self):  # Added **kwargs to ignore unused args
        try:
            ds_2d = load_dataset("nyu-visionx/CV-Bench", "2D", split="test")
            ds_3d = load_dataset("nyu-visionx/CV-Bench", "3D", split="test")
            self.dataset = concatenate_datasets([ds_2d, ds_3d])
        except Exception as e:
            logger.error("Failed to load 'nyu-visionx/CV-Bench' from Hugging Face: %s", e)
            raise

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Fetches and maps a sample from the CV-Bench dataset."""
        example = self.dataset[idx]

        image = example["image"]  # Already a PIL Image
        question = example["prompt"]
        correct_answer = example["answer"]
        task_name = example["task"]
        sample_id = f"cvbench_{idx}"  # Add prefix

        return {
            "image": image,
            "question": question,
            "correct_answer": correct_answer,
            "task_name": task_name,
            "sample_id": sample_id,
        }
