import json
import logging
import os
from typing import Any

from PIL import Image

from .base import BaseDatasetLoader

logger = logging.getLogger(__name__)


class MMELoader(BaseDatasetLoader):
    """Loads the MME-RealWorld-Lite dataset from local JSON and image files."""

    def __init__(
        self,
        data_path: str = "/data/datasets/MME-RealWorld-Lite/data/MME-RealWorld-Lite.json",
        image_dir: str = "/data/datasets/MME-RealWorld-Lite/data/imgs",
    ):
        self.data_path = data_path
        self.image_dir = image_dir

        if not data_path or not image_dir:
            raise ValueError("MME loader requires --data_path and --image_dir arguments.")

        try:
            with open(data_path) as f:
                self.dataset = json.load(f)
        except Exception as e:
            logger.error("Failed to load MME data from %s: %s", data_path, e)
            raise

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Fetches and maps a sample from the MME dataset."""
        example = self.dataset[idx]

        # Load image from file path
        image_file = example["Image"]
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path)

        # Build the full prompt/question
        question = example["Text"]
        answer_choices = example["Answer choices"]
        prompt = question + "\n" + "\n".join(answer_choices)

        correct_answer = example["Ground truth"]
        task_name = f"{example['Task']}/{example['Subtask']}"
        sample_id = f"mme_{idx}"  # Add prefix

        return {
            "image": image,
            "question": prompt,
            "correct_answer": correct_answer,
            "task_name": task_name,
            "sample_id": sample_id,
        }
