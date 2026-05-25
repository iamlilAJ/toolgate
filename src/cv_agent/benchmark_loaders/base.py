from abc import ABC, abstractmethod
from typing import Any


class BaseDatasetLoader(ABC):
    """
    Abstract base class for a dataset loader.
    Defines the interface for __len__ and __getitem__
    to provide a unified data structure.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Fetches a single sample by its index and returns it in a
        standardized dictionary format.

        Standardized Format:
        {
            "image": PIL.Image.Image,
            "question": str,
            "correct_answer": str,
            "task_name": str,
            "sample_id": str  # Must be unique (e.g., "mme_123")
        }
        """
        pass
