import logging

from .base import BaseDatasetLoader
from .cv_bench_loader import CVBenchLoader
from .hr_bench_loader import HRBenchLoader
from .mme_loader import MMELoader
from .probe_loader import ProbeDatasetLoader
from .vstar_loader import VStarLoader

logger = logging.getLogger(__name__)


def get_dataset_loader(name: str) -> BaseDatasetLoader:
    """
    Factory function to get the specified dataset loader.
    """
    if name == "mme":
        logger.info("Instantiating MME dataset loader.")
        return MMELoader()
    elif name == "cvbench":
        logger.info("Instantiating CV-Bench dataset loader.")
        return CVBenchLoader()
    elif name == "vstar":
        logger.info("Instantiating VStar dataset loader.")
        return VStarLoader()
    elif name == "hrbench-4k":
        logger.info("Instantiating HR-Bench dataset loader.")
        return HRBenchLoader(split_name="hrbench_4k")
    elif name == "hrbench-8k":
        logger.info("Instantiating HR-Bench dataset loader.")
        return HRBenchLoader(split_name="hrbench_8k")
    elif name == "probe-all":
        logger.info("Instantiating Probe dataset loader (all sources).")
        return ProbeDatasetLoader("all")
    elif name == "probe-max":
        logger.info("Instantiating Probe MAX dataset loader.")
        return ProbeDatasetLoader("max")
    elif name == "probe-max-p1":
        logger.info("Instantiating Probe MAX Part1 loader.")
        return ProbeDatasetLoader("max-p1")
    elif name == "probe-max-p2":
        logger.info("Instantiating Probe MAX Part2 loader.")
        return ProbeDatasetLoader("max-p2")
    elif name == "probe-all-v2":
        logger.info("Instantiating Probe v2 dataset loader (all sources).")
        return ProbeDatasetLoader("all_v2")
    elif name == "probe-all-v3":
        logger.info("Instantiating Probe v3 dataset loader (GQA+TextVQA+LHRS+SEED+AOKVQA+DocVQA).")
        return ProbeDatasetLoader("all_v3")
    elif name == "probe-cub":
        logger.info("Instantiating Probe CUB dataset loader (fine-grained bird attribute MCQ).")
        return ProbeDatasetLoader("cub")
    elif name == "probe-gqa":
        logger.info("Instantiating Probe dataset loader (GQA).")
        return ProbeDatasetLoader("gqa")
    elif name == "probe-textvqa":
        logger.info("Instantiating Probe dataset loader (TextVQA).")
        return ProbeDatasetLoader("textvqa")
    elif name == "probe-rsvlmqa":
        logger.info("Instantiating Probe dataset loader (RSVLM-QA).")
        return ProbeDatasetLoader("rsvlmqa")
    else:
        raise ValueError(
            f"Unknown dataset loader: {name}. Must be 'mme', 'mme-og', 'realworldqa',"
            f" 'vstar', 'cvbench', 'hrbench-*', 'emb', or 'probe-*'."
        )


__all__ = ["get_dataset_loader", "BaseDatasetLoader"]
