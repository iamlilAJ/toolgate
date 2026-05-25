#!/usr/bin/env python3
"""Feature-ablation training for probe_v3_skipsafe.

Mirrors the original skipsafe probe:
  - MiniLM (all-MiniLM-L6-v2) trajectory-prefix embeddings (384-d)
  - 9-d struct features: step_idx_norm + is_first_step + is_repeated_tool + 6-d tool_onehot
  - class-balanced LogisticRegression on must_execute (= tool_flipped)
    label: not correct_before and correct_after
  - 5-fold StratifiedKFold CV AUROC

--feature_type selects which features the classifier sees:
  full              -> np.hstack([emb, struct])   (393-d)
  text_only         -> emb                        (384-d)
  struct_only       -> struct                     (9-d)
  tool_onehot_only  -> struct[:, 3:9]             (6-d)
"""
import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, "/workspace/cv-agent/scripts/probing")
from train_probe import load_info_gain_records, build_trajectory_prefix_from_record  # noqa: E402
from train_probe_noconf import build_struct_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Fixed tool list to guarantee identical ordering with the headline probe.
TOOL_NAMES = [
    "cropping",
    "depth_estimation",
    "detection",
    "detection_small_object",
    "ocr",
    "segmentation",
]

STRUCT_SPEC = ["step_idx_norm", "is_first_step", "is_repeated_tool", "tool_onehot"]

INPUT_FILES = [
    "/workspace/skill-guard/data/probe_info_gain/vstar_full.jsonl",
    "/workspace/skill-guard/data/probe_info_gain/cvbench_500.jsonl",
    "/workspace/skill-guard/data/probe_info_gain/hrbench4k_full.jsonl",
    "/workspace/skill-guard/data/probe_info_gain/hrbench8k_full.jsonl",
    "/workspace/skill-guard/data/probe_info_gain/mme_full.jsonl",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_type",
        required=True,
        choices=["full", "text_only", "struct_only", "tool_onehot_only"],
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--input", nargs="+", default=INPUT_FILES)
    args = parser.parse_args()

    records = load_info_gain_records(args.input)
    labels = np.array(
        [
            int((not r["correct_before"]) and r["correct_after"])
            for r in records
        ]
    )
    logger.info(
        "Loaded %d records, must_execute=1: %d (%.1f%%)",
        len(records),
        int(labels.sum()),
        100.0 * labels.mean(),
    )

    # Struct features are always computed (so metadata stays consistent), but
    # only the configured slice is handed to the classifier.
    struct = build_struct_features(records, TOOL_NAMES)
    assert struct.shape[1] == 9, f"unexpected struct shape: {struct.shape}"

    feat_type = args.feature_type
    needs_emb = feat_type in ("full", "text_only")

    if needs_emb:
        from sentence_transformers import SentenceTransformer

        texts = [build_trajectory_prefix_from_record(r) for r in records]
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Encoding %d texts with all-MiniLM-L6-v2", len(texts))
        emb = encoder.encode(texts, batch_size=64, show_progress_bar=True)
        emb = np.asarray(emb)
        logger.info("Embedding shape: %s", emb.shape)
    else:
        emb = None

    if feat_type == "full":
        X = np.hstack([emb, struct])
        struct_spec = STRUCT_SPEC
        struct_dim = 9
    elif feat_type == "text_only":
        X = emb
        struct_spec = []
        struct_dim = 0
    elif feat_type == "struct_only":
        X = struct
        struct_spec = STRUCT_SPEC
        struct_dim = 9
    elif feat_type == "tool_onehot_only":
        X = struct[:, 3:9]
        struct_spec = ["tool_onehot"]
        struct_dim = 6
    else:
        raise ValueError(f"bad feature_type {feat_type}")

    logger.info("X shape: %s (feature_type=%s)", X.shape, feat_type)
    logger.info("Tool names (%d): %s", len(TOOL_NAMES), TOOL_NAMES)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, labels, cv=cv, scoring="roc_auc")
    cv_mean = float(scores.mean())
    cv_std = float(scores.std())
    logger.info(
        "must_execute (skip_safe) CV AUROC = %.4f +/- %.4f (folds=%s)",
        cv_mean,
        cv_std,
        np.round(scores, 4).tolist(),
    )

    clf.fit(X, labels)
    y_prob = clf.predict_proba(X)[:, 1]
    logger.info("Full AUROC: %.4f", roc_auc_score(labels, y_prob))

    save = {
        "model": clf,
        "encoder_name": "all-MiniLM-L6-v2",
        "tool_names": TOOL_NAMES,
        "threshold": 0.5,
        "struct_feature_spec": struct_spec,
        "struct_dim": struct_dim,
        "feature_type": feat_type,
        "label_type": "skip_safe",
        "cv_auroc_mean": cv_mean,
        "cv_auroc_std": cv_std,
        "use_extra_features": struct_dim > 0,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(save, out)
    logger.info("Saved to %s", out)


if __name__ == "__main__":
    main()
