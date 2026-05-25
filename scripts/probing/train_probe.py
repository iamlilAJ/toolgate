#!/usr/bin/env python3
"""Train probing classifier on info gain labels.

Reads info_gain_labels JSONL files, embeds trajectory prefixes with
sentence-transformers, trains logistic regression to predict tool_useful.

Usage:
    pip install sentence-transformers scikit-learn joblib
    python scripts/probing/train_probe.py \
        --input data/probe_info_gain/vstar_full.jsonl \
                data/probe_info_gain/hrbench4k_full.jsonl \
                data/probe_info_gain/cvbench_500.jsonl \
        --output models/probe_gate/probe_model.joblib
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_info_gain_records(paths: list[str]) -> list[dict]:
    """Load tool-call level records with info_gain labels from trajectory JSONL."""
    records = []
    for path in paths:
        for line in open(path):
            if not line.strip():
                continue
            episode = json.loads(line)
            for step in episode.get("trajectory", []):
                ig = step.get("info_gain")
                if ig is None:
                    continue
                records.append({
                    "task_id": episode.get("task_id", ""),
                    "step_idx": step.get("step", 0),
                    "tool_name": step.get("tool_name", ""),
                    "tool_params": step.get("tool_params", {}),
                    "thinking": step.get("thinking", ""),
                    "ground_truth": episode.get("ground_truth", ""),
                    "question": episode.get("question", ""),
                    # Info gain fields
                    "tool_useful": ig["tool_useful"],
                    "conf_before": ig.get("conf_before", 0),
                    "conf_after": ig.get("conf_after", 0),
                    "correct_before": ig.get("correct_before", False),
                    "correct_after": ig.get("correct_after", False),
                    "kl_divergence": ig.get("kl_divergence", 0),
                    "conf_gain": ig.get("conf_gain", 0),
                    # Trajectory prefix for embedding
                    "trajectory_prefix": ig.get("trajectory_prefix", ""),
                    # Previous steps for building prefix if not stored
                    "_episode_trajectory": episode.get("trajectory", []),
                })
    return records


def build_trajectory_prefix_from_record(record: dict) -> str:
    """Build trajectory prefix if not already stored in info_gain."""
    if record.get("trajectory_prefix"):
        return record["trajectory_prefix"]

    # Fallback: build from episode trajectory
    parts = [f"[Q] {record.get('question', '')}"]
    for i, step in enumerate(record.get("_episode_trajectory", [])):
        if i >= record["step_idx"]:
            break
        thinking = step.get("thinking", "")
        tool = step.get("tool_name", "")
        if thinking:
            parts.append(f"[T{i+1}] {thinking[:200]}")
        if tool and tool != "__final_answer__":
            parts.append(f"[TOOL] {tool}")

    parts.append(f"[T{record['step_idx']+1}] {record.get('thinking', '')[:200]}")
    parts.append(f"[PENDING] {record.get('tool_name', '')}")

    text = "\n".join(parts)
    if len(text) > 1500:
        text = "..." + text[-1497:]
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Info gain JSONL files")
    parser.add_argument("--output", default="models/probe_gate/probe_model.joblib")
    parser.add_argument("--encoder", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    # Load records
    records = load_info_gain_records(args.input)
    logger.info("Loaded %d tool-call records from %d files", len(records), len(args.input))

    if len(records) < 50:
        logger.error("Too few records (%d). Need at least 50.", len(records))
        sys.exit(1)

    # Label distribution
    n_useful = sum(1 for r in records if r["tool_useful"])
    n_not = len(records) - n_useful
    logger.info("Labels: useful=%d (%.1f%%), not_useful=%d (%.1f%%)",
                n_useful, 100 * n_useful / len(records),
                n_not, 100 * n_not / len(records))

    # Build trajectory prefixes
    texts = [build_trajectory_prefix_from_record(r) for r in records]
    labels = np.array([int(r["tool_useful"]) for r in records])

    # Embed
    logger.info("Encoding %d texts with %s...", len(texts), args.encoder)
    encoder = SentenceTransformer(args.encoder)
    embeddings = encoder.encode(texts, batch_size=64, show_progress_bar=True)
    logger.info("Embeddings shape: %s", embeddings.shape)

    # Optional: add hand-crafted features
    tool_names = sorted(set(r["tool_name"] for r in records))
    tool_to_idx = {t: i for i, t in enumerate(tool_names)}

    extra = np.zeros((len(records), 2 + len(tool_names)))
    for i, r in enumerate(records):
        extra[i, 0] = r["step_idx"] / 10.0  # normalized step
        extra[i, 1] = r["conf_before"]  # confidence before
        if r["tool_name"] in tool_to_idx:
            extra[i, 2 + tool_to_idx[r["tool_name"]]] = 1.0  # one-hot tool

    X_emb = embeddings
    X_combined = np.hstack([embeddings, extra])

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "logistic_emb": (LogisticRegression(max_iter=1000, class_weight="balanced"), X_emb),
        "logistic_combined": (LogisticRegression(max_iter=1000, class_weight="balanced"), X_combined),
        "mlp_combined": (MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=42), X_combined),
    }

    print("\n=== Cross-validation Results ===")
    print(f"{'Model':25s}  {'AUROC':>12s}  {'Accuracy':>12s}")
    best_name, best_auroc = None, 0
    for name, (model, X) in models.items():
        try:
            auroc = cross_val_score(model, X, labels, cv=cv, scoring="roc_auc")
            acc = cross_val_score(model, X, labels, cv=cv, scoring="accuracy")
            print(f"{name:25s}  {auroc.mean():.3f}±{auroc.std():.3f}  {acc.mean():.3f}±{acc.std():.3f}")
            if auroc.mean() > best_auroc:
                best_auroc = auroc.mean()
                best_name = name
        except Exception as e:
            print(f"{name:25s}  ERROR: {e}")

    print(f"\nBest model: {best_name} (AUROC={best_auroc:.3f})")

    # Train best model on all data
    best_model_cls, best_X = models[best_name]
    best_model_cls.fit(best_X, labels)

    # Full evaluation
    y_pred = best_model_cls.predict(best_X)
    y_prob = best_model_cls.predict_proba(best_X)[:, 1]
    print(f"\nFull dataset evaluation:")
    print(f"  AUROC: {roc_auc_score(labels, y_prob):.3f}")
    print(f"  Accuracy: {accuracy_score(labels, y_pred):.3f}")
    print(classification_report(labels, y_pred, target_names=["not_useful", "useful"]))

    # Per-benchmark breakdown
    print("=== Per-source Breakdown ===")
    sources = {}
    for i, r in enumerate(records):
        src = r["task_id"].split("_")[0] if "_" in r["task_id"] else "unknown"
        sources.setdefault(src, {"indices": [], "labels": []})
        sources[src]["indices"].append(i)
        sources[src]["labels"].append(labels[i])

    for src, data in sorted(sources.items()):
        idx = data["indices"]
        y_true = np.array(data["labels"])
        y_p = y_prob[idx]
        try:
            auc = roc_auc_score(y_true, y_p)
        except ValueError:
            auc = float("nan")
        n_pos = y_true.sum()
        print(f"  {src:15s}: n={len(idx):4d}, useful={n_pos:3d} ({100*n_pos/len(idx):.0f}%), AUROC={auc:.3f}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "model": best_model_cls,
        "model_name": best_name,
        "encoder_name": args.encoder,
        "threshold": 0.5,
        "tool_names": tool_names,
        "auroc": best_auroc,
        "n_records": len(records),
        "n_useful": int(n_useful),
        "use_extra_features": "combined" in best_name,
    }
    joblib.dump(save_data, out_path)
    logger.info("Model saved to %s", out_path)


if __name__ == "__main__":
    main()
