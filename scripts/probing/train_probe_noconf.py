#!/usr/bin/env python3
"""Train probe using ONLY inference-time features (no conf_before).

Struct features (9 dims):
    0: step_idx / 10.0
    1: is_first_step (step_idx == 0)
    2: is_repeated_tool (tool_name appeared in earlier steps)
    3..: one-hot tool_name
"""
import argparse, json, logging, sys
from pathlib import Path
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, '/workspace/cv-agent/scripts/probing')
from train_probe import load_info_gain_records, build_trajectory_prefix_from_record

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


def build_struct_features(records, tool_names):
    """Inference-time struct features only (no conf_before)."""
    tool_to_idx = {t: i for i, t in enumerate(tool_names)}
    n_tools = len(tool_names)
    X = np.zeros((len(records), 3 + n_tools))

    # Group trajectories by task_id to resolve previous tools
    for i, r in enumerate(records):
        X[i, 0] = r['step_idx'] / 10.0
        X[i, 1] = float(r["step_idx"] <= 1)  # 1-indexed first step

        # is_repeated_tool: tool_name appeared in prior steps
        prior_tools = set()
        for prior_step in r.get('_episode_trajectory', []):
            ps = prior_step.get('step', 0)
            if ps >= r['step_idx']:
                break
            prior_tools.add(prior_step.get('tool_name', ''))
        X[i, 2] = float(r['tool_name'] in prior_tools)

        if r['tool_name'] in tool_to_idx:
            X[i, 3 + tool_to_idx[r['tool_name']]] = 1.0
    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--encoder', default='all-MiniLM-L6-v2')
    args = parser.parse_args()

    records = load_info_gain_records(args.input)
    n_useful = sum(1 for r in records if r['tool_useful'])
    logger.info('Loaded %d records, %d useful (%.1f%%)',
                len(records), n_useful, 100*n_useful/len(records))

    # Build prefixes + embed
    texts = [build_trajectory_prefix_from_record(r) for r in records]
    labels = np.array([int(r['tool_useful']) for r in records])

    from sentence_transformers import SentenceTransformer
    logger.info('Encoding %d texts with %s', len(texts), args.encoder)
    encoder = SentenceTransformer(args.encoder)
    embeddings = encoder.encode(texts, batch_size=64, show_progress_bar=True)
    logger.info('Embeddings shape: %s', embeddings.shape)

    # Struct features (NO conf_before)
    tool_names = sorted(set(r['tool_name'] for r in records))
    logger.info('Tool names (%d): %s', len(tool_names), tool_names)
    struct = build_struct_features(records, tool_names)
    logger.info('Struct shape: %s (expected 3 + %d tools = %d)',
                struct.shape, len(tool_names), 3 + len(tool_names))

    X = np.hstack([embeddings, struct])
    logger.info('Combined X shape: %s', X.shape)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        'logistic': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'mlp128': MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=42),
    }

    print('\n=== CV Results (no conf_before) ===')
    best_name, best_auroc = None, 0
    for name, clf in candidates.items():
        auroc = cross_val_score(clf, X, labels, cv=cv, scoring='roc_auc')
        print(f'{name:15s}  AUROC={auroc.mean():.3f}±{auroc.std():.3f}')
        if auroc.mean() > best_auroc:
            best_auroc = auroc.mean()
            best_name = name

    print(f'\nBest: {best_name} AUROC={best_auroc:.3f}')
    best_model = candidates[best_name]
    best_model.fit(X, labels)

    y_prob = best_model.predict_proba(X)[:, 1]
    print(f'Full AUROC: {roc_auc_score(labels, y_prob):.3f}')
    print(f'Full Acc:   {accuracy_score(labels, best_model.predict(X)):.3f}')

    # Per-source
    print('\n=== Per-source ===')
    sources = {}
    for i, r in enumerate(records):
        src = r['task_id'].split('_')[0] if '_' in r['task_id'] else 'unknown'
        sources.setdefault(src, []).append(i)
    for src, idx in sorted(sources.items()):
        y_true = labels[idx]
        y_p = y_prob[idx]
        try:
            auc = roc_auc_score(y_true, y_p)
        except ValueError:
            auc = float('nan')
        print(f'  {src:15s}  n={len(idx):5d}  useful={int(y_true.sum()):4d}  AUROC={auc:.3f}')

    save = {
        'model': best_model,
        'model_name': best_name,
        'encoder_name': args.encoder,
        'threshold': 0.5,
        'tool_names': tool_names,
        'struct_feature_spec': ['step_idx_norm', 'is_first_step', 'is_repeated_tool', 'tool_onehot'],
        'struct_dim': 3 + len(tool_names),
        'auroc': best_auroc,
        'n_records': len(records),
        'n_useful': n_useful,
        'use_extra_features': True,
        'use_conf_before': False,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(save, out)
    logger.info('Saved to %s', out)


if __name__ == '__main__':
    main()
