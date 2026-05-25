#!/usr/bin/env python3
"""Focal-loss variant of skipsafe ablation.

Same 393-d features, same must_execute label as train_skipsafe_ablation.py
(full feature_type). Classifier = focal-BCE linear head trained with Adam,
then folded into an sklearn LogisticRegression for drop-in compatibility
with probe_gate.py. Standard scaler (per-fold for CV, full-data for final)
is folded into the linear weights so predict_proba works on raw features.
"""
import argparse, logging, sys
from pathlib import Path

import joblib, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, '/workspace/cv-agent/scripts/probing')
from train_probe import load_info_gain_records, build_trajectory_prefix_from_record
from train_probe_noconf import build_struct_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

TOOL_NAMES = [
    'cropping', 'depth_estimation', 'detection',
    'detection_small_object', 'ocr', 'segmentation',
]
STRUCT_SPEC = ['step_idx_norm', 'is_first_step', 'is_repeated_tool', 'tool_onehot']

INPUT_FILES = [
    '/workspace/skill-guard/data/probe_info_gain/vstar_full.jsonl',
    '/workspace/skill-guard/data/probe_info_gain/cvbench_500.jsonl',
    '/workspace/skill-guard/data/probe_info_gain/hrbench4k_full.jsonl',
    '/workspace/skill-guard/data/probe_info_gain/hrbench8k_full.jsonl',
    '/workspace/skill-guard/data/probe_info_gain/mme_full.jsonl',
]


class Linear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
    def forward(self, x):
        return self.fc(x).squeeze(-1)


def focal_bce(logits, y, alpha, gamma):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    pt = p*y + (1-p)*(1-y)
    at = alpha*y + (1-alpha)*(1-y)
    return (at * (1-pt).pow(gamma) * ce).mean()


def train_linear(Xtr, ytr, alpha, gamma, epochs=400, lr=5e-3, wd=1e-4):
    torch.manual_seed(42)
    model = Linear(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.from_numpy(Xtr).float()
    yt = torch.from_numpy(ytr).float()
    for _ in range(epochs):
        opt.zero_grad()
        focal_bce(model(Xt), yt, alpha, gamma).backward()
        opt.step()
    return model


def fold_scaler_into_lr(linear, mean, scale):
    W = linear.fc.weight.detach().numpy().squeeze()
    b = float(linear.fc.bias.detach().numpy().squeeze())
    Wp = W / scale
    bp = b - (mean * Wp).sum()
    return Wp, bp


def build_sklearn_lr(Wp, bp, n):
    lr = LogisticRegression(max_iter=1, class_weight='balanced')
    lr.fit(np.vstack([np.zeros(n), np.ones(n)]), np.array([0, 1]))
    lr.coef_ = Wp.reshape(1, -1).astype(np.float64)
    lr.intercept_ = np.array([bp], dtype=np.float64)
    lr.classes_ = np.array([0, 1])
    return lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--input', nargs='+', default=INPUT_FILES)
    ap.add_argument('--gamma', type=float, default=2.0)
    ap.add_argument('--alpha', type=float, default=None,
                    help='pos-weight; defaults to 1 - pos_rate (class balance)')
    ap.add_argument('--epochs', type=int, default=400)
    args = ap.parse_args()

    records = load_info_gain_records(args.input)
    labels = np.array([int((not r['correct_before']) and r['correct_after']) for r in records])
    pos_rate = float(labels.mean())
    alpha = args.alpha if args.alpha is not None else (1.0 - pos_rate)
    log.info('records=%d pos=%d pos_rate=%.3f alpha=%.3f gamma=%.1f',
             len(records), int(labels.sum()), pos_rate, alpha, args.gamma)

    struct = build_struct_features(records, TOOL_NAMES)
    assert struct.shape[1] == 9

    from sentence_transformers import SentenceTransformer
    texts = [build_trajectory_prefix_from_record(r) for r in records]
    enc = SentenceTransformer('all-MiniLM-L6-v2')
    log.info('encoding %d texts', len(texts))
    emb = np.asarray(enc.encode(texts, batch_size=64, show_progress_bar=True))

    X = np.hstack([emb, struct]).astype(np.float32)
    log.info('X=%s', X.shape)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for i, (tr, te) in enumerate(cv.split(X, labels)):
        mean = X[tr].mean(0); scale = X[tr].std(0) + 1e-8
        Xtr_s = (X[tr] - mean) / scale
        Xte_s = (X[te] - mean) / scale
        m = train_linear(Xtr_s, labels[tr], alpha, args.gamma, args.epochs)
        with torch.no_grad():
            p_te = torch.sigmoid(m(torch.from_numpy(Xte_s).float())).numpy()
        auc = roc_auc_score(labels[te], p_te)
        aucs.append(auc)
        log.info('fold %d AUROC=%.4f', i, auc)
    cv_mean = float(np.mean(aucs)); cv_std = float(np.std(aucs))
    log.info('CV AUROC = %.4f +/- %.4f', cv_mean, cv_std)

    mean = X.mean(0); scale = X.std(0) + 1e-8
    Xs = (X - mean) / scale
    m = train_linear(Xs, labels, alpha, args.gamma, args.epochs)
    Wp, bp = fold_scaler_into_lr(m, mean, scale)
    lr = build_sklearn_lr(Wp, bp, X.shape[1])
    full_auc = roc_auc_score(labels, lr.predict_proba(X)[:, 1])
    log.info('folded full-data AUROC=%.4f', full_auc)

    save = {
        'model': lr,
        'encoder_name': 'all-MiniLM-L6-v2',
        'tool_names': TOOL_NAMES,
        'threshold': 0.5,
        'struct_feature_spec': STRUCT_SPEC,
        'struct_dim': 9,
        'feature_type': 'full',
        'label_type': 'skip_safe',
        'cv_auroc_mean': cv_mean,
        'cv_auroc_std': cv_std,
        'use_extra_features': True,
        'classifier': 'focal_bce',
        'focal_alpha': float(alpha),
        'focal_gamma': float(args.gamma),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(save, out)
    log.info('saved %s', out)


if __name__ == '__main__':
    main()
