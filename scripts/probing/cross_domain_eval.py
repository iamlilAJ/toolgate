#!/usr/bin/env python3
"""Train on probe data, test on eval benchmarks (and vice versa)."""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer


def load_records(*paths):
    records = []
    for path in paths:
        for line in open(path):
            ep = json.loads(line)
            prev_tool = None
            for step in ep.get("trajectory", []):
                ig = step.get("info_gain")
                if ig and ig.get("trajectory_prefix"):
                    tool = step.get("tool_name", "")
                    records.append({
                        "text": ig["trajectory_prefix"],
                        "useful": int(ig["tool_useful"]),
                        "step_idx": step.get("step", 0),
                        "tool_name": tool,
                        "is_repeat": 1 if tool == prev_tool else 0,
                        "thinking_len": len(step.get("thinking", "")),
                    })
                    prev_tool = tool
    return records


probe = load_records("/workspace/skill-guard/data/probe_info_gain/probe_p1_full.jsonl")
eval_paths = {
    "vstar": "/workspace/skill-guard/data/probe_info_gain/vstar_full.jsonl",
    "cvbench": "/workspace/skill-guard/data/probe_info_gain/cvbench_500.jsonl",
    "hr4k": "/workspace/skill-guard/data/probe_info_gain/hrbench4k_full.jsonl",
    "hr8k": "/workspace/skill-guard/data/probe_info_gain/hrbench8k_full.jsonl",
    "mme": "/workspace/skill-guard/data/probe_info_gain/mme_full.jsonl",
}
eval_all = load_records(*eval_paths.values())

n_probe_useful = sum(r["useful"] for r in probe)
n_eval_useful = sum(r["useful"] for r in eval_all)
print(f"Probe: {len(probe)} records, useful={n_probe_useful} ({100*n_probe_useful/len(probe):.0f}%)")
print(f"Eval:  {len(eval_all)} records, useful={n_eval_useful} ({100*n_eval_useful/len(eval_all):.0f}%)")

# Build features
all_recs = probe + eval_all
texts = [r["text"] for r in all_recs]
tool_names = sorted(set(r["tool_name"] for r in all_recs))
tool_idx = {t: i for i, t in enumerate(tool_names)}

extra = np.zeros((len(all_recs), 3 + len(tool_names)))
for i, r in enumerate(all_recs):
    extra[i, 0] = r["step_idx"] / 10.0
    extra[i, 1] = r["is_repeat"]
    extra[i, 2] = r["thinking_len"] / 500.0
    if r["tool_name"] in tool_idx:
        extra[i, 3 + tool_idx[r["tool_name"]]] = 1.0

tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(texts).toarray()
X = np.hstack([X_tfidf, extra])

n_p = len(probe)
X_probe, X_eval = X[:n_p], X[n_p:]
y_probe = np.array([r["useful"] for r in probe])
y_eval = np.array([r["useful"] for r in eval_all])

# Train probe -> Test eval
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(X_probe, y_probe)
auroc = roc_auc_score(y_eval, lr.predict_proba(X_eval)[:, 1])
print(f"\nTrain PROBE -> Test EVAL (all): AUROC={auroc:.3f}")

# Train eval -> Test probe
lr2 = LogisticRegression(max_iter=1000, class_weight="balanced")
lr2.fit(X_eval, y_eval)
auroc2 = roc_auc_score(y_probe, lr2.predict_proba(X_probe)[:, 1])
print(f"Train EVAL -> Test PROBE:       AUROC={auroc2:.3f}")

# Per-benchmark breakdown
print(f"\nTrain PROBE -> Test per-benchmark:")
offset = n_p
for name, path in eval_paths.items():
    recs = load_records(path)
    n = len(recs)
    y = y_eval[offset - n_p:offset - n_p + n] if False else np.array([r["useful"] for r in recs])
    Xt = X[n_p + sum(len(load_records(p)) for p2, p in list(eval_paths.items())[:list(eval_paths.keys()).index(name)]):][:n]
    # Simpler: just recompute
    t = [r["text"] for r in recs]
    e = np.zeros((n, 3 + len(tool_names)))
    for i, r in enumerate(recs):
        e[i, 0] = r["step_idx"] / 10.0
        e[i, 1] = r["is_repeat"]
        e[i, 2] = r["thinking_len"] / 500.0
        if r["tool_name"] in tool_idx:
            e[i, 3 + tool_idx[r["tool_name"]]] = 1.0
    Xt = np.hstack([tfidf.transform(t).toarray(), e])
    yp = lr.predict_proba(Xt)[:, 1]
    y_true = np.array([r["useful"] for r in recs])
    try:
        a = roc_auc_score(y_true, yp)
    except ValueError:
        a = float("nan")
    u = y_true.sum()
    print(f"  {name:10s}: n={n:5d} useful={u:4d} ({100*u/n:.0f}%) AUROC={a:.3f}")
