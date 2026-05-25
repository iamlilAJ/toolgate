#!/usr/bin/env python3
"""Aggregate benchmark results into acc/tools/tokens metrics table.

Pulls from:
  - judged_by_235b.json : 235B VL judged is_valid / is_correct / tool_usage
  - trajectories/*.jsonl: per-episode total_input_tokens / total_output_tokens

Produces:
  - verified_metrics.json with {cv_auroc, deployment_results}
  - markdown table of acc / tools / tokens ratio per (run, benchmark)

Usage:
    python compute_metrics.py --output verified_metrics.json
    python compute_metrics.py --print-table

Assumes CWD is the cv-agent root (or adjust RESULTS_ROOT below).
"""
import argparse
import json
import glob
import os
from collections import OrderedDict

RESULTS_ROOT = 'results'


def stats_from_judged(judged_file):
    """Parse judged_by_235b.json -> acc + tools."""
    if not os.path.exists(judged_file):
        return None
    d = json.load(open(judged_file))
    n = len(d)
    n_valid = sum(1 for r in d if r.get('is_valid'))
    n_correct = sum(1 for r in d if r.get('is_correct'))
    tools_total = sum(sum(r.get('tool_usage', {}).values()) for r in d)
    return {
        'n': n,
        'n_valid': n_valid,
        'n_correct': n_correct,
        'valid_acc': round(n_correct / n_valid * 100 if n_valid else 0, 2),
        'overall_acc': round(n_correct / n * 100 if n else 0, 2),
        'tools_per_sample': round(tools_total / n, 3) if n else 0,
    }


def avg_tokens_per_sample(run_dir):
    """Sum total_input_tokens+total_output_tokens from trajectories, div by episode count."""
    traj_dir = os.path.join(run_dir, 'trajectories')
    if not os.path.isdir(traj_dir):
        return None
    tot_in = tot_out = n = 0
    for f in glob.glob(os.path.join(traj_dir, 'trajectories_worker_*.jsonl')):
        for line in open(f):
            if not line.strip():
                continue
            ep = json.loads(line)
            tot_in += ep.get('total_input_tokens', 0)
            tot_out += ep.get('total_output_tokens', 0)
            n += 1
    return (tot_in + tot_out) // n if n else None


def resolve_latest(pattern):
    """Glob + pick lexically latest matching dir."""
    paths = sorted(glob.glob(pattern))
    return paths[-1] if paths else None


# ------------------------------------------------------------------
# Manifest of all runs to aggregate.
# Keyed by run name. Each value maps benchmark -> run_dir pattern.
# Glob '*' is resolved to the lexically latest match.
# ------------------------------------------------------------------

RUNS = OrderedDict([
    ('baseline_no_probe', {
        'cvbench':    f'{RESULTS_ROOT}/qwen-30b/agent/cvbench/20260412_045608',
        'vstar':      f'{RESULTS_ROOT}/qwen-30b/agent/vstar/20260412_050242',
        'hrbench-4k': f'{RESULTS_ROOT}/qwen-30b/agent/hrbench-4k/20260412_082600',
        'hrbench-8k': f'{RESULTS_ROOT}/qwen-30b-ep2/agent/hrbench-8k/20260412_082602',
        'mme':        f'{RESULTS_ROOT}/qwen-30b/agent/mme/20260412_081519',
    }),
    ('probe_v2_crossdomain', {
        'cvbench':    f'{RESULTS_ROOT}/qwen-30b-probe-crossdomain/agent/cvbench/20260412_171015',
        'vstar':      f'{RESULTS_ROOT}/qwen-30b-probe-crossdomain/agent/vstar/20260412_171015',
        'hrbench-4k': f'{RESULTS_ROOT}/qwen-30b-probe-crossdomain/agent/hrbench-4k/20260412_171019',
        'hrbench-8k': f'{RESULTS_ROOT}/qwen-30b-probe-crossdomain-ep1/agent/hrbench-8k/20260412_171020',
        'mme':        f'{RESULTS_ROOT}/qwen-30b-probe-crossdomain-ep1/agent/mme/20260412_171023',
    }),
    ('probe_v3_t50', {
        'cvbench':    f'{RESULTS_ROOT}/qwen-30b-probe-cross-pv3/agent/cvbench/20260414_090851',
        'vstar':      f'{RESULTS_ROOT}/qwen-30b-probe-cross-pv3/agent/vstar/20260414_090854',
        'hrbench-4k': f'{RESULTS_ROOT}/qwen-30b-probe-cross-pv3/agent/hrbench-4k/20260414_090857',
        'hrbench-8k': f'{RESULTS_ROOT}/qwen-30b-probe-cross-pv3-ep1/agent/hrbench-8k/20260414_090900',
        'mme':        f'{RESULTS_ROOT}/qwen-30b-probe-cross-pv3-ep1/agent/mme/20260414_090904',
    }),
    ('probe_v3_t60', {
        'cvbench':    f'{RESULTS_ROOT}/qwen-30b-probe-cross-pv3-t60/agent/cvbench/*',
        'vstar':      f'{RESULTS_ROOT}/qwen-30b-probe-cross-pv3-t60/agent/vstar/*',
        'hrbench-4k': f'{RESULTS_ROOT}/qwen-30b-probe-cross-pv3-t60/agent/hrbench-4k/*',
        'hrbench-8k': f'{RESULTS_ROOT}/qwen-30b-probe-cross-pv3-t60-ep1/agent/hrbench-8k/*',
        'mme':        f'{RESULTS_ROOT}/qwen-30b-probe-cross-pv3-t60-ep1/agent/mme/*',
    }),
    ('probe_v3_t70', {
        'cvbench':    f'{RESULTS_ROOT}/qwen-30b-probe-cross-pv3-t70/agent/cvbench/*',
        'vstar':      f'{RESULTS_ROOT}/qwen-30b-probe-cross-pv3-t70/agent/vstar/*',
        'hrbench-4k': f'{RESULTS_ROOT}/qwen-30b-probe-cross-pv3-t70/agent/hrbench-4k/*',
        'hrbench-8k': f'{RESULTS_ROOT}/qwen-30b-probe-cross-pv3-t70-ep1/agent/hrbench-8k/*',
        'mme':        f'{RESULTS_ROOT}/qwen-30b-probe-cross-pv3-t70-ep1/agent/mme/*',
    }),
    ('probe_v3_vstar_max2', {
        'vstar': f'{RESULTS_ROOT}/qwen-30b-probe-cross-pv3/agent/vstar/20260414_232931',
    }),
    ('probe_v3_vstar_max3', {
        'vstar': f'{RESULTS_ROOT}/qwen-30b-probe-cross-pv3/agent/vstar/20260414_232933',
    }),
    ('probe_v3_qwen_frozen', {  # running as of 2026-04-15, may be partial
        'cvbench':    f'{RESULTS_ROOT}/qwen-30b-probe-qwen-frozen/agent/cvbench/*',
        'vstar':      f'{RESULTS_ROOT}/qwen-30b-probe-qwen-frozen/agent/vstar/*',
        'hrbench-4k': f'{RESULTS_ROOT}/qwen-30b-probe-qwen-frozen/agent/hrbench-4k/*',
        'hrbench-8k': f'{RESULTS_ROOT}/qwen-30b-probe-qwen-frozen-ep1/agent/hrbench-8k/*',
        'mme':        f'{RESULTS_ROOT}/qwen-30b-probe-qwen-frozen-ep1/agent/mme/*',
    }),
])


# CV AUROC from training logs (scripts/probing/train_*.py stdout)
CV_AUROC = OrderedDict([
    # (model_name_for_reference, CV_AUROC_logistic)
    ('probe_v2 (probe_crossdomain.joblib)', 0.669),
    ('probe_v3 (probe_v3.joblib)', 0.700),
    ('probe_v3_qwen_frozen (Qwen3-Emb-0.6B frozen + LR)', 0.703),
    ('probe_v3_ctx (+ image_res + question_type, 18d struct)', 0.699),
    ('probe_v3_imgdelta (+ DINOv2-small 5d delta)', 0.703),
    ('probe_v3_pixeldelta (+ edge/pHash/hist/size 5d)', 0.704),
    ('probe_v3_qwen_lora (LoRA rank 16, 2ep)', 0.671),
    ('probe_v3_qwen_full (full FT, 2ep, bs=4)', 0.660),
])


def aggregate():
    out = {}
    for run_name, benches in RUNS.items():
        out[run_name] = {}
        for bench, pattern in benches.items():
            run_dir = resolve_latest(pattern) if '*' in pattern else pattern
            if not run_dir or not os.path.isdir(run_dir):
                continue
            s = stats_from_judged(os.path.join(run_dir, 'judged_by_235b.json'))
            if s is None:
                continue
            s['tokens_per_sample'] = avg_tokens_per_sample(run_dir)
            s['run_dir'] = run_dir
            out[run_name][bench] = s

    # Normalize token ratio to baseline
    base = out.get('baseline_no_probe', {})
    base_tok = {b: base[b]['tokens_per_sample'] for b in base if base[b].get('tokens_per_sample')}
    for run_name, benches in out.items():
        for bench, s in benches.items():
            if s.get('tokens_per_sample') and bench in base_tok and base_tok[bench]:
                s['token_ratio_vs_baseline'] = round(s['tokens_per_sample'] / base_tok[bench], 3)

    return {'cv_auroc': dict(CV_AUROC), 'deployment_results': out}


def print_table(data):
    """Pretty-print main results table."""
    benches = ['cvbench', 'vstar', 'hrbench-4k', 'hrbench-8k', 'mme']
    header = ['run', 'metric'] + [b[:10] for b in benches] + ['AVG']
    print(f'{header[0]:28s} {header[1]:14s}' + ''.join(f'{c:>14s}' for c in header[2:]))
    print('-' * 150)
    for run_name, benches_stats in data['deployment_results'].items():
        accs = []
        tools = []
        ratios = []
        for b in benches:
            s = benches_stats.get(b)
            if s is None:
                accs.append(None)
                tools.append(None)
                ratios.append(None)
            else:
                accs.append(s['valid_acc'])
                tools.append(s['tools_per_sample'])
                ratios.append(s.get('token_ratio_vs_baseline'))

        def fmt(vals, fstr, na='—'):
            return ''.join(f'{v:>14{fstr[-1]}}'.replace('nanf','N/A') if isinstance(v,(int,float))
                           else f'{na:>14}' for v in vals)

        def fmt_pct(vals):
            return ''.join(f'{v:>13.2f}%' if isinstance(v,(int,float)) else f'{"—":>14}' for v in vals)

        def fmt_f(vals, digits=2):
            return ''.join(f'{v:>14.{digits}f}' if isinstance(v,(int,float)) else f'{"—":>14}' for v in vals)

        valid_accs = [a for a in accs if a is not None]
        valid_tools = [t for t in tools if t is not None]
        valid_ratios = [r for r in ratios if r is not None]
        avg_acc = sum(valid_accs) / len(valid_accs) if valid_accs else None
        avg_tools = sum(valid_tools) / len(valid_tools) if valid_tools else None
        avg_ratio = sum(valid_ratios) / len(valid_ratios) if valid_ratios else None

        print(f'{run_name:28s} {"acc%":14s}' + fmt_pct(accs) +
              (f'{avg_acc:>13.2f}%' if avg_acc else f'{"—":>14}'))
        print(f'{"":28s} {"tools/sample":14s}' + fmt_f(tools, 2) +
              (f'{avg_tools:>14.2f}' if avg_tools else f'{"—":>14}'))
        print(f'{"":28s} {"token_ratio":14s}' + fmt_f(ratios, 2) +
              (f'{avg_ratio:>14.2f}' if avg_ratio else f'{"—":>14}'))
        print()


def main():
    global RESULTS_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='verified_metrics.json')
    ap.add_argument('--print-table', action='store_true')
    ap.add_argument('--results-root', default=RESULTS_ROOT)
    args = ap.parse_args()
    RESULTS_ROOT = args.results_root

    data = aggregate()

    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Saved {args.output}')

    if args.print_table:
        print_table(data)


if __name__ == '__main__':
    main()
