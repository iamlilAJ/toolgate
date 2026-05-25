#!/usr/bin/env python3
"""Build 50/50 train/test splits of task_ids per benchmark.

Seed=42, over UNIQUE task_ids (not records) so each question goes to one split.
"""
import argparse, json, os, random, sys
sys.path.insert(0, '/workspace/cv-agent/scripts/probing')

BENCHES = ['cvbench_500','hrbench4k_full','hrbench8k_full','mme_full','vstar_full']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/workspace/skill-guard/data/probe_info_gain')
    parser.add_argument('--out-dir', default='/workspace/skill-guard/splits')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for bench in BENCHES:
        path = os.path.join(args.data_dir, f'{bench}.jsonl')
        task_ids = set()
        for line in open(path):
            if not line.strip(): continue
            ep = json.loads(line)
            task_ids.add(ep['task_id'])
        task_ids = sorted(task_ids)
        rng = random.Random(args.seed)
        rng.shuffle(task_ids)
        n = len(task_ids)
        split = n // 2
        train = task_ids[:split]
        test = task_ids[split:]
        short = bench.replace('_full','').replace('_500','')
        with open(f'{args.out_dir}/{short}_train.txt','w') as f:
            f.write('\n'.join(train)+'\n')
        with open(f'{args.out_dir}/{short}_test.txt','w') as f:
            f.write('\n'.join(test)+'\n')
        print(f'{short:12s} total={n:5d}  train={len(train):5d}  test={len(test):5d}')
        # sanity: disjoint union
        assert set(train).isdisjoint(set(test))
        assert set(train) | set(test) == set(task_ids)

if __name__ == '__main__':
    main()
