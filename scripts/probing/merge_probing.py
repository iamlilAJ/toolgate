#!/usr/bin/env python3
"""Merge annotated probing datasets from all sources.

Usage::

    python scripts/probing/merge_probing.py \
        --sources data/probe_gqa/annotated.jsonl data/probe_textvqa/annotated.jsonl data/probe_rsvlmqa/annotated.jsonl \
        --output data/probe_all.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge probing datasets")
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--output", default="data/probe_all.jsonl")
    args = parser.parse_args()

    all_records = []
    source_counts = {}

    for source_path in args.sources:
        p = Path(source_path)
        if not p.exists():
            print(f"Warning: {source_path} not found, skipping")
            continue

        count = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))
                    count += 1
        source_counts[p.name] = count
        print(f"Loaded {count} from {source_path}")

    # Save merged
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nMerged {len(all_records)} records → {out_path}")

    # Coverage matrix
    matrix = defaultdict(lambda: defaultdict(int))
    for r in all_records:
        ctx = r.get("context", {})
        qt = ctx.get("query_type", r.get("query_type", "unknown"))
        domain = ctx.get("domain", "unknown")
        matrix[qt][domain] += 1

    domains = sorted({d for row in matrix.values() for d in row})
    query_types = sorted(matrix.keys())

    print("\n--- Coverage Matrix (query_type × domain) ---")
    header = f"{'':>20}" + "".join(f"{d:>15}" for d in domains) + f"{'Total':>10}"
    print(header)
    print("-" * len(header))
    for qt in query_types:
        row = f"{qt:>20}"
        total = 0
        for d in domains:
            c = matrix[qt][d]
            row += f"{c:>15}"
            total += c
        row += f"{total:>10}"
        print(row)

    # Source distribution
    print("\n--- Per source ---")
    for name, count in source_counts.items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
