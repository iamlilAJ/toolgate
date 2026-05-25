#!/usr/bin/env python3
"""Build remote sensing probing dataset from LHRS-Bench.

Uses `lipoi/LHRS_Bench` from HuggingFace — already MCQ formatted with
counting, spatial, and attribute questions on remote sensing imagery.

Covers: counting, spatial_relation, attribute × remote_sensing.

Usage::

    python scripts/probing/build_rsvlmqa.py \
        --output-dir data/probe_rsvlmqa \
        --target 500
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


# Map LHRS-Bench question types to our schema
def classify_query_type(question: str, q_type: str = "") -> str:
    """Classify a question into our query_type schema."""
    q_lower = question.lower()

    # Counting indicators
    counting_keywords = ["how many", "count", "number of", "total number"]
    if any(kw in q_lower for kw in counting_keywords):
        return "counting"

    # Spatial indicators
    spatial_keywords = [
        "where", "position", "located", "location", "direction",
        "left", "right", "above", "below", "next to", "near",
        "relative", "between", "surrounding",
    ]
    if any(kw in q_lower for kw in spatial_keywords):
        return "spatial_relation"

    # Default to attribute (color, shape, type, material, etc.)
    return "attribute"


def main():
    parser = argparse.ArgumentParser(description="Build remote sensing probing dataset")
    parser.add_argument("--output-dir", type=str, default="data/probe_rsvlmqa")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    print("Loading LHRS-Bench dataset...")
    ds = load_dataset("lipoi/LHRS_Bench", split="val")
    print(f"Total LHRS-Bench: {len(ds)} samples")

    # Explore structure
    ex = ds[0]
    print(f"Fields: {list(ex.keys())}")
    print(f"Sample: question={ex.get('question', '')[:80]}, answer={ex.get('answer', '')}, type={ex.get('type', '')}")

    records = []
    type_counts = defaultdict(int)

    for idx in range(len(ds)):
        ex = ds[idx]
        question = ex.get("question", "")
        answer = ex.get("answer", "")
        choices = ex.get("choices", [])
        image = ex.get("image")
        q_type = ex.get("type", "")

        if not question or not answer or not image:
            continue

        # Classify query type
        query_type = classify_query_type(question, q_type)
        type_counts[query_type] += 1

        # Parse LHRS-Bench choices format: "A. Factory; B. Park; C. Farm; D. Forest."
        import re
        if not choices or not isinstance(choices, str):
            continue

        # Split by semicolon and parse "X. text" pattern
        parts = [p.strip().rstrip(".") for p in choices.split(";")]
        option_lines = []
        parsed_options = {}
        for part in parts:
            match = re.match(r"^([A-F])\.\s*(.+)$", part)
            if match:
                letter = match.group(1)
                text = match.group(2)
                option_lines.append(f"({letter}) {text}")
                parsed_options[letter] = text

        if not option_lines:
            continue

        # Get correct answer letter from "C." format
        ans_letter = answer.strip().rstrip(".").upper()
        if ans_letter not in parsed_options:
            continue
        correct_option = f"({ans_letter})"

        mcq_text = question + "\n" + "\n".join(option_lines)

        # Save image (convert RGBA to RGB for JPEG)
        img_filename = f"lhrs_{idx}.jpg"
        img_path = out_dir / "images" / img_filename
        if not img_path.exists():
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image.save(img_path, "JPEG")

        record = {
            "task_id": f"probe_rsvlmqa_{idx}",
            "image_path": str(img_path),
            "question": mcq_text,
            "correct_answer": correct_option,
            "source": "lhrs_bench",
            "original_id": str(ex.get("question_id", idx)),
            "query_type": query_type,
            "original_answer": answer,
            "original_question": question,
            "original_type": q_type,
        }
        records.append(record)

    print(f"\nProcessed {len(records)} valid MCQ records")
    print("Query type distribution:")
    for qt, c in sorted(type_counts.items()):
        print(f"  {qt}: {c}")

    # Subsample if needed
    if len(records) > args.target:
        # Balance by query type
        by_type = defaultdict(list)
        for r in records:
            by_type[r["query_type"]].append(r)

        per_type = args.target // len(by_type)
        selected = []
        for qt, recs in by_type.items():
            rng.shuffle(recs)
            selected.extend(recs[:per_type])

        # Fill remaining with random from leftovers
        remaining = args.target - len(selected)
        if remaining > 0:
            leftovers = [r for r in records if r not in selected]
            rng.shuffle(leftovers)
            selected.extend(leftovers[:remaining])

        records = selected
        print(f"\nSubsampled to {len(records)} records")

    # Save
    raw_path = out_dir / "raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(records)} records to {raw_path}")

    # Final stats
    final_counts = defaultdict(int)
    for r in records:
        final_counts[r["query_type"]] += 1
    print("Final query type distribution:")
    for qt, c in sorted(final_counts.items()):
        print(f"  {qt}: {c}")


if __name__ == "__main__":
    main()
