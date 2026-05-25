#!/usr/bin/env python3
"""Build A-OKVQA probing dataset (MCQ-native, no distractor generation needed).

Usage:
    python scripts/probing/build_aokvqa.py --target 2000 --seed 42 --output-dir data/probe_v3/aokvqa
"""
import argparse
import json
import random
from pathlib import Path
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    print("Loading HuggingFaceM4/A-OKVQA train split...")
    ds = load_dataset("HuggingFaceM4/A-OKVQA", split="train")
    print(f"Train has {len(ds)} samples")

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(ds)), min(args.target, len(ds)))

    letters = ["A", "B", "C", "D"]
    records = []
    errors = 0

    for i, idx in enumerate(indices):
        try:
            s = ds[idx]
            choices = s["choices"]
            correct_idx = s["correct_choice_idx"]
            if len(choices) != 4:
                errors += 1
                continue
            if correct_idx < 0 or correct_idx >= 4:
                errors += 1
                continue

            task_id = f"probe_aokvqa_{i:04d}"
            img_path = out_dir / "images" / f"{task_id}.jpg"

            img = s["image"].convert("RGB")
            img.save(img_path, "JPEG", quality=90)

            options_text = "\n".join([f"({letters[j]}) {choices[j]}" for j in range(4)])
            mcq_text = f"{s['question']}\n{options_text}"

            record = {
                "task_id": task_id,
                "image_path": str(img_path),
                "question": mcq_text,
                "correct_answer": f"({letters[correct_idx]})",
                "source": "aokvqa",
                "query_type": "reasoning",
                "original_question": s["question"],
                "original_answer": choices[correct_idx],
                "question_id": s.get("question_id", ""),
            }
            records.append(record)

            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(indices)} processed")
        except Exception as e:
            errors += 1
            print(f"  error on idx {idx}: {e}")

    raw_path = out_dir / "samples.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone! {len(records)} records saved to {raw_path}")
    print(f"Images: {out_dir / 'images'}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
