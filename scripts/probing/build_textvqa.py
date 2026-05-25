#!/usr/bin/env python3
"""Build TextVQA probing dataset: download, filter, convert to MCQ.

Covers: reading_ocr × natural_scene.

Usage::

    python scripts/probing/build_textvqa.py \
        --output-dir data/probe_textvqa \
        --target 500 \
        --vlm-endpoint http://localhost:10010/v1
"""

import argparse
import base64
import io
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset


def _call_vlm(endpoint: str, api_key: str, model: str, prompt: str, image=None, max_tokens: int = 256) -> str:
    """Call VLM endpoint."""
    import requests

    messages = []
    if image is not None:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        })
    else:
        messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        f"{endpoint}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.7},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def get_consensus_answer(answers: list[str]) -> str | None:
    """Get the most common answer if it has ≥ 3 votes (out of 10)."""
    counter = Counter(a.strip().lower() for a in answers)
    most_common, count = counter.most_common(1)[0]
    if count >= 3 and len(most_common) <= 30:  # Short, consensus answer
        # Return the original casing of the most common
        for a in answers:
            if a.strip().lower() == most_common:
                return a.strip()
    return None


def format_mcq(question: str, correct_answer: str, distractors: list[str], rng: random.Random) -> tuple[str, str]:
    """Format as MCQ with randomized option order."""
    options = [correct_answer] + distractors[:3]
    rng.shuffle(options)
    correct_idx = options.index(correct_answer)
    labels = ["(A)", "(B)", "(C)", "(D)"]
    option_lines = [f"{labels[i]} {options[i]}" for i in range(len(options))]
    mcq_text = question + "\n" + "\n".join(option_lines)
    return mcq_text, labels[correct_idx]


def main():
    parser = argparse.ArgumentParser(description="Build TextVQA probing dataset")
    parser.add_argument("--output-dir", type=str, default="data/probe_textvqa")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--vlm-endpoint", type=str, default="http://localhost:10010/v1")
    parser.add_argument("--vlm-model", type=str, default="Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--vlm-api-key", type=str, default="anything")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    print("Loading TextVQA dataset (streaming)...")
    ds = load_dataset("facebook/textvqa", split="validation", streaming=True, trust_remote_code=True)

    # Phase 1: Stream and filter — keep questions with short, consensus answers
    candidates = []
    scanned = 0
    for ex in ds:
        scanned += 1
        answer = get_consensus_answer(ex["answers"])
        if answer is None:
            continue
        if len(answer) < 2 or len(answer) > 25:
            continue
        candidates.append((ex, answer))
        if scanned % 1000 == 0:
            print(f"  Scanned {scanned}, found {len(candidates)} candidates...")
        # Collect enough candidates (2x target for safety)
        if len(candidates) >= args.target * 2:
            break

    print(f"Scanned {scanned}, found {len(candidates)} candidates with consensus answers")

    # Phase 2: Subsample
    rng.shuffle(candidates)
    selected = candidates[: args.target]
    print(f"Selected {len(selected)} for MCQ conversion")

    # Phase 3: Convert to MCQ with VLM distractors
    records = []
    errors = 0

    for i, (ex, answer) in enumerate(selected):
        if (i + 1) % 50 == 0:
            print(f"  Processing {i + 1}/{len(selected)}...")

        image = ex["image"]
        img_filename = f"{ex['image_id']}.jpg"
        img_path = out_dir / "images" / img_filename
        if not img_path.exists():
            image.save(img_path, "JPEG")

        # Generate distractors via VLM
        prompt = (
            f"Given this image, the question '{ex['question']}' has the correct answer '{answer}'. "
            f"Generate 3 plausible but INCORRECT alternatives that someone might confuse with the actual answer. "
            f"Consider OCR-like errors, similar-looking words, or other text visible in the image. "
            f"Return ONLY a JSON array: [\"alt1\", \"alt2\", \"alt3\"]"
        )

        try:
            response = _call_vlm(args.vlm_endpoint, args.vlm_api_key, args.vlm_model, prompt, image=image)
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            distractors = json.loads(text)
            if not isinstance(distractors, list) or len(distractors) < 3:
                raise ValueError(f"Got {distractors}")
            distractors = [str(d) for d in distractors[:3]]
        except Exception as e:
            errors += 1
            distractors = [f"not {answer}", "unclear", "other text"]

        mcq_text, correct_option = format_mcq(ex["question"], answer, distractors, rng)

        record = {
            "task_id": f"probe_textvqa_{ex['question_id']}",
            "image_path": str(img_path),
            "question": mcq_text,
            "correct_answer": correct_option,
            "source": "textvqa",
            "original_id": str(ex["question_id"]),
            "query_type": "reading_ocr",
            "original_answer": answer,
            "original_question": ex["question"],
        }
        records.append(record)

    # Save
    raw_path = out_dir / "raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone! Saved {len(records)} records to {raw_path}")
    print(f"VLM distractor errors: {errors}")


if __name__ == "__main__":
    main()
