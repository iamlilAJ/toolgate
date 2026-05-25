#!/usr/bin/env python3
"""Build DocVQA probing dataset: short-answer filter + VLM distractor MCQ."""
import argparse
import json
import random
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset


def call_vlm(endpoint: str, api_key: str, model: str, prompt: str, timeout: int = 30) -> str:
    resp = requests.post(
        f"{endpoint}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.2,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def generate_distractors(question: str, answer: str, endpoint: str, api_key: str, model: str, n: int = 3) -> list[str]:
    prompt = (
        f"This is a question about a DOCUMENT image: '{question}' "
        f"with correct answer: '{answer}'.\n"
        f"Generate {n} plausible but INCORRECT alternative answers for a multiple-choice version. "
        f"Keep each under 30 characters, same format as the correct answer "
        f"(number if correct is a number, short phrase otherwise). "
        f'Return ONLY a JSON array of {n} strings: ["alt1","alt2","alt3"]'
    )
    for _ in range(3):
        try:
            text = call_vlm(endpoint, api_key, model, prompt).strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            arr = json.loads(text)
            if isinstance(arr, list) and len(arr) >= n:
                cleaned = [str(x)[:40] for x in arr[:n] if str(x) and str(x) != answer]
                if len(cleaned) >= n:
                    return cleaned[:n]
        except Exception:
            pass
    # fallback
    return ["N/A", "not specified", "unknown"][:n]


def format_mcq(question: str, correct: str, distractors: list[str], rng: random.Random) -> tuple[str, str]:
    options = [correct] + distractors
    rng.shuffle(options)
    idx = options.index(correct)
    letters = ["A", "B", "C", "D"]
    opts_text = "\n".join([f"({letters[j]}) {options[j]}" for j in range(len(options))])
    return f"{question}\n{opts_text}", f"({letters[idx]})"


def process_one(args_tuple):
    i, sample, out_dir, endpoint, api_key, model, seed = args_tuple
    try:
        task_id = f"probe_docvqa_{i:04d}"
        img_path = out_dir / "images" / f"{task_id}.jpg"
        img = sample["image"].convert("RGB")
        img.save(img_path, "JPEG", quality=88)
        rng = random.Random(seed + i)
        answer = sample["answers"][0] if sample.get("answers") else ""
        distractors = generate_distractors(sample["question"], answer, endpoint, api_key, model)
        mcq_text, correct_opt = format_mcq(sample["question"], answer, distractors, rng)
        return {
            "task_id": task_id,
            "image_path": str(img_path),
            "question": mcq_text,
            "correct_answer": correct_opt,
            "source": "docvqa",
            "query_type": "reading_ocr",
            "original_question": sample["question"],
            "original_answer": answer,
            "question_types": sample.get("question_types", []),
        }
    except Exception as e:
        print(f"  error i={i}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--vlm-endpoint", default="http://localhost:52308/v1")
    parser.add_argument("--vlm-api-key", default="anything")
    parser.add_argument("--vlm-model", default="Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    print("Loading lmms-lab/DocVQA validation split...")
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    print(f"Total: {len(ds)} samples")

    # Filter: short answers (< 30 chars, non-empty)
    candidate_indices = []
    for i, s in enumerate(ds):
        if s.get("answers"):
            a = s["answers"][0]
            if 0 < len(a) < 30:
                candidate_indices.append(i)
    print(f"With short answers: {len(candidate_indices)}")

    rng = random.Random(args.seed)
    rng.shuffle(candidate_indices)
    chosen = candidate_indices[: args.target]
    print(f"Sampling {len(chosen)} for MCQ conversion...")

    records = [None] * len(chosen)
    work_items = [(i, ds[idx], out_dir, args.vlm_endpoint, args.vlm_api_key, args.vlm_model, args.seed)
                  for i, idx in enumerate(chosen)]

    done = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, w): w[0] for w in work_items}
        for f in as_completed(futs):
            i = futs[f]
            r = f.result()
            if r is None:
                errors += 1
            else:
                records[i] = r
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(chosen)}  errors={errors}")

    records = [r for r in records if r is not None]
    raw_path = out_dir / "samples.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nDone! {len(records)} records saved to {raw_path}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
