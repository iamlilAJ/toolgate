#!/usr/bin/env python3
"""Build GQA probing dataset: download, filter, convert to MCQ.

Covers: counting, spatial_relation, attribute × natural_scene/indoor.

Usage::

    python scripts/probing/build_gqa.py \
        --output-dir data/probe_gqa \
        --target-per-type 400 \
        --vlm-endpoint http://localhost:10010/v1
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

# Mapping GQA types.detailed → our query_type
# Based on actual GQA testdev_balanced types
DETAIL_TO_QUERY_TYPE = {
    # Spatial relation types
    "relS": "spatial_relation",
    "relO": "spatial_relation",
    "categoryRelS": "spatial_relation",
    "categoryRelO": "spatial_relation",
    "relVerify": "spatial_relation",
    "relVerifyCo": "spatial_relation",
    "relChooser": "spatial_relation",
    "directOf": "spatial_relation",
    "directWhich": "spatial_relation",
    "positionQuery": "spatial_relation",
    "positionChoose": "spatial_relation",
    "positionVerify": "spatial_relation",
    "positionVerifyC": "spatial_relation",
    "existRelS": "spatial_relation",
    "existRelSC": "spatial_relation",
    # Counting types
    "how": "counting",
    # Attribute types
    "verifyAttrsC": "attribute",
    "verifyAttrs": "attribute",
    "chooseAttr": "attribute",
    "verifyAttrK": "attribute",
    "verifyAttrKC": "attribute",
    "verifyAttr": "attribute",
    "verifyAttrC": "attribute",
    "material": "attribute",
    "categoryAttr": "attribute",
    "twoSameMaterialC": "attribute",
    "twoCommon": "attribute",
    "twoSame": "attribute",
    "twoSameC": "attribute",
    "twoDifferentC": "attribute",
    "verifyAttrCThis": "attribute",
    "verifyMaterialAnd": "attribute",
    "existAttrNot": "attribute",
    "existAttrC": "attribute",
    "existAttrOrC": "attribute",
    "existAttrOr": "attribute",
}


def _call_vlm(endpoint: str, api_key: str, model: str, prompt: str, image=None, max_tokens: int = 256) -> str:
    """Call VLM endpoint for distractor generation."""
    import requests
    import base64
    import io

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


def generate_distractors(
    question: str,
    answer: str,
    query_type: str,
    structural_type: str,
    endpoint: str,
    api_key: str,
    model: str,
    image=None,
    num_distractors: int = 3,
) -> list[str]:
    """Generate plausible MCQ distractors using VLM."""

    if structural_type == "verify":
        # Yes/No questions — generate plausible alternative descriptions
        prompt = (
            f"This is a visual question: '{question}' with correct answer: '{answer}'.\n"
            f"Generate {num_distractors} plausible but incorrect alternative answers for a multiple choice question. "
            f"Each alternative should be a short phrase that someone might reasonably guess. "
            f"Return ONLY a JSON array of strings, no markdown: [\"alt1\", \"alt2\", \"alt3\"]"
        )
    elif structural_type == "choose":
        prompt = (
            f"This is a visual question: '{question}' with correct answer: '{answer}'.\n"
            f"Generate {num_distractors} more plausible but incorrect alternatives. "
            f"They should be the same type as the correct answer (e.g., if answer is a color, distractors should be colors). "
            f"Return ONLY a JSON array of strings: [\"alt1\", \"alt2\", \"alt3\"]"
        )
    else:  # query
        if query_type == "counting":
            # For counting, generate nearby numbers
            try:
                n = int(answer)
                candidates = [str(x) for x in range(max(0, n - 3), n + 4) if x != n]
                random.shuffle(candidates)
                return candidates[:num_distractors]
            except ValueError:
                pass

        prompt = (
            f"This is a visual question: '{question}' with correct answer: '{answer}'.\n"
            f"Generate {num_distractors} plausible but incorrect alternatives of the same type. "
            f"Return ONLY a JSON array of strings: [\"alt1\", \"alt2\", \"alt3\"]"
        )

    try:
        response = _call_vlm(endpoint, api_key, model, prompt, image=image)
        # Try to parse JSON from response
        # Strip markdown fences if present
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        distractors = json.loads(text)
        if isinstance(distractors, list) and len(distractors) >= num_distractors:
            return [str(d) for d in distractors[:num_distractors]]
    except Exception as e:
        print(f"  VLM distractor generation failed: {e}", file=sys.stderr)

    # Fallback: generic distractors
    if query_type == "counting":
        return ["0", "1", "2"][:num_distractors]
    elif query_type == "attribute":
        return ["unknown", "other", "different"][:num_distractors]
    else:
        return ["none of the above", "cannot determine", "opposite"][:num_distractors]


def format_mcq(question: str, correct_answer: str, distractors: list[str], rng: random.Random) -> tuple[str, str]:
    """Format as MCQ with randomized option order. Returns (mcq_text, correct_option)."""
    options = [correct_answer] + distractors
    rng.shuffle(options)
    correct_idx = options.index(correct_answer)
    labels = ["(A)", "(B)", "(C)", "(D)"]

    option_lines = [f"{labels[i]} {options[i]}" for i in range(len(options))]
    mcq_text = question + "\n" + "\n".join(option_lines)
    return mcq_text, labels[correct_idx]


def main():
    parser = argparse.ArgumentParser(description="Build GQA probing dataset")
    parser.add_argument("--output-dir", type=str, default="data/probe_gqa")
    parser.add_argument("--target-per-type", type=int, default=400)
    parser.add_argument("--vlm-endpoint", type=str, default="http://localhost:10010/v1")
    parser.add_argument("--vlm-model", type=str, default="Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--vlm-api-key", type=str, default="anything")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    print("Loading GQA dataset (instructions for filtering)...")
    ds_instructions = load_dataset(
        "lmms-lab/GQA", "testdev_balanced_instructions", split="testdev"
    )

    # Phase 1: Filter by query_type
    print(f"Total GQA testdev_balanced: {len(ds_instructions)} questions")

    type_buckets: dict[str, list] = defaultdict(list)
    type_counts: dict[str, int] = defaultdict(int)
    skipped_types: dict[str, int] = defaultdict(int)

    for idx, ex in enumerate(ds_instructions):
        detailed = ex["types"]["detailed"]
        qt = DETAIL_TO_QUERY_TYPE.get(detailed)
        if qt is None:
            skipped_types[detailed] = skipped_types.get(detailed, 0) + 1
            continue
        type_counts[qt] += 1
        type_buckets[qt].append(idx)

    print(f"\nFiltered by query_type:")
    for qt, count in sorted(type_counts.items()):
        print(f"  {qt}: {count} questions")
    print(f"  skipped: {sum(skipped_types.values())} (types: {len(skipped_types)})")

    # Phase 2: Subsample
    selected_indices = []
    for qt in ["counting", "spatial_relation", "attribute"]:
        indices = type_buckets[qt]
        rng.shuffle(indices)
        selected = indices[: args.target_per_type]
        if len(selected) < args.target_per_type:
            print(f"  WARNING: {qt} has only {len(indices)} candidates, requested {args.target_per_type}")
        selected_indices.extend(selected)
        print(f"  Selected {len(selected)} for {qt}")

    print(f"\nTotal selected: {len(selected_indices)}")

    # Phase 3: Load images and convert to MCQ
    print("\nLoading GQA images dataset...")
    ds_images = load_dataset(
        "lmms-lab/GQA", "testdev_balanced_images", split="testdev"
    )
    # Build imageId → index mapping
    print("Building image index...")
    image_id_to_idx = {}
    for i, ex in enumerate(ds_images):
        image_id_to_idx[ex["id"]] = i

    print(f"Image index built: {len(image_id_to_idx)} entries")

    records = []
    errors = 0

    for i, idx in enumerate(selected_indices):
        ex = ds_instructions[idx]
        detailed = ex["types"]["detailed"]
        qt = DETAIL_TO_QUERY_TYPE[detailed]
        structural = ex["types"]["structural"]

        if (i + 1) % 50 == 0:
            print(f"  Processing {i + 1}/{len(selected_indices)}...")

        # Get image
        image_id = ex["imageId"]
        img_idx = image_id_to_idx.get(image_id)
        if img_idx is None:
            errors += 1
            continue

        image = ds_images[img_idx]["image"]

        # Save image locally
        img_filename = f"{image_id}.jpg"
        img_path = out_dir / "images" / img_filename
        if not img_path.exists():
            image.save(img_path, "JPEG")

        # Generate distractors
        distractors = generate_distractors(
            question=ex["question"],
            answer=ex["answer"],
            query_type=qt,
            structural_type=structural,
            endpoint=args.vlm_endpoint,
            api_key=args.vlm_api_key,
            model=args.vlm_model,
            image=image,
        )

        # Format MCQ
        mcq_text, correct_option = format_mcq(ex["question"], ex["answer"], distractors, rng)

        record = {
            "task_id": f"probe_gqa_{ex['id']}",
            "image_path": str(img_path),
            "question": mcq_text,
            "correct_answer": correct_option,
            "source": "gqa",
            "original_id": ex["id"],
            "query_type": qt,
            "original_answer": ex["answer"],
            "original_question": ex["question"],
            "structural_type": structural,
            "detailed_type": detailed,
        }
        records.append(record)

    # Save
    raw_path = out_dir / "raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone! Saved {len(records)} records to {raw_path}")
    print(f"Errors (missing images): {errors}")

    # Summary
    qt_counts = defaultdict(int)
    for r in records:
        qt_counts[r["query_type"]] += 1
    print("\nPer query_type:")
    for qt, c in sorted(qt_counts.items()):
        print(f"  {qt}: {c}")


if __name__ == "__main__":
    main()
