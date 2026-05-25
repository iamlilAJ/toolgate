#!/usr/bin/env python3
"""Build surveillance/driving probing dataset from SEED-Bench-2.

SEED-Bench-2 already has MCQ format. We filter for outdoor/driving/surveillance
scenes and run unified ξ annotation to select samples that fill the gap in
our probing pool (surveillance_traffic_cam, driving_road_view).

Usage::

    python scripts/probing/build_seedbench.py \
        --output-dir data/probe_seedbench \
        --target 500
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
from PIL import Image, ImageFile

# Handle large images and truncated files
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _call_vlm_scene(endpoint: str, api_key: str, model: str, image) -> str:
    """Quick VLM call to classify scene type."""
    import requests

    buf = io.BytesIO()
    if image.mode == "RGBA":
        image = image.convert("RGB")
    # Resize for speed
    max_dim = 512
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        image = image.resize((int(image.width * ratio), int(image.height * ratio)), Image.LANCZOS)
    image.save(buf, format="JPEG", quality=75)
    b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = (
        'Classify this image scene. Reply with ONLY one of: '
        'indoor_room, outdoor_street, outdoor_park_nature, aerial_satellite, '
        'surveillance_traffic_cam, driving_road_view, document_chart_table, '
        'sports_event, food_product, shop_storefront, construction_industrial'
    )

    resp = requests.post(
        f"{endpoint}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}],
            "max_tokens": 30,
            "temperature": 0.0,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip().lower()


def main():
    parser = argparse.ArgumentParser(description="Build SEED-Bench-2 probing dataset")
    parser.add_argument("--output-dir", type=str, default="data/probe_seedbench")
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

    print("Loading SEED-Bench-2...")
    ds = load_dataset("lmms-lab/SEED-Bench-2", split="test", trust_remote_code=True)
    print(f"Loaded {len(ds)} samples")

    # Phase 1: Collect all single-image samples
    print("Scanning for outdoor/driving/surveillance scenes...")
    candidates = []
    scene_counts = Counter()
    scanned = 0

    target_scenes = {
        "surveillance_traffic_cam", "driving_road_view",
        "outdoor_street", "construction_industrial",
    }

    for ex in ds:
        if ex.get("data_type") != "Single Image":
            continue
        img_field = ex.get("image")
        if img_field is None:
            continue
        # SEED-Bench-2 stores images as list[PIL.Image]
        if isinstance(img_field, list):
            if not img_field:
                continue
            image = img_field[0]
        else:
            image = img_field

        scanned += 1
        if scanned % 200 == 0:
            print(f"  Scanned {scanned}, found {len(candidates)} candidates...")

        # Quick VLM scene classification
        try:
            scene = _call_vlm_scene(args.vlm_endpoint, args.vlm_api_key, args.vlm_model, image)
        except Exception:
            continue

        scene_counts[scene] += 1

        # Keep if it's a target scene type
        if any(t in scene for t in target_scenes):
            candidates.append({
                "question": ex["question"],
                "choice_a": ex.get("choice_a", ""),
                "choice_b": ex.get("choice_b", ""),
                "choice_c": ex.get("choice_c", ""),
                "choice_d": ex.get("choice_d", ""),
                "answer": ex.get("answer", ""),
                "image": image,
                "question_type_id": ex.get("question_type_id", ""),
                "data_source": ex.get("data_source", ""),
                "question_id": ex.get("question_id", ""),
                "scene_type": scene,
            })

        # Stop after enough candidates
        if len(candidates) >= args.target * 2:
            break

        # Safety: don't scan forever
        if scanned >= 20000:
            break

    print(f"\nScanned {scanned}, found {len(candidates)} target-scene candidates")
    print(f"Scene distribution (all scanned):")
    for k, v in scene_counts.most_common():
        print(f"  {k}: {v}")

    if not candidates:
        print("No candidates found! Try relaxing scene filter.")
        sys.exit(1)

    # Phase 2: Subsample
    rng.shuffle(candidates)
    selected = candidates[:args.target]
    print(f"\nSelected {len(selected)} samples")

    # Phase 3: Save
    records = []
    for i, c in enumerate(selected):
        # Format MCQ
        labels = ["(A)", "(B)", "(C)", "(D)"]
        choices = [c["choice_a"], c["choice_b"], c["choice_c"], c["choice_d"]]
        option_lines = [f"{labels[j]} {choices[j]}" for j in range(4) if choices[j]]
        mcq_text = c["question"] + "\n" + "\n".join(option_lines)

        correct_option = f"({c['answer']})" if len(c["answer"]) == 1 else c["answer"]

        # Save image
        img = c["image"]
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img_filename = f"seedbench_{i}.jpg"
        img_path = out_dir / "images" / img_filename
        img.save(img_path, "JPEG")

        record = {
            "task_id": f"probe_seedbench_{c.get('question_id', i)}",
            "image_path": str(img_path.resolve()),
            "question": mcq_text,
            "correct_answer": correct_option,
            "source": "seed_bench_2",
            "original_id": str(c.get("question_id", i)),
            "query_type": "unknown",  # Will be set by unified annotation
            "original_question": c["question"],
            "scene_type_quick": c["scene_type"],
        }
        records.append(record)

    raw_path = out_dir / "raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(records)} records to {raw_path}")

    scene_final = Counter(r["scene_type_quick"] for r in records)
    print("Final scene distribution:")
    for k, v in scene_final.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
