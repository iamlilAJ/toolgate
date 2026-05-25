#!/usr/bin/env python3
"""Build CUB-200 attribute MCQ dataset.

Each image has 5-10 English captions describing its visual attributes.
Parse captions → extract (body_part, color/shape) patterns → build MCQ.

Usage:
    python build_cub_mcq.py --target 3000 --output-dir data/probe_v3/cub
"""
import argparse, json, random, re
from collections import Counter
from pathlib import Path
from datasets import load_dataset


COLORS = ['black', 'white', 'brown', 'gray', 'grey', 'red', 'blue', 'yellow', 'green',
          'orange', 'pink', 'purple', 'tan', 'beige', 'cream', 'rust', 'olive']
SHAPES = ['long', 'short', 'curved', 'straight', 'pointed', 'hooked', 'thick', 'thin',
          'flat', 'rounded', 'sharp']
BODY_PARTS = ['beak', 'bill', 'body', 'wing', 'wings', 'belly', 'head', 'throat',
              'tail', 'breast', 'crown', 'feathers', 'feet', 'legs', 'neck', 'eye']


def extract_attrs(captions):
    """Extract (body_part, attribute) from concatenated captions."""
    text = ' '.join(captions).lower()
    attrs = {}   # body_part -> Counter of attribute
    # Patterns like "ATTR BODY" or "BODY is ATTR" or "ATTR, BODY"
    for bp in BODY_PARTS:
        for attr in COLORS + SHAPES:
            # "ATTR BODY" (e.g., "brown body", "long bill")
            if re.search(rf'\b{attr}\s+{bp}\b', text):
                attrs.setdefault(bp, Counter())[attr] += 1
            # "BODY is/are ATTR" (e.g., "bill is black")
            if re.search(rf'\b{bp}\s+(?:is|are|has)\s+(?:a\s+|an\s+)?{attr}\b', text):
                attrs.setdefault(bp, Counter())[attr] += 2
    # Normalize: beak/bill, wing/wings, feet/legs
    aliases = {'bill': 'beak', 'wings': 'wing', 'legs': 'feet'}
    for src, dst in aliases.items():
        if src in attrs:
            attrs[dst] = attrs.get(dst, Counter()) + attrs[src]
            del attrs[src]
    return attrs


def pick_mcq(attrs, rng):
    """Pick one salient (body_part, attr) → MCQ."""
    if not attrs: return None
    # Pick body part with highest-confidence attr (top count)
    parts_sorted = sorted(attrs.items(), key=lambda kv: -max(kv[1].values()))
    for bp, counter in parts_sorted:
        top_attr, top_cnt = counter.most_common(1)[0]
        if top_cnt < 2: continue  # need at least 2 mentions
        # Determine domain (color or shape)
        is_color = top_attr in COLORS
        pool = COLORS if is_color else SHAPES
        # Distractors: 3 unused values from same domain
        used = set(counter.keys())
        distractors = [x for x in pool if x not in used]
        rng.shuffle(distractors)
        distractors = distractors[:3]
        if len(distractors) < 3: continue
        # Build question
        noun = 'color' if is_color else 'shape'
        article = 'an' if bp[0] in 'aeiou' else 'a'
        q = f"What is the {noun} of the bird's {bp}?"
        options = [top_attr] + distractors
        rng.shuffle(options)
        idx = options.index(top_attr)
        letters = ['A','B','C','D']
        opts_text = '\n'.join(f'({letters[i]}) {options[i]}' for i in range(4))
        return {
            'question_text': f'{q}\n{opts_text}',
            'correct_answer': f'({letters[idx]})',
            'body_part': bp,
            'attribute_type': noun,
            'original_answer': top_attr,
        }
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / 'images').mkdir(parents=True, exist_ok=True)

    print('Loading CUB-200...')
    # Multimodal-Fatima/CUB_train has 5994 + CUB_test 5794
    train = load_dataset('Multimodal-Fatima/CUB_train', split='train')
    test = load_dataset('Multimodal-Fatima/CUB_test', split='test')
    print(f'Train: {len(train)}, Test: {len(test)}')

    rng = random.Random(args.seed)
    all_idx = [('train', i) for i in range(len(train))] + [('test', i) for i in range(len(test))]
    rng.shuffle(all_idx)

    records = []
    fail = 0
    for split, idx in all_idx:
        if len(records) >= args.target: break
        s = (train if split == 'train' else test)[idx]
        desc = s.get('description', '')
        # Descriptions are joined with \n
        captions = desc.split('\n') if isinstance(desc, str) else desc
        captions = [c.strip() for c in captions if c.strip()]
        if len(captions) < 3:
            fail += 1; continue
        attrs = extract_attrs(captions)
        mcq = pick_mcq(attrs, rng)
        if mcq is None:
            fail += 1; continue
        tid = f'probe_cub_{len(records):04d}'
        img_path = out_dir / 'images' / f'{tid}.jpg'
        try:
            s['image'].convert('RGB').save(img_path, 'JPEG', quality=90)
        except Exception as e:
            fail += 1; continue
        records.append({
            'task_id': tid,
            'image_path': str(img_path),
            'question': mcq['question_text'],
            'correct_answer': mcq['correct_answer'],
            'source': 'cub',
            'query_type': 'fine_grained_attribute',
            'body_part': mcq['body_part'],
            'attribute_type': mcq['attribute_type'],
            'original_answer': mcq['original_answer'],
            'species_label': s.get('label', -1),
        })
        if len(records) % 200 == 0:
            print(f'  {len(records)}/{args.target}  (fail={fail})')

    out_path = out_dir / 'samples.jsonl'
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'\nDone! {len(records)} records → {out_path}   (fail={fail})')
    # Quick diversity stats
    from collections import Counter
    print('Body parts:', Counter(r['body_part'] for r in records).most_common())
    print('Attr types:', Counter(r['attribute_type'] for r in records).most_common())
    print('Orig answers (top 10):', Counter(r['original_answer'] for r in records).most_common(10))


if __name__ == '__main__':
    main()
