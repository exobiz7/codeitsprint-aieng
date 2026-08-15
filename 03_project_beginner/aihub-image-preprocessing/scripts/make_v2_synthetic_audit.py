#!/usr/bin/env python3
"""Create contact sheets for synthetic v2 combo shards."""

from __future__ import annotations

import argparse
import csv
import json
import random
import tarfile
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from v2_common import DEFAULT_PROCESSED_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--samples-per-size", type=int, default=24)
    parser.add_argument("--thumb-width", type=int, default=260)
    parser.add_argument("--seed", type=int, default=20260703)
    return parser


def reservoir_manifest_samples(processed_root: Path, samples_per_size: int, rng: random.Random):
    manifest_path = processed_root / "manifests" / "combo_synth_v1_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing synthetic manifest: {manifest_path}")
    by_size = {2: [], 3: [], 4: []}
    seen = {2: 0, 3: 0, 4: 0}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                size = int(row["combo_size"])
            except (KeyError, ValueError):
                continue
            if size not in by_size:
                continue
            seen[size] += 1
            bucket = by_size[size]
            if len(bucket) < samples_per_size:
                bucket.append(row)
            else:
                replace_at = rng.randrange(seen[size])
                if replace_at < samples_per_size:
                    bucket[replace_at] = row
    return by_size, seen


def read_selected_tar_samples(rows: list[dict[str, str]]):
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["source_zip"], []).append(row)
    for tar_path, tar_rows in sorted(grouped.items()):
        with tarfile.open(tar_path, "r") as tar:
            members = {member.name: member for member in tar if member.isfile()}
            for row in tar_rows:
                image_member = row["image_member"]
                key = Path(image_member).stem
                label_member = f"{key}.json"
                image_data = tar.extractfile(members[image_member]).read()
                label = json.loads(tar.extractfile(members[label_member]).read())
                yield key, image_data, label


def draw_sample(image_data: bytes, label: dict, thumb_width: int) -> Image.Image:
    with Image.open(BytesIO(image_data)) as image:
        image = image.convert("RGB")
    scale = thumb_width / image.width
    thumb_height = int(round(image.height * scale))
    thumb = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(thumb)
    palette = [(255, 30, 30), (30, 120, 255), (30, 180, 80), (255, 160, 20)]
    for index, ann in enumerate(label.get("annotations", [])):
        x, y, w, h = ann["bbox"]
        box = [x * scale, y * scale, (x + w) * scale, (y + h) * scale]
        for offset in range(2):
            draw.rectangle(
                [box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset],
                outline=palette[index % len(palette)],
            )
    return thumb


def make_sheet(samples, output_path: Path, title: str, thumb_width: int) -> None:
    if not samples:
        return
    cols = 4
    gap = 14
    title_h = 28
    thumbs = [draw_sample(image_data, label, thumb_width) for _, image_data, label in samples]
    cell_h = max(thumb.height for thumb in thumbs) + title_h
    rows = (len(thumbs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb_width + (cols - 1) * gap, rows * cell_h + title_h), (244, 244, 240))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(20, 20, 20), font=font)
    for index, ((key, _, label), thumb) in enumerate(zip(samples, thumbs)):
        col = index % cols
        row = index // cols
        x = col * (thumb_width + gap)
        y = title_h + row * cell_h
        draw.text((x + 4, y + 4), f"{key} n={len(label.get('annotations', []))}", fill=(20, 20, 20), font=font)
        canvas.paste(thumb, (x, y + title_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=95, subsampling=0, optimize=True)


def main() -> int:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    by_size, seen = reservoir_manifest_samples(args.processed_root, args.samples_per_size, rng)
    outputs = []
    for size, rows in by_size.items():
        selected = list(read_selected_tar_samples(rows))
        output_path = args.processed_root / "audit" / f"combo_synth_size_{size}.jpg"
        make_sheet(selected, output_path, f"combo_synth_v1 size={size}", args.thumb_width)
        if selected:
            outputs.append(str(output_path))
            print(f"wrote {output_path}")
    summary_path = args.processed_root / "audit" / "synthetic_audit_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"outputs": outputs, "seen_by_size": seen}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
