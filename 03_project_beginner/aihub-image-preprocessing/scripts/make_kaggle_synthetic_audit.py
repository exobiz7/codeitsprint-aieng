#!/usr/bin/env python3
"""Create audit contact sheets for Kaggle-specialized synthetic runs."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_OUTPUT_ROOT = Path("/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_sam2_synth_v1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--samples-per-size", type=int, default=12)
    parser.add_argument("--asset-samples", type=int, default=32)
    parser.add_argument("--thumb-width", type=int, default=260)
    parser.add_argument("--seed", type=int, default=20260704)
    return parser


def draw_sample(image_path: Path, annotations: list[dict], width: int, thumb_width: int) -> Image.Image:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    scale = thumb_width / width
    thumb_height = int(round(image.height * scale))
    thumb = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(thumb)
    palette = [(255, 30, 30), (30, 120, 255), (30, 180, 80), (255, 160, 20)]
    for index, ann in enumerate(annotations):
        x, y, w, h = ann["bbox"]
        box = [x * scale, y * scale, (x + w) * scale, (y + h) * scale]
        color = palette[index % len(palette)]
        for offset in range(2):
            draw.rectangle(
                [box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset],
                outline=color,
            )
    return thumb


def make_sheet(samples, output_path: Path, title: str, thumb_width: int) -> None:
    if not samples:
        return
    cols = 4
    gap = 14
    title_h = 30
    label_h = 26
    thumbs = [draw_sample(path, anns, width, thumb_width) for _, path, anns, width in samples]
    cell_h = max(thumb.height for thumb in thumbs) + label_h
    rows = (len(thumbs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb_width + (cols - 1) * gap, rows * cell_h + title_h), (244, 244, 240))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(20, 20, 20), font=font)
    for index, ((label, _, _, _), thumb) in enumerate(zip(samples, thumbs)):
        col = index % cols
        row = index // cols
        x = col * (thumb_width + gap)
        y = title_h + row * cell_h
        draw.text((x + 4, y + 4), label[:42], fill=(20, 20, 20), font=font)
        canvas.paste(thumb, (x, y + label_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=95, subsampling=0, optimize=True)


def checkerboard(size: tuple[int, int], cell: int = 12) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, (210, 210, 210))
    draw = ImageDraw.Draw(image)
    for y in range(0, height, cell):
        for x in range(0, width, cell):
            if (x // cell + y // cell) % 2 == 0:
                draw.rectangle([x, y, x + cell - 1, y + cell - 1], fill=(245, 245, 245))
    return image


def fit(image: Image.Image, box_size: int) -> Image.Image:
    scale = min(box_size / image.width, box_size / image.height)
    size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    return image.resize(size, Image.Resampling.LANCZOS)


def make_asset_sheet(output_root: Path, output_path: Path, rng: random.Random, sample_count: int) -> None:
    manifest_path = output_root / "assets" / "assets_manifest.csv"
    if not manifest_path.exists():
        return
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rng.shuffle(rows)
    rows = rows[:sample_count]
    thumb = 170
    cols = 4
    gap = 12
    title_h = 30
    label_h = 24
    cell_w = thumb
    cell_h = thumb + label_h
    sheet_rows = (len(rows) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w + (cols - 1) * gap, sheet_rows * cell_h + title_h), (244, 244, 240))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), "SAM2 cutout assets", fill=(20, 20, 20), font=font)
    for index, row in enumerate(rows):
        col = index % cols
        sheet_row = index // cols
        x = col * (cell_w + gap)
        y = title_h + sheet_row * cell_h
        draw.text((x + 2, y + 2), f"{row['product_id']} {row['shape']}"[:30], fill=(20, 20, 20), font=font)
        with Image.open(row["asset_path"]) as image:
            cutout = fit(image.convert("RGBA"), thumb)
        board = checkerboard((thumb, thumb))
        ox = (thumb - cutout.width) // 2
        oy = (thumb - cutout.height) // 2
        board.paste(cutout, (ox, oy), cutout.getchannel("A"))
        canvas.paste(board, (x, y + label_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=95, subsampling=0, optimize=True)


def main() -> int:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    run_root = args.output_root / "runs" / args.run_name
    image_dir = run_root / "images"
    coco = json.loads((run_root / "annotations_coco.json").read_text(encoding="utf-8"))
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    by_size = {2: [], 3: [], 4: []}
    for image in coco["images"]:
        size = int(image.get("combo_size") or len(anns_by_image.get(image["id"], [])))
        if size not in by_size:
            continue
        label = f"{image['file_name']} n={size}"
        by_size[size].append((label, image_dir / image["file_name"], anns_by_image.get(image["id"], []), image["width"]))
    outputs = []
    for size, samples in by_size.items():
        rng.shuffle(samples)
        selected = samples[: args.samples_per_size]
        output_path = run_root / "audit" / f"kaggle_synth_size_{size}.jpg"
        make_sheet(selected, output_path, f"kaggle_sam2_synth size={size}", args.thumb_width)
        if selected:
            outputs.append(str(output_path))
            print(f"wrote {output_path}")
    asset_output = run_root / "audit" / "kaggle_sam2_cutout_assets.jpg"
    make_asset_sheet(args.output_root, asset_output, rng, args.asset_samples)
    if asset_output.exists():
        outputs.append(str(asset_output))
        print(f"wrote {asset_output}")
    summary_path = run_root / "audit" / "audit_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"outputs": outputs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
