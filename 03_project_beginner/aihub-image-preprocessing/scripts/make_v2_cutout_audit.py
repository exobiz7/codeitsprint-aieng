#!/usr/bin/env python3
"""Create a checkerboard audit sheet for SAM2 pill cutouts before composition."""

from __future__ import annotations

import argparse
import csv
import json
import random
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageFont

from v2_common import DEFAULT_PROCESSED_ROOT, Sam2MaskProvider, crop_cutout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--thumb-size", type=int, default=180)
    parser.add_argument("--margin-ratio", type=float, default=0.22)
    parser.add_argument("--min-mask-score", type=float, default=0.86)
    parser.add_argument("--sam2-checkpoint", type=Path, default=Path("models/sam2/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2-device", default="auto")
    parser.add_argument("--sam2-logit-threshold", type=float, default=0.8)
    parser.add_argument("--sam2-box-expansion-ratio", type=float, default=0.035)
    parser.add_argument("--include-transparent-sources", action="store_true")
    return parser


class ZipCache:
    def __init__(self) -> None:
        self._cache: dict[str, ZipFile] = {}

    def get(self, path: str) -> ZipFile:
        if path not in self._cache:
            self._cache[path] = ZipFile(path)
        return self._cache[path]

    def close(self) -> None:
        for handle in self._cache.values():
            handle.close()
        self._cache.clear()


def row_bbox(row: dict[str, str]) -> list[float]:
    return json.loads(row["annotations"])[0]["bbox"]


def risky_transparent_source(row: dict[str, str]) -> bool:
    text = " ".join([row.get("shape", ""), row.get("color", ""), row.get("product_id", "")])
    return any(token in text for token in ("투명", "반투명", "연질"))


def crop_with_local_box(image: Image.Image, bbox: list[float], margin_ratio: float):
    width, height = image.size
    x, y, box_w, box_h = bbox
    margin = int(round(max(box_w, box_h) * margin_ratio))
    left = max(0, int(x - margin))
    top = max(0, int(y - margin))
    right = min(width, int(x + box_w + margin))
    bottom = min(height, int(y + box_h + margin))
    local_box = [int(round(x - left)), int(round(y - top)), int(round(x + box_w - left)), int(round(y + box_h - top))]
    return image.crop((left, top, right, bottom)).convert("RGB"), local_box


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


def draw_pair(original: Image.Image, local_box: list[int], cutout: Image.Image, quality: dict, thumb_size: int) -> Image.Image:
    gap = 8
    title_h = 34
    cell = Image.new("RGB", (thumb_size * 2 + gap, thumb_size + title_h), (244, 244, 240))
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(cell)
    label = f"{quality.get('method')} score={quality.get('score')} fill={quality.get('fill_ratio')}"
    draw.text((4, 4), label[:58], fill=(20, 20, 20), font=font)

    original_thumb = fit(original, thumb_size)
    original_canvas = Image.new("RGB", (thumb_size, thumb_size), (230, 230, 226))
    ox = (thumb_size - original_thumb.width) // 2
    oy = (thumb_size - original_thumb.height) // 2
    original_canvas.paste(original_thumb, (ox, oy))
    scale = min(thumb_size / original.width, thumb_size / original.height)
    box = [
        ox + local_box[0] * scale,
        oy + local_box[1] * scale,
        ox + local_box[2] * scale,
        oy + local_box[3] * scale,
    ]
    ImageDraw.Draw(original_canvas).rectangle(box, outline=(255, 40, 40), width=2)

    cutout_thumb = fit(cutout, thumb_size)
    cutout_canvas = checkerboard((thumb_size, thumb_size))
    cx = (thumb_size - cutout_thumb.width) // 2
    cy = (thumb_size - cutout_thumb.height) // 2
    cutout_canvas.paste(cutout_thumb.convert("RGBA"), (cx, cy), cutout_thumb.getchannel("A"))

    cell.paste(original_canvas, (0, title_h))
    cell.paste(cutout_canvas, (thumb_size + gap, title_h))
    return cell


def main() -> int:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    rows = []
    manifest_path = args.processed_root / "manifests" / "split_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") == "train_seen" and row.get("dataset_kind") == "single":
                if not args.include_transparent_sources and risky_transparent_source(row):
                    continue
                rows.append(row)
    rng.shuffle(rows)

    provider = Sam2MaskProvider(
        args.sam2_checkpoint,
        args.sam2_config,
        args.sam2_device,
        multimask=True,
        logit_threshold=args.sam2_logit_threshold,
        box_expansion_ratio=args.sam2_box_expansion_ratio,
    )
    zip_cache = ZipCache()
    samples = []
    scanned = 0
    try:
        for row in rows:
            if len(samples) >= args.samples:
                break
            scanned += 1
            with Image.open(BytesIO(zip_cache.get(row["source_zip"]).read(row["image_member"]))) as image:
                image = image.convert("RGB")
                bbox = row_bbox(row)
                original_crop, local_box = crop_with_local_box(image, bbox, args.margin_ratio)
                cutout, quality = crop_cutout(
                    image,
                    bbox,
                    row.get("shape", ""),
                    margin_ratio=args.margin_ratio,
                    mask_provider=provider,
                )
            if quality.get("method") == "sam2_bbox" and quality.get("score", 0.0) >= args.min_mask_score:
                samples.append((original_crop, local_box, cutout, quality))
    finally:
        zip_cache.close()

    cols = 2
    gap = 14
    title_h = 30
    cells = [draw_pair(*sample, args.thumb_size) for sample in samples]
    if not cells:
        raise SystemExit("No accepted SAM2 cutouts found for audit.")
    cell_w = cells[0].width
    cell_h = cells[0].height
    rows_count = (len(cells) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w + (cols - 1) * gap, rows_count * cell_h + title_h), (244, 244, 240))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"sam2 cutout audit accepted={len(samples)} scanned={scanned}", fill=(20, 20, 20))
    for index, cell in enumerate(cells):
        x = (index % cols) * (cell_w + gap)
        y = title_h + (index // cols) * cell_h
        canvas.paste(cell, (x, y))

    output_path = args.processed_root / "audit" / "sam2_cutout_audit.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=95, subsampling=0, optimize=True)
    summary_path = args.processed_root / "audit" / "sam2_cutout_audit_summary.json"
    summary_path.write_text(
        json.dumps({"output": str(output_path), "accepted": len(samples), "scanned": scanned}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
