#!/usr/bin/env python3
"""Preprocess clean 384x384 background chips for Kaggle realcopy generation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance


DEFAULT_INPUT_DIR = Path("codex-handoff/autoclean_bg_clean_384")
DEFAULT_OUTPUT_DIR = Path("/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg_clean64_preprocessed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--safe-crop", type=int, default=288)
    parser.add_argument("--canvas-width", type=int, default=976)
    parser.add_argument("--canvas-height", type=int, default=1280)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def edge_metrics(rgb: np.ndarray) -> dict:
    lum = luma(rgb.astype(np.float32))
    h, w = lum.shape
    center = lum[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    border = np.concatenate([lum[:24, :].ravel(), lum[-24:, :].ravel(), lum[:, :24].ravel(), lum[:, -24:].ravel()])
    corners = [
        lum[:72, :72],
        lum[:72, -72:],
        lum[-72:, :72],
        lum[-72:, -72:],
    ]
    center_med = float(np.median(center))
    corner_delta_max = float(max(np.median(corner) - center_med for corner in corners))
    border_delta_p95 = float(np.percentile(border, 95) - center_med)
    return {
        "center_luma_median": round(center_med, 3),
        "border_luma_p95_delta": round(border_delta_p95, 3),
        "corner_luma_max_delta": round(corner_delta_max, 3),
        "has_bright_edge_risk": bool(corner_delta_max > 13 or border_delta_p95 > 18),
    }


def clean_chip(image: Image.Image, safe_crop: int) -> tuple[Image.Image, dict]:
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    if h != 384 or w != 384:
        raise ValueError(f"Expected 384x384 chip, got {w}x{h}")
    m = edge_metrics(rgb)
    crop = min(safe_crop, h, w)
    x0 = (w - crop) // 2
    y0 = (h - crop) // 2
    central = rgb[y0 : y0 + crop, x0 : x0 + crop]
    # Gentle denoise only. Keep the real paper texture, but suppress sharp
    # screenshot/corner transitions that can become synthetic shortcuts.
    clean = cv2.bilateralFilter(central, d=5, sigmaColor=10, sigmaSpace=8)
    clean_img = Image.fromarray(clean, "RGB").resize((384, 384), Image.Resampling.BICUBIC)
    clean_img = ImageEnhance.Contrast(clean_img).enhance(0.985)
    m.update({"crop_box": [x0, y0, crop, crop], "output_size": [384, 384]})
    return clean_img, m


def make_canvas(clean_img: Image.Image, width: int, height: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = clean_img.resize((width, height), Image.Resampling.BICUBIC)
    arr = np.asarray(base, dtype=np.float32)
    # Add very light camera-like texture so upscaling does not become sterile.
    noise = rng.normal(0, 0.45, arr.shape)
    yy, xx = np.mgrid[0:height, 0:width]
    cx = width * (0.5 + rng.uniform(-0.08, 0.08))
    cy = height * (0.5 + rng.uniform(-0.08, 0.08))
    rr = ((xx - cx) / width) ** 2 + ((yy - cy) / height) ** 2
    vignette = 1.0 - np.clip(rr * rng.uniform(0.025, 0.055), 0, 0.035)
    arr = arr * vignette[..., None] + noise
    return Image.fromarray(np.uint8(np.clip(arr, 0, 255)), "RGB")


def contact_sheet(files: list[Path], output_path: Path, title: str, thumb: int = 128) -> None:
    label_h = 18
    cols = 8
    rows = math.ceil(len(files) / cols)
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), (245, 245, 242))
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(files):
        image = Image.open(path).convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
        x = (i % cols) * thumb
        y = (i // cols) * (thumb + label_h)
        sheet.paste(image, (x, y + label_h))
        draw.text((x + 3, y + 3), path.stem, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def build(args) -> dict:
    input_dir = args.input_dir
    output_dir = args.output_dir
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}. Pass --overwrite.")
        shutil.rmtree(output_dir)
    chips_dir = output_dir / "chips_384"
    canvases_dir = output_dir / "canvases_976x1280"
    reports_dir = output_dir / "reports"
    chips_dir.mkdir(parents=True)
    canvases_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)

    source_files = sorted(input_dir.glob("*.png"))
    if len(source_files) != 64:
        raise RuntimeError(f"Expected 64 png background chips, got {len(source_files)}")

    rows = []
    clean_paths = []
    canvas_paths = []
    for idx, source in enumerate(source_files):
        with Image.open(source) as image:
            clean, metrics = clean_chip(image, args.safe_crop)
        clean_name = f"bg_clean64_{idx:02d}.png"
        canvas_name = f"bg_canvas64_{idx:02d}.png"
        clean_path = chips_dir / clean_name
        canvas_path = canvases_dir / canvas_name
        clean.save(clean_path)
        make_canvas(clean, args.canvas_width, args.canvas_height, seed=20260705 + idx).save(canvas_path)
        clean_paths.append(clean_path)
        canvas_paths.append(canvas_path)
        rows.append(
            {
                "index": idx,
                "source_file": source.name,
                "clean_chip": f"chips_384/{clean_name}",
                "canvas": f"canvases_976x1280/{canvas_name}",
                **metrics,
            }
        )

    contact_sheet(source_files, reports_dir / "raw_bg64_contact_sheet.jpg", "raw")
    contact_sheet(clean_paths, reports_dir / "cleaned_bg64_contact_sheet.jpg", "cleaned")
    # Canvas contact sheet is portrait; keep a taller thumbnail.
    canvas_thumb_w, canvas_thumb_h, label_h, cols = 122, 160, 18, 8
    canvas_sheet = Image.new("RGB", (cols * canvas_thumb_w, math.ceil(len(canvas_paths) / cols) * (canvas_thumb_h + label_h)), (245, 245, 242))
    draw = ImageDraw.Draw(canvas_sheet)
    for i, path in enumerate(canvas_paths):
        image = Image.open(path).convert("RGB").resize((canvas_thumb_w, canvas_thumb_h), Image.Resampling.LANCZOS)
        x = (i % cols) * canvas_thumb_w
        y = (i // cols) * (canvas_thumb_h + label_h)
        canvas_sheet.paste(image, (x, y + label_h))
        draw.text((x + 3, y + 3), path.stem, fill=(0, 0, 0))
    canvas_sheet.save(reports_dir / "canvas_bg64_contact_sheet.jpg", quality=95)

    with (reports_dir / "background_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "chips": len(rows),
        "safe_crop": args.safe_crop,
        "chip_output": "chips_384/*.png",
        "canvas_output": "canvases_976x1280/*.png",
        "bright_edge_risk_count": sum(1 for row in rows if row["has_bright_edge_risk"]),
        "bright_edge_risk_files": [row["source_file"] for row in rows if row["has_bright_edge_risk"]],
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
