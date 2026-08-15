#!/usr/bin/env python3
"""Generate v2 synthetic combo WebDataset shards from a prebuilt SAM2 asset bank."""

from __future__ import annotations

import argparse
import collections
import csv
import heapq
import json
import random
import tarfile
from io import BytesIO
from pathlib import Path

from PIL import Image

from v2_common import (
    CANONICAL_FIELDS,
    DEFAULT_PROCESSED_ROOT,
    add_tar_bytes,
    alpha_bbox,
    apply_common_photometric,
    canonical_json,
    iou_xywh,
    json_compact,
    paste_with_shadow,
    sanitize_tar_key,
    synthetic_background,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--bank-name", default="sam2_large_a6")
    parser.add_argument("--run-name", default="combo_synth_1k")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--width", type=int, default=976)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--combo2-ratio", type=float, default=0.30)
    parser.add_argument("--combo3-ratio", type=float, default=0.40)
    parser.add_argument("--combo4-ratio", type=float, default=0.30)
    parser.add_argument("--max-iou", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-parquet", action="store_true")
    return parser


def read_assets(processed_root: Path, bank_name: str) -> list[dict[str, str]]:
    path = processed_root / "asset_banks" / bank_name / "assets_manifest.csv"
    if not path.exists():
        raise SystemExit(f"Missing asset bank manifest: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def choose_combo_size(rng: random.Random, ratios: tuple[float, float, float]) -> int:
    value = rng.random()
    if value < ratios[0]:
        return 2
    if value < ratios[0] + ratios[1]:
        return 3
    return 4


def choose_products(
    rng: random.Random,
    heap: list[tuple[int, float, str]],
    counts: collections.Counter,
    combo_size: int,
) -> list[str]:
    chosen = []
    skipped = []
    while heap and len(chosen) < combo_size:
        count, tie, product_id = heapq.heappop(heap)
        if count != counts[product_id]:
            continue
        if product_id in chosen:
            skipped.append((count, tie, product_id))
            continue
        chosen.append(product_id)
    for item in skipped:
        heapq.heappush(heap, item)
    for product_id in chosen:
        counts[product_id] += 1
        heapq.heappush(heap, (counts[product_id], rng.random(), product_id))
    if len(chosen) != combo_size:
        raise RuntimeError(f"Could not choose {combo_size} unique products")
    return chosen


def placement_cells(combo_size: int, width: int, height: int) -> list[tuple[int, int, int, int]]:
    pad_x, pad_y = 72, 92
    if combo_size == 2:
        return [(pad_x, pad_y, width // 2 - pad_x, height - pad_y), (width // 2, pad_y, width - pad_x, height - pad_y)]
    if combo_size == 3:
        return [
            (pad_x, pad_y, width // 2 - pad_x, height // 2),
            (width // 2, pad_y, width - pad_x, height // 2),
            (width // 4, height // 2, width * 3 // 4, height - pad_y),
        ]
    return [
        (pad_x, pad_y, width // 2 - pad_x, height // 2),
        (width // 2, pad_y, width - pad_x, height // 2),
        (pad_x, height // 2, width // 2 - pad_x, height - pad_y),
        (width // 2, height // 2, width - pad_x, height - pad_y),
    ]


def read_cutout(asset: dict[str, str]) -> Image.Image:
    with Image.open(asset["asset_path"]) as image:
        return image.convert("RGBA")


def read_profile(asset: dict[str, str]) -> Image.Image | None:
    path = asset.get("profile_path")
    if not path:
        return None
    with Image.open(path) as image:
        return image.convert("RGB")


def transform_cutout(cutout: Image.Image, rng: random.Random) -> Image.Image:
    angle = rng.uniform(-10.0, 10.0)
    return cutout.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)


def place_cutouts(cutouts: list[Image.Image], width: int, height: int, max_iou: float, rng: random.Random):
    cells = placement_cells(len(cutouts), width, height)
    rng.shuffle(cells)
    placements = []
    bboxes = []
    for cutout, cell in zip(cutouts, cells):
        x1, y1, x2, y2 = cell
        cell_w = max(24, x2 - x1)
        cell_h = max(24, y2 - y1)
        max_scale = min(1.12, (cell_w * 0.82) / max(1, cutout.width), (cell_h * 0.82) / max(1, cutout.height))
        min_scale = max(0.42, min(0.86, max_scale * 0.78))
        max_scale = max(max_scale, min_scale)
        scale = rng.uniform(min_scale, max_scale)
        scaled = cutout.resize(
            (max(12, int(round(cutout.width * scale))), max(12, int(round(cutout.height * scale)))),
            Image.Resampling.LANCZOS,
        )
        placed = False
        for _ in range(100):
            max_x = max(x1, x2 - scaled.width)
            max_y = max(y1, y2 - scaled.height)
            x = rng.randint(x1, max_x) if max_x > x1 else x1
            y = rng.randint(y1, max_y) if max_y > y1 else y1
            fg_bbox = alpha_bbox(scaled.getchannel("A")) or [0, 0, scaled.width, scaled.height]
            bbox = [x + fg_bbox[0], y + fg_bbox[1], fg_bbox[2], fg_bbox[3]]
            if x < 0 or y < 0 or x + scaled.width > width or y + scaled.height > height:
                continue
            if all(iou_xywh(bbox, existing) <= max_iou for existing in bboxes):
                placements.append((scaled, (x, y), [int(round(v)) for v in bbox]))
                bboxes.append(bbox)
                placed = True
                break
        if not placed:
            raise RuntimeError("Could not place synthetic cutout without overlap")
    return placements


def open_shard(output_dir: Path, shard_index: int, overwrite: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"train_combo_synth-{shard_index:06d}.tar"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Shard exists: {path}. Pass --overwrite to replace.")
    return path, tarfile.open(path, "w")


def load_assets_by_product(assets: list[dict[str, str]], rng: random.Random) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for asset in assets:
        grouped[asset["product_id"]].append(asset)
    for product_assets in grouped.values():
        rng.shuffle(product_assets)
    return dict(grouped)


def generate(args) -> tuple[list[dict], dict]:
    rng = random.Random(args.seed)
    assets = read_assets(args.processed_root, args.bank_name)
    assets_by_product = load_assets_by_product(assets, rng)
    if len(assets_by_product) < 4:
        raise RuntimeError("Asset bank must contain at least four products")
    counts = collections.Counter()
    cursors = collections.Counter()
    heap = [(0, rng.random(), product_id) for product_id in assets_by_product]
    heapq.heapify(heap)

    run_root = args.processed_root / "runs" / args.run_name
    output_dir = run_root / "webdataset" / "train_combo_synth"
    if run_root.exists() and not args.overwrite:
        raise FileExistsError(f"Run exists: {run_root}. Pass --overwrite to replace.")
    if args.overwrite and output_dir.exists():
        for path in output_dir.glob("*.tar"):
            path.unlink()
    manifest_dir = run_root / "manifests"
    csv_path = manifest_dir / "combo_synth_v1_manifest.csv"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_handle = csv_path.open("w", newline="", encoding="utf-8")
    manifest_writer = csv.DictWriter(manifest_handle, fieldnames=CANONICAL_FIELDS)
    manifest_writer.writeheader()
    rows_written = 0
    annotation_total = 0
    combo_size_counts = collections.Counter()
    shard_index = 0
    count_in_shard = 0
    shard_paths = []
    shard_path, tar = open_shard(output_dir, shard_index, args.overwrite)
    shard_paths.append(str(shard_path))
    ratios = (args.combo2_ratio, args.combo3_ratio, args.combo4_ratio)
    attempts = 0
    placement_failures = 0
    try:
        while rows_written < args.num_images:
            attempts += 1
            if attempts > args.num_images * 20:
                raise RuntimeError(f"Too many placement failures: {placement_failures}")
            combo_size = choose_combo_size(rng, ratios)
            product_ids = choose_products(rng, heap, counts, combo_size)
            selected_assets = []
            cutouts = []
            for product_id in product_ids:
                product_assets = assets_by_product[product_id]
                cursor = cursors[product_id]
                if cursor % len(product_assets) == 0:
                    rng.shuffle(product_assets)
                asset = product_assets[cursor % len(product_assets)]
                cursors[product_id] += 1
                selected_assets.append(asset)
                cutouts.append(transform_cutout(read_cutout(asset), rng))
            try:
                placements = place_cutouts(cutouts, args.width, args.height, args.max_iou, rng)
            except RuntimeError:
                placement_failures += 1
                continue
            profile_asset = rng.choice(selected_assets)
            canvas = synthetic_background(args.width, args.height, rng, read_profile(profile_asset)).convert("RGBA")
            annotations = []
            source_refs = []
            mask_qualities = []
            for ann_index, (asset, (cutout, xy, bbox)) in enumerate(zip(selected_assets, placements), start=1):
                paste_with_shadow(canvas, cutout, xy, rng)
                annotations.append(
                    {
                        "id": ann_index,
                        "bbox": [int(v) for v in bbox],
                        "area": int(bbox[2] * bbox[3]),
                        "category_id": 1,
                        "product_id": asset["product_id"],
                        "class_index": asset.get("class_index", ""),
                        "ignore_for_id": False,
                        "source_sample_id": asset["sample_id"],
                        "source_asset_id": asset["asset_id"],
                    }
                )
                source_refs.append(
                    {
                        "sample_id": asset["sample_id"],
                        "asset_id": asset["asset_id"],
                        "source_zip": asset["source_zip"],
                        "image_member": asset["image_member"],
                        "product_id": asset["product_id"],
                    }
                )
                mask_qualities.append(json.loads(asset["quality"]))
            final_image = apply_common_photometric(canvas.convert("RGB"), rng)
            image_buffer = BytesIO()
            final_image.save(image_buffer, format="JPEG", quality=95, subsampling=0, optimize=True)
            image_bytes = image_buffer.getvalue()
            sample_index = rows_written
            sample_id = f"{args.run_name}_{sample_index:09d}"
            key = sanitize_tar_key(sample_id)
            add_tar_bytes(tar, f"{key}.jpg", image_bytes)
            row = {
                "sample_id": sample_id,
                "dataset_kind": "combo_synth",
                "split_source": "combo_synth_asset_bank",
                "split": "train_combo_synth",
                "image_path": "",
                "image_member": f"{key}.jpg",
                "label_path": "",
                "label_members": json_compact([]),
                "product_id": "",
                "combo_product_ids": json_compact(product_ids),
                "combo_size": combo_size,
                "bbox": json_compact(annotations[0]["bbox"]),
                "annotations": json_compact(annotations),
                "width": args.width,
                "height": args.height,
                "shape": "",
                "color": "",
                "back_color": "synthetic_profiled",
                "light_color": "synthetic_profiled",
                "camera_la": "",
                "camera_lo": "",
                "drug_dir": "",
                "source_zip": str(shard_path),
                "label_zip": "",
                "source_zip_name": shard_path.name,
                "label_zip_name": "",
                "set_id": "",
                "original_stem": sample_id,
                "sha256": "",
                "phash": "",
                "synthetic": "true",
                "source_refs": json_compact(source_refs),
                "transform": json_compact(
                    {
                        "placement": "grid_jitter",
                        "photometric": "global_jitter_noise",
                        "asset_bank": args.bank_name,
                    }
                ),
                "mask_quality": json_compact({"instances": mask_qualities}),
                "ignore_for_id": "false",
                "class_index": "",
            }
            add_tar_bytes(tar, f"{key}.json", json.dumps(canonical_json(row), ensure_ascii=False).encode("utf-8"))
            manifest_writer.writerow({field: row.get(field, "") for field in CANONICAL_FIELDS})
            rows_written += 1
            annotation_total += combo_size
            combo_size_counts[combo_size] += 1
            count_in_shard += 1
            if count_in_shard >= args.shard_size:
                tar.close()
                shard_index += 1
                count_in_shard = 0
                shard_path, tar = open_shard(output_dir, shard_index, args.overwrite)
                shard_paths.append(str(shard_path))
            if rows_written % 1000 == 0:
                manifest_handle.flush()
                print(f"generated {rows_written}/{args.num_images}", flush=True)
        tar.close()
        tar = None
        if count_in_shard == 0 and rows_written:
            empty_path = Path(shard_paths.pop())
            empty_path.unlink(missing_ok=True)
    finally:
        if tar is not None:
            tar.close()
        manifest_handle.close()
    summary = {
        "run_name": args.run_name,
        "num_images": rows_written,
        "annotations": annotation_total,
        "asset_bank": args.bank_name,
        "asset_products": len(assets_by_product),
        "products_used": len(counts),
        "product_instance_count_min": min(counts.values()) if counts else 0,
        "product_instance_count_max": max(counts.values()) if counts else 0,
        "combo_size_counts": dict(combo_size_counts),
        "attempts": attempts,
        "placement_failures": placement_failures,
        "shards": len(shard_paths),
        "shard_paths": shard_paths,
        "manifest_csv": str(csv_path),
    }
    summary_path = manifest_dir / "combo_synth_v1_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [], summary


def main() -> int:
    args = build_parser().parse_args()
    _, summary = generate(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
