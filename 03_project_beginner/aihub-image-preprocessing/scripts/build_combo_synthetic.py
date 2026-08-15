#!/usr/bin/env python3
"""Generate balanced synthetic combo WebDataset shards from train_seen single pills."""

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
from zipfile import ZipFile

from PIL import Image

from v2_common import (
    CANONICAL_FIELDS,
    DEFAULT_PROCESSED_ROOT,
    add_tar_bytes,
    alpha_bbox,
    apply_common_photometric,
    canonical_json,
    crop_cutout,
    iou_xywh,
    json_compact,
    optional_write_parquet,
    paste_with_shadow,
    read_csv_rows,
    Sam2MaskProvider,
    sanitize_tar_key,
    synthetic_background,
    write_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--num-images", type=int, default=600_000)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--width", type=int, default=976)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--combo2-ratio", type=float, default=0.30)
    parser.add_argument("--combo3-ratio", type=float, default=0.40)
    parser.add_argument("--combo4-ratio", type=float, default=0.30)
    parser.add_argument("--hard-mix-ratio", type=float, default=0.30)
    parser.add_argument("--max-iou", type=float, default=0.02)
    parser.add_argument("--mask-provider", choices=("sam2", "grabcut"), default="sam2")
    parser.add_argument("--min-mask-score", type=float, default=0.86)
    parser.add_argument("--cutout-retries-per-product", type=int, default=24)
    parser.add_argument("--allow-grabcut-fallback", action="store_true")
    parser.add_argument("--sam2-checkpoint", type=Path, default=Path("models/sam2/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2-device", default="auto", help="auto, mps, cuda, or cpu")
    parser.add_argument("--sam2-logit-threshold", type=float, default=0.8)
    parser.add_argument("--sam2-box-expansion-ratio", type=float, default=0.035)
    parser.add_argument("--include-transparent-sources", action="store_true")
    parser.add_argument("--skip-parquet", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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


def risky_transparent_source(row: dict[str, str]) -> bool:
    text = " ".join([row.get("shape", ""), row.get("color", ""), row.get("product_id", "")])
    return any(token in text for token in ("투명", "반투명", "연질"))


def load_train_seen_rows(processed_root: Path, include_transparent_sources: bool) -> tuple[dict[str, list[dict[str, str]]], dict]:
    manifest_path = processed_root / "manifests" / "split_manifest.csv"
    rows_by_product: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    excluded_rows = 0
    excluded_products = set()
    for row in read_csv_rows(manifest_path):
        if row["split"] == "train_seen" and row["dataset_kind"] == "single" and row["product_id"]:
            if not include_transparent_sources and risky_transparent_source(row):
                excluded_rows += 1
                excluded_products.add(row["product_id"])
                continue
            rows_by_product[row["product_id"]].append(row)
    if not rows_by_product:
        raise SystemExit(f"No train_seen single rows found in {manifest_path}")
    stats = {
        "excluded_transparent_rows": excluded_rows,
        "excluded_transparent_products": len(excluded_products),
        "include_transparent_sources": include_transparent_sources,
    }
    return rows_by_product, stats


def metadata_groups(rows_by_product: dict[str, list[dict[str, str]]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for product_id, rows in rows_by_product.items():
        row = rows[0]
        key = f"{row.get('shape','')}|{row.get('color','')}"
        groups[key].append(product_id)
    return {key: values for key, values in groups.items() if len(values) >= 2}


def choose_combo_size(rng: random.Random, ratios: tuple[float, float, float]) -> int:
    value = rng.random()
    if value < ratios[0]:
        return 2
    if value < ratios[0] + ratios[1]:
        return 3
    return 4


def product_heap(rows_by_product: dict[str, list[dict[str, str]]]) -> list[tuple[int, str]]:
    return [(0, product_id) for product_id in rows_by_product]


def choose_products(
    rng: random.Random,
    heap: list[tuple[int, str]],
    counts: collections.Counter,
    rows_by_product: dict[str, list[dict[str, str]]],
    groups: dict[str, list[str]],
    combo_size: int,
    hard_mix: bool,
) -> list[str]:
    chosen: list[str] = []
    skipped: list[tuple[int, str]] = []
    while heap and len(chosen) < combo_size:
        count, product_id = heapq.heappop(heap)
        if count != counts[product_id]:
            continue
        if product_id in chosen:
            skipped.append((count, product_id))
            continue
        chosen.append(product_id)
        if hard_mix and len(chosen) == 1:
            anchor_row = rows_by_product[product_id][0]
            key = f"{anchor_row.get('shape','')}|{anchor_row.get('color','')}"
            candidates = [item for item in groups.get(key, []) if item != product_id]
            rng.shuffle(candidates)
            for candidate in sorted(candidates[:16], key=lambda item: counts[item]):
                if len(chosen) >= combo_size:
                    break
                if candidate not in chosen:
                    chosen.append(candidate)
    for item in skipped:
        heapq.heappush(heap, item)
    for product_id in chosen:
        counts[product_id] += 1
        heapq.heappush(heap, (counts[product_id], product_id))
    return chosen


def choose_source_row(rng: random.Random, product_rows: list[dict[str, str]], per_product_cursor: dict[str, int], product_id: str):
    cursor = per_product_cursor[product_id]
    if cursor % len(product_rows) == 0:
        rng.shuffle(product_rows)
    row = product_rows[cursor % len(product_rows)]
    per_product_cursor[product_id] += 1
    return row


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


def place_cutouts(cutouts, width: int, height: int, max_iou: float, rng: random.Random):
    cells = placement_cells(len(cutouts), width, height)
    rng.shuffle(cells)
    placements = []
    bboxes = []
    for cutout, cell in zip(cutouts, cells):
        x1, y1, x2, y2 = cell
        cell_w = max(24, x2 - x1)
        cell_h = max(24, y2 - y1)
        max_scale = min(1.12, (cell_w * 0.82) / max(1, cutout.width), (cell_h * 0.82) / max(1, cutout.height))
        min_scale = min(0.86, max_scale)
        scale = rng.uniform(max(0.42, min_scale), max(0.44, max_scale))
        scaled = cutout.resize(
            (max(12, int(cutout.width * scale)), max(12, int(cutout.height * scale))),
            Image.Resampling.LANCZOS,
        )
        for _ in range(80):
            max_x = max(x1, x2 - scaled.width)
            max_y = max(y1, y2 - scaled.height)
            x = rng.randint(x1, max_x) if max_x > x1 else x1
            y = rng.randint(y1, max_y) if max_y > y1 else y1
            fg_bbox = alpha_bbox(scaled.getchannel("A")) or [0, 0, scaled.width, scaled.height]
            bbox = [x + fg_bbox[0], y + fg_bbox[1], fg_bbox[2], fg_bbox[3]]
            if x < 0 or y < 0 or x + scaled.width > width or y + scaled.height > height:
                continue
            if all(iou_xywh(bbox, existing) <= max_iou for existing in bboxes):
                placements.append((scaled, (x, y), bbox))
                bboxes.append(bbox)
                break
        else:
            fallback_scale = min(scale, (cell_w * 0.62) / max(1, cutout.width), (cell_h * 0.62) / max(1, cutout.height))
            scaled = cutout.resize(
                (max(12, int(cutout.width * fallback_scale)), max(12, int(cutout.height * fallback_scale))),
                Image.Resampling.LANCZOS,
            )
            x = x1 + max(0, (cell_w - scaled.width) // 2)
            y = y1 + max(0, (cell_h - scaled.height) // 2)
            fg_bbox = alpha_bbox(scaled.getchannel("A")) or [0, 0, scaled.width, scaled.height]
            bbox = [x + fg_bbox[0], y + fg_bbox[1], fg_bbox[2], fg_bbox[3]]
            if any(iou_xywh(bbox, existing) > max_iou for existing in bboxes):
                raise RuntimeError("Could not place synthetic cutout without overlap")
            placements.append((scaled, (x, y), bbox))
            bboxes.append(bbox)
    return placements


def open_shard(output_dir: Path, shard_index: int, overwrite: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"train_combo_synth-{shard_index:06d}.tar"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Shard exists: {path}. Pass --overwrite to replace.")
    return path, tarfile.open(path, "w")


def row_bbox(row: dict[str, str]) -> list[float]:
    annotations = json.loads(row["annotations"])
    return annotations[0]["bbox"]


def build_mask_provider(args) -> Sam2MaskProvider | None:
    if args.mask_provider != "sam2":
        return None
    return Sam2MaskProvider(
        checkpoint=args.sam2_checkpoint,
        config=args.sam2_config,
        device=args.sam2_device,
        multimask=True,
        logit_threshold=args.sam2_logit_threshold,
        box_expansion_ratio=args.sam2_box_expansion_ratio,
    )


def accepted_mask_methods(args) -> set[str]:
    if args.mask_provider == "grabcut":
        return {"opencv_grabcut_bbox"}
    methods = {"sam2_bbox"}
    if args.allow_grabcut_fallback:
        methods.add("opencv_grabcut_bbox")
    return methods


def choose_good_cutout(
    rng: random.Random,
    zip_cache: ZipCache,
    rows_by_product: dict[str, list[dict[str, str]]],
    cursors: dict[str, int],
    product_id: str,
    min_mask_score: float,
    retries: int,
    mask_provider: Sam2MaskProvider | None,
    accepted_methods: set[str],
):
    best = None
    for _ in range(max(1, retries)):
        source_row = choose_source_row(rng, rows_by_product[product_id], cursors, product_id)
        image_zip = zip_cache.get(source_row["source_zip"])
        image_data = image_zip.read(source_row["image_member"])
        with Image.open(BytesIO(image_data)) as image:
            image = image.convert("RGB")
            cutout, quality = crop_cutout(
                image,
                row_bbox(source_row),
                source_row.get("shape", ""),
                margin_ratio=rng.uniform(0.18, 0.26),
                mask_provider=mask_provider,
            )
            profile_image = image.copy()
        if best is None or quality.get("score", 0.0) > best[2].get("score", 0.0):
            best = (source_row, cutout, quality, profile_image)
        if quality.get("score", 0.0) >= min_mask_score and quality.get("method") in accepted_methods:
            return source_row, cutout, quality, profile_image
    return None


def generate(args) -> tuple[list[dict], dict]:
    rng = random.Random(args.seed)
    rows_by_product, source_filter_stats = load_train_seen_rows(
        args.processed_root,
        include_transparent_sources=args.include_transparent_sources,
    )
    groups = metadata_groups(rows_by_product)
    heap = product_heap(rows_by_product)
    heapq.heapify(heap)
    counts: collections.Counter = collections.Counter()
    cursors = {product_id: 0 for product_id in rows_by_product}
    zip_cache = ZipCache()
    mask_provider = build_mask_provider(args)
    accepted_methods = accepted_mask_methods(args)
    output_dir = args.processed_root / "webdataset" / "train_combo_synth"
    manifest_rows: list[dict] = []
    shard_index = 0
    count_in_shard = 0
    shard_paths = []
    shard_path, tar = open_shard(output_dir, shard_index, args.overwrite)
    shard_paths.append(str(shard_path))
    try:
        ratios = (args.combo2_ratio, args.combo3_ratio, args.combo4_ratio)
        attempts = 0
        skipped_low_quality = 0
        while len(manifest_rows) < args.num_images:
            attempts += 1
            if attempts > args.num_images * 50:
                raise RuntimeError(
                    f"Too many skipped synthetic attempts: generated={len(manifest_rows)} "
                    f"target={args.num_images} skipped_low_quality={skipped_low_quality}"
                )
            combo_size = choose_combo_size(rng, ratios)
            product_ids = choose_products(
                rng,
                heap,
                counts,
                rows_by_product,
                groups,
                combo_size,
                hard_mix=rng.random() < args.hard_mix_ratio,
            )
            source_rows = []
            cutouts = []
            mask_qualities = []
            profile_image = None
            cutout_failed = False
            for product_id in product_ids:
                result = choose_good_cutout(
                    rng,
                    zip_cache,
                    rows_by_product,
                    cursors,
                    product_id,
                    args.min_mask_score,
                    args.cutout_retries_per_product,
                    mask_provider,
                    accepted_methods,
                )
                if result is None:
                    cutout_failed = True
                    break
                source_row, cutout, quality, candidate_profile = result
                source_rows.append(source_row)
                cutouts.append(cutout)
                mask_qualities.append(quality)
                if profile_image is None:
                    profile_image = candidate_profile
            if cutout_failed:
                skipped_low_quality += 1
                for product_id in product_ids:
                    counts[product_id] -= 1
                continue
            canvas = synthetic_background(args.width, args.height, rng, profile_image).convert("RGBA")
            placements = place_cutouts(cutouts, args.width, args.height, args.max_iou, rng)
            annotations = []
            source_refs = []
            for ann_index, (source_row, product_id, (_, xy, bbox), quality) in enumerate(
                zip(source_rows, product_ids, placements, mask_qualities), start=1
            ):
                paste_with_shadow(canvas, placements[ann_index - 1][0], xy, rng)
                annotations.append(
                    {
                        "id": ann_index,
                        "bbox": [int(v) for v in bbox],
                        "area": int(bbox[2] * bbox[3]),
                        "category_id": 1,
                        "product_id": product_id,
                        "class_index": source_row.get("class_index", ""),
                        "ignore_for_id": False,
                        "source_sample_id": source_row["sample_id"],
                    }
                )
                source_refs.append(
                    {
                        "sample_id": source_row["sample_id"],
                        "source_zip": source_row["source_zip"],
                        "image_member": source_row["image_member"],
                        "product_id": product_id,
                    }
                )
            final_image = apply_common_photometric(canvas.convert("RGB"), rng)
            image_buffer = BytesIO()
            final_image.save(image_buffer, format="JPEG", quality=95, subsampling=0, optimize=True)
            image_bytes = image_buffer.getvalue()
            sample_index = len(manifest_rows)
            sample_id = f"combo_synth_v1_{sample_index:09d}"
            key = sanitize_tar_key(sample_id)
            add_tar_bytes(tar, f"{key}.jpg", image_bytes)
            row = {
                "sample_id": sample_id,
                "dataset_kind": "combo_synth",
                "split_source": "combo_synth_v1",
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
                "transform": json_compact({"placement": "grid_jitter", "photometric": "global_jitter_noise"}),
                "mask_quality": json_compact({"instances": mask_qualities}),
                "ignore_for_id": "false",
                "class_index": "",
            }
            add_tar_bytes(tar, f"{key}.json", json.dumps(canonical_json(row), ensure_ascii=False).encode("utf-8"))
            manifest_rows.append(row)
            count_in_shard += 1
            if count_in_shard >= args.shard_size:
                tar.close()
                shard_index += 1
                count_in_shard = 0
                shard_path, tar = open_shard(output_dir, shard_index, args.overwrite)
                shard_paths.append(str(shard_path))
            if len(manifest_rows) % 1000 == 0:
                print(f"generated {len(manifest_rows)}/{args.num_images}", flush=True)
        tar.close()
        tar = None
        if count_in_shard == 0 and manifest_rows:
            empty_path = Path(shard_paths.pop())
            empty_path.unlink(missing_ok=True)
    finally:
        if tar is not None:
            tar.close()
        zip_cache.close()
    summary = {
        "num_images": len(manifest_rows),
        "shards": len(shard_paths),
        "shard_paths": shard_paths,
        "product_instance_count_min": min(counts.values()) if counts else 0,
        "product_instance_count_max": max(counts.values()) if counts else 0,
        "product_count": len(counts),
        "attempts": attempts,
        "skipped_low_quality": skipped_low_quality,
        "min_mask_score": args.min_mask_score,
        "mask_provider": args.mask_provider,
        "accepted_mask_methods": sorted(accepted_methods),
        "sam2_checkpoint": str(args.sam2_checkpoint) if args.mask_provider == "sam2" else "",
        "sam2_config": args.sam2_config if args.mask_provider == "sam2" else "",
        "sam2_device": getattr(mask_provider, "device", "") if mask_provider else "",
        "sam2_logit_threshold": args.sam2_logit_threshold if args.mask_provider == "sam2" else "",
        "sam2_box_expansion_ratio": args.sam2_box_expansion_ratio if args.mask_provider == "sam2" else "",
        **source_filter_stats,
    }
    return manifest_rows, summary


def main() -> int:
    args = build_parser().parse_args()
    manifest_rows, summary = generate(args)
    manifest_dir = args.processed_root / "manifests"
    csv_path = manifest_dir / "combo_synth_v1_manifest.csv"
    write_csv(csv_path, CANONICAL_FIELDS, manifest_rows)
    parquet_written = False
    if not args.skip_parquet:
        parquet_written = optional_write_parquet(csv_path, manifest_dir / "combo_synth_v1_manifest.parquet")
    summary.update({"manifest_csv": str(csv_path), "parquet_written": parquet_written})
    summary_path = manifest_dir / "combo_synth_v1_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
