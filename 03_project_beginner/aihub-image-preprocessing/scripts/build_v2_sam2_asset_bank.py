#!/usr/bin/env python3
"""Build a reusable SAM2 cutout asset bank from v2 train_seen single-pill rows."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import shutil
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from v2_common import DEFAULT_PROCESSED_ROOT, Sam2MaskProvider, crop_cutout, read_csv_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--bank-name", default="sam2_large_a6")
    parser.add_argument("--assets-per-product", type=int, default=6)
    parser.add_argument("--candidates-per-product", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--margin-ratio", type=float, default=0.22)
    parser.add_argument("--min-mask-score", type=float, default=0.86)
    parser.add_argument("--include-transparent-sources", action="store_true")
    parser.add_argument("--only-products-file", type=Path)
    parser.add_argument("--append-existing", action="store_true")
    parser.add_argument("--sam2-checkpoint", type=Path, default=Path("models/sam2/sam2.1_hiera_large.pt"))
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-device", default="auto")
    parser.add_argument("--sam2-logit-threshold", type=float, default=0.8)
    parser.add_argument("--sam2-box-expansion-ratio", type=float, default=0.035)
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


def compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def risky_transparent_source(row: dict[str, str]) -> bool:
    text = " ".join([row.get("shape", ""), row.get("color", ""), row.get("product_id", "")])
    return any(token in text for token in ("투명", "반투명", "연질"))


def read_product_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    products = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not products:
        raise SystemExit(f"Empty product filter: {path}")
    return products


def row_bbox(row: dict[str, str]) -> list[float]:
    return json.loads(row["annotations"])[0]["bbox"]


def reservoir_sample_candidates(args) -> tuple[dict[str, list[dict[str, str]]], dict]:
    rng = random.Random(args.seed)
    product_filter = read_product_filter(args.only_products_file)
    manifest_path = args.processed_root / "manifests" / "split_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing split manifest: {manifest_path}")
    rows_by_product: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    seen = collections.Counter()
    excluded_transparent = collections.Counter()
    total_train_seen = 0
    for row in read_csv_rows(manifest_path):
        if row.get("split") != "train_seen" or row.get("dataset_kind") != "single" or not row.get("product_id"):
            continue
        if product_filter is not None and row["product_id"] not in product_filter:
            continue
        total_train_seen += 1
        if not args.include_transparent_sources and risky_transparent_source(row):
            excluded_transparent[row["product_id"]] += 1
            continue
        product_id = row["product_id"]
        seen[product_id] += 1
        bucket = rows_by_product[product_id]
        if len(bucket) < args.candidates_per_product:
            bucket.append(row)
        else:
            replace_at = rng.randrange(seen[product_id])
            if replace_at < args.candidates_per_product:
                bucket[replace_at] = row
    for bucket in rows_by_product.values():
        rng.shuffle(bucket)
    stats = {
        "total_train_seen_rows": total_train_seen,
        "candidate_products": len(rows_by_product),
        "candidate_rows": sum(len(rows) for rows in rows_by_product.values()),
        "excluded_transparent_rows": sum(excluded_transparent.values()),
        "excluded_transparent_products": len(excluded_transparent),
        "include_transparent_sources": args.include_transparent_sources,
        "candidates_per_product": args.candidates_per_product,
        "requested_products": len(product_filter) if product_filter is not None else 0,
        "requested_products_without_candidates": sorted(product_filter - set(rows_by_product)) if product_filter else [],
    }
    return rows_by_product, stats


def build_profile_image(image: Image.Image) -> Image.Image:
    return image.convert("RGB").resize((160, 210), Image.Resampling.BICUBIC)


def build_assets(args) -> dict:
    bank_root = args.processed_root / "asset_banks" / args.bank_name
    cutout_dir = bank_root / "cutouts"
    profile_dir = bank_root / "profiles"
    manifest_path = bank_root / "assets_manifest.csv"
    summary_path = bank_root / "assets_summary.json"
    if manifest_path.exists() and not args.overwrite and not args.append_existing:
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        print(json.dumps({"asset_bank": str(bank_root), "reused": True, **summary}, ensure_ascii=False, indent=2))
        return summary
    if args.overwrite and args.append_existing:
        raise SystemExit("--overwrite and --append-existing cannot be used together")
    if bank_root.exists() and args.overwrite:
        shutil.rmtree(bank_root)
    cutout_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    rows_by_product, sampling_stats = reservoir_sample_candidates(args)
    provider = Sam2MaskProvider(
        checkpoint=args.sam2_checkpoint,
        config=args.sam2_config,
        device=args.sam2_device,
        multimask=True,
        logit_threshold=args.sam2_logit_threshold,
        box_expansion_ratio=args.sam2_box_expansion_ratio,
    )
    zip_cache = ZipCache()
    manifest_rows = []
    failures = []
    accepted_by_product = collections.Counter()
    if args.append_existing and manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            manifest_rows = list(csv.DictReader(handle))
        accepted_by_product.update(row["product_id"] for row in manifest_rows)
        failures_path = bank_root / "asset_failures.json"
        if failures_path.exists():
            failures = json.loads(failures_path.read_text(encoding="utf-8"))
    try:
        for product_index, product_id in enumerate(sorted(rows_by_product), start=1):
            for candidate_index, row in enumerate(rows_by_product[product_id]):
                if accepted_by_product[product_id] >= args.assets_per_product:
                    break
                image_zip = zip_cache.get(row["source_zip"])
                image_data = image_zip.read(row["image_member"])
                with Image.open(BytesIO(image_data)) as image:
                    image = image.convert("RGB")
                    cutout, quality = crop_cutout(
                        image,
                        row_bbox(row),
                        row.get("shape", ""),
                        margin_ratio=args.margin_ratio,
                        mask_provider=provider,
                    )
                    profile = build_profile_image(image)
                if quality.get("method") != "sam2_bbox" or quality.get("score", 0.0) < args.min_mask_score:
                    failures.append(
                        {
                            "sample_id": row["sample_id"],
                            "product_id": product_id,
                            "quality": quality,
                        }
                    )
                    continue
                asset_id = f"{product_id}_{accepted_by_product[product_id]:02d}_{row['sample_id']}"
                safe_asset_id = asset_id.replace("/", "_")
                cutout_path = cutout_dir / f"{safe_asset_id}.png"
                profile_path = profile_dir / f"{safe_asset_id}.jpg"
                cutout.save(cutout_path, format="PNG", optimize=True)
                profile.save(profile_path, format="JPEG", quality=90, subsampling=0, optimize=True)
                manifest_rows.append(
                    {
                        "asset_id": safe_asset_id,
                        "asset_path": str(cutout_path),
                        "profile_path": str(profile_path),
                        "sample_id": row["sample_id"],
                        "product_id": product_id,
                        "class_index": row.get("class_index", ""),
                        "source_zip": row["source_zip"],
                        "image_member": row["image_member"],
                        "source_bbox": row["bbox"],
                        "width": cutout.width,
                        "height": cutout.height,
                        "shape": row.get("shape", ""),
                        "color": row.get("color", ""),
                        "back_color": row.get("back_color", ""),
                        "light_color": row.get("light_color", ""),
                        "camera_la": row.get("camera_la", ""),
                        "camera_lo": row.get("camera_lo", ""),
                        "quality": compact_json(quality),
                    }
                )
                accepted_by_product[product_id] += 1
            if product_index % 100 == 0:
                print(
                    f"products {product_index}/{len(rows_by_product)} assets={len(manifest_rows)} "
                    f"failed={len(failures)}",
                    flush=True,
                )
    finally:
        zip_cache.close()

    fields = [
        "asset_id",
        "asset_path",
        "profile_path",
        "sample_id",
        "product_id",
        "class_index",
        "source_zip",
        "image_member",
        "source_bbox",
        "width",
        "height",
        "shape",
        "color",
        "back_color",
        "light_color",
        "camera_la",
        "camera_lo",
        "quality",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    (bank_root / "asset_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "asset_bank": str(bank_root),
        "manifest_csv": str(manifest_path),
        "accepted_assets": len(manifest_rows),
        "failed_attempts": len(failures),
        "products_with_assets": len(accepted_by_product),
        "products_missing_assets": sorted(set(rows_by_product) - set(accepted_by_product)),
        "assets_per_product_target": args.assets_per_product,
        "assets_per_product_min": min(accepted_by_product.values()) if accepted_by_product else 0,
        "assets_per_product_max": max(accepted_by_product.values()) if accepted_by_product else 0,
        "sam2_checkpoint": str(args.sam2_checkpoint),
        "sam2_config": args.sam2_config,
        "sam2_device": getattr(provider, "device", args.sam2_device),
        "sam2_logit_threshold": args.sam2_logit_threshold,
        "sam2_box_expansion_ratio": args.sam2_box_expansion_ratio,
        "min_mask_score": args.min_mask_score,
        **sampling_stats,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    args = build_parser().parse_args()
    build_assets(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
