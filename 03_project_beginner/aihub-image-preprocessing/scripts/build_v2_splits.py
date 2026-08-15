#!/usr/bin/env python3
"""Create leakage-safe v2 splits and product-balanced sampler metadata."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

from v2_common import CANONICAL_FIELDS, DEFAULT_PROCESSED_ROOT, optional_write_parquet, read_csv_rows, write_csv


SPLIT_FIELDS = CANONICAL_FIELDS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--single-effective-target", type=int, default=432)
    parser.add_argument("--val-seen-ratio", type=float, default=0.10)
    parser.add_argument("--val-seen-min", type=int, default=24)
    parser.add_argument("--val-seen-max", type=int, default=128)
    parser.add_argument("--effective-beta", type=float, default=0.9999)
    parser.add_argument("--skip-parquet", action="store_true")
    return parser


def stable_score(sample_id: str, seed: str = "v2_split") -> int:
    digest = hashlib.sha1(f"{seed}|{sample_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def val_count_for_product(count: int, ratio: float, min_count: int, max_count: int) -> int:
    return min(max_count, max(min_count, int(round(count * ratio))))


def class_weight(count: int, beta: float) -> float:
    effective = (1.0 - beta**count) / (1.0 - beta)
    return 1.0 / effective


def load_base_rows(manifest_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name in ("train_single_raw", "val_official_single_ood", "val_official_combo_real"):
        path = manifest_dir / f"{name}.csv"
        if path.exists():
            rows.extend(read_csv_rows(path))
    return rows


def assign_splits(rows: list[dict[str, str]], args) -> tuple[list[dict[str, str]], dict]:
    by_product: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        if row["split_source"] == "train_single_raw":
            by_product[row["product_id"]].append(row)

    class_index = {product_id: index for index, product_id in enumerate(sorted(by_product))}
    val_seen_ids = set()
    for product_id, product_rows in by_product.items():
        sorted_rows = sorted(product_rows, key=lambda item: stable_score(item["sample_id"]))
        val_count = val_count_for_product(
            len(sorted_rows), args.val_seen_ratio, args.val_seen_min, args.val_seen_max
        )
        val_seen_ids.update(row["sample_id"] for row in sorted_rows[:val_count])

    output_rows: list[dict[str, str]] = []
    for row in rows:
        row = dict(row)
        if row["split_source"] == "train_single_raw":
            row["split"] = "val_seen_id" if row["sample_id"] in val_seen_ids else "train_seen"
            row["class_index"] = class_index[row["product_id"]]
            row["ignore_for_id"] = "false"
        elif row["split_source"] == "val_official_single_ood":
            row["split"] = "val_official_single_ood"
            row["class_index"] = ""
            row["ignore_for_id"] = "true"
        elif row["split_source"] == "val_official_combo_real":
            annotations = json.loads(row["annotations"])
            for annotation in annotations:
                product_id = annotation.get("product_id", "")
                annotation["class_index"] = class_index.get(product_id, "")
                annotation["ignore_for_id"] = product_id not in class_index
            row["split"] = "val_official_combo_real"
            row["annotations"] = json.dumps(annotations, ensure_ascii=False, separators=(",", ":"))
            row["ignore_for_id"] = "false"
        output_rows.append(row)

    train_counts = {product_id: len(product_rows) for product_id, product_rows in by_product.items()}
    sampler = {
        "single_effective_target": args.single_effective_target,
        "product_count": len(by_product),
        "class_index": class_index,
        "class_weights": {
            product_id: class_weight(count, args.effective_beta) for product_id, count in train_counts.items()
        },
        "train_seen_counts": collections.Counter(row["product_id"] for row in output_rows if row["split"] == "train_seen"),
        "val_seen_id_counts": collections.Counter(row["product_id"] for row in output_rows if row["split"] == "val_seen_id"),
        "sampler_policy": {
            "under_target": "replacement_sampling_plus_on_the_fly_augmentation",
            "over_target": "epoch_seeded_rotating_subsampling",
            "batch_sampler": "product_balanced",
        },
    }
    summary = {
        "rows_by_split": collections.Counter(row["split"] for row in output_rows),
        "product_count": len(by_product),
        "class_index_count": len(class_index),
        "val_seen_ratio": args.val_seen_ratio,
        "val_seen_min": args.val_seen_min,
        "val_seen_max": args.val_seen_max,
        "single_effective_target": args.single_effective_target,
    }
    return output_rows, {"summary": summary, "sampler": sampler}


def main() -> int:
    args = build_parser().parse_args()
    manifest_dir = args.processed_root / "manifests"
    rows = load_base_rows(manifest_dir)
    if not rows:
        raise SystemExit(f"No base manifests found in {manifest_dir}. Run build_v2_manifest.py first.")
    split_rows, docs = assign_splits(rows, args)
    split_csv = manifest_dir / "split_manifest.csv"
    count = write_csv(split_csv, SPLIT_FIELDS, split_rows)
    parquet_written = False
    if not args.skip_parquet:
        parquet_written = optional_write_parquet(split_csv, manifest_dir / "split_manifest.parquet")
    sampler_path = manifest_dir / "sampler_plan.json"
    sampler_json = json.dumps(docs["sampler"], ensure_ascii=False, indent=2, default=dict)
    sampler_path.write_text(sampler_json + "\n", encoding="utf-8")
    summary = {
        **docs["summary"],
        "split_manifest_csv": str(split_csv),
        "split_manifest_rows": count,
        "split_manifest_parquet_written": parquet_written,
        "sampler_plan": str(sampler_path),
    }
    summary_path = manifest_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=dict) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
