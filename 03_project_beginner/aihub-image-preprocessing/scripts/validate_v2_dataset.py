#!/usr/bin/env python3
"""Validate v2 manifests and WebDataset-style tar shards."""

from __future__ import annotations

import argparse
import collections
import json
import tarfile
from io import BytesIO
from pathlib import Path

from PIL import Image

from v2_common import DEFAULT_PROCESSED_ROOT, iou_xywh, json_loads_or, read_csv_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--validate-shards", action="store_true")
    parser.add_argument("--max-shard-samples", type=int, help="Decode at most N samples per shard for quick checks.")
    parser.add_argument("--max-iou", type=float, default=0.02)
    return parser


def validate_manifest(path: Path, max_iou: float) -> dict:
    issues = []
    rows = 0
    split_counts = collections.Counter()
    synthetic_validation_refs = 0
    product_id_failures = 0
    bbox_failures = 0
    combo_size_failures = 0
    official_combo_label_docs = 0
    for row in read_csv_rows(path):
        rows += 1
        split_counts[row.get("split", "")] += 1
        width = int(row["width"])
        height = int(row["height"])
        annotations = json_loads_or(row.get("annotations"), [])
        if row.get("dataset_kind") == "single" and not row.get("product_id"):
            product_id_failures += 1
        for annotation in annotations:
            bbox = annotation.get("bbox", [])
            if len(bbox) != 4:
                bbox_failures += 1
                continue
            x, y, w, h = bbox
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
                bbox_failures += 1
        if row.get("split") == "train_combo_synth":
            refs = json_loads_or(row.get("source_refs"), [])
            for ref in refs:
                source_zip = ref.get("source_zip", "")
                if "2.Validation" in source_zip or "Validation_jpeg_q95" in source_zip:
                    synthetic_validation_refs += 1
            combo_size = int(row.get("combo_size") or 0)
            if combo_size != len(annotations):
                combo_size_failures += 1
            for index, left in enumerate(annotations):
                for right in annotations[index + 1 :]:
                    if iou_xywh(left["bbox"], right["bbox"]) > max_iou:
                        combo_size_failures += 1
                        break
        if row.get("split") == "val_official_combo_real":
            official_combo_label_docs += len(json_loads_or(row.get("label_members"), []))

    if product_id_failures:
        issues.append(f"product_id extraction failures: {product_id_failures}")
    if bbox_failures:
        issues.append(f"bbox failures: {bbox_failures}")
    if synthetic_validation_refs:
        issues.append(f"synthetic validation refs: {synthetic_validation_refs}")
    if combo_size_failures:
        issues.append(f"combo annotation/overlap failures: {combo_size_failures}")
    return {
        "path": str(path),
        "rows": rows,
        "split_counts": dict(split_counts),
        "product_id_failures": product_id_failures,
        "bbox_failures": bbox_failures,
        "synthetic_validation_refs": synthetic_validation_refs,
        "combo_size_failures": combo_size_failures,
        "official_combo_label_docs": official_combo_label_docs,
        "issues": issues,
    }


def validate_tar(path: Path, max_samples: int | None) -> dict:
    jpg_keys = set()
    json_keys = set()
    decoded = 0
    issues = []
    with tarfile.open(path, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            stem = Path(member.name).stem
            if member.name.endswith(".jpg"):
                jpg_keys.add(stem)
                if max_samples is None or decoded < max_samples:
                    data = tar.extractfile(member).read()
                    with Image.open(BytesIO(data)) as image:
                        image.load()
                    decoded += 1
            elif member.name.endswith(".json"):
                json_keys.add(stem)
    missing_json = jpg_keys - json_keys
    missing_jpg = json_keys - jpg_keys
    if missing_json:
        issues.append(f"missing json count: {len(missing_json)}")
    if missing_jpg:
        issues.append(f"missing jpg count: {len(missing_jpg)}")
    return {
        "path": str(path),
        "jpg": len(jpg_keys),
        "json": len(json_keys),
        "decoded": decoded,
        "issues": issues,
    }


def main() -> int:
    args = build_parser().parse_args()
    manifest_dir = args.processed_root / "manifests"
    manifest_reports = []
    for name in ("split_manifest.csv", "combo_synth_v1_manifest.csv"):
        path = manifest_dir / name
        if path.exists():
            manifest_reports.append(validate_manifest(path, args.max_iou))
    shard_reports = []
    if args.validate_shards:
        for path in sorted((args.processed_root / "webdataset").glob("*/*.tar")):
            shard_reports.append(validate_tar(path, args.max_shard_samples))
    total_issues = sum(len(report["issues"]) for report in manifest_reports + shard_reports)
    report = {
        "manifest_reports": manifest_reports,
        "shard_reports": shard_reports,
        "total_issue_count": total_issues,
    }
    report_path = args.processed_root / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if total_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
