#!/usr/bin/env python3
"""Validate Kaggle-specialized synthetic pill combo runs."""

from __future__ import annotations

import argparse
import collections
import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from v2_common import iou_xywh


DEFAULT_OUTPUT_ROOT = Path("/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_sam2_synth_v1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-iou", type=float, default=0.02)
    parser.add_argument("--max-decode", type=int, default=0, help="Decode all images when 0.")
    parser.add_argument("--expect-classes", type=int, default=56)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_root = args.output_root / "runs" / args.run_name
    annotation_path = run_root / "annotations_coco.json"
    image_dir = run_root / "images"
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in coco["images"]}
    anns_by_image: dict[int, list[dict]] = collections.defaultdict(list)
    class_counter = collections.Counter()
    bbox_failures = 0
    overlap_failures = 0
    missing_images = 0
    decode_failures = 0
    decoded = 0
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
        class_counter[ann.get("product_id", str(ann["category_id"]))] += 1
    for image_id, image in images.items():
        width = int(image["width"])
        height = int(image["height"])
        image_path = image_dir / image["file_name"]
        if not image_path.exists():
            missing_images += 1
        elif args.max_decode == 0 or decoded < args.max_decode:
            try:
                with Image.open(image_path) as im:
                    im.load()
                    if im.size != (width, height):
                        decode_failures += 1
                decoded += 1
            except Exception:
                decode_failures += 1
        annotations = anns_by_image[image_id]
        if image.get("combo_size") != len(annotations):
            bbox_failures += 1
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
                bbox_failures += 1
        for index, left in enumerate(annotations):
            for right in annotations[index + 1 :]:
                if iou_xywh(left["bbox"], right["bbox"]) > args.max_iou:
                    overlap_failures += 1
    aihub_json_count = len(list((run_root / "annotations_aihub_like").rglob("*.json")))
    expected_aihub_json = len(coco["annotations"]) if (run_root / "annotations_aihub_like").exists() else 0
    aihub_json_failures = 0 if aihub_json_count == expected_aihub_json else abs(aihub_json_count - expected_aihub_json)
    issues = []
    if len(coco["categories"]) != args.expect_classes:
        issues.append(f"category count {len(coco['categories'])} != {args.expect_classes}")
    if len(class_counter) != args.expect_classes:
        issues.append(f"used class count {len(class_counter)} != {args.expect_classes}")
    if missing_images:
        issues.append(f"missing images: {missing_images}")
    if decode_failures:
        issues.append(f"decode failures: {decode_failures}")
    if bbox_failures:
        issues.append(f"bbox/combo failures: {bbox_failures}")
    if overlap_failures:
        issues.append(f"overlap failures: {overlap_failures}")
    if aihub_json_failures:
        issues.append(f"aihub json count mismatch: {aihub_json_count} vs {expected_aihub_json}")
    report = {
        "run_name": args.run_name,
        "images": len(images),
        "annotations": len(coco["annotations"]),
        "categories": len(coco["categories"]),
        "classes_used": len(class_counter),
        "class_instance_min": min(class_counter.values()) if class_counter else 0,
        "class_instance_max": max(class_counter.values()) if class_counter else 0,
        "combo_size_counts": dict(collections.Counter(image["combo_size"] for image in images.values())),
        "decoded": decoded,
        "missing_images": missing_images,
        "decode_failures": decode_failures,
        "bbox_failures": bbox_failures,
        "overlap_failures": overlap_failures,
        "aihub_json_count": aihub_json_count,
        "expected_aihub_json": expected_aihub_json,
        "issues": issues,
        "total_issue_count": len(issues),
    }
    report_path = run_root / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
