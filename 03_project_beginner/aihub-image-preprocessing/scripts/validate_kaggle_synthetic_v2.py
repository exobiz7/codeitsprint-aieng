#!/usr/bin/env python3
"""Strict validator for Kaggle SAM2 synthetic v2 runs."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from v2_common import alpha_bbox, iou_xywh


DEFAULT_DATA_ROOT = Path("/Volumes/SSD 4T/01_sprint_ai_project1_data")
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "processed" / "kaggle_sam2_synth_v2"
K041768 = "K-041768"
FORBIDDEN_SOURCE_TOKENS = (
    "TL_2_조합",
    "TS_2_조합",
    "경구약제조합",
    "/01_sprint_ai_project1_data/train_images",
    "/01_sprint_ai_project1_data/train_annotations",
    "kaggle_sam2_synth_v1",
    "kaggle_30k",
)


@dataclass(frozen=True)
class ClassRow:
    class_index: int
    category_id: int
    product_id: str
    product_name: str
    shape: str
    color: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--spec-dir", type=Path, default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--expect-images", type=int, default=None)
    parser.add_argument("--expect-classes", type=int, default=56)
    parser.add_argument("--max-iou", type=float, default=0.02)
    parser.add_argument("--max-decode", type=int, default=0, help="Decode all images when 0.")
    parser.add_argument("--min-real-plus-synth", type=int, default=0)
    parser.add_argument("--assets-per-class", type=int, default=12)
    parser.add_argument("--min-mask-score", type=float, default=0.86)
    parser.add_argument("--background-tolerance", type=float, default=15.0)
    parser.add_argument("--skip-assets", action="store_true")
    parser.add_argument("--skip-domain", action="store_true")
    return parser


def read_class_map(spec_dir: Path) -> tuple[list[ClassRow], dict[str, ClassRow], dict[int, ClassRow]]:
    rows: list[ClassRow] = []
    with (spec_dir / "class_map_56.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                ClassRow(
                    class_index=int(row["class_index"]),
                    category_id=int(row["category_id"]),
                    product_id=row["K_code"],
                    product_name=row["product_name"],
                    shape=row.get("shape", ""),
                    color=row.get("color", ""),
                )
            )
    by_product = {row.product_id: row for row in rows}
    by_category = {row.category_id: row for row in rows}
    return rows, by_product, by_category


def read_real_counts(data_root: Path, class_by_product: dict[str, ClassRow]) -> tuple[collections.Counter, list[str]]:
    counts = collections.Counter()
    issues: list[str] = []
    for json_path in (data_root / "train_annotations").rglob("*.json"):
        try:
            doc = json.loads(json_path.read_text(encoding="utf-8"))
            image_meta = doc["images"][0]
            ann = doc["annotations"][0]
            product_id = image_meta.get("drug_N") or image_meta.get("dl_mapping_code")
            expected = class_by_product.get(product_id)
            if expected is None:
                issues.append(f"real annotation product not in class map: {json_path}")
                continue
            if int(ann["category_id"]) != expected.category_id:
                issues.append(f"real annotation category mismatch: {json_path}")
                continue
            counts[product_id] += 1
        except Exception as exc:  # noqa: BLE001
            issues.append(f"real annotation read failure: {json_path}: {exc}")
    return counts, issues[:20]


def load_domain_profile(spec_dir: Path) -> dict:
    path = spec_dir / "domain_profile.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def expand_bbox(bbox: list[float], width: int, height: int, pad: int = 24) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x1 = max(0, int(math.floor(x - pad)))
    y1 = max(0, int(math.floor(y - pad)))
    x2 = min(width, int(math.ceil(x + w + pad)))
    y2 = min(height, int(math.ceil(y + h + pad)))
    return x1, y1, x2, y2


def background_mean(image_path: Path, annotations: list[dict], width: int, height: int) -> list[float]:
    with Image.open(image_path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    keep = np.ones((height, width), dtype=bool)
    for ann in annotations:
        x1, y1, x2, y2 = expand_bbox([float(v) for v in ann["bbox"]], width, height)
        keep[y1:y2, x1:x2] = False
    if keep.sum() < width * height * 0.35:
        keep[:] = True
    mean = arr[keep].reshape(-1, 3).mean(axis=0)
    return [round(float(v), 3) for v in mean]


def validate_categories(coco: dict, class_rows: list[ClassRow]) -> tuple[list[str], dict[int, ClassRow]]:
    issues = []
    expected_by_category = {row.category_id: row for row in class_rows}
    expected_by_product = {row.product_id: row for row in class_rows}
    categories = coco.get("categories", [])
    if len(categories) != len(class_rows):
        issues.append(f"category count {len(categories)} != {len(class_rows)}")
    seen_ids = set()
    for category in categories:
        category_id = int(category.get("id", -1))
        product_id = category.get("product_id", "")
        class_index = category.get("class_index")
        if category_id == 1:
            issues.append("category_id=1 exists in categories")
        expected = expected_by_product.get(product_id) or expected_by_category.get(category_id)
        if expected is None:
            issues.append(f"category not in class map: id={category_id} product_id={product_id}")
            continue
        if category_id != expected.category_id:
            issues.append(f"category id mismatch for {product_id}: {category_id} != {expected.category_id}")
        try:
            class_index_int = int(class_index)
        except (TypeError, ValueError):
            class_index_int = -1
        if class_index_int != expected.class_index:
            issues.append(f"category class_index mismatch for {product_id}: {class_index} != {expected.class_index}")
        seen_ids.add(category_id)
    missing = sorted(set(expected_by_category) - seen_ids)
    if missing:
        issues.append(f"categories missing ids: {missing[:10]} count={len(missing)}")
    return issues, expected_by_category


def validate_assets(
    output_root: Path,
    class_by_product: dict[str, ClassRow],
    assets_per_class: int,
    min_mask_score: float,
) -> tuple[dict[str, dict[str, str]], dict]:
    manifest_path = output_root / "assets" / "assets_manifest.csv"
    report = {
        "manifest_path": str(manifest_path),
        "rows": 0,
        "classes": 0,
        "min_assets_per_class": 0,
        "max_assets_per_class": 0,
        "source_kind_counts": {},
        "issues": [],
        "issue_samples": [],
    }
    if not manifest_path.exists():
        report["issues"].append(f"missing asset manifest: {manifest_path}")
        return {}, report

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_asset_id: dict[str, dict[str, str]] = {}
    counts = collections.Counter()
    source_kind_counts = collections.Counter()
    for row in rows:
        report["rows"] += 1
        asset_id = row.get("asset_id", "")
        product_id = row.get("product_id", "")
        source_kind = row.get("source_kind", "")
        source_kind_counts[source_kind] += 1
        if asset_id in by_asset_id:
            report["issues"].append(f"duplicate asset_id: {asset_id}")
        by_asset_id[asset_id] = row
        counts[product_id] += 1
        expected = class_by_product.get(product_id)
        if expected is None:
            report["issues"].append(f"asset product not in class map: {product_id}")
            continue
        try:
            category_id = int(row.get("category_id", -1))
            class_index = int(row.get("class_index", -1))
        except ValueError:
            category_id = -1
            class_index = -1
        if category_id == 1:
            report["issues"].append(f"asset has forbidden category_id=1: {asset_id}")
        if category_id != expected.category_id or class_index != expected.class_index:
            report["issues"].append(f"asset class mapping mismatch: {asset_id}")
        source_text = "|".join(
            row.get(field, "")
            for field in ("source_zip", "image_member", "source_asset_path", "asset_path")
        )
        for token in FORBIDDEN_SOURCE_TOKENS:
            if token in source_text:
                report["issues"].append(f"forbidden asset source token {token}: {asset_id}")
                break
        if source_kind not in {"service_asset_relinked", "aihub_single_topup"}:
            report["issues"].append(f"unexpected asset source_kind={source_kind}: {asset_id}")
        if "단일경구약제" not in source_text and source_kind == "aihub_single_topup":
            report["issues"].append(f"top-up asset source is not AI Hub single: {asset_id}")
        source_split = row.get("source_split", "")
        if product_id == K041768:
            if source_split != "val_official_single_ood":
                report["issues"].append(f"K-041768 asset must use validation single exception: {asset_id}")
        elif source_kind == "aihub_single_topup" and source_split != "train_seen":
            report["issues"].append(f"top-up asset must use train_seen: {asset_id}")
        elif source_kind == "service_asset_relinked" and "1.Training_jpeg_q95" not in source_text:
            report["issues"].append(f"service asset did not originate from training JPEG single: {asset_id}")
        quality_raw = row.get("quality") or "{}"
        try:
            quality = json.loads(quality_raw)
        except json.JSONDecodeError:
            quality = {}
            report["issues"].append(f"asset quality is not valid JSON: {asset_id}")
        if quality.get("method") and quality.get("method") != "sam2_bbox":
            report["issues"].append(f"asset was not masked by sam2_bbox: {asset_id}")
        if quality.get("score") is not None and float(quality.get("score", 0.0)) < min_mask_score:
            report["issues"].append(f"asset SAM2 score below threshold: {asset_id}")
        asset_path = Path(row.get("asset_path", ""))
        if not asset_path.exists():
            report["issues"].append(f"asset file missing: {asset_id}")
            continue
        try:
            with Image.open(asset_path) as image:
                rgba = image.convert("RGBA")
                bbox = alpha_bbox(rgba.getchannel("A"), threshold=24)
                if bbox is None or bbox[2] <= 0 or bbox[3] <= 0:
                    report["issues"].append(f"asset alpha bbox invalid: {asset_id}")
        except Exception as exc:  # noqa: BLE001
            report["issues"].append(f"asset decode failure: {asset_id}: {exc}")

    bad_counts = {
        product_id: counts[product_id]
        for product_id in sorted(class_by_product)
        if counts[product_id] != assets_per_class
    }
    if bad_counts:
        report["issues"].append(f"asset count per class mismatch: {bad_counts}")
    report["classes"] = len(counts)
    report["min_assets_per_class"] = min(counts.values()) if counts else 0
    report["max_assets_per_class"] = max(counts.values()) if counts else 0
    report["source_kind_counts"] = dict(source_kind_counts)
    report["issue_samples"] = report["issues"][:20]
    return by_asset_id, report


def validate_run(args, spec_dir: Path) -> dict:
    class_rows, class_by_product, _ = read_class_map(spec_dir)
    real_counts, real_count_issues = read_real_counts(args.data_root, class_by_product)
    run_root = args.output_root / "runs" / args.run_name
    annotation_path = run_root / "annotations_coco.json"
    image_dir = run_root / "images"
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    warnings: list[str] = []
    if len(class_rows) != args.expect_classes:
        issues.append(f"class_map rows {len(class_rows)} != {args.expect_classes}")
    if real_count_issues:
        issues.extend(real_count_issues)
    category_issues, _ = validate_categories(coco, class_rows)
    issues.extend(category_issues)

    asset_by_id = {}
    asset_report = {}
    if not args.skip_assets:
        asset_by_id, asset_report = validate_assets(
            args.output_root,
            class_by_product,
            args.assets_per_class,
            args.min_mask_score,
        )
        issues.extend(asset_report.get("issues", []))

    images = {int(image["id"]): image for image in coco.get("images", [])}
    anns_by_image: dict[int, list[dict]] = collections.defaultdict(list)
    synth_counts = collections.Counter()
    mapping_failures = 0
    bbox_failures = 0
    overlap_failures = 0
    duplicate_product_failures = 0
    source_asset_failures = 0
    missing_images = 0
    decode_failures = 0
    decoded = 0
    bbox_sides = []
    bbox_rel_areas = []
    for ann in coco.get("annotations", []):
        image_id = int(ann.get("image_id", -1))
        anns_by_image[image_id].append(ann)
        product_id = ann.get("product_id")
        expected = class_by_product.get(product_id or "")
        if expected is None:
            mapping_failures += 1
            continue
        if int(ann.get("category_id", -1)) == 1:
            mapping_failures += 1
        if int(ann.get("category_id", -1)) != expected.category_id:
            mapping_failures += 1
        if "class_index" not in ann or int(ann.get("class_index", -1)) != expected.class_index:
            mapping_failures += 1
        synth_counts[product_id] += 1
        if asset_by_id:
            asset = asset_by_id.get(ann.get("source_asset_id", ""))
            if asset is None or asset.get("product_id") != product_id:
                source_asset_failures += 1

    background_means = []
    domain_checked = 0
    for image_id, image in images.items():
        width = int(image.get("width", 0))
        height = int(image.get("height", 0))
        image_path = image_dir / image.get("file_name", "")
        if not image_path.exists():
            missing_images += 1
        elif args.max_decode == 0 or decoded < args.max_decode:
            try:
                with Image.open(image_path) as im:
                    im.load()
                    if im.size != (width, height):
                        decode_failures += 1
                decoded += 1
            except Exception:  # noqa: BLE001
                decode_failures += 1
        annotations = anns_by_image.get(image_id, [])
        if int(image.get("combo_size", len(annotations))) != len(annotations):
            bbox_failures += 1
        product_ids = [ann.get("product_id") for ann in annotations]
        if len(product_ids) != len(set(product_ids)):
            duplicate_product_failures += 1
        for ann in annotations:
            x, y, w, h = [float(v) for v in ann.get("bbox", [])]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
                bbox_failures += 1
            bbox_sides.append(math.sqrt(max(0.0, w * h)))
            bbox_rel_areas.append((w * h) / max(1, width * height))
        for index, left in enumerate(annotations):
            for right in annotations[index + 1 :]:
                if iou_xywh(left["bbox"], right["bbox"]) > args.max_iou:
                    overlap_failures += 1
        if not args.skip_domain and image_path.exists():
            background_means.append(background_mean(image_path, annotations, width, height))
            domain_checked += 1

    if args.expect_images is not None and len(images) != args.expect_images:
        issues.append(f"image count {len(images)} != expected {args.expect_images}")
    if len(coco.get("annotations", [])) != sum(len(v) for v in anns_by_image.values()):
        issues.append("annotation grouping count mismatch")
    if mapping_failures:
        issues.append(f"class/product/category/class_index mapping failures: {mapping_failures}")
    if source_asset_failures:
        issues.append(f"annotation source_asset mapping failures: {source_asset_failures}")
    if missing_images:
        issues.append(f"missing images: {missing_images}")
    if decode_failures:
        issues.append(f"decode failures: {decode_failures}")
    if bbox_failures:
        issues.append(f"bbox/combo failures: {bbox_failures}")
    if overlap_failures:
        issues.append(f"instance IoU failures over {args.max_iou}: {overlap_failures}")
    if duplicate_product_failures:
        issues.append(f"duplicate product in image failures: {duplicate_product_failures}")
    if len(synth_counts) != args.expect_classes:
        missing = sorted(set(class_by_product) - set(synth_counts))
        issues.append(f"used class count {len(synth_counts)} != {args.expect_classes}; missing={missing}")
    total_counts = collections.Counter(real_counts)
    total_counts.update(synth_counts)
    if args.min_real_plus_synth:
        low = {product_id: total_counts[product_id] for product_id in class_by_product if total_counts[product_id] < args.min_real_plus_synth}
        if low:
            issues.append(f"real+synth class count below {args.min_real_plus_synth}: {low}")

    profile = load_domain_profile(spec_dir)
    target = np.asarray(profile.get("background_rgb_target_RGB", [112, 130, 154]), dtype=np.float32)
    background_report = {}
    if background_means:
        bg = np.asarray(background_means, dtype=np.float32)
        bg_min = bg.min(axis=0).round(3).tolist()
        bg_max = bg.max(axis=0).round(3).tolist()
        bg_mean = bg.mean(axis=0).round(3).tolist()
        out_of_range = np.abs(bg - target[None, :]) > args.background_tolerance
        blue_fail = bg[:, 2] <= bg[:, 0]
        fail_count = int(out_of_range.any(axis=1).sum() + blue_fail.sum())
        if fail_count:
            issues.append(
                f"background RGB target failures: tolerance={args.background_tolerance}, "
                f"out_of_range_images={int(out_of_range.any(axis=1).sum())}, blue_fail={int(blue_fail.sum())}"
            )
        background_report = {
            "checked": domain_checked,
            "target_rgb": target.astype(int).tolist(),
            "tolerance": args.background_tolerance,
            "mean_rgb": bg_mean,
            "min_rgb": bg_min,
            "max_rgb": bg_max,
            "out_of_range_images": int(out_of_range.any(axis=1).sum()),
            "blue_fail_images": int(blue_fail.sum()),
        }
    if bbox_sides and profile:
        side = np.asarray(bbox_sides, dtype=np.float32)
        rel_area = np.asarray(bbox_rel_areas, dtype=np.float32)
        side_q = np.quantile(side, [0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist()
        area_q = np.quantile(rel_area, [0.05, 0.25, 0.5, 0.75, 0.95]).round(5).tolist()
        expected_side = profile.get("bbox_side_px_pct", {})
        if expected_side:
            lo = float(expected_side["0.05"]) - 35.0
            hi = float(expected_side["0.95"]) + 35.0
            if side_q[0] < lo or side_q[-1] > hi:
                warnings.append(f"bbox side distribution outside relaxed profile range: p05/p95={side_q[0]}/{side_q[-1]}")
    else:
        side_q = []
        area_q = []

    report = {
        "run_name": args.run_name,
        "run_root": str(run_root),
        "images": len(images),
        "annotations": len(coco.get("annotations", [])),
        "categories": len(coco.get("categories", [])),
        "classes_used": len(synth_counts),
        "combo_size_counts": dict(collections.Counter(int(image.get("combo_size", 0)) for image in images.values())),
        "synth_class_instance_min": min(synth_counts.values()) if synth_counts else 0,
        "synth_class_instance_max": max(synth_counts.values()) if synth_counts else 0,
        "real_plus_synth_class_instance_min": min(total_counts[p] for p in class_by_product),
        "real_plus_synth_class_instance_max": max(total_counts[p] for p in class_by_product),
        "decoded": decoded,
        "mapping_failures": mapping_failures,
        "source_asset_failures": source_asset_failures,
        "missing_images": missing_images,
        "decode_failures": decode_failures,
        "bbox_failures": bbox_failures,
        "overlap_failures": overlap_failures,
        "duplicate_product_failures": duplicate_product_failures,
        "bbox_side_px_quantiles": side_q,
        "bbox_rel_area_quantiles": area_q,
        "background_report": background_report,
        "asset_report": asset_report,
        "issues": issues,
        "warnings": warnings,
        "total_issue_count": len(issues),
    }
    report_path = run_root / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    args = build_parser().parse_args()
    spec_dir = args.spec_dir or (args.output_root / "spec" / "codex_handoff")
    report = validate_run(args, spec_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
