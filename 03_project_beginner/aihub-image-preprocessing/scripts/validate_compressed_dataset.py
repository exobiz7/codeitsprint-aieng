#!/usr/bin/env python3
"""Validate compressed AI Hub JPEG/label zip pairs against original labels and manifests."""

from __future__ import annotations

import argparse
import json
import random
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from aihub_common import (
    DEFAULT_OUTPUT_TRAINING_ROOT,
    DEFAULT_TRAINING_ROOT,
    add_path_args,
    add_zip_args,
    existing_manifest_set_ids,
    extract_annotations,
    extract_primary_image,
    image_zip_name,
    iter_members,
    label_zip_name,
    load_json_bytes,
    manifest_value_to_member_list,
    manifest_dir_for,
    manifest_path_for,
    member_stem,
    read_manifest_rows,
    require_set_selection,
    resolve_dataset_paths,
    zip_path_for,
)


def import_pillow():
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required. Install it with: python -m pip install -r requirements.txt") from exc
    return Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_args(parser)
    add_zip_args(parser)
    parser.set_defaults(training_root=DEFAULT_TRAINING_ROOT, output_training_root=DEFAULT_OUTPUT_TRAINING_ROOT)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--set-ids", type=int, nargs="+")
    parser.add_argument("--all", action="store_true", help="Validate every manifest set.")
    parser.add_argument("--limit", type=int, help="Validate the first N manifest rows per set.")
    parser.add_argument("--sample-size", type=int, help="Validate N deterministic random rows per set.")
    parser.add_argument("--seed", type=int, default=20260703)
    return parser


def stems_in_zip(zip_file: ZipFile, suffix: str) -> dict[str, str]:
    return {member_stem(member): member for member in iter_members(zip_file.namelist(), suffix)}


def selected_rows(rows: list[dict], sample_size: int | None, seed: int, set_id: int) -> list[dict]:
    if sample_size is None or sample_size >= len(rows):
        return rows
    rng = random.Random(seed + set_id)
    indices = sorted(rng.sample(range(len(rows)), sample_size))
    return [rows[index] for index in indices]


def compare_label_docs(original_doc: dict, output_doc: dict, source: str) -> list[str]:
    issues: list[str] = []
    original_image = extract_primary_image(original_doc, source)
    output_image = extract_primary_image(output_doc, source)
    for key in ("width", "height"):
        if original_image.get(key) != output_image.get(key):
            issues.append(f"{source}: image {key} changed {original_image.get(key)} -> {output_image.get(key)}")
    for key in ("file_name", "imgfile"):
        value = output_image.get(key)
        if isinstance(value, str) and not value.lower().endswith(".jpg"):
            issues.append(f"{source}: output image field {key} does not end with .jpg: {value}")

    original_annotations = extract_annotations(original_doc, source)
    output_annotations = extract_annotations(output_doc, source)
    if len(original_annotations) != len(output_annotations):
        issues.append(f"{source}: annotation count changed")
        return issues
    for index, (original_annotation, output_annotation) in enumerate(zip(original_annotations, output_annotations)):
        for key in ("bbox", "area", "category_id", "iscrowd", "ignore", "image_id"):
            if original_annotation.get(key) != output_annotation.get(key):
                issues.append(
                    f"{source}: annotation[{index}].{key} changed "
                    f"{original_annotation.get(key)} -> {output_annotation.get(key)}"
                )
    return issues


def validate_jpeg_size(image_zip: ZipFile, member: str, expected_size: tuple[int, int], Image) -> str | None:
    try:
        with Image.open(BytesIO(image_zip.read(member))) as image:
            image.load()
            size = image.size
    except Exception as exc:
        return f"{member}: JPEG decode failed: {exc}"
    if size != expected_size:
        return f"{member}: JPEG size {size} != expected {expected_size}"
    return None


def validate_set(args, paths, set_id: int, Image) -> dict:
    manifest_dir = args.manifest_dir or manifest_dir_for(args.reports_dir)
    rows = list(read_manifest_rows(manifest_path_for(manifest_dir, set_id, args.image_prefix), limit=args.limit))
    rows_to_check = selected_rows(rows, args.sample_size, args.seed, set_id)
    expected_image_members = {row["output_image_member"] for row in rows}
    expected_label_members = {
        label_member
        for row in rows
        for label_member in manifest_value_to_member_list(row.get("label_members"), row.get("label_member"))
    }

    output_image_zip_path = zip_path_for(paths.output_source_dir, args.image_prefix, set_id, args.zip_kind)
    output_label_zip_path = zip_path_for(paths.output_label_dir, args.label_prefix, set_id, args.zip_kind)
    source_label_zip_path = zip_path_for(paths.label_dir, args.label_prefix, set_id, args.zip_kind)
    issues: list[str] = []

    with (
        ZipFile(output_image_zip_path) as output_image_zip,
        ZipFile(output_label_zip_path) as output_label_zip,
        ZipFile(source_label_zip_path) as source_label_zip,
    ):
        output_image_members = set(iter_members(output_image_zip.namelist(), ".jpg"))
        output_label_members = set(iter_members(output_label_zip.namelist(), ".json"))
        for member in sorted(expected_image_members - output_image_members)[:50]:
            issues.append(f"missing output image member: {member}")
        for member in sorted(expected_label_members - output_label_members)[:50]:
            issues.append(f"missing output label member: {member}")
        for member in sorted(output_image_members - expected_image_members)[:50]:
            issues.append(f"extra output image member: {member}")
        for member in sorted(output_label_members - expected_label_members)[:50]:
            issues.append(f"extra output label member: {member}")

        checked = 0
        checked_label_docs = 0
        for row in rows_to_check:
            source = f"{args.image_prefix}_{set_id}:{row['stem']}"
            first_original_doc = None
            for label_member in manifest_value_to_member_list(row.get("label_members"), row.get("label_member")):
                original_doc = load_json_bytes(
                    source_label_zip.read(label_member), f"{source_label_zip_path.name}:{label_member}"
                )
                output_doc = load_json_bytes(
                    output_label_zip.read(label_member),
                    f"{output_label_zip_path.name}:{label_member}",
                )
                issues.extend(compare_label_docs(original_doc, output_doc, f"{source}:{label_member}"))
                first_original_doc = first_original_doc or original_doc
                checked_label_docs += 1
            if first_original_doc is None:
                issues.append(f"{source}: no label members to validate")
                continue
            image = extract_primary_image(first_original_doc, source)
            expected_size = (int(image["width"]), int(image["height"]))
            jpeg_issue = validate_jpeg_size(output_image_zip, row["output_image_member"], expected_size, Image)
            if jpeg_issue:
                issues.append(jpeg_issue)
            checked += 1

    report = {
        "set_id": set_id,
        "image_prefix": args.image_prefix,
        "label_prefix": args.label_prefix,
        "zip_kind": args.zip_kind,
        "manifest_rows": len(rows),
        "checked_rows": checked,
        "checked_label_docs": checked_label_docs,
        "expected_label_docs": len(expected_label_members),
        "sampled": args.sample_size is not None,
        "issue_count": len(issues),
        "issues": issues[:200],
        "truncated_issues": len(issues) > 200,
    }
    validation_dir = args.reports_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / f"validation_{args.image_prefix}_{set_id}_{args.zip_kind}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    args = build_parser().parse_args()
    Image = import_pillow()
    paths = resolve_dataset_paths(args)
    manifest_dir = args.manifest_dir or manifest_dir_for(args.reports_dir)
    selected_set_ids = require_set_selection(args.set_ids, args.all)
    if selected_set_ids is None:
        selected_set_ids = existing_manifest_set_ids(manifest_dir, args.image_prefix)
    if not selected_set_ids:
        raise SystemExit(f"No manifest sets found under {manifest_dir}")

    reports = []
    for set_id in selected_set_ids:
        report = validate_set(args, paths, set_id, Image)
        reports.append(report)
        print(f"{args.image_prefix}_{set_id}: checked={report['checked_rows']} issues={report['issue_count']}")

    summary = {"reports": reports, "total_issue_count": sum(item["issue_count"] for item in reports)}
    validation_dir = args.reports_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 1 if summary["total_issue_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
