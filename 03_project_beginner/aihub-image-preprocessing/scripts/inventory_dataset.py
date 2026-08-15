#!/usr/bin/env python3
"""Inventory AI Hub source/label zip pairs and write paired stem manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from aihub_common import (
    DEFAULT_OUTPUT_TRAINING_ROOT,
    DEFAULT_TRAINING_ROOT,
    IMAGE_PREFIX,
    LABEL_PREFIX,
    MANIFEST_FIELDS,
    UNMATCHED_FIELDS,
    add_path_args,
    add_zip_args,
    bbox_to_manifest_value,
    image_zip_name,
    iter_members,
    label_zip_name,
    load_json_bytes,
    manifest_dir_for,
    manifest_path_for,
    member_list_to_manifest_value,
    member_stem,
    replace_member_suffix,
    resolve_dataset_paths,
    validate_bbox,
    write_tsv,
    zip_map,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_args(parser)
    add_zip_args(parser)
    parser.set_defaults(training_root=DEFAULT_TRAINING_ROOT, output_training_root=DEFAULT_OUTPUT_TRAINING_ROOT)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--set-ids", type=int, nargs="+")
    parser.add_argument("--limit-per-set", type=int, help="Write at most N paired rows per set for quick tests.")
    parser.add_argument(
        "--skip-label-validation",
        action="store_true",
        help="Skip JSON bbox validation while still writing paired manifests.",
    )
    return parser


def members_by_stem(zip_file: ZipFile, suffix: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    grouped, duplicates = members_list_by_stem(zip_file, suffix)
    return {stem: members[0] for stem, members in grouped.items()}, duplicates


def members_list_by_stem(zip_file: ZipFile, suffix: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    result: dict[str, list[str]] = {}
    for member in iter_members(zip_file.namelist(), suffix):
        stem = member_stem(member)
        result.setdefault(stem, []).append(member)
    duplicates = {stem: members for stem, members in result.items() if len(members) > 1}
    return result, duplicates


def inventory_set(
    set_id: int,
    source_zip_path: Path,
    label_zip_path: Path,
    manifest_dir: Path,
    unmatched_rows: list[dict],
    validate_labels: bool,
    limit_per_set: int | None,
    image_prefix: str,
) -> dict:
    with ZipFile(source_zip_path) as source_zip, ZipFile(label_zip_path) as label_zip:
        image_members, image_duplicates = members_by_stem(source_zip, ".png")
        label_member_lists, label_duplicates = members_list_by_stem(label_zip, ".json")
        image_stems = set(image_members)
        label_stems = set(label_member_lists)
        paired_stems = sorted(image_stems & label_stems)
        image_only_stems = sorted(image_stems - label_stems)
        label_only_stems = sorted(label_stems - image_stems)

        for stem in image_only_stems:
            unmatched_rows.append(
                {"set_id": set_id, "side": "image_only", "stem": stem, "member": image_members[stem]}
            )
        for stem in label_only_stems:
            unmatched_rows.append(
                {"set_id": set_id, "side": "label_only", "stem": stem, "member": label_member_lists[stem][0]}
            )

        invalid_json_count = 0
        invalid_bbox_count = 0
        manifest_rows: list[dict] = []
        for index, stem in enumerate(paired_stems):
            if limit_per_set is not None and index >= limit_per_set:
                break
            image_member = image_members[stem]
            label_members = label_member_lists[stem]
            label_member = label_members[0]
            width = ""
            height = ""
            bbox = ""
            if validate_labels:
                try:
                    label_doc = load_json_bytes(
                        label_zip.read(label_member), f"{label_zip_path.name}:{label_member}"
                    )
                    parsed_width, parsed_height, parsed_bbox = validate_bbox(
                        label_doc, f"{label_zip_path.name}:{label_member}"
                    )
                    width = parsed_width
                    height = parsed_height
                    bbox = bbox_to_manifest_value(parsed_bbox)
                except ValueError as exc:
                    if str(exc).startswith("Invalid JSON"):
                        invalid_json_count += 1
                    else:
                        invalid_bbox_count += 1

            manifest_rows.append(
                {
                    "set_id": set_id,
                    "stem": stem,
                    "image_member": image_member,
                    "label_member": label_member,
                    "output_image_member": replace_member_suffix(image_member, ".jpg"),
                    "output_label_member": label_member,
                    "label_members": member_list_to_manifest_value(label_members),
                    "width": width,
                    "height": height,
                    "bbox": bbox,
                }
            )

    written_rows = write_tsv(manifest_path_for(manifest_dir, set_id, image_prefix), MANIFEST_FIELDS, manifest_rows)
    status = "perfect"
    if image_only_stems or label_only_stems:
        status = "paired_with_unmatched"
    if invalid_json_count or invalid_bbox_count:
        status = "invalid_labels"

    return {
        "set_id": set_id,
        "status": status,
        "source_zip": str(source_zip_path),
        "label_zip": str(label_zip_path),
        "source_size_bytes": source_zip_path.stat().st_size,
        "label_size_bytes": label_zip_path.stat().st_size,
        "image_count": len(image_stems),
        "json_count": len(label_stems),
        "json_member_count": sum(len(members) for members in label_member_lists.values()),
        "paired_count": len(paired_stems),
        "manifest_rows": written_rows,
        "image_only_count": len(image_only_stems),
        "label_only_count": len(label_only_stems),
        "image_duplicate_stem_count": len(image_duplicates),
        "label_duplicate_stem_count": len(label_duplicates),
        "invalid_json_count": invalid_json_count,
        "invalid_bbox_count": invalid_bbox_count,
    }


def main() -> int:
    args = build_parser().parse_args()
    paths = resolve_dataset_paths(args)
    reports_dir = Path(args.reports_dir)
    manifest_dir = manifest_dir_for(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    source_zips = zip_map(paths.source_dir, args.image_prefix, args.zip_kind)
    label_zips = zip_map(paths.label_dir, args.label_prefix, args.zip_kind)
    selected_ids = set(args.set_ids) if args.set_ids else (set(source_zips) | set(label_zips))

    unmatched_rows: list[dict] = []
    set_summaries: list[dict] = []
    for set_id in sorted(selected_ids):
        source_zip = source_zips.get(set_id)
        label_zip = label_zips.get(set_id)
        if source_zip is None:
            set_summaries.append(
                {
                    "set_id": set_id,
                    "status": "label_only_set",
                    "source_zip": str(paths.source_dir / image_zip_name(set_id, args.zip_kind, args.image_prefix)),
                    "label_zip": str(label_zip)
                    if label_zip
                    else str(paths.label_dir / label_zip_name(set_id, args.zip_kind, args.label_prefix)),
                }
            )
            continue
        if label_zip is None:
            set_summaries.append(
                {
                    "set_id": set_id,
                    "status": "source_only_set",
                    "source_zip": str(source_zip),
                    "label_zip": str(paths.label_dir / label_zip_name(set_id, args.zip_kind, args.label_prefix)),
                }
            )
            continue
        summary = inventory_set(
            set_id=set_id,
            source_zip_path=source_zip,
            label_zip_path=label_zip,
            manifest_dir=manifest_dir,
            unmatched_rows=unmatched_rows,
            validate_labels=not args.skip_label_validation,
            limit_per_set=args.limit_per_set,
            image_prefix=args.image_prefix,
        )
        set_summaries.append(summary)
        print(
            f"{args.image_prefix}_{set_id}: paired={summary['paired_count']} "
            f"image_only={summary['image_only_count']} label_only={summary['label_only_count']} "
            f"status={summary['status']}"
        )

    write_tsv(reports_dir / "unmatched.tsv", UNMATCHED_FIELDS, unmatched_rows)
    summary_doc = {
        "source_dir": str(paths.source_dir),
        "label_dir": str(paths.label_dir),
        "output_source_dir": str(paths.output_source_dir),
        "output_label_dir": str(paths.output_label_dir),
        "image_prefix": args.image_prefix,
        "label_prefix": args.label_prefix,
        "zip_kind": args.zip_kind,
        "source_zip_count": len(source_zips),
        "label_zip_count": len(label_zips),
        "source_total_size_bytes": sum(path.stat().st_size for path in source_zips.values()),
        "label_total_size_bytes": sum(path.stat().st_size for path in label_zips.values()),
        "set_summaries": set_summaries,
        "unmatched_tsv": str(reports_dir / "unmatched.tsv"),
        "manifest_dir": str(manifest_dir),
    }
    (reports_dir / "inventory_summary.json").write_text(
        json.dumps(summary_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    processable = [item for item in set_summaries if item.get("paired_count")]
    print(
        f"inventory complete: source_zips={len(source_zips)} label_zips={len(label_zips)} "
        f"processable_sets={len(processable)} unmatched_rows={len(unmatched_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
