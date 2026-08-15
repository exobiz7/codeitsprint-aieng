#!/usr/bin/env python3
"""Build canonical v2 manifests from JPEG q95 AI Hub zip datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from aihub_common import manifest_value_to_member_list
from v2_common import (
    CANONICAL_FIELDS,
    DEFAULT_PROCESSED_ROOT,
    TRAIN_MANIFEST_DIR,
    TRAIN_SINGLE_LABEL_DIR,
    TRAIN_SINGLE_SOURCE_DIR,
    VAL_COMBO_LABEL_DIR,
    VAL_COMBO_MANIFEST_DIR,
    VAL_COMBO_SOURCE_DIR,
    VAL_SINGLE_LABEL_DIR,
    VAL_SINGLE_MANIFEST_DIR,
    VAL_SINGLE_SOURCE_DIR,
    ManifestInput,
    bbox_in_bounds,
    canonical_annotations_from_labels,
    extract_product_id,
    json_compact,
    label_metadata,
    load_json_bytes,
    optional_write_parquet,
    parse_combo_product_ids,
    phash_bytes,
    read_manifest_rows,
    stable_sample_id,
    write_csv,
    zip_map,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--limit-per-source", type=int)
    parser.add_argument("--no-hashes", action="store_true", help="Skip sha256/phash for quick pilot runs.")
    parser.add_argument("--skip-parquet", action="store_true")
    parser.add_argument("--include", choices=["all", "train", "validation"], default="all")
    return parser


def default_inputs() -> list[ManifestInput]:
    return [
        ManifestInput(
            split_source="train_single_raw",
            dataset_kind="single",
            paired_manifest_dir=TRAIN_MANIFEST_DIR,
            image_zip_dir=TRAIN_SINGLE_SOURCE_DIR,
            label_zip_dir=TRAIN_SINGLE_LABEL_DIR,
            image_prefix="TS",
            label_prefix="TL",
            zip_kind="단일",
        ),
        ManifestInput(
            split_source="val_official_single_ood",
            dataset_kind="single",
            paired_manifest_dir=VAL_SINGLE_MANIFEST_DIR,
            image_zip_dir=VAL_SINGLE_SOURCE_DIR,
            label_zip_dir=VAL_SINGLE_LABEL_DIR,
            image_prefix="VS",
            label_prefix="VL",
            zip_kind="단일",
        ),
        ManifestInput(
            split_source="val_official_combo_real",
            dataset_kind="combo",
            paired_manifest_dir=VAL_COMBO_MANIFEST_DIR,
            image_zip_dir=VAL_COMBO_SOURCE_DIR,
            label_zip_dir=VAL_COMBO_LABEL_DIR,
            image_prefix="VS",
            label_prefix="VL",
            zip_kind="조합",
        ),
    ]


def selected_inputs(include: str) -> list[ManifestInput]:
    inputs = default_inputs()
    if include == "train":
        return [item for item in inputs if item.split_source.startswith("train")]
    if include == "validation":
        return [item for item in inputs if item.split_source.startswith("val")]
    return inputs


def build_rows(input_spec: ManifestInput, limit_per_source: int | None, compute_hashes: bool):
    image_zips = zip_map(input_spec.image_zip_dir, input_spec.image_prefix, input_spec.zip_kind)
    label_zips = zip_map(input_spec.label_zip_dir, input_spec.label_prefix, input_spec.zip_kind)
    for manifest_path in sorted(input_spec.paired_manifest_dir.glob(f"{input_spec.image_prefix}_*_paired.tsv")):
        set_id = int(manifest_path.stem.split("_")[1])
        image_zip_path = image_zips.get(set_id)
        label_zip_path = label_zips.get(set_id)
        if image_zip_path is None or label_zip_path is None:
            raise FileNotFoundError(f"Missing image/label zip for {input_spec.image_prefix}_{set_id}")
        emitted = 0
        with ZipFile(image_zip_path) as image_zip, ZipFile(label_zip_path) as label_zip:
            for paired in read_manifest_rows(manifest_path):
                if limit_per_source is not None and emitted >= limit_per_source:
                    break
                label_members = manifest_value_to_member_list(paired.get("label_members"), paired.get("label_member"))
                label_docs = [
                    load_json_bytes(label_zip.read(member), f"{label_zip_path.name}:{member}")
                    for member in label_members
                ]
                first_doc = label_docs[0]
                meta = label_metadata(first_doc)
                width = int(meta["width"])
                height = int(meta["height"])
                if input_spec.dataset_kind == "combo":
                    combo_product_ids = parse_combo_product_ids(paired["stem"])
                    product_id = ""
                    annotations = canonical_annotations_from_labels(label_docs, combo_product_ids)
                else:
                    product_id = extract_product_id(paired["stem"], paired["image_member"], paired["label_member"])
                    combo_product_ids = []
                    annotations = canonical_annotations_from_labels(label_docs, [product_id])
                for annotation, member in zip(annotations, label_members):
                    annotation["source_label_member"] = member
                    if input_spec.split_source == "val_official_combo_real":
                        annotation["ignore_for_id"] = annotation["product_id"] not in _TRAIN_PRODUCT_IDS
                invalid_label_members = []
                valid_annotations = []
                for annotation in annotations:
                    bbox = annotation.get("bbox")
                    if isinstance(bbox, list) and bbox_in_bounds(bbox, width, height):
                        valid_annotations.append(annotation)
                    else:
                        invalid_label_members.append(annotation.get("source_label_member", ""))
                if invalid_label_members and input_spec.split_source != "val_official_combo_real":
                    raise ValueError(
                        f"Invalid bbox in {manifest_path}:{paired['stem']}: {invalid_label_members[:3]}"
                    )
                annotations = valid_annotations
                primary_bbox = annotations[0]["bbox"] if annotations else []
                if compute_hashes:
                    image_data = image_zip.read(paired["output_image_member"])
                    sha256 = hashlib.sha256(image_data).hexdigest()
                    phash = phash_bytes(image_data)
                else:
                    sha256 = ""
                    phash = ""
                sample_id = stable_sample_id(input_spec.split_source, image_zip_path.name, paired["stem"])
                ignore_for_id = input_spec.split_source == "val_official_single_ood"
                row = {
                    "sample_id": sample_id,
                    "dataset_kind": input_spec.dataset_kind,
                    "split_source": input_spec.split_source,
                    "split": input_spec.split_source,
                    "image_path": str(image_zip_path),
                    "image_member": paired["output_image_member"],
                    "label_path": str(label_zip_path),
                    "label_members": json_compact(label_members),
                    "product_id": product_id,
                    "combo_product_ids": json_compact(combo_product_ids),
                    "combo_size": len(combo_product_ids) if combo_product_ids else 1,
                    "bbox": json_compact(primary_bbox),
                    "annotations": json_compact(annotations),
                    "width": width,
                    "height": height,
                    "source_zip": str(image_zip_path),
                    "label_zip": str(label_zip_path),
                    "source_zip_name": image_zip_path.name,
                    "label_zip_name": label_zip_path.name,
                    "set_id": set_id,
                    "original_stem": paired["stem"],
                    "sha256": sha256,
                    "phash": phash,
                    "synthetic": "false",
                    "source_refs": json_compact([]),
                    "transform": json_compact(
                        {
                            "invalid_label_members": invalid_label_members,
                            "invalid_label_policy": "dropped_from_canonical_annotations"
                            if invalid_label_members
                            else "",
                        }
                    ),
                    "mask_quality": json_compact({}),
                    "ignore_for_id": "true" if ignore_for_id else "false",
                    "class_index": "",
                    **meta,
                }
                yield row
                emitted += 1


def write_manifest(processed_root: Path, name: str, rows, skip_parquet: bool) -> dict:
    manifest_dir = processed_root / "manifests"
    csv_path = manifest_dir / f"{name}.csv"
    count = write_csv(csv_path, CANONICAL_FIELDS, rows)
    parquet_path = manifest_dir / f"{name}.parquet"
    parquet_written = False
    if not skip_parquet:
        parquet_written = optional_write_parquet(csv_path, parquet_path)
    return {
        "name": name,
        "csv": str(csv_path),
        "parquet": str(parquet_path) if parquet_written else "",
        "rows": count,
        "parquet_written": parquet_written,
    }


def collect_train_product_ids() -> set[str]:
    ids = set()
    for manifest_path in sorted(TRAIN_MANIFEST_DIR.glob("TS_*_paired.tsv")):
        for paired in read_manifest_rows(manifest_path):
            product_id = extract_product_id(paired["stem"], paired["image_member"], paired["label_member"])
            if product_id:
                ids.add(product_id)
    return ids


_TRAIN_PRODUCT_IDS: set[str] = set()


def main() -> int:
    global _TRAIN_PRODUCT_IDS
    args = build_parser().parse_args()
    args.processed_root.mkdir(parents=True, exist_ok=True)
    _TRAIN_PRODUCT_IDS = collect_train_product_ids()
    compute_hashes = not args.no_hashes
    summaries = []
    for input_spec in selected_inputs(args.include):
        rows = build_rows(input_spec, args.limit_per_source, compute_hashes)
        summaries.append(write_manifest(args.processed_root, input_spec.split_source, rows, args.skip_parquet))
    summary = {
        "processed_root": str(args.processed_root),
        "compute_hashes": compute_hashes,
        "train_product_count": len(_TRAIN_PRODUCT_IDS),
        "manifests": summaries,
    }
    summary_path = args.processed_root / "manifests" / "base_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
