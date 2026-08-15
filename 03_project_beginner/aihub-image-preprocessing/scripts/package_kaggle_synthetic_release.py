#!/usr/bin/env python3
"""Package a Kaggle synthetic run as COCO + WebDataset release folder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path("/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_sam2_synth_v2")
DEFAULT_RELEASE_ROOT = Path("/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_sam2_synth_v2_release")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--release-name", default=None)
    parser.add_argument("--shard-prefix", default=None)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def package(args) -> dict:
    run_root = args.output_root / "runs" / args.run_name
    release_name = args.release_name or f"kaggle_sam2_synth_v2_{args.run_name}"
    shard_prefix = args.shard_prefix or release_name
    release = args.release_root / release_name
    if release.exists():
        if not args.overwrite:
            raise FileExistsError(f"Release exists: {release}. Pass --overwrite to replace.")
        shutil.rmtree(release)

    (release / "coco" / "images").mkdir(parents=True)
    (release / "webdataset" / "train").mkdir(parents=True)
    (release / "reports" / "audit").mkdir(parents=True)
    (release / "spec").mkdir(parents=True)

    copy_if_exists(run_root / "annotations_coco.json", release / "coco" / "annotations_coco.json")
    for name in ["summary.json", "validation_report.json"]:
        copy_if_exists(run_root / name, release / "reports" / name)
    for name in ["synthetic_manifest.csv", "class_instance_distribution.csv"]:
        copy_if_exists(run_root / "manifests" / name, release / "reports" / name)
    for path in sorted((run_root / "images").glob("*.jpg")):
        shutil.copy2(path, release / "coco" / "images" / path.name)
    for path in sorted((run_root / "audit").glob("*.jpg")):
        shutil.copy2(path, release / "reports" / "audit" / path.name)
    for name in ["class_map_56.csv", "class_map_56.json", "domain_profile.json", "SPEC.md"]:
        copy_if_exists(args.output_root / "spec" / "codex_handoff" / name, release / "spec" / name)
    copy_if_exists(args.output_root / "assets" / "assets_manifest.csv", release / "reports" / "assets_manifest.csv")

    coco = json.loads((run_root / "annotations_coco.json").read_text(encoding="utf-8"))
    images = sorted(coco["images"], key=lambda image: int(image["id"]))
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)
    cat_by_id = {int(cat["id"]): cat for cat in coco["categories"]}

    manifest_rows = []
    for shard_idx, start in enumerate(range(0, len(images), args.shard_size)):
        shard_images = images[start : start + args.shard_size]
        shard_name = f"{shard_prefix}-{shard_idx:06d}.tar"
        shard_path = release / "webdataset" / "train" / shard_name
        sample_count = 0
        ann_count = 0
        with tarfile.open(shard_path, "w") as tf:
            for image in shard_images:
                image_id = int(image["id"])
                key = Path(image["file_name"]).stem
                image_path = run_root / "images" / image["file_name"]
                jpg_bytes = image_path.read_bytes()
                sample_anns = sorted(anns_by_image.get(image_id, []), key=lambda ann: int(ann["id"]))
                sample_json = {
                    "__key__": key,
                    "image": image,
                    "annotations": sample_anns,
                    "categories": [cat_by_id[int(ann["category_id"])] for ann in sample_anns],
                    "synthetic": True,
                    "dataset": release_name,
                }
                jpg_info = tarfile.TarInfo(f"{key}.jpg")
                jpg_info.size = len(jpg_bytes)
                tf.addfile(jpg_info, io.BytesIO(jpg_bytes))
                json_bytes = json.dumps(sample_json, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                json_info = tarfile.TarInfo(f"{key}.json")
                json_info.size = len(json_bytes)
                tf.addfile(json_info, io.BytesIO(json_bytes))
                sample_count += 1
                ann_count += len(sample_anns)
        digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        manifest_rows.append(
            {
                "shard": f"webdataset/train/{shard_name}",
                "samples": sample_count,
                "annotations": ann_count,
                "sha256": digest,
                "bytes": shard_path.stat().st_size,
            }
        )

    with (release / "webdataset" / "shards_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["shard", "samples", "annotations", "sha256", "bytes"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    info = {
        "dataset_name": release_name,
        "format": ["COCO + images", "WebDataset"],
        "images": len(images),
        "annotations": len(coco["annotations"]),
        "classes": len(coco["categories"]),
        "source": "Synthetic combo images generated from AI Hub single-pill sources; K-041768 uses AI Hub validation single exception.",
        "class_map": "spec/class_map_56.csv",
        "coco_annotations": "coco/annotations_coco.json",
        "webdataset": "webdataset/train/*.tar",
    }
    (release / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = f"""# {release_name}

Ready-to-share Kaggle synthetic pill combo dataset.

## Contents
- `coco/images/*.jpg`: {len(images)} synthetic images
- `coco/annotations_coco.json`: COCO annotations with `category_id`, `class_index`, and `product_id`
- `webdataset/train/*.tar`: self-contained WebDataset shards, each sample has `.jpg` + `.json`
- `spec/class_map_56.csv`: 56-class SSOT
- `reports/validation_report.json`: strict validation report
- `reports/class_instance_distribution.csv`: per-class synthetic instance distribution
- `reports/audit/*.jpg`: visual audit contact sheets

## Counts
- images: {len(images)}
- annotations: {len(coco["annotations"])}
- classes: {len(coco["categories"])}
- WebDataset shards: {len(manifest_rows)}

This release excludes intermediate pilot runs and cutout PNG assets.
"""
    (release / "README.md").write_text(readme, encoding="utf-8")
    return {
        "release": str(release),
        "webdataset_shards": len(manifest_rows),
        "images": len(images),
        "annotations": len(coco["annotations"]),
    }


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(package(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
