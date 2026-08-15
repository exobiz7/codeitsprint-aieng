#!/usr/bin/env python3
"""Build photometric-only real augmentation to balance synthetic Kaggle releases."""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import random
import shutil
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


DEFAULT_DATA_ROOT = Path("/Volumes/SSD 4T/01_sprint_ai_project1_data")
DEFAULT_SYNTH_RELEASE = (
    DEFAULT_DATA_ROOT
    / "processed"
    / "kaggle_sam2_synth_v2_release"
    / "kaggle_sam2_synth_v2_kaggle_1500_696style_floor10_realbalanced"
)
DEFAULT_SPEC_DIR = (
    DEFAULT_DATA_ROOT
    / "processed"
    / "kaggle_sam2_synth_v2"
    / "spec"
    / "codex_handoff"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_SYNTH_RELEASE)
    parser.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    parser.add_argument("--num-aug-images", type=int, default=208)
    parser.add_argument("--target-total-real", type=int, default=0)
    parser.add_argument("--include-original-real", action="store_true")
    parser.add_argument("--output-name", default="real_balancer_aug208")
    parser.add_argument("--file-prefix", default="kaggle_real_aug_for_696plus1500")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def read_class_map(spec_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = []
    with (spec_dir / "class_map_56.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "class_index": int(row["class_index"]),
                    "category_id": int(row["category_id"]),
                    "product_id": row["K_code"],
                    "name": row["product_name"],
                    "shape": row.get("shape", ""),
                    "color": row.get("color", ""),
                    "imprint_front": row.get("imprint_front", ""),
                }
            )
    return rows, {row["product_id"]: row for row in rows}


def bbox_inside_image(bbox: list[float], width: int, height: int) -> bool:
    x, y, w, h = bbox
    return w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= width and y + h <= height


def collect_real_annotations(
    data_root: Path, class_by_product: dict[str, dict]
) -> tuple[list[dict], dict[str, list[dict]], list[dict]]:
    images: dict[str, dict] = {}
    anns_by_file: dict[str, list[dict]] = collections.defaultdict(list)
    dropped_invalid: list[dict] = []
    ann_id = 1
    for json_path in sorted((data_root / "train_annotations").rglob("*.json")):
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        image_meta = doc["images"][0]
        ann = doc["annotations"][0]
        file_name = image_meta["file_name"]
        width = int(image_meta["width"])
        height = int(image_meta["height"])
        product_id = image_meta.get("drug_N") or image_meta.get("dl_mapping_code")
        class_row = class_by_product.get(product_id)
        if class_row is None:
            raise RuntimeError(f"Product not in class map: {product_id} at {json_path}")
        if int(ann["category_id"]) != int(class_row["category_id"]):
            raise RuntimeError(f"Category mismatch: {json_path}")
        bbox = [float(v) for v in ann["bbox"]]
        if not bbox_inside_image(bbox, width, height):
            dropped_invalid.append(
                {
                    "annotation_path": str(json_path),
                    "file_name": file_name,
                    "product_id": product_id,
                    "bbox": bbox,
                    "width": width,
                    "height": height,
                }
            )
            continue
        images.setdefault(
            file_name,
            {
                "file_name": file_name,
                "width": width,
                "height": height,
                "source_image_path": str(data_root / "train_images" / file_name),
            },
        )
        anns_by_file[file_name].append(
            {
                "id": ann_id,
                "category_id": int(class_row["category_id"]),
                "class_index": int(class_row["class_index"]),
                "product_id": product_id,
                "bbox": bbox,
                "area": float(ann.get("area", ann["bbox"][2] * ann["bbox"][3])),
                "iscrowd": 0,
                "segmentation": [],
                "source_annotation_path": str(json_path),
            }
        )
        ann_id += 1
    ordered_images = [images[name] for name in sorted(images)]
    return ordered_images, anns_by_file, dropped_invalid


def image_weight(image_row: dict, anns_by_file: dict[str, list[dict]], class_counts: collections.Counter) -> float:
    anns = anns_by_file[image_row["file_name"]]
    rarity = [1.0 / max(1, class_counts[ann["product_id"]]) for ann in anns]
    return max(rarity) + 0.15 * sum(rarity)


def weak_photometric(image: Image.Image, rng: random.Random) -> Image.Image:
    image = image.convert("RGB")
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.94, 1.06))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.95, 1.07))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.96, 1.06))
    if rng.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.05, 0.25)))
    arr = np.asarray(image, dtype=np.float32).copy()
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(0.25, 0.9), arr.shape)
    return Image.fromarray(np.uint8(np.clip(arr + noise, 0, 255)), "RGB")


def write_coco_and_images(
    args,
    class_rows: list[dict],
    real_images: list[dict],
    anns_by_file: dict[str, list[dict]],
    dropped_invalid: list[dict],
) -> dict:
    rng = random.Random(args.seed)
    output_root = args.release_root / args.output_name
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Real balancer exists: {output_root}. Pass --overwrite.")
        shutil.rmtree(output_root)
    image_dir = output_root / "coco" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)
    (output_root / "webdataset" / "train").mkdir(parents=True, exist_ok=True)

    class_counts = collections.Counter()
    for anns in anns_by_file.values():
        class_counts.update(ann["product_id"] for ann in anns)
    weights = [image_weight(image, anns_by_file, class_counts) for image in real_images]
    num_aug_images = args.num_aug_images
    if args.target_total_real:
        original_count = len(real_images) if args.include_original_real else 0
        num_aug_images = args.target_total_real - original_count
        if num_aug_images < 0:
            raise ValueError(
                f"target_total_real={args.target_total_real} is smaller than original_count={original_count}"
            )
    selected_aug = rng.choices(real_images, weights=weights, k=num_aug_images)
    selected_original = list(real_images) if args.include_original_real else []

    coco_images = []
    coco_annotations = []
    ann_id = 1
    source_counter = collections.Counter()
    class_aug_counter = collections.Counter()

    def add_sample(index: int, source: dict, image: Image.Image, real_augmented: bool, prefix: str) -> None:
        nonlocal ann_id
        file_name = f"{prefix}_{index:06d}.jpg"
        image.save(image_dir / file_name, format="JPEG", quality=95, subsampling=0, optimize=True)
        image_id = len(coco_images) + 1
        coco_images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": source["width"],
                "height": source["height"],
                "real_augmented": real_augmented,
                "real_original_reencoded": not real_augmented,
                "source_file_name": source["file_name"],
            }
        )
        source_counter[source["file_name"]] += 1
        for source_ann in anns_by_file[source["file_name"]]:
            ann = dict(source_ann)
            ann["id"] = ann_id
            ann["image_id"] = image_id
            coco_annotations.append(ann)
            class_aug_counter[ann["product_id"]] += 1
            ann_id += 1

    for index, source in enumerate(selected_original, start=1):
        source_path = Path(source["source_image_path"])
        with Image.open(source_path) as image:
            original = image.convert("RGB")
        add_sample(index, source, original, False, f"{args.file_prefix}_original")

    for index, source in enumerate(selected_aug, start=1):
        source_path = Path(source["source_image_path"])
        with Image.open(source_path) as image:
            aug = weak_photometric(image, rng)
        add_sample(index, source, aug, True, f"{args.file_prefix}_aug")

    categories = [
        {
            "id": row["category_id"],
            "name": row["name"],
            "supercategory": "pill",
            "product_id": row["product_id"],
            "class_index": row["class_index"],
            "shape": row["shape"],
            "color": row["color"],
            "imprint_front": row["imprint_front"],
        }
        for row in sorted(class_rows, key=lambda item: item["class_index"])
    ]
    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
        "info": {
            "name": args.output_name,
            "augmentation": "photometric_only_no_geometry_bbox_unchanged",
            "num_original_images": len(selected_original),
            "num_aug_images": len(selected_aug),
            "target_total_real": args.target_total_real or len(coco_images),
            "intended_ratio": "For synthetic 1500, target_total_real=750 gives real:synth image ratio 1:2.",
        },
    }
    (output_root / "coco" / "annotations_coco.json").write_text(
        json.dumps(coco, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with (output_root / "reports" / "class_aug_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_index", "product_id", "category_id", "real_balancer_instances"])
        writer.writeheader()
        for row in sorted(class_rows, key=lambda item: item["class_index"]):
            writer.writerow(
                {
                    "class_index": row["class_index"],
                    "product_id": row["product_id"],
                    "category_id": row["category_id"],
                    "real_balancer_instances": class_aug_counter[row["product_id"]],
                }
            )

    summary = {
        "images": len(coco_images),
        "annotations": len(coco_annotations),
        "original_images": len(selected_original),
        "augmented_images": len(selected_aug),
        "dropped_invalid_source_annotations": len(dropped_invalid),
        "source_unique_images": len(source_counter),
        "source_reuse_min": min(source_counter.values()) if source_counter else 0,
        "source_reuse_max": max(source_counter.values()) if source_counter else 0,
        "class_aug_instance_min": min(class_aug_counter.values()) if class_aug_counter else 0,
        "class_aug_instance_max": max(class_aug_counter.values()) if class_aug_counter else 0,
    }
    (output_root / "reports" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "reports" / "dropped_invalid_source_annotations.json").write_text(
        json.dumps(dropped_invalid, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    build_webdataset(output_root, coco, args.shard_size)
    return {"output_root": str(output_root), **summary}


def build_webdataset(output_root: Path, coco: dict, shard_size: int) -> None:
    images = sorted(coco["images"], key=lambda image: int(image["id"]))
    anns_by_image: dict[int, list[dict]] = collections.defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)
    cat_by_id = {int(cat["id"]): cat for cat in coco["categories"]}
    rows = []
    for shard_idx, start in enumerate(range(0, len(images), shard_size)):
        shard_images = images[start : start + shard_size]
        shard_name = f"{output_root.name}-{shard_idx:06d}.tar"
        shard_path = output_root / "webdataset" / "train" / shard_name
        sample_count = 0
        ann_count = 0
        with tarfile.open(shard_path, "w") as tf:
            for image in shard_images:
                key = Path(image["file_name"]).stem
                jpg_bytes = (output_root / "coco" / "images" / image["file_name"]).read_bytes()
                sample_anns = sorted(anns_by_image[int(image["id"])], key=lambda ann: int(ann["id"]))
                sample_json = {
                    "__key__": key,
                    "image": image,
                    "annotations": sample_anns,
                    "categories": [cat_by_id[int(ann["category_id"])] for ann in sample_anns],
                    "real_augmented": bool(image.get("real_augmented")),
                    "real_original_reencoded": bool(image.get("real_original_reencoded")),
                    "dataset": output_root.name,
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
        rows.append({"shard": f"webdataset/train/{shard_name}", "samples": sample_count, "annotations": ann_count})
    with (output_root / "webdataset" / "shards_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["shard", "samples", "annotations"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = build_parser().parse_args()
    class_rows, class_by_product = read_class_map(args.spec_dir)
    real_images, anns_by_file, dropped_invalid = collect_real_annotations(args.data_root, class_by_product)
    result = write_coco_and_images(args, class_rows, real_images, anns_by_file, dropped_invalid)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
