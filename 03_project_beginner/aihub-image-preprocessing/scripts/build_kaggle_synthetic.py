#!/usr/bin/env python3
"""Build Kaggle-specialized synthetic pill combo data from labeled combo images."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import random
import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from v2_common import (
    Sam2MaskProvider,
    alpha_bbox,
    crop_cutout,
    iou_xywh,
    paste_with_shadow,
)


DEFAULT_KAGGLE_ROOT = Path("/Volumes/SSD 4T/01_sprint_ai_project1_data")
DEFAULT_OUTPUT_ROOT = DEFAULT_KAGGLE_ROOT / "processed" / "kaggle_sam2_synth_v1"


@dataclass
class SourceInstance:
    image_name: str
    image_path: Path
    json_path: Path
    product_id: str
    category_id: int
    category_name: str
    bbox: list[int]
    width: int
    height: int
    shape: str
    color: str
    back_color: str
    light_color: str
    camera_la: int
    camera_lo: int
    size: int
    image_meta: dict
    annotation: dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_KAGGLE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="pilot_100")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--width", type=int, default=976)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--min-mask-score", type=float, default=0.86)
    parser.add_argument("--margin-ratio", type=float, default=0.22)
    parser.add_argument("--max-iou", type=float, default=0.02)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--sam2-checkpoint", type=Path, default=Path("models/sam2/sam2.1_hiera_large.pt"))
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-device", default="auto")
    parser.add_argument("--sam2-logit-threshold", type=float, default=0.8)
    parser.add_argument("--sam2-box-expansion-ratio", type=float, default=0.035)
    parser.add_argument("--rebuild-assets", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-aihub-json", action=argparse.BooleanOptionalAction, default=True)
    return parser


def compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_source_instances(data_root: Path) -> list[SourceInstance]:
    image_dir = data_root / "train_images"
    annotation_dir = data_root / "train_annotations"
    instances = []
    for json_path in sorted(annotation_dir.rglob("*.json")):
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        image_meta = doc["images"][0]
        annotation = doc["annotations"][0]
        category = doc["categories"][0]
        image_name = image_meta["file_name"]
        image_path = image_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Annotation image not found: {image_path}")
        color = "|".join(part for part in [image_meta.get("color_class1", ""), image_meta.get("color_class2", "")] if part)
        instances.append(
            SourceInstance(
                image_name=image_name,
                image_path=image_path,
                json_path=json_path,
                product_id=image_meta.get("drug_N") or image_meta.get("dl_mapping_code", ""),
                category_id=int(annotation["category_id"]),
                category_name=category.get("name", ""),
                bbox=[int(round(v)) for v in annotation["bbox"]],
                width=int(image_meta.get("width", 976)),
                height=int(image_meta.get("height", 1280)),
                shape=str(image_meta.get("drug_shape") or ""),
                color=color,
                back_color=image_meta.get("back_color", ""),
                light_color=image_meta.get("light_color", ""),
                camera_la=int(image_meta.get("camera_la") or 0),
                camera_lo=int(image_meta.get("camera_lo") or 0),
                size=int(image_meta.get("size") or 0),
                image_meta=image_meta,
                annotation=annotation,
            )
        )
    return instances


def group_instances_by_image(instances: list[SourceInstance]) -> dict[str, list[SourceInstance]]:
    grouped: dict[str, list[SourceInstance]] = collections.defaultdict(list)
    for instance in instances:
        grouped[instance.image_name].append(instance)
    return dict(grouped)


def write_source_reports(instances: list[SourceInstance], output_root: Path) -> None:
    output_root.joinpath("manifests").mkdir(parents=True, exist_ok=True)
    rows = []
    for inst in instances:
        rows.append(
            {
                "image_name": inst.image_name,
                "json_path": str(inst.json_path),
                "product_id": inst.product_id,
                "category_id": inst.category_id,
                "category_name": inst.category_name,
                "bbox": compact_json(inst.bbox),
                "shape": inst.shape,
                "color": inst.color,
                "back_color": inst.back_color,
                "light_color": inst.light_color,
                "camera_la": inst.camera_la,
                "camera_lo": inst.camera_lo,
            }
        )
    fields = list(rows[0].keys()) if rows else []
    with (output_root / "manifests" / "source_instances.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source_images": len({inst.image_name for inst in instances}),
        "source_instances": len(instances),
        "classes": len({inst.product_id for inst in instances}),
        "instances_per_image": dict(collections.Counter(len(v) for v in group_instances_by_image(instances).values())),
        "back_color": dict(collections.Counter(inst.back_color for inst in instances)),
        "light_color": dict(collections.Counter(inst.light_color for inst in instances)),
        "shape": dict(collections.Counter(inst.shape for inst in instances)),
        "color": dict(collections.Counter(inst.color for inst in instances)),
        "class_instances": dict(collections.Counter(inst.product_id for inst in instances)),
    }
    (output_root / "manifests" / "source_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def asset_key(instance: SourceInstance, index: int) -> str:
    stem = Path(instance.image_name).stem
    return f"{instance.product_id}_{index:05d}_{stem}"


def valid_source_bbox(instance: SourceInstance) -> bool:
    x, y, w, h = instance.bbox
    return x >= 0 and y >= 0 and w > 0 and h > 0 and x < instance.width and y < instance.height and x + w > 0 and y + h > 0


def build_assets(args, instances: list[SourceInstance]) -> list[dict]:
    asset_dir = args.output_root / "assets" / "cutouts"
    manifest_path = args.output_root / "assets" / "assets_manifest.csv"
    if manifest_path.exists() and not args.rebuild_assets:
        return read_assets(manifest_path)
    if asset_dir.exists() and args.rebuild_assets:
        shutil.rmtree(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.joinpath("assets").mkdir(parents=True, exist_ok=True)

    provider = Sam2MaskProvider(
        checkpoint=args.sam2_checkpoint,
        config=args.sam2_config,
        device=args.sam2_device,
        multimask=True,
        logit_threshold=args.sam2_logit_threshold,
        box_expansion_ratio=args.sam2_box_expansion_ratio,
    )
    rows = []
    failures = []
    for index, instance in enumerate(instances):
        if not valid_source_bbox(instance):
            failures.append(
                {
                    "image_name": instance.image_name,
                    "product_id": instance.product_id,
                    "bbox": instance.bbox,
                    "quality": {"failure": "invalid_source_bbox"},
                }
            )
            continue
        with Image.open(instance.image_path) as image:
            image = image.convert("RGB")
            cutout, quality = crop_cutout(
                image,
                instance.bbox,
                instance.shape,
                margin_ratio=args.margin_ratio,
                mask_provider=provider,
            )
        if quality.get("method") != "sam2_bbox" or quality.get("score", 0.0) < args.min_mask_score:
            failures.append(
                {
                    "image_name": instance.image_name,
                    "product_id": instance.product_id,
                    "bbox": instance.bbox,
                    "quality": quality,
                }
            )
            continue
        key = asset_key(instance, index)
        asset_path = asset_dir / f"{key}.png"
        cutout.save(asset_path, format="PNG", optimize=True)
        row = {
            "asset_id": key,
            "asset_path": str(asset_path),
            "image_name": instance.image_name,
            "image_path": str(instance.image_path),
            "json_path": str(instance.json_path),
            "product_id": instance.product_id,
            "category_id": instance.category_id,
            "category_name": instance.category_name,
            "source_bbox": compact_json(instance.bbox),
            "width": cutout.width,
            "height": cutout.height,
            "shape": instance.shape,
            "color": instance.color,
            "back_color": instance.back_color,
            "light_color": instance.light_color,
            "camera_la": instance.camera_la,
            "camera_lo": instance.camera_lo,
            "size": instance.size,
            "quality": compact_json(quality),
            "image_meta": compact_json(instance.image_meta),
        }
        rows.append(row)
        if (index + 1) % 100 == 0:
            print(f"built assets {index + 1}/{len(instances)} accepted={len(rows)}", flush=True)
    fields = [
        "asset_id",
        "asset_path",
        "image_name",
        "image_path",
        "json_path",
        "product_id",
        "category_id",
        "category_name",
        "source_bbox",
        "width",
        "height",
        "shape",
        "color",
        "back_color",
        "light_color",
        "camera_la",
        "camera_lo",
        "size",
        "quality",
        "image_meta",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    failure_path = args.output_root / "assets" / "asset_failures.json"
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "accepted_assets": len(rows),
        "failed_assets": len(failures),
        "classes_with_assets": len({row["product_id"] for row in rows}),
        "classes_failed_all": sorted(
            set(inst.product_id for inst in instances) - set(row["product_id"] for row in rows)
        ),
        "sam2_checkpoint": str(args.sam2_checkpoint),
        "sam2_config": args.sam2_config,
        "sam2_device": getattr(provider, "device", args.sam2_device),
        "sam2_logit_threshold": args.sam2_logit_threshold,
        "sam2_box_expansion_ratio": args.sam2_box_expansion_ratio,
        "min_mask_score": args.min_mask_score,
    }
    (args.output_root / "assets" / "assets_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return rows


def read_assets(manifest_path: Path) -> list[dict]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def combo_size_distribution(instances: list[SourceInstance]) -> list[int]:
    counts = collections.Counter(len(v) for v in group_instances_by_image(instances).values())
    sizes = []
    for size in (2, 3, 4):
        sizes.extend([size] * counts[size])
    return sizes or [3]


def placement_cells(combo_size: int, width: int, height: int) -> list[tuple[int, int, int, int]]:
    pad_x, pad_y = 64, 86
    if combo_size == 2:
        return [(pad_x, pad_y, width // 2 - pad_x, height - pad_y), (width // 2, pad_y, width - pad_x, height - pad_y)]
    if combo_size == 3:
        return [
            (pad_x, pad_y, width // 2 - pad_x, height // 2),
            (width // 2, pad_y, width - pad_x, height // 2),
            (width // 4, height // 2, width * 3 // 4, height - pad_y),
        ]
    return [
        (pad_x, pad_y, width // 2 - pad_x, height // 2),
        (width // 2, pad_y, width - pad_x, height // 2),
        (pad_x, height // 2, width // 2 - pad_x, height - pad_y),
        (width // 2, height // 2, width - pad_x, height - pad_y),
    ]


def read_cutout(asset: dict) -> Image.Image:
    with Image.open(asset["asset_path"]) as image:
        return image.convert("RGBA")


def rotate_cutout(cutout: Image.Image, rng: random.Random) -> Image.Image:
    angle = rng.uniform(-8.0, 8.0)
    return cutout.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)


def place_cutouts(cutouts: list[Image.Image], width: int, height: int, max_iou: float, rng: random.Random):
    cells = placement_cells(len(cutouts), width, height)
    rng.shuffle(cells)
    placements = []
    bboxes = []
    for cutout, cell in zip(cutouts, cells):
        x1, y1, x2, y2 = cell
        cell_w = max(24, x2 - x1)
        cell_h = max(24, y2 - y1)
        max_scale = min(1.05, (cell_w * 0.84) / max(1, cutout.width), (cell_h * 0.84) / max(1, cutout.height))
        min_scale = max(0.42, min(0.88, max_scale * 0.78))
        if max_scale < 0.42:
            max_scale = 0.42
        scale = rng.uniform(min_scale, max_scale)
        scaled = cutout.resize(
            (max(12, int(round(cutout.width * scale))), max(12, int(round(cutout.height * scale)))),
            Image.Resampling.LANCZOS,
        )
        placed = False
        for _ in range(100):
            max_x = max(x1, x2 - scaled.width)
            max_y = max(y1, y2 - scaled.height)
            x = rng.randint(x1, max_x) if max_x > x1 else x1
            y = rng.randint(y1, max_y) if max_y > y1 else y1
            fg_bbox = alpha_bbox(scaled.getchannel("A")) or [0, 0, scaled.width, scaled.height]
            bbox = [x + fg_bbox[0], y + fg_bbox[1], fg_bbox[2], fg_bbox[3]]
            if x < 0 or y < 0 or x + scaled.width > width or y + scaled.height > height:
                continue
            if all(iou_xywh(bbox, existing) <= max_iou for existing in bboxes):
                placements.append((scaled, (x, y), [int(round(v)) for v in bbox]))
                bboxes.append(bbox)
                placed = True
                break
        if not placed:
            raise RuntimeError("Could not place cutout without overlap")
    return placements


def build_background_profile(image_path: Path, instances: list[SourceInstance], width: int, height: int) -> Image.Image:
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((160, 210), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32)
    mask = np.zeros(arr.shape[:2], dtype=bool)
    sx = arr.shape[1] / width
    sy = arr.shape[0] / height
    for inst in instances:
        x, y, w, h = inst.bbox
        margin = max(w, h) * 0.08
        x1 = max(0, int((x - margin) * sx))
        y1 = max(0, int((y - margin) * sy))
        x2 = min(arr.shape[1], int((x + w + margin) * sx))
        y2 = min(arr.shape[0], int((y + h + margin) * sy))
        mask[y1:y2, x1:x2] = True
    bg_pixels = arr[~mask]
    if len(bg_pixels) == 0:
        bg_pixels = arr.reshape(-1, 3)
    median = np.percentile(bg_pixels, 50, axis=0)
    std = np.clip(np.percentile(np.abs(bg_pixels - median), 70, axis=0), 2, 18)
    seed = int(hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    profile = rng.normal(median, std, arr.shape)
    yy = np.linspace(-1.0, 1.0, arr.shape[0], dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, arr.shape[1], dtype=np.float32)[None, :]
    grad_x = rng.uniform(-7.0, 7.0)
    grad_y = rng.uniform(-7.0, 7.0)
    profile += (xx[..., None] * grad_x + yy[..., None] * grad_y)
    image_profile = Image.fromarray(np.uint8(np.clip(profile, 0, 255)), "RGB").filter(
        ImageFilter.GaussianBlur(radius=2.2)
    )
    return image_profile


def make_background(
    rng: random.Random,
    profiles: list[Image.Image],
    width: int,
    height: int,
) -> Image.Image:
    profile = profiles[rng.randrange(len(profiles))]
    base = profile.resize((width, height), Image.Resampling.BICUBIC)
    arr = np.asarray(profile, dtype=np.float32)
    flat = arr.reshape(-1, 3)
    mean = np.percentile(flat, 50, axis=0)
    std = np.clip(np.percentile(np.abs(flat - mean), 70, axis=0), 2, 20)
    noise_shape = (max(16, height // 16), max(16, width // 16), 3)
    noise_rng = np.random.default_rng(rng.randrange(2**32))
    noise = noise_rng.normal(mean, std, noise_shape)
    noise_image = Image.fromarray(np.uint8(np.clip(noise, 0, 255)), "RGB").filter(ImageFilter.GaussianBlur(radius=1.2))
    noise_image = noise_image.resize((width, height), Image.Resampling.BICUBIC)
    blended = Image.blend(base, noise_image, rng.uniform(0.18, 0.32))
    bg = np.asarray(blended, dtype=np.float32)
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    radius = np.sqrt(xx * xx + yy * yy)
    vignette_strength = rng.uniform(-10.0, 8.0)
    vignette = (radius - radius.mean()) * vignette_strength
    bg += vignette[..., None]
    return Image.fromarray(np.uint8(np.clip(bg, 0, 255)), "RGB").filter(ImageFilter.GaussianBlur(radius=0.35))


def apply_kaggle_photometric(image: Image.Image, rng: random.Random) -> Image.Image:
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.96, 1.05))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.96, 1.07))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.97, 1.05))
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(0.35, 1.2), arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def product_heap(assets_by_product: dict[str, list[dict]]) -> list[tuple[int, str]]:
    return [(0, product_id) for product_id in assets_by_product]


def choose_products(
    rng: random.Random,
    heap: list[tuple[int, str]],
    counts: collections.Counter,
    combo_size: int,
) -> list[str]:
    import heapq

    chosen = []
    skipped = []
    while heap and len(chosen) < combo_size:
        count, product_id = heapq.heappop(heap)
        if count != counts[product_id]:
            continue
        if product_id in chosen:
            skipped.append((count, product_id))
            continue
        chosen.append(product_id)
    for item in skipped:
        heapq.heappush(heap, item)
    for product_id in chosen:
        counts[product_id] += 1
        heapq.heappush(heap, (counts[product_id], product_id))
    if len(chosen) != combo_size:
        raise RuntimeError(f"Could not choose {combo_size} unique products")
    return chosen


def category_rows(instances: list[SourceInstance]) -> list[dict]:
    by_id = {}
    for inst in instances:
        by_id[inst.category_id] = {
            "id": inst.category_id,
            "name": inst.category_name,
            "supercategory": "pill",
            "product_id": inst.product_id,
        }
    return [by_id[key] for key in sorted(by_id)]


def image_metadata_from_assets(selected_assets: list[dict], camera_la: int, width: int, height: int, file_name: str) -> dict:
    first_meta = json.loads(selected_assets[0].get("image_meta", "{}"))
    meta = dict(first_meta)
    meta.update(
        {
            "file_name": file_name,
            "imgfile": file_name,
            "width": width,
            "height": height,
            "drug_N": ",".join(asset["product_id"] for asset in selected_assets),
            "back_color": "연회색 배경",
            "light_color": "주백색",
            "camera_la": camera_la,
            "camera_lo": 0,
            "size": 200,
            "synthetic": True,
        }
    )
    return meta


def write_aihub_jsons(run_root: Path, file_name: str, image_meta: dict, annotations: list[dict], categories: list[dict]) -> None:
    combo_key = "-".join(category["product_id"] for category in categories)
    for annotation, category in zip(annotations, categories):
        doc = {
            "images": [dict(image_meta, drug_N=category["product_id"])],
            "type": "instances",
            "annotations": [
                {
                    "area": int(annotation["area"]),
                    "iscrowd": 0,
                    "bbox": annotation["bbox"],
                    "category_id": category["id"],
                    "ignore": 0,
                    "segmentation": [],
                    "id": annotation["id"],
                    "image_id": annotation["image_id"],
                }
            ],
            "categories": [
                {
                    "supercategory": "pill",
                    "id": category["id"],
                    "name": category["name"],
                }
            ],
        }
        out = run_root / "annotations_aihub_like" / f"{combo_key}_json" / category["product_id"] / f"{Path(file_name).stem}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def generate(args, instances: list[SourceInstance], assets: list[dict]) -> dict:
    rng = random.Random(args.seed)
    run_root = args.output_root / "runs" / args.run_name
    if run_root.exists() and args.overwrite:
        shutil.rmtree(run_root)
    if run_root.exists():
        raise FileExistsError(f"Run exists: {run_root}. Pass --overwrite to replace.")
    image_dir = run_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    (run_root / "manifests").mkdir(parents=True, exist_ok=True)

    grouped_images = group_instances_by_image(instances)
    profiles = [
        build_background_profile(insts[0].image_path, insts, args.width, args.height)
        for insts in grouped_images.values()
    ]
    assets_by_product: dict[str, list[dict]] = collections.defaultdict(list)
    for asset in assets:
        assets_by_product[asset["product_id"]].append(asset)
    if len(assets_by_product) < 2:
        raise RuntimeError("Need at least two classes with accepted assets")
    for product_assets in assets_by_product.values():
        rng.shuffle(product_assets)

    size_distribution = combo_size_distribution(instances)
    camera_distribution = [inst.camera_la for inst in instances if inst.camera_la] or [70, 75, 90]
    counts = collections.Counter()
    import heapq

    heap = product_heap(assets_by_product)
    heapq.heapify(heap)

    coco_images = []
    coco_annotations = []
    manifest_rows = []
    product_asset_cursor = collections.Counter()
    annotation_id = 1
    attempts = 0
    while len(coco_images) < args.num_images:
        attempts += 1
        if attempts > args.num_images * 20:
            raise RuntimeError("Too many failed generation attempts")
        combo_size = rng.choice(size_distribution)
        combo_size = min(combo_size, len(assets_by_product))
        product_ids = choose_products(rng, heap, counts, combo_size)
        selected_assets = []
        cutouts = []
        for product_id in product_ids:
            product_assets = assets_by_product[product_id]
            cursor = product_asset_cursor[product_id]
            if cursor % len(product_assets) == 0:
                rng.shuffle(product_assets)
            asset = product_assets[cursor % len(product_assets)]
            product_asset_cursor[product_id] += 1
            selected_assets.append(asset)
            cutouts.append(rotate_cutout(read_cutout(asset), rng))
        try:
            placements = place_cutouts(cutouts, args.width, args.height, args.max_iou, rng)
        except RuntimeError:
            continue
        canvas = make_background(rng, profiles, args.width, args.height).convert("RGBA")
        sample_index = len(coco_images)
        file_name = f"kaggle_synth_v1_{sample_index:06d}.jpg"
        image_id = sample_index + 1
        annotations = []
        for local_index, (asset, (cutout, xy, bbox)) in enumerate(zip(selected_assets, placements), start=1):
            paste_with_shadow(canvas, cutout, xy, rng)
            area = int(bbox[2] * bbox[3])
            ann = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(asset["category_id"]),
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": [],
                "product_id": asset["product_id"],
                "source_asset_id": asset["asset_id"],
            }
            annotations.append(ann)
            coco_annotations.append(ann)
            annotation_id += 1
        final_image = apply_kaggle_photometric(canvas.convert("RGB"), rng)
        image_path = image_dir / file_name
        final_image.save(image_path, format="JPEG", quality=args.jpeg_quality, subsampling=0, optimize=True)
        camera_la = rng.choice(camera_distribution)
        image_row = {
            "id": image_id,
            "file_name": file_name,
            "width": args.width,
            "height": args.height,
            "combo_size": combo_size,
            "product_ids": [asset["product_id"] for asset in selected_assets],
            "back_color": "연회색 배경",
            "light_color": "주백색",
            "camera_la": camera_la,
            "camera_lo": 0,
            "synthetic": True,
        }
        coco_images.append(image_row)
        selected_categories = [
            {
                "id": int(asset["category_id"]),
                "name": asset["category_name"],
                "product_id": asset["product_id"],
            }
            for asset in selected_assets
        ]
        image_meta = image_metadata_from_assets(selected_assets, camera_la, args.width, args.height, file_name)
        if args.write_aihub_json:
            write_aihub_jsons(run_root, file_name, image_meta, annotations, selected_categories)
        manifest_rows.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "combo_size": combo_size,
                "product_ids": compact_json(image_row["product_ids"]),
                "category_ids": compact_json([asset["category_id"] for asset in selected_assets]),
                "source_asset_ids": compact_json([asset["asset_id"] for asset in selected_assets]),
                "annotations": compact_json(annotations),
                "back_color": "연회색 배경",
                "light_color": "주백색",
                "camera_la": camera_la,
                "camera_lo": 0,
            }
        )
        if len(coco_images) % 1000 == 0:
            print(f"generated {len(coco_images)}/{args.num_images}", flush=True)

    categories = category_rows(instances)
    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
        "info": {
            "name": "kaggle_sam2_synth_v1",
            "source": str(args.data_root),
            "sam2_checkpoint": str(args.sam2_checkpoint),
            "sam2_config": args.sam2_config,
            "min_mask_score": args.min_mask_score,
            "back_color": "연회색 배경",
            "light_color": "주백색",
        },
    }
    annotation_path = run_root / "annotations_coco.json"
    annotation_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_fields = list(manifest_rows[0].keys())
    with (run_root / "manifests" / "synthetic_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary = {
        "run_name": args.run_name,
        "num_images": len(coco_images),
        "annotations": len(coco_annotations),
        "classes_total": len(categories),
        "classes_used": len({ann["product_id"] for ann in coco_annotations}),
        "combo_size_counts": dict(collections.Counter(image["combo_size"] for image in coco_images)),
        "class_instance_min": min(counts.values()) if counts else 0,
        "class_instance_max": max(counts.values()) if counts else 0,
        "attempts": attempts,
        "images_dir": str(image_dir),
        "annotation_path": str(annotation_path),
        "write_aihub_json": args.write_aihub_json,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = build_parser().parse_args()
    instances = read_source_instances(args.data_root)
    write_source_reports(instances, args.output_root)
    assets = build_assets(args, instances)
    summary = generate(args, instances, assets)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
