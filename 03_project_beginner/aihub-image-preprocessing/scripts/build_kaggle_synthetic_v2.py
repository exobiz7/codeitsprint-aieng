#!/usr/bin/env python3
"""Build Kaggle v2 synthetic combo data from AI Hub single-pill sources only."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import random
import shutil
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from v2_common import (
    DEFAULT_PROCESSED_ROOT,
    Sam2MaskProvider,
    alpha_bbox,
    crop_cutout,
    iou_xywh,
    json_compact,
    paste_with_shadow,
    trim_transparent,
)


DEFAULT_DATA_ROOT = Path("/Volumes/SSD 4T/01_sprint_ai_project1_data")
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "processed" / "kaggle_sam2_synth_v2"
DEFAULT_HANDOFF_ZIP = Path(
    "/Volumes/SSD 4T/166.약품식별 인공지능 개발을 위한 "
    "경구약제 이미지 데이터/01.데이터/processed/v2_pilot/codex_handoff.zip"
)
DEFAULT_AIHUB_ROOT = Path(
    "/Volumes/SSD 4T/166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터/01.데이터/processed/v2_sam2_full"
)
DEFAULT_SERVICE_BANK = DEFAULT_AIHUB_ROOT / "asset_banks" / "sam2_large_a6" / "assets_manifest.csv"
K041768 = "K-041768"
TARGET_RGB = np.array([112.0, 130.0, 154.0], dtype=np.float32)


@dataclass(frozen=True)
class ClassRow:
    class_index: int
    category_id: int
    product_id: str
    product_name: str
    shape: str
    color: str
    imprint_front: str


class ZipCache:
    def __init__(self) -> None:
        self._cache: dict[str, ZipFile] = {}

    def get(self, path: str) -> ZipFile:
        if path not in self._cache:
            self._cache[path] = ZipFile(path)
        return self._cache[path]

    def close(self) -> None:
        for handle in self._cache.values():
            handle.close()
        self._cache.clear()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--handoff-zip", type=Path, default=DEFAULT_HANDOFF_ZIP)
    parser.add_argument("--aihub-root", type=Path, default=DEFAULT_AIHUB_ROOT)
    parser.add_argument("--service-bank", type=Path, default=DEFAULT_SERVICE_BANK)
    parser.add_argument("--run-name", default="pilot_100")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--file-prefix", default="kaggle_synth_v2")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--width", type=int, default=976)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument(
        "--background-dir",
        type=Path,
        default=None,
        help="Optional directory of clean background PNGs. When set, this overrides spec/codex_handoff/backgrounds.",
    )
    parser.add_argument("--assets-per-class", type=int, default=12)
    parser.add_argument("--reuse-service-assets", type=int, default=6)
    parser.add_argument("--candidate-limit", type=int, default=192)
    parser.add_argument("--min-mask-score", type=float, default=0.86)
    parser.add_argument("--margin-ratio", type=float, default=0.22)
    parser.add_argument("--max-iou", type=float, default=0.02)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--sam2-checkpoint", type=Path, default=Path("models/sam2/sam2.1_hiera_large.pt"))
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-device", default="auto")
    parser.add_argument("--sam2-logit-threshold", type=float, default=0.8)
    parser.add_argument("--sam2-box-expansion-ratio", type=float, default=0.035)
    parser.add_argument(
        "--class-balance-mode",
        choices=["deficit_balanced", "synthetic_balanced"],
        default="deficit_balanced",
        help="deficit_balanced uses real+synth counts; synthetic_balanced balances synthetic instances only.",
    )
    parser.add_argument(
        "--synthetic-min-instances-per-class",
        type=int,
        default=0,
        help="Guarantee this minimum synthetic instance count per class before returning to the selected balance mode.",
    )
    parser.add_argument("--rebuild-assets", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def ensure_spec(output_root: Path, handoff_zip: Path) -> Path:
    spec_root = output_root / "spec"
    target_dir = spec_root / "codex_handoff"
    if not target_dir.exists():
        if not handoff_zip.exists():
            raise FileNotFoundError(f"Missing handoff zip: {handoff_zip}")
        spec_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(handoff_zip) as archive:
            archive.extractall(spec_root)
    return target_dir


def read_class_map(spec_dir: Path) -> tuple[list[ClassRow], dict[str, ClassRow]]:
    path = spec_dir / "class_map_56.csv"
    rows: list[ClassRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                ClassRow(
                    class_index=int(row["class_index"]),
                    category_id=int(row["category_id"]),
                    product_id=row["K_code"],
                    product_name=row["product_name"],
                    shape=row.get("shape", ""),
                    color=row.get("color", ""),
                    imprint_front=row.get("imprint_front", ""),
                )
            )
    if len(rows) != 56:
        raise RuntimeError(f"class_map_56.csv must contain 56 rows, got {len(rows)}")
    if sorted(row.class_index for row in rows) != list(range(56)):
        raise RuntimeError("class_index must be exactly 0..55")
    by_product = {row.product_id: row for row in rows}
    if len(by_product) != 56:
        raise RuntimeError("Duplicate K_code in class map")
    return rows, by_product


def read_domain_profile(spec_dir: Path) -> dict:
    return json.loads((spec_dir / "domain_profile.json").read_text(encoding="utf-8"))


def read_backgrounds(spec_dir: Path, background_dir: Path | None = None) -> tuple[list[Image.Image], str]:
    clean_names = {"bg_01.png", "bg_04.png", "bg_12.png"}
    if background_dir is not None:
        paths = sorted(background_dir.glob("*.png"))
        selected_paths = paths
        source_label = str(background_dir)
    else:
        paths = sorted((spec_dir / "backgrounds").glob("*.png"))
        clean_paths = [path for path in paths if path.name in clean_names]
        selected_paths = clean_paths if clean_paths else paths
        source_label = "codex_handoff/backgrounds"
    images = []
    for path in selected_paths:
        image = Image.open(path).convert("RGB")
        image.info["source_name"] = path.name
        images.append(image)
    if not images:
        raise RuntimeError(f"No background patches found in {background_dir or (spec_dir / 'backgrounds')}")
    return images, source_label


def read_real_counts(data_root: Path, class_by_product: dict[str, ClassRow]) -> collections.Counter:
    counts = collections.Counter()
    bad = []
    for json_path in (data_root / "train_annotations").rglob("*.json"):
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        image_meta = doc["images"][0]
        ann = doc["annotations"][0]
        product_id = image_meta.get("drug_N") or image_meta.get("dl_mapping_code")
        class_row = class_by_product.get(product_id)
        if class_row is None or int(ann["category_id"]) != class_row.category_id:
            bad.append(str(json_path))
            continue
        counts[product_id] += 1
    if bad:
        raise RuntimeError(f"Kaggle train annotation/class_map mismatch. Sample: {bad[:3]}")
    missing = sorted(set(class_by_product) - set(counts))
    if missing:
        raise RuntimeError(f"Kaggle train annotations missing classes: {missing}")
    return counts


def parse_bbox(row: dict[str, str]) -> list[float]:
    if row.get("annotations"):
        anns = json.loads(row["annotations"])
        if anns:
            return [float(v) for v in anns[0]["bbox"]]
    return [float(v) for v in json.loads(row["bbox"])]


def read_source_rows(aihub_root: Path, class_by_product: dict[str, ClassRow], candidate_limit: int, seed: int):
    rng = random.Random(seed)
    manifest = aihub_root / "manifests" / "split_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing AI Hub split manifest: {manifest}")
    wanted = set(class_by_product)
    rows_by_product: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    seen = collections.Counter()
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            product_id = row.get("product_id", "")
            if product_id not in wanted or row.get("dataset_kind") != "single":
                continue
            required_split = "val_official_single_ood" if product_id == K041768 else "train_seen"
            if row.get("split") != required_split:
                continue
            seen[product_id] += 1
            bucket = rows_by_product[product_id]
            if len(bucket) < candidate_limit:
                bucket.append(row)
            else:
                replace_at = rng.randrange(seen[product_id])
                if replace_at < candidate_limit:
                    bucket[replace_at] = row
    missing = sorted(wanted - set(rows_by_product))
    if missing:
        raise RuntimeError(f"Missing AI Hub single source rows for classes: {missing}")
    for product_id, rows in rows_by_product.items():
        rng.shuffle(rows)
        rows.sort(key=lambda row: (row.get("drug_dir") != "앞면", row.get("camera_la") not in {"70", "75", "90"}))
    return rows_by_product, seen


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def erode_alpha(cutout: Image.Image, iterations: int = 1) -> Image.Image:
    image = cutout.convert("RGBA")
    alpha = image.getchannel("A")
    for _ in range(iterations):
        alpha = alpha.filter(ImageFilter.MinFilter(3))
    image.putalpha(alpha)
    return trim_transparent(image, padding=5)


def normalize_cutout_wb(cutout: Image.Image) -> Image.Image:
    image = cutout.convert("RGBA")
    rgba = np.asarray(image, dtype=np.float32).copy()
    alpha = rgba[:, :, 3] > 24
    if alpha.sum() < 20:
        return image
    fg = rgba[:, :, :3][alpha]
    mean = np.maximum(fg.mean(axis=0), 1.0)
    gray = float(mean.mean())
    gains = np.clip(gray / mean, 0.92, 1.08)
    rgba[:, :, :3] = np.clip(rgba[:, :, :3] * gains, 0, 255)
    return Image.fromarray(rgba.astype(np.uint8), "RGBA")


def prepare_cutout(cutout: Image.Image) -> Image.Image:
    return erode_alpha(normalize_cutout_wb(cutout), iterations=1)


def read_assets(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_asset_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "asset_id",
        "asset_path",
        "product_id",
        "category_id",
        "class_index",
        "product_name",
        "shape",
        "color",
        "source_kind",
        "source_split",
        "source_sample_id",
        "source_zip",
        "image_member",
        "source_asset_id",
        "source_asset_path",
        "source_bbox",
        "width",
        "height",
        "quality",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def valid_alpha_cutout(path: Path) -> tuple[bool, list[int] | None]:
    with Image.open(path) as image:
        alpha = image.convert("RGBA").getchannel("A")
        bbox = alpha_bbox(alpha, threshold=24)
        return bbox is not None, bbox


def add_service_assets(
    args,
    class_by_product: dict[str, ClassRow],
    existing_rows: list[dict[str, str]],
    cutout_dir: Path,
) -> list[dict[str, str]]:
    if existing_rows:
        return existing_rows
    service_rows = read_assets(args.service_bank)
    grouped: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in service_rows:
        if row.get("product_id") in class_by_product:
            grouped[row["product_id"]].append(row)
    output_rows = []
    for product_id, class_row in sorted(class_by_product.items(), key=lambda item: item[1].class_index):
        for index, source in enumerate(grouped.get(product_id, [])[: args.reuse_service_assets]):
            src = Path(source["asset_path"])
            if not src.exists():
                continue
            with Image.open(src) as image:
                cutout = prepare_cutout(image.convert("RGBA"))
            asset_id = f"{product_id}_service_{index:02d}_{source['asset_id']}"
            dst = cutout_dir / f"{asset_id}.png"
            cutout.save(dst, format="PNG", optimize=True)
            ok, bbox = valid_alpha_cutout(dst)
            if not ok:
                dst.unlink(missing_ok=True)
                continue
            quality = json.loads(source.get("quality") or "{}")
            quality["reused_service_asset"] = True
            output_rows.append(
                {
                    "asset_id": asset_id,
                    "asset_path": str(dst),
                    "product_id": product_id,
                    "category_id": class_row.category_id,
                    "class_index": class_row.class_index,
                    "product_name": class_row.product_name,
                    "shape": source.get("shape") or class_row.shape,
                    "color": source.get("color") or class_row.color,
                    "source_kind": "service_asset_relinked",
                    "source_split": "train_seen",
                    "source_sample_id": source.get("sample_id", ""),
                    "source_zip": source.get("source_zip", ""),
                    "image_member": source.get("image_member", ""),
                    "source_asset_id": source.get("asset_id", ""),
                    "source_asset_path": source.get("asset_path", ""),
                    "source_bbox": source.get("source_bbox", ""),
                    "width": bbox[2] if bbox else cutout.width,
                    "height": bbox[3] if bbox else cutout.height,
                    "quality": json_compact(quality),
                }
            )
    return output_rows


def build_topup_assets(args, class_by_product: dict[str, ClassRow], asset_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    counts = collections.Counter(row["product_id"] for row in asset_rows)
    need = {product_id: args.assets_per_class - counts[product_id] for product_id in class_by_product}
    need = {product_id: count for product_id, count in need.items() if count > 0}
    if not need:
        return asset_rows

    rows_by_product, source_seen = read_source_rows(
        args.aihub_root,
        class_by_product,
        args.candidate_limit,
        args.seed + 17,
    )
    provider = Sam2MaskProvider(
        checkpoint=args.sam2_checkpoint,
        config=args.sam2_config,
        device=args.sam2_device,
        multimask=True,
        logit_threshold=args.sam2_logit_threshold,
        box_expansion_ratio=args.sam2_box_expansion_ratio,
    )
    zip_cache = ZipCache()
    cutout_dir = args.output_root / "assets" / "cutouts"
    failures = []
    used_samples = collections.defaultdict(set)
    for row in asset_rows:
        if row.get("source_sample_id"):
            used_samples[row["product_id"]].add(row["source_sample_id"])
    try:
        for product_id, needed in sorted(need.items(), key=lambda item: class_by_product[item[0]].class_index):
            class_row = class_by_product[product_id]
            accepted = 0
            for source in rows_by_product[product_id]:
                if accepted >= needed:
                    break
                if source.get("sample_id") in used_samples[product_id]:
                    continue
                with Image.open(BytesIO(zip_cache.get(source["source_zip"]).read(source["image_member"]))) as image:
                    image = image.convert("RGB")
                    cutout, quality = crop_cutout(
                        image,
                        parse_bbox(source),
                        source.get("shape") or class_row.shape,
                        margin_ratio=args.margin_ratio,
                        mask_provider=provider,
                    )
                if quality.get("method") != "sam2_bbox" or float(quality.get("score", 0.0)) < args.min_mask_score:
                    failures.append(
                        {
                            "product_id": product_id,
                            "sample_id": source.get("sample_id"),
                            "quality": quality,
                        }
                    )
                    continue
                cutout = prepare_cutout(cutout)
                asset_id = f"{product_id}_topup_{counts[product_id] + accepted:02d}_{source['sample_id']}"
                dst = cutout_dir / f"{asset_id}.png"
                cutout.save(dst, format="PNG", optimize=True)
                ok, alpha_box = valid_alpha_cutout(dst)
                if not ok:
                    failures.append(
                        {
                            "product_id": product_id,
                            "sample_id": source.get("sample_id"),
                            "quality": {"failure": "invalid_alpha_after_prepare", **quality},
                        }
                    )
                    dst.unlink(missing_ok=True)
                    continue
                accepted += 1
                used_samples[product_id].add(source.get("sample_id", ""))
                split = source.get("split", "")
                quality["source_split"] = split
                quality["k041768_validation_exception"] = product_id == K041768 and split == "val_official_single_ood"
                asset_rows.append(
                    {
                        "asset_id": asset_id,
                        "asset_path": str(dst),
                        "product_id": product_id,
                        "category_id": class_row.category_id,
                        "class_index": class_row.class_index,
                        "product_name": class_row.product_name,
                        "shape": source.get("shape") or class_row.shape,
                        "color": source.get("color") or class_row.color,
                        "source_kind": "aihub_single_topup",
                        "source_split": split,
                        "source_sample_id": source.get("sample_id", ""),
                        "source_zip": source.get("source_zip", ""),
                        "image_member": source.get("image_member", ""),
                        "source_asset_id": "",
                        "source_asset_path": "",
                        "source_bbox": source.get("bbox", ""),
                        "width": alpha_box[2] if alpha_box else cutout.width,
                        "height": alpha_box[3] if alpha_box else cutout.height,
                        "quality": json_compact(quality),
                    }
                )
            if accepted < needed:
                raise RuntimeError(
                    f"Could not build enough top-up assets for {product_id}: {accepted}/{needed}; "
                    f"source_rows={source_seen[product_id]}"
                )
            print(f"top-up {product_id}: +{accepted}", flush=True)
    finally:
        zip_cache.close()

    failure_path = args.output_root / "assets" / "asset_failures.json"
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return asset_rows


def build_asset_bank(args, class_by_product: dict[str, ClassRow]) -> list[dict[str, str]]:
    cutout_dir = args.output_root / "assets" / "cutouts"
    manifest_path = args.output_root / "assets" / "assets_manifest.csv"
    if manifest_path.exists() and not args.rebuild_assets:
        rows = read_assets(manifest_path)
    else:
        if (args.output_root / "assets").exists() and args.rebuild_assets:
            shutil.rmtree(args.output_root / "assets")
        cutout_dir.mkdir(parents=True, exist_ok=True)
        rows = add_service_assets(args, class_by_product, [], cutout_dir)
        rows = build_topup_assets(args, class_by_product, rows)
        rows.sort(key=lambda row: (int(row["class_index"]), row["asset_id"]))
        write_asset_manifest(manifest_path, rows)

    counts = collections.Counter(row["product_id"] for row in rows)
    bad = [product_id for product_id in class_by_product if counts[product_id] != args.assets_per_class]
    if bad:
        raise RuntimeError(f"Asset bank must contain {args.assets_per_class}/class. Bad classes: {bad}")
    summary = {
        "assets": len(rows),
        "classes": len(counts),
        "assets_per_class": args.assets_per_class,
        "min_assets_per_class": min(counts.values()),
        "max_assets_per_class": max(counts.values()),
        "source_kind_counts": dict(collections.Counter(row["source_kind"] for row in rows)),
        "k041768_source_splits": dict(collections.Counter(row["source_split"] for row in rows if row["product_id"] == K041768)),
        "manifest_csv": str(manifest_path),
    }
    (args.output_root / "assets" / "assets_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return rows


def exact_combo_sizes(num_images: int) -> list[int]:
    if num_images == 100:
        sizes = [2] * 3 + [3] * 65 + [4] * 32
    elif num_images == 696:
        sizes = [2] * 21 + [3] * 453 + [4] * 222
    elif num_images == 2500:
        # Keeps the 696/domain object-size mix while making total instances
        # exactly divisible by 56 classes: 8232 = 56 * 147.
        sizes = [2] * 76 + [3] * 1616 + [4] * 808
    else:
        base = {2: 0.0302, 3: 0.6509, 4: 0.3189}
        n2 = round(num_images * base[2])
        n4 = round(num_images * base[4])
        n3 = num_images - n2 - n4
        sizes = [2] * n2 + [3] * n3 + [4] * n4
    return sizes


def sample_target_side(profile: dict, rng: random.Random) -> float:
    pct = profile["bbox_side_px_pct"]
    points = [(0.05, pct["0.05"]), (0.25, pct["0.25"]), (0.5, pct["0.5"]), (0.75, pct["0.75"]), (0.95, pct["0.95"])]
    u = rng.uniform(0.05, 0.95)
    for (p0, v0), (p1, v1) in zip(points, points[1:]):
        if p0 <= u <= p1:
            t = (u - p0) / (p1 - p0)
            return float(v0 + (v1 - v0) * t)
    return float(points[-1][1])


def read_cutout(asset: dict[str, str]) -> Image.Image:
    with Image.open(asset["asset_path"]) as image:
        return image.convert("RGBA")


def transform_cutout(cutout: Image.Image, target_side: float, rng: random.Random) -> Image.Image:
    angle = rng.uniform(-7.0, 7.0)
    rotated = cutout.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    bbox = alpha_bbox(rotated.getchannel("A"), threshold=24) or [0, 0, rotated.width, rotated.height]
    current_side = math.sqrt(max(1, bbox[2] * bbox[3]))
    scale = float(np.clip(target_side / current_side, 0.45, 1.35))
    return rotated.resize(
        (max(12, int(round(rotated.width * scale))), max(12, int(round(rotated.height * scale)))),
        Image.Resampling.LANCZOS,
    )


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


def place_cutouts(cutouts: list[Image.Image], width: int, height: int, max_iou: float, rng: random.Random):
    cells = placement_cells(len(cutouts), width, height)
    rng.shuffle(cells)
    placements = []
    bboxes = []
    for cutout, cell in zip(cutouts, cells):
        x1, y1, x2, y2 = cell
        cell_w = max(24, x2 - x1)
        cell_h = max(24, y2 - y1)
        fit_scale = min(1.0, (cell_w * 0.86) / max(1, cutout.width), (cell_h * 0.86) / max(1, cutout.height))
        if fit_scale < 1.0:
            cutout = cutout.resize(
                (max(12, int(round(cutout.width * fit_scale))), max(12, int(round(cutout.height * fit_scale)))),
                Image.Resampling.LANCZOS,
            )
        placed = False
        for _ in range(140):
            max_x = max(x1, x2 - cutout.width)
            max_y = max(y1, y2 - cutout.height)
            x = rng.randint(x1, max_x) if max_x > x1 else x1
            y = rng.randint(y1, max_y) if max_y > y1 else y1
            fg_bbox = alpha_bbox(cutout.getchannel("A"), threshold=24) or [0, 0, cutout.width, cutout.height]
            bbox = [x + fg_bbox[0], y + fg_bbox[1], fg_bbox[2], fg_bbox[3]]
            if x < 0 or y < 0 or x + cutout.width > width or y + cutout.height > height:
                continue
            if all(iou_xywh(bbox, existing) <= max_iou for existing in bboxes):
                placements.append((cutout, (x, y), [int(round(v)) for v in bbox]))
                bboxes.append(bbox)
                placed = True
                break
        if not placed:
            raise RuntimeError("Could not place cutout without overlap")
    return placements


def make_domain_background(backgrounds: list[Image.Image], width: int, height: int, rng: random.Random) -> Image.Image:
    patch = backgrounds[rng.randrange(len(backgrounds))]
    scale = max(width / patch.width, height / patch.height) * rng.uniform(1.08, 1.35)
    resized = patch.resize(
        (max(width, int(round(patch.width * scale))), max(height, int(round(patch.height * scale)))),
        Image.Resampling.BICUBIC,
    )
    if rng.random() < 0.5:
        resized = resized.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        resized = resized.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    ox = rng.randrange(max(1, resized.width - width + 1))
    oy = rng.randrange(max(1, resized.height - height + 1))
    image = resized.crop((ox, oy, ox + width, oy + height)).filter(ImageFilter.GaussianBlur(radius=0.25))
    arr = np.asarray(image, dtype=np.float32).copy()
    bg_mean = arr.reshape(-1, 3).mean(axis=0)
    target = TARGET_RGB + np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5)], dtype=np.float32)
    arr += target - bg_mean
    noise_rng = np.random.default_rng(rng.randrange(2**32))
    low = noise_rng.normal(0, rng.uniform(1.0, 2.2), (max(8, height // 32), max(8, width // 32), 3))
    low_img = Image.fromarray(np.uint8(np.clip(low + 127, 0, 255)), "RGB").resize((width, height), Image.Resampling.BICUBIC)
    low_noise = np.asarray(low_img, dtype=np.float32) - 127.0
    fine_noise = noise_rng.normal(0, rng.uniform(0.25, 0.8), arr.shape)
    arr = np.clip(arr + low_noise + fine_noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def apply_light_photometric(image: Image.Image, rng: random.Random) -> Image.Image:
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.98, 1.03))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.98, 1.04))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.99, 1.03))
    return image


def choose_products(
    products: list[str],
    real_counts: collections.Counter,
    synth_counts: collections.Counter,
    combo_size: int,
    rng: random.Random,
    balance_mode: str,
    synthetic_min_instances_per_class: int,
) -> list[str]:
    selected: list[str] = []
    if synthetic_min_instances_per_class > 0:
        below_floor = [
            (synth_counts[product_id], rng.random(), product_id)
            for product_id in products
            if synth_counts[product_id] < synthetic_min_instances_per_class
        ]
        below_floor.sort()
        selected = [item[2] for item in below_floor[:combo_size]]
        if len(selected) == combo_size:
            return selected

    selected_set = set(selected)
    ranked = []
    for product_id in products:
        if product_id in selected_set:
            continue
        tie = rng.random()
        if balance_mode == "synthetic_balanced":
            ranked.append((synth_counts[product_id], tie, product_id))
        else:
            ranked.append((synth_counts[product_id] > 0, real_counts[product_id] + synth_counts[product_id], tie, product_id))
    ranked.sort()
    selected.extend(item[-1] for item in ranked[: combo_size - len(selected)])
    return selected


def group_assets(asset_rows: list[dict[str, str]], rng: random.Random) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in asset_rows:
        grouped[row["product_id"]].append(row)
    for rows in grouped.values():
        rng.shuffle(rows)
    return dict(grouped)


def write_manifest(path: Path, rows: list[dict]) -> None:
    fields = [
        "image_id",
        "file_name",
        "combo_size",
        "product_ids",
        "category_ids",
        "class_indices",
        "source_asset_ids",
        "annotations",
        "background_rgb_mean",
        "background_source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate(args, class_rows: list[ClassRow], class_by_product: dict[str, ClassRow], profile: dict, backgrounds: list[Image.Image], background_source_label: str, asset_rows: list[dict[str, str]], real_counts: collections.Counter) -> dict:
    rng = random.Random(args.seed)
    run_root = args.output_root / "runs" / args.run_name
    if run_root.exists() and args.overwrite:
        shutil.rmtree(run_root)
    if run_root.exists():
        raise FileExistsError(f"Run exists: {run_root}. Pass --overwrite to replace.")
    image_dir = run_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    (run_root / "manifests").mkdir(parents=True, exist_ok=True)

    products = [row.product_id for row in sorted(class_rows, key=lambda row: row.class_index)]
    assets_by_product = group_assets(asset_rows, rng)
    cursors = collections.Counter()
    synth_counts = collections.Counter()
    combo_sizes = exact_combo_sizes(args.num_images)
    rng.shuffle(combo_sizes)
    coco_images = []
    coco_annotations = []
    manifest_rows = []
    ann_id = 1
    attempts = 0
    placement_failures = 0
    size_index = 0
    while len(coco_images) < args.num_images:
        combo_size = combo_sizes[size_index]
        attempts += 1
        if attempts > args.num_images * 30:
            raise RuntimeError(f"Too many failed generation attempts: placement_failures={placement_failures}")
        selected_products = choose_products(
            products,
            real_counts,
            synth_counts,
            combo_size,
            rng,
            args.class_balance_mode,
            args.synthetic_min_instances_per_class,
        )
        selected_assets = []
        cutouts = []
        for product_id in selected_products:
            product_assets = assets_by_product[product_id]
            cursor = cursors[product_id]
            if cursor % len(product_assets) == 0:
                rng.shuffle(product_assets)
            asset = product_assets[cursor % len(product_assets)]
            cursors[product_id] += 1
            selected_assets.append(asset)
            cutouts.append(transform_cutout(read_cutout(asset), sample_target_side(profile, rng), rng))
        try:
            placements = place_cutouts(cutouts, args.width, args.height, args.max_iou, rng)
        except RuntimeError:
            placement_failures += 1
            continue

        canvas = make_domain_background(backgrounds, args.width, args.height, rng).convert("RGBA")
        background_rgb_mean = [round(float(x), 2) for x in np.asarray(canvas.convert("RGB")).reshape(-1, 3).mean(axis=0)]
        image_id = len(coco_images) + 1
        file_name = f"{args.file_prefix}_{image_id:06d}.jpg"
        anns_for_image = []
        for local_id, (asset, (cutout, xy, bbox)) in enumerate(zip(selected_assets, placements), start=1):
            paste_with_shadow(canvas, cutout, xy, rng)
            class_row = class_by_product[asset["product_id"]]
            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": class_row.category_id,
                "class_index": class_row.class_index,
                "product_id": class_row.product_id,
                "bbox": bbox,
                "area": int(bbox[2] * bbox[3]),
                "iscrowd": 0,
                "segmentation": [],
                "source_asset_id": asset["asset_id"],
                "source_sample_id": asset.get("source_sample_id", ""),
            }
            anns_for_image.append(ann)
            coco_annotations.append(ann)
            ann_id += 1
            synth_counts[class_row.product_id] += 1
        final_image = apply_light_photometric(canvas.convert("RGB"), rng)
        final_image.save(image_dir / file_name, format="JPEG", quality=args.jpeg_quality, subsampling=0, optimize=True)
        image_row = {
            "id": image_id,
            "file_name": file_name,
            "width": args.width,
            "height": args.height,
            "combo_size": combo_size,
            "product_ids": selected_products,
            "synthetic": True,
            "background": background_source_label,
            "background_rgb_mean": background_rgb_mean,
        }
        coco_images.append(image_row)
        manifest_rows.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "combo_size": combo_size,
                "product_ids": json_compact(selected_products),
                "category_ids": json_compact([class_by_product[p].category_id for p in selected_products]),
                "class_indices": json_compact([class_by_product[p].class_index for p in selected_products]),
                "source_asset_ids": json_compact([asset["asset_id"] for asset in selected_assets]),
                "annotations": json_compact(anns_for_image),
                "background_rgb_mean": json_compact(background_rgb_mean),
                "background_source": background_source_label,
            }
        )
        size_index += 1
        if len(coco_images) % 100 == 0:
            print(f"generated {len(coco_images)}/{args.num_images}", flush=True)

    categories = [
        {
            "id": row.category_id,
            "name": row.product_name,
            "supercategory": "pill",
            "product_id": row.product_id,
            "class_index": row.class_index,
            "shape": row.shape,
            "color": row.color,
            "imprint_front": row.imprint_front,
        }
        for row in sorted(class_rows, key=lambda row: row.class_index)
    ]
    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
        "info": {
            "name": "kaggle_sam2_synth_v2",
            "source": "AI Hub single only; K-041768 uses AI Hub official validation single exception",
            "class_map": "codex_handoff/class_map_56.csv",
            "domain_profile": "codex_handoff/domain_profile.json",
            "backgrounds": background_source_label,
            "sam2_checkpoint": str(args.sam2_checkpoint),
            "sam2_config": args.sam2_config,
            "min_mask_score": args.min_mask_score,
        },
    }
    annotation_path = run_root / "annotations_coco.json"
    annotation_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_manifest(run_root / "manifests" / "synthetic_manifest.csv", manifest_rows)
    class_distribution_path = run_root / "manifests" / "class_instance_distribution.csv"
    with class_distribution_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_index", "product_id", "category_id", "synthetic_instances", "real_instances", "real_plus_synth"],
        )
        writer.writeheader()
        for row in sorted(class_rows, key=lambda item: item.class_index):
            writer.writerow(
                {
                    "class_index": row.class_index,
                    "product_id": row.product_id,
                    "category_id": row.category_id,
                    "synthetic_instances": synth_counts[row.product_id],
                    "real_instances": real_counts[row.product_id],
                    "real_plus_synth": real_counts[row.product_id] + synth_counts[row.product_id],
                }
            )
    total_counts = collections.Counter(real_counts)
    total_counts.update(synth_counts)
    summary = {
        "run_name": args.run_name,
        "num_images": len(coco_images),
        "annotations": len(coco_annotations),
        "classes_total": len(categories),
        "classes_used": len(synth_counts),
        "combo_size_counts": dict(collections.Counter(image["combo_size"] for image in coco_images)),
        "synth_class_instance_min": min(synth_counts.values()) if synth_counts else 0,
        "synth_class_instance_max": max(synth_counts.values()) if synth_counts else 0,
        "real_plus_synth_class_instance_min": min(total_counts.values()) if total_counts else 0,
        "real_plus_synth_class_instance_max": max(total_counts.values()) if total_counts else 0,
        "placement_failures": placement_failures,
        "attempts": attempts,
        "class_balance_mode": args.class_balance_mode,
        "synthetic_min_instances_per_class": args.synthetic_min_instances_per_class,
        "backgrounds": background_source_label,
        "background_count": len(backgrounds),
        "class_distribution_csv": str(class_distribution_path),
        "images_dir": str(image_dir),
        "annotation_path": str(annotation_path),
    }
    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    args = build_parser().parse_args()
    spec_dir = ensure_spec(args.output_root, args.handoff_zip)
    class_rows, class_by_product = read_class_map(spec_dir)
    profile = read_domain_profile(spec_dir)
    backgrounds, background_source_label = read_backgrounds(spec_dir, args.background_dir)
    real_counts = read_real_counts(args.data_root, class_by_product)
    asset_rows = build_asset_bank(args, class_by_product)
    generate(args, class_rows, class_by_product, profile, backgrounds, background_source_label, asset_rows, real_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
