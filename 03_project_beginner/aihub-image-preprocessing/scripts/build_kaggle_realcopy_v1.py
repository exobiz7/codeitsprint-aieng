#!/usr/bin/env python3
"""Build in-domain real Copy-Paste augmentation for the Kaggle pill dataset."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import random
import shutil
import tarfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from v2_common import Sam2MaskProvider, crop_cutout


DEFAULT_DATA_ROOT = Path("/Volumes/SSD 4T/01_sprint_ai_project1_data")
DEFAULT_HANDOFF_DIR = Path("codex-handoff/handoff_realcopy")
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "processed" / "kaggle_realcopy_v1_1500"
DEFAULT_VALIDATED_BACKGROUND_DIR = (
    DEFAULT_DATA_ROOT / "processed" / "kaggle_sam2_synth_v2" / "spec" / "codex_handoff" / "backgrounds"
)
DEFAULT_CLEAN64_PREPROCESSED_DIR = DEFAULT_DATA_ROOT / "processed" / "kaggle_realcopy_bg_clean64_preprocessed"
DEFAULT_SAM2_CHECKPOINT = Path("models/sam2/sam2.1_hiera_large.pt")
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--handoff-dir", type=Path, default=DEFAULT_HANDOFF_DIR)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--categories-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-images", type=int, default=1500)
    parser.add_argument("--file-prefix", default="realcopy_sam2_v2")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument(
        "--background-mode",
        choices=["source_silhouette_inpaint", "validated_v2_clean", "clean64_preprocessed", "fold_train_bg64"],
        default="clean64_preprocessed",
    )
    parser.add_argument("--fold-bg-count", type=int, default=64)
    parser.add_argument("--fold-bg-chip-size", type=int, default=384)
    parser.add_argument("--validated-background-dir", type=Path, default=DEFAULT_VALIDATED_BACKGROUND_DIR)
    parser.add_argument("--clean64-background-dir", type=Path, default=DEFAULT_CLEAN64_PREPROCESSED_DIR)
    parser.add_argument(
        "--mask-provider",
        choices=["sam2", "cv2"],
        default="sam2",
        help="sam2 rejects fallback masks; cv2 keeps the old GrabCut/Otsu path for regression checks.",
    )
    parser.add_argument("--min-mask-score", type=float, default=0.86)
    parser.add_argument("--margin-ratio", type=float, default=0.22)
    parser.add_argument("--min-edge-margin", type=int, default=16)
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--sam2-device", default="auto")
    parser.add_argument("--sam2-logit-threshold", type=float, default=0.8)
    parser.add_argument("--sam2-box-expansion-ratio", type=float, default=0.035)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def load_inputs(
    handoff_dir: Path,
    source_manifest: Path | None = None,
    categories_path: Path | None = None,
) -> tuple[list[dict], dict, dict[int, dict]]:
    categories_file = categories_path or (handoff_dir / "target_categories_schema.json")
    manifest_file = source_manifest or (handoff_dir / "realcopy_source_manifest_fold0train.json")
    categories = json.loads(categories_file.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["_source_manifest_path"] = str(manifest_file)
    manifest["_categories_path"] = str(categories_file)
    by_category = {int(row["id"]): row for row in categories}
    if len(categories) != 56:
        raise RuntimeError(f"Expected 56 categories, got {len(categories)}")
    return categories, manifest, by_category


def bbox_inside(bbox: list[float], width: int, height: int) -> bool:
    x, y, w, h = bbox
    return w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= width and y + h <= height


def expanded_box(bbox: list[float], width: int, height: int, margin_frac: float = 0.18) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    margin = max(10.0, margin_frac * max(w, h))
    x0 = max(0, int(math.floor(x - margin)))
    y0 = max(0, int(math.floor(y - margin)))
    x1 = min(width, int(math.ceil(x + w + margin)))
    y1 = min(height, int(math.ceil(y + h + margin)))
    return x0, y0, x1, y1


def otsu_mask(crop_rgb: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = rect
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_mask = np.ones(crop_rgb.shape[:2], dtype=bool)
    bg_mask[max(0, y - 3) : min(bg_mask.shape[0], y + h + 3), max(0, x - 3) : min(bg_mask.shape[1], x + w + 3)] = False
    border = np.zeros_like(bg_mask)
    border[:6, :] = True
    border[-6:, :] = True
    border[:, :6] = True
    border[:, -6:] = True
    bg_pixels = lab[bg_mask | border]
    if bg_pixels.size == 0:
        bg_pixels = lab.reshape(-1, 3)
    bg = np.median(bg_pixels, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    rect_mask = np.zeros(crop_rgb.shape[:2], dtype=np.uint8)
    pad = 4
    rect_mask[max(0, y - pad) : min(rect_mask.shape[0], y + h + pad), max(0, x - pad) : min(rect_mask.shape[1], x + w + pad)] = 1
    values = np.uint8(np.clip(dist / max(1e-6, np.percentile(dist, 98)) * 255, 0, 255))
    _, thresholded = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (thresholded > 0).astype(np.uint8) * rect_mask
    return mask


def largest_component(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num <= 1:
        return mask.astype(np.uint8)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == best).astype(np.uint8)


def refine_mask(mask: np.ndarray) -> np.ndarray:
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel5, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    mask = largest_component(mask)
    contour_mask = np.zeros_like(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(contour_mask, contours, -1, 1, thickness=-1)
        mask = contour_mask
    # Erode 1 px to remove blue-gray fringe as requested in the handoff guide.
    mask = cv2.erode(mask, kernel3, iterations=1)
    return mask.astype(np.uint8)


def mask_quality(mask: np.ndarray, source_bbox_area: float, crop_shape: tuple[int, int]) -> tuple[bool, dict]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return False, {"reason": "empty"}
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    area = int(mask.sum())
    h, w = crop_shape
    touches_edge = x0 <= 1 or y0 <= 1 or x1 >= w - 1 or y1 >= h - 1
    area_ratio = area / max(1.0, source_bbox_area)
    ok = (
        not touches_edge
        and area_ratio >= 0.18
        and area_ratio <= 1.10
        and (x1 - x0) >= 24
        and (y1 - y0) >= 24
    )
    return ok, {
        "alpha_bbox": [x0, y0, x1 - x0, y1 - y0],
        "mask_area": area,
        "mask_area_to_source_bbox": round(area_ratio, 4),
        "touches_edge": touches_edge,
    }


def extract_alpha_cutout(image_rgb: np.ndarray, bbox: list[float]) -> tuple[Image.Image | None, dict]:
    height, width = image_rgb.shape[:2]
    x0, y0, x1, y1 = expanded_box(bbox, width, height)
    crop = image_rgb[y0:y1, x0:x1].copy()
    rx = int(round(bbox[0] - x0))
    ry = int(round(bbox[1] - y0))
    rw = int(round(bbox[2]))
    rh = int(round(bbox[3]))
    rect = (
        max(1, rx),
        max(1, ry),
        min(crop.shape[1] - max(1, rx) - 2, max(2, rw)),
        min(crop.shape[0] - max(1, ry) - 2, max(2, rh)),
    )
    source_bbox_area = float(bbox[2] * bbox[3])

    masks: list[tuple[str, np.ndarray]] = []
    try:
        grab = np.zeros(crop.shape[:2], dtype=np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(crop, grab, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        grab_mask = np.where((grab == cv2.GC_FGD) | (grab == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        masks.append(("grabcut", grab_mask))
    except Exception:
        pass
    masks.append(("otsu_lab_bgdist", otsu_mask(crop, rect)))

    best: tuple[str, np.ndarray, dict] | None = None
    for method, raw_mask in masks:
        mask = refine_mask(raw_mask)
        ok, quality = mask_quality(mask, source_bbox_area, crop.shape[:2])
        quality["method"] = method
        if ok:
            best = (method, mask, quality)
            break
        if best is None or quality.get("mask_area", 0) > best[2].get("mask_area", 0):
            best = (method, mask, quality)

    if best is None:
        return None, {"reason": "no_mask"}
    method, mask, quality = best
    ok, quality = mask_quality(mask, source_bbox_area, crop.shape[:2])
    quality["method"] = method
    if not ok:
        quality["reason"] = quality.get("reason", "quality_gate_failed")
        return None, quality

    alpha = cv2.GaussianBlur((mask * 255).astype(np.uint8), (0, 0), 0.55)
    ys, xs = np.where(alpha > 8)
    ax0, ax1 = int(xs.min()), int(xs.max()) + 1
    ay0, ay1 = int(ys.min()), int(ys.max()) + 1
    pad = 3
    ax0 = max(0, ax0 - pad)
    ay0 = max(0, ay0 - pad)
    ax1 = min(crop.shape[1], ax1 + pad)
    ay1 = min(crop.shape[0], ay1 + pad)
    rgba = np.dstack([crop, alpha])
    cutout = Image.fromarray(rgba[ay0:ay1, ax0:ax1], "RGBA")
    quality["cutout_size"] = [cutout.width, cutout.height]
    return cutout, quality


def trim_rgba(image: Image.Image, padding: int = 5, threshold: int = 2) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    ys, xs = np.where(alpha > threshold)
    if len(xs) == 0:
        return rgba
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(rgba.width, int(xs.max()) + 1 + padding)
    y1 = min(rgba.height, int(ys.max()) + 1 + padding)
    return rgba.crop((x0, y0, x1, y1))


def erode_alpha(cutout: Image.Image, iterations: int = 1) -> Image.Image:
    image = cutout.convert("RGBA")
    alpha = image.getchannel("A")
    for _ in range(iterations):
        alpha = alpha.filter(ImageFilter.MinFilter(3))
    image.putalpha(alpha)
    return trim_rgba(image, padding=5)


def normalize_cutout_wb(cutout: Image.Image) -> Image.Image:
    image = cutout.convert("RGBA")
    rgba = np.asarray(image, dtype=np.float32).copy()
    alpha = rgba[:, :, 3] > 24
    if int(alpha.sum()) < 20:
        return image
    foreground = rgba[:, :, :3][alpha]
    mean = np.maximum(foreground.mean(axis=0), 1.0)
    gray = float(mean.mean())
    gains = np.clip(gray / mean, 0.92, 1.08)
    rgba[:, :, :3] = np.clip(rgba[:, :, :3] * gains, 0, 255)
    return Image.fromarray(rgba.astype(np.uint8), "RGBA")


def prepare_cutout(cutout: Image.Image) -> Image.Image:
    return erode_alpha(normalize_cutout_wb(cutout), iterations=1)


def prepared_cutout_quality(cutout: Image.Image, bbox: list[float]) -> tuple[bool, dict]:
    alpha = np.asarray(cutout.convert("RGBA").getchannel("A"))
    ys, xs = np.where(alpha > 24)
    if len(xs) == 0:
        return False, {"failure": "prepared_alpha_empty"}
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    foreground_area = int((alpha > 24).sum())
    source_bbox_area = max(1.0, float(bbox[2] * bbox[3]))
    alpha_bbox_area = max(1, (x1 - x0) * (y1 - y0))
    area_ratio = foreground_area / source_bbox_area
    fill_ratio = foreground_area / alpha_bbox_area
    touches_edge = bool(
        (alpha[0, :] > 24).any()
        or (alpha[-1, :] > 24).any()
        or (alpha[:, 0] > 24).any()
        or (alpha[:, -1] > 24).any()
    )
    ok = (
        not touches_edge
        and 0.14 <= area_ratio <= 1.22
        and fill_ratio <= 0.94
        and (x1 - x0) >= 24
        and (y1 - y0) >= 24
    )
    return ok, {
        "prepared_alpha_bbox": [x0, y0, x1 - x0, y1 - y0],
        "prepared_foreground_area": foreground_area,
        "prepared_area_ratio": round(area_ratio, 4),
        "prepared_fill_ratio": round(fill_ratio, 4),
        "prepared_touches_edge": touches_edge,
        **({} if ok else {"failure": "prepared_quality_gate_failed"}),
    }


def build_source_records(manifest: dict, by_category: dict[int, dict]) -> tuple[list[dict], list[dict]]:
    width, height = [int(v) for v in manifest["image_size"]]
    records: list[dict] = []
    dropped: list[dict] = []
    for image in manifest["images"]:
        bbox_groups: dict[tuple[float, float, float, float], list[int]] = collections.defaultdict(list)
        for idx, pill in enumerate(image["pills"]):
            bbox = [float(v) for v in pill["bbox_px"]]
            bbox_groups[tuple(round(v, 2) for v in bbox)].append(idx)
        duplicate_indexes = {idx for indexes in bbox_groups.values() if len(indexes) > 1 for idx in indexes}
        for idx, pill in enumerate(image["pills"]):
            category_id = int(pill["category_id"])
            bbox = [float(v) for v in pill["bbox_px"]]
            base = {"file": image["file"], "category_id": category_id, "bbox_px": bbox}
            if category_id not in by_category:
                dropped.append({**base, "reason": "category_not_in_schema"})
                continue
            if not bbox_inside(bbox, width, height):
                dropped.append({**base, "reason": "bbox_outside_image"})
                continue
            if idx in duplicate_indexes:
                dropped.append({**base, "reason": "duplicate_bbox_ambiguous_instance"})
                continue
            records.append(base)
    return records, dropped


def build_cutout_bank(
    data_root: Path,
    manifest: dict,
    source_records: list[dict],
    by_category: dict[int, dict],
    mask_provider: Sam2MaskProvider | None,
    mask_provider_name: str,
    min_mask_score: float,
    margin_ratio: float,
) -> tuple[list[dict], list[dict]]:
    image_dir = Path(manifest.get("image_dir_macstudio") or data_root / "train_images")
    by_file: dict[str, list[dict]] = collections.defaultdict(list)
    for record in source_records:
        by_file[record["file"]].append(record)

    assets: list[dict] = []
    failed: list[dict] = []
    asset_id = 1
    for file_name, records in sorted(by_file.items()):
        image_path = image_dir / file_name
        if not image_path.exists():
            for record in records:
                failed.append({**record, "reason": "missing_source_image"})
            continue
        with Image.open(image_path) as image:
            pil_rgb = image.convert("RGB")
            image_rgb = np.asarray(pil_rgb)
        for local_index, record in enumerate(records):
            category = by_category[int(record["category_id"])]
            if mask_provider_name == "sam2":
                cutout, quality = crop_cutout(
                    pil_rgb,
                    record["bbox_px"],
                    category.get("shape") or "",
                    margin_ratio=margin_ratio,
                    mask_provider=mask_provider,
                )
                if quality.get("method") != "sam2_bbox":
                    failed.append({**record, **quality, "reason": "sam2_required_but_fallback_returned"})
                    continue
                if float(quality.get("score", 0.0)) < min_mask_score:
                    failed.append({**record, **quality, "reason": "sam2_score_below_threshold"})
                    continue
                cutout = prepare_cutout(cutout)
                ok, prepared_quality = prepared_cutout_quality(cutout, record["bbox_px"])
                quality = {**quality, **prepared_quality}
                if not ok:
                    failed.append({**record, **quality, "reason": "prepared_cutout_quality_failed"})
                    continue
            else:
                cutout, quality = extract_alpha_cutout(image_rgb, record["bbox_px"])
            if cutout is None:
                failed.append({**record, **quality})
                continue
            assets.append(
                {
                    "asset_id": asset_id,
                    "file": record["file"],
                    "category_id": int(record["category_id"]),
                    "class_index": int(category["class_index"]),
                    "product_id": category["product_id"],
                    "bbox_px": record["bbox_px"],
                    "cutout": cutout,
                    "quality": quality,
                    "source_instance_index": local_index,
                }
            )
            asset_id += 1
    return assets, failed


def make_inpaint_backgrounds(data_root: Path, manifest: dict) -> tuple[list[dict], int]:
    image_dir = Path(manifest.get("image_dir_macstudio") or data_root / "train_images")
    width, height = [int(v) for v in manifest["image_size"]]
    backgrounds: list[dict] = []
    contamination_count = 0

    def local_pill_mask(image_rgb: np.ndarray, bbox: list[float]) -> tuple[int, int, np.ndarray] | None:
        x0, y0, x1, y1 = expanded_box(bbox, width, height, margin_frac=0.20)
        crop = image_rgb[y0:y1, x0:x1].copy()
        rx = int(round(bbox[0] - x0))
        ry = int(round(bbox[1] - y0))
        rw = int(round(bbox[2]))
        rh = int(round(bbox[3]))
        rect = (
            max(1, rx),
            max(1, ry),
            min(crop.shape[1] - max(1, rx) - 2, max(2, rw)),
            min(crop.shape[0] - max(1, ry) - 2, max(2, rh)),
        )
        candidates: list[np.ndarray] = []
        try:
            grab = np.zeros(crop.shape[:2], dtype=np.uint8)
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            cv2.grabCut(crop, grab, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
            candidates.append(np.where((grab == cv2.GC_FGD) | (grab == cv2.GC_PR_FGD), 1, 0).astype(np.uint8))
        except Exception:
            pass
        candidates.append(otsu_mask(crop, rect))
        best: np.ndarray | None = None
        best_area = 0
        for candidate in candidates:
            mask = refine_mask(candidate)
            ok, quality = mask_quality(mask, float(bbox[2] * bbox[3]), crop.shape[:2])
            area = int(mask.sum())
            if ok:
                best = mask
                break
            if area > best_area:
                best = mask
                best_area = area
        if best is None or best.sum() == 0:
            return None
        return x0, y0, best

    for image in manifest["images"]:
        image_path = image_dir / image["file"]
        if not image_path.exists():
            continue
        with Image.open(image_path) as pil_image:
            rgb = np.asarray(pil_image.convert("RGB"))
        if rgb.shape[:2] != (height, width):
            continue
        full_mask = np.zeros((height, width), dtype=np.uint8)
        for pill in image["pills"]:
            bbox = [float(v) for v in pill["bbox_px"]]
            if not bbox_inside(bbox, width, height):
                continue
            result = local_pill_mask(rgb, bbox)
            if result is None:
                x, y, w, h = bbox
                margin = max(18, int(0.06 * max(w, h)))
                bx0 = max(0, int(x) - margin)
                by0 = max(0, int(y) - margin)
                bx1 = min(width, int(math.ceil(x + w)) + margin)
                by1 = min(height, int(math.ceil(y + h)) + margin)
                full_mask[by0:by1, bx0:bx1] = 255
                continue
            x0, y0, local = result
            h, w = local.shape[:2]
            full_mask[y0 : y0 + h, x0 : x0 + w] = np.maximum(
                full_mask[y0 : y0 + h, x0 : x0 + w],
                (local * 255).astype(np.uint8),
            )
        full_mask = cv2.dilate(full_mask, np.ones((19, 19), np.uint8), iterations=1)
        full_mask = cv2.GaussianBlur(full_mask, (0, 0), 1.2)
        hard_mask = np.uint8(full_mask > 12) * 255
        if hard_mask.sum() > 0:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            telea = cv2.inpaint(bgr, hard_mask, 7, cv2.INPAINT_TELEA)
            ns = cv2.inpaint(bgr, hard_mask, 7, cv2.INPAINT_NS)
            inpainted = cv2.addWeighted(telea, 0.65, ns, 0.35, 0)
            bg_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
            soft = cv2.GaussianBlur((hard_mask.astype(np.float32) / 255.0), (0, 0), 8)
            soft = np.clip(soft[..., None], 0.0, 1.0)
            smooth = cv2.GaussianBlur(bg_rgb, (0, 0), 4)
            bg_rgb = np.uint8(np.clip(bg_rgb.astype(np.float32) * (1 - soft * 0.25) + smooth.astype(np.float32) * (soft * 0.25), 0, 255))
        else:
            bg_rgb = rgb.copy()
        backgrounds.append(
            {
                "file": image["file"],
                "rgb": bg_rgb,
                "masked_pixels": int((hard_mask > 0).sum()),
                "background_mode": "silhouette_mask_inpaint",
            }
        )
    return backgrounds, contamination_count


def make_validated_v2_clean_backgrounds(background_dir: Path, manifest: dict, seed: int) -> tuple[list[dict], int]:
    width, height = [int(v) for v in manifest["image_size"]]
    clean_names = ["bg_01.png", "bg_04.png", "bg_12.png"]
    patches = []
    for name in clean_names:
        path = background_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing validated clean background: {path}")
        with Image.open(path) as image:
            patches.append((name, np.asarray(image.convert("RGB"))))
    backgrounds: list[dict] = []
    rng = random.Random(seed)
    for variant in range(48):
        name, patch = rng.choice(patches)
        scale = rng.randint(288, 448)
        tile = np.asarray(Image.fromarray(patch, "RGB").resize((scale, scale), Image.Resampling.BICUBIC))
        reps_y = math.ceil((height + scale) / scale) + 2
        reps_x = math.ceil((width + scale) / scale) + 2
        rows = []
        for yy in range(reps_y):
            row = []
            for xx in range(reps_x):
                cur = tile
                if (xx + yy + variant) % 2:
                    cur = np.fliplr(cur)
                if (xx * 3 + yy + variant) % 3 == 0:
                    cur = np.flipud(cur)
                row.append(cur)
            rows.append(np.concatenate(row, axis=1))
        tiled = np.concatenate(rows, axis=0)
        max_y = tiled.shape[0] - height
        max_x = tiled.shape[1] - width
        y0 = rng.randint(0, max(0, max_y))
        x0 = rng.randint(0, max(0, max_x))
        bg = tiled[y0 : y0 + height, x0 : x0 + width].copy()
        bg = cv2.GaussianBlur(bg, (0, 0), rng.uniform(0.35, 0.9))
        pil = Image.fromarray(bg, "RGB")
        pil = ImageEnhance.Brightness(pil).enhance(rng.uniform(0.985, 1.015))
        pil = ImageEnhance.Contrast(pil).enhance(rng.uniform(0.985, 1.018))
        arr = np.asarray(pil, dtype=np.float32)
        noise = np.random.default_rng(seed + variant).normal(0, rng.uniform(0.25, 0.65), arr.shape)
        bg = np.uint8(np.clip(arr + noise, 0, 255))
        backgrounds.append(
            {
                "file": f"validated_v2_clean_{Path(name).stem}_{variant:02d}",
                "rgb": bg,
                "masked_pixels": 0,
                "background_mode": "validated_v2_clean_subset_bg_01_04_12",
            }
        )
    return backgrounds, 0


def make_clean64_preprocessed_backgrounds(background_root: Path, manifest: dict) -> tuple[list[dict], int]:
    width, height = [int(v) for v in manifest["image_size"]]
    canvas_dir = background_root / "canvases_976x1280"
    paths = sorted(canvas_dir.glob("*.png"))
    if len(paths) != 64:
        raise RuntimeError(f"Expected 64 preprocessed canvases in {canvas_dir}, got {len(paths)}")
    backgrounds: list[dict] = []
    for path in paths:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        if rgb.shape[:2] != (height, width):
            raise RuntimeError(f"Bad clean64 canvas size {path}: {rgb.shape[1]}x{rgb.shape[0]}")
        backgrounds.append(
            {
                "file": path.name,
                "rgb": rgb,
                "masked_pixels": 0,
                "background_mode": "clean64_preprocessed_canvas",
            }
        )
    return backgrounds, 0


def _background_forbidden_mask(width: int, height: int, pills: list[dict], pad: int = 64) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for pill in pills:
        bbox = [float(v) for v in pill["bbox_px"]]
        x, y, w, h = bbox
        x0 = max(0, int(math.floor(x - pad)))
        y0 = max(0, int(math.floor(y - pad)))
        x1 = min(width, int(math.ceil(x + w + pad)))
        y1 = min(height, int(math.ceil(y + h + pad)))
        mask[y0:y1, x0:x1] = 1
    return mask


def _crop_clean_background_chip(
    image_rgb: np.ndarray,
    forbidden_mask: np.ndarray,
    rng: random.Random,
    requested_size: int,
) -> tuple[np.ndarray | None, dict]:
    height, width = image_rgb.shape[:2]
    for chip_size in [requested_size, 352, 320, 288, 256, 224, 192, 160, 128, 96, 64]:
        if chip_size > width or chip_size > height:
            continue
        max_x = width - chip_size
        max_y = height - chip_size
        best: tuple[int, int, int] | None = None
        for attempt in range(900):
            x = rng.randint(0, max_x)
            y = rng.randint(0, max_y)
            mixed = int(forbidden_mask[y : y + chip_size, x : x + chip_size].sum())
            if mixed == 0:
                chip = image_rgb[y : y + chip_size, x : x + chip_size].copy()
                return chip, {
                    "chip_size": chip_size,
                    "x": x,
                    "y": y,
                    "forbidden_pixels": 0,
                    "attempt": attempt + 1,
                }
            if best is None or mixed < best[0]:
                best = (mixed, x, y)
        # Never accept a mixed crop; try a smaller chip.
    return None, {"reason": "no_clean_background_crop"}


def _make_bg_canvas_from_chip(chip: np.ndarray, width: int, height: int, rng: random.Random) -> np.ndarray:
    pixels = chip.reshape(-1, 3).astype(np.float32)
    median = np.percentile(pixels, 50, axis=0)
    mad = np.percentile(np.abs(pixels - median), 65, axis=0)
    channel_bias = np.clip(median - median.mean(), -8.0, 8.0) * 0.18
    target = np.array(
        [112.0 + rng.uniform(-5, 5), 130.0 + rng.uniform(-5, 5), 154.0 + rng.uniform(-5, 5)],
        dtype=np.float32,
    ) + channel_bias
    noise_rng = np.random.default_rng(rng.randrange(2**32))

    def smooth_noise(grid_div: int, sigma: float) -> np.ndarray:
        small_h = max(8, height // grid_div)
        small_w = max(8, width // grid_div)
        noise = noise_rng.normal(0, sigma, (small_h, small_w, 3)).astype(np.float32)
        noise_img = Image.fromarray(np.uint8(np.clip(noise + 127, 0, 255)), "RGB")
        up = noise_img.resize((width, height), Image.Resampling.BICUBIC)
        return np.asarray(up, dtype=np.float32) - 127.0

    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None, None]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :, None]
    gradient = (
        yy * np.array([rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0)], dtype=np.float32)
        + xx * np.array([rng.uniform(-1.4, 1.4), rng.uniform(-1.4, 1.4), rng.uniform(-1.4, 1.4)], dtype=np.float32)
    )
    texture_sigma = float(np.clip(mad.mean() * 0.10, 0.8, 2.4))
    arr = np.zeros((height, width, 3), dtype=np.float32)
    arr[:] = target
    arr += gradient
    arr += smooth_noise(56, texture_sigma * 1.25)
    arr += smooth_noise(24, texture_sigma * 0.65)
    arr += noise_rng.normal(0, rng.uniform(0.18, 0.45), arr.shape)
    arr = cv2.GaussianBlur(arr, (0, 0), rng.uniform(0.15, 0.35))
    return np.uint8(np.clip(arr, 0, 255))


def make_fold_train_bg64_backgrounds(
    data_root: Path,
    manifest: dict,
    output_root: Path,
    seed: int,
    background_count: int,
    chip_size: int,
) -> tuple[list[dict], int]:
    image_dir = Path(manifest.get("image_dir_macstudio") or data_root / "train_images")
    width, height = [int(v) for v in manifest["image_size"]]
    manifest_files = {image["file"] for image in manifest["images"]}
    rng = random.Random(seed + int(manifest.get("fold", 0)) * 1009 + 71)

    chip_dir = output_root / "backgrounds" / "chips_fold_train"
    canvas_dir = output_root / "backgrounds" / "bg64"
    reports_dir = output_root / "reports"
    chip_dir.mkdir(parents=True, exist_ok=True)
    canvas_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    chip_rows: list[dict] = []
    chips: list[dict] = []
    contamination_count = 0
    for index, image_meta in enumerate(manifest["images"]):
        file_name = image_meta["file"]
        image_path = image_dir / file_name
        if not image_path.exists():
            chip_rows.append({"source_file": file_name, "status": "missing_source_image"})
            continue
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        if rgb.shape[:2] != (height, width):
            chip_rows.append({"source_file": file_name, "status": "bad_image_size"})
            continue
        chip_rng = random.Random(seed + index * 7919)
        chip = None
        meta: dict = {}
        forbidden = None
        for pad in (64, 48, 32, 24, 16, 8, 0):
            forbidden = _background_forbidden_mask(width, height, image_meta["pills"], pad=pad)
            chip, meta = _crop_clean_background_chip(rgb, forbidden, chip_rng, chip_size)
            if chip is not None:
                meta["bbox_exclusion_pad"] = pad
                break
        if chip is None:
            chip_rows.append({"source_file": file_name, "status": "no_clean_crop", **meta})
            continue
        assert forbidden is not None
        mixed = int(forbidden[meta["y"] : meta["y"] + meta["chip_size"], meta["x"] : meta["x"] + meta["chip_size"]].sum())
        if mixed:
            contamination_count += 1
        chip_name = f"bg_chip_{index:04d}_{Path(file_name).stem}.png"
        Image.fromarray(chip, "RGB").save(chip_dir / chip_name, format="PNG", optimize=True)
        row = {
            "source_file": file_name,
            "chip": str((chip_dir / chip_name).relative_to(output_root)),
            "status": "ok",
            "x": meta["x"],
            "y": meta["y"],
            "chip_size": meta["chip_size"],
            "bbox_exclusion_pad": meta["bbox_exclusion_pad"],
            "forbidden_pixels": mixed,
            "source_in_manifest": file_name in manifest_files,
        }
        chip_rows.append(row)
        chips.append({"source_file": file_name, "rgb": chip, "row": row})

    if len(chips) < background_count:
        raise RuntimeError(f"Too few clean fold-train background chips: {len(chips)}/{background_count}")

    backgrounds: list[dict] = []
    canvas_rows: list[dict] = []
    for idx in range(background_count):
        chip_record = chips[idx % len(chips)]
        bg = _make_bg_canvas_from_chip(chip_record["rgb"], width, height, rng)
        canvas_name = f"fold{manifest.get('fold', 'x')}_bg64_{idx:02d}.png"
        Image.fromarray(bg, "RGB").save(canvas_dir / canvas_name, format="PNG", optimize=True)
        backgrounds.append(
            {
                "file": canvas_name,
                "rgb": bg,
                "masked_pixels": 0,
                "background_mode": "fold_train_bg64",
                "source_file": chip_record["source_file"],
            }
        )
        canvas_rows.append(
            {
                "background": str((canvas_dir / canvas_name).relative_to(output_root)),
                "source_file": chip_record["source_file"],
                "source_in_manifest": chip_record["source_file"] in manifest_files,
                "rgb_mean": json.dumps([round(float(v), 3) for v in bg.reshape(-1, 3).mean(axis=0)]),
            }
        )

    with (reports_dir / "background_chip_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["source_file", "chip", "status", "x", "y", "chip_size", "bbox_exclusion_pad", "forbidden_pixels", "source_in_manifest", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(chip_rows)
    with (reports_dir / "background_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["background", "source_file", "source_in_manifest", "rgb_mean"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(canvas_rows)

    return backgrounds, contamination_count


def sample_combo_size(rng: random.Random, manifest: dict) -> int:
    counts = collections.Counter(len(image["pills"]) for image in manifest["images"])
    choices = sorted(k for k in counts if 2 <= k <= 4)
    weights = [counts[k] for k in choices]
    return rng.choices(choices, weights=weights, k=1)[0]


def transform_cutout(cutout: Image.Image, rng: random.Random) -> Image.Image:
    scale = rng.uniform(0.92, 1.08)
    new_size = (max(1, int(round(cutout.width * scale))), max(1, int(round(cutout.height * scale))))
    transformed = cutout.resize(new_size, Image.Resampling.LANCZOS)
    angle = rng.uniform(0, 360)
    return transformed.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)


def alpha_bbox(image_rgba: Image.Image) -> tuple[int, int, int, int] | None:
    alpha = np.asarray(image_rgba.getchannel("A"))
    ys, xs = np.where(alpha > 12)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def group_assets_by_category(assets: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = collections.defaultdict(list)
    for asset in assets:
        grouped[int(asset["category_id"])].append(asset)
    return dict(grouped)


def natural_category_choices(manifest: dict, assets_by_category: dict[int, list[dict]]) -> tuple[list[int], list[int]]:
    natural = {int(k): int(v) for k, v in manifest.get("class_count_natural", {}).items()}
    categories = [category for category, count in sorted(natural.items()) if count > 0 and category in assets_by_category]
    weights = [natural[category] for category in categories]
    if not categories:
        categories = sorted(assets_by_category)
        weights = [len(assets_by_category[category]) for category in categories]
    return categories, weights


def choose_natural_assets(
    assets_by_category: dict[int, list[dict]],
    categories: list[int],
    weights: list[int],
    combo_size: int,
    rng: random.Random,
) -> list[dict]:
    selected_categories: list[int] = []
    selected_set: set[int] = set()
    for _ in range(combo_size):
        available = [(category, weight) for category, weight in zip(categories, weights) if category not in selected_set]
        if not available:
            available = list(zip(categories, weights))
        choice_categories = [category for category, _ in available]
        choice_weights = [weight for _, weight in available]
        category = rng.choices(choice_categories, weights=choice_weights, k=1)[0]
        selected_categories.append(category)
        selected_set.add(category)
    return [rng.choice(assets_by_category[category]) for category in selected_categories]


def paste_with_shadow(canvas: Image.Image, item: Image.Image, x: int, y: int, rng: random.Random) -> None:
    alpha = item.getchannel("A")
    shadow = Image.new("RGBA", item.size, (0, 0, 0, 0))
    shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=rng.uniform(3.0, 6.5)))
    opacity = rng.randint(22, 48)
    shadow.putalpha(shadow_alpha.point(lambda value: int(value * opacity / 255)))
    canvas.alpha_composite(shadow, (x + rng.randint(3, 8), y + rng.randint(4, 10)))
    canvas.alpha_composite(item, (x, y))


def rect_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def candidate_centers(combo_size: int, width: int, height: int, rng: random.Random) -> list[tuple[int, int]]:
    if combo_size == 2:
        patterns = [
            [(0.28, 0.34), (0.72, 0.66)],
            [(0.30, 0.66), (0.72, 0.34)],
            [(0.32, 0.50), (0.70, 0.50)],
        ]
    elif combo_size == 3:
        patterns = [
            [(0.28, 0.30), (0.72, 0.34), (0.50, 0.72)],
            [(0.30, 0.68), (0.50, 0.30), (0.74, 0.68)],
            [(0.28, 0.36), (0.70, 0.50), (0.34, 0.74)],
        ]
    else:
        patterns = [
            [(0.28, 0.28), (0.72, 0.30), (0.30, 0.72), (0.72, 0.72)],
            [(0.34, 0.25), (0.72, 0.40), (0.28, 0.62), (0.66, 0.78)],
        ]
    centers = rng.choice(patterns)
    jitter_x = int(width * 0.055)
    jitter_y = int(height * 0.055)
    return [
        (
            int(round(cx * width + rng.randint(-jitter_x, jitter_x))),
            int(round(cy * height + rng.randint(-jitter_y, jitter_y))),
        )
        for cx, cy in centers
    ]


def render_sample(
    sample_id: int,
    assets: list[dict],
    backgrounds: list[dict],
    manifest: dict,
    rng: random.Random,
    min_edge_margin: int,
) -> tuple[Image.Image, list[dict], dict]:
    width, height = [int(v) for v in manifest["image_size"]]
    combo_size = sample_combo_size(rng, manifest)
    assets_by_category = group_assets_by_category(assets)
    categories, weights = natural_category_choices(manifest, assets_by_category)
    selected = choose_natural_assets(assets_by_category, categories, weights, combo_size, rng)

    bg_record = rng.choice(backgrounds)
    bg = Image.fromarray(bg_record["rgb"].copy(), "RGB")
    bg = ImageEnhance.Brightness(bg).enhance(rng.uniform(0.985, 1.015))
    bg = ImageEnhance.Contrast(bg).enhance(rng.uniform(0.985, 1.015))
    canvas = bg.convert("RGBA")
    centers = candidate_centers(combo_size, width, height, rng)
    rng.shuffle(centers)

    annotations: list[dict] = []
    placed_boxes: list[list[float]] = []
    placement_failures = 0
    for order, asset in enumerate(selected):
        placed = False
        item = transform_cutout(asset["cutout"], rng)
        for attempt in range(160):
            if attempt > 0 and attempt % 40 == 0:
                item = transform_cutout(asset["cutout"], rng)
            abox = alpha_bbox(item)
            if abox is None:
                continue
            ax0, ay0, ax1, ay1 = abox
            alpha_w, alpha_h = ax1 - ax0, ay1 - ay0
            cx, cy = centers[order % len(centers)]
            if attempt > 0:
                cx += rng.randint(-120, 120)
                cy += rng.randint(-150, 150)
            px = int(round(cx - item.width / 2))
            py = int(round(cy - item.height / 2))
            bbox = [px + ax0, py + ay0, alpha_w, alpha_h]
            if (
                bbox[0] < min_edge_margin
                or bbox[1] < min_edge_margin
                or bbox[0] + bbox[2] > width - min_edge_margin
                or bbox[1] + bbox[3] > height - min_edge_margin
            ):
                continue
            area_norm = (bbox[2] * bbox[3]) / (width * height)
            if area_norm < 0.015 or area_norm > 0.14:
                continue
            if any(rect_iou(bbox, existing) > 0.02 for existing in placed_boxes):
                continue
            paste_with_shadow(canvas, item, px, py, rng)
            placed_boxes.append(bbox)
            annotations.append(
                {
                    "category_id": int(asset["category_id"]),
                    "class_index": int(asset["class_index"]),
                    "product_id": asset["product_id"],
                    "bbox": [round(float(v), 2) for v in bbox],
                    "area": round(float(bbox[2] * bbox[3]), 2),
                    "iscrowd": 0,
                    "segmentation": [],
                    "source_ref": {
                        "file": asset["file"],
                        "bbox_px": asset["bbox_px"],
                        "asset_id": asset["asset_id"],
                    },
                }
            )
            placed = True
            break
        if not placed:
            placement_failures += 1
    if len(annotations) != combo_size:
        raise RuntimeError(f"Sample {sample_id}: placement failed {placement_failures}, got {len(annotations)}/{combo_size}")
    final = canvas.convert("RGB")
    arr = np.asarray(final, dtype=np.float32)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(0.15, 0.55), arr.shape)
    final = Image.fromarray(np.uint8(np.clip(arr + noise, 0, 255)), "RGB")
    meta = {"combo_size": combo_size, "background_source": bg_record["file"], "placement_failures": placement_failures}
    return final, annotations, meta


def rank(values: dict[int, int]) -> dict[int, float]:
    sorted_items = sorted(values.items(), key=lambda item: item[1])
    ranks: dict[int, float] = {}
    i = 0
    while i < len(sorted_items):
        j = i
        while j + 1 < len(sorted_items) and sorted_items[j + 1][1] == sorted_items[i][1]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_items[k][0]] = avg
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    if not xs or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (denx * deny) if denx > 0 and deny > 0 else 0.0


def write_webdataset(output_root: Path, coco: dict, shard_size: int) -> list[dict]:
    train_dir = output_root / "webdataset" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(coco["images"], key=lambda image: int(image["id"]))
    anns_by_image: dict[int, list[dict]] = collections.defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)
    cat_by_id = {int(cat["id"]): cat for cat in coco["categories"]}
    file_prefix = coco.get("info", {}).get("file_prefix", "realcopy_v1")
    rows = []
    for shard_idx, start in enumerate(range(0, len(images), shard_size)):
        shard_images = images[start : start + shard_size]
        shard_name = f"{file_prefix}-{shard_idx:06d}.tar"
        shard_path = train_dir / shard_name
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
                    "synthetic": True,
                    "realcopy": True,
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
        rows.append(
            {
                "shard": f"webdataset/train/{shard_name}",
                "samples": sample_count,
                "annotations": ann_count,
                "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
                "bytes": shard_path.stat().st_size,
            }
        )
    with (output_root / "webdataset" / "shards_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["shard", "samples", "annotations", "sha256", "bytes"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def make_contact_sheet(output_root: Path, coco: dict, rng: random.Random) -> None:
    images = coco["images"]
    anns_by_image: dict[int, list[dict]] = collections.defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)
    selected = rng.sample(images, k=min(20, len(images)))
    thumb_w, thumb_h = 244, 320
    label_h = 22
    cols = 5
    rows = math.ceil(len(selected) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (245, 245, 242))
    colors = [(255, 50, 60), (30, 130, 255), (30, 180, 90), (255, 170, 0)]
    for idx, image_meta in enumerate(selected):
        image = Image.open(output_root / "coco" / "images" / image_meta["file_name"]).convert("RGB")
        image = image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)
        sx, sy = thumb_w / image_meta["width"], thumb_h / image_meta["height"]
        for j, ann in enumerate(anns_by_image[int(image_meta["id"])]):
            x, y, w, h = ann["bbox"]
            draw.rectangle([x * sx, y * sy, (x + w) * sx, (y + h) * sy], outline=colors[j % len(colors)], width=2)
        x0 = (idx % cols) * thumb_w
        y0 = (idx // cols) * (thumb_h + label_h)
        sheet.paste(image, (x0, y0 + label_h))
        ImageDraw.Draw(sheet).text((x0 + 3, y0 + 4), f"{image_meta['file_name']} n={len(anns_by_image[int(image_meta['id'])])}", fill=(0, 0, 0))
    (output_root / "reports" / "audit").mkdir(parents=True, exist_ok=True)
    sheet.save(output_root / "reports" / "audit" / "realcopy_v1_contact_sheet.png")


def checkerboard(size: tuple[int, int], cell: int = 12) -> Image.Image:
    width, height = size
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    light = np.array([232, 235, 238], dtype=np.uint8)
    dark = np.array([198, 204, 210], dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x] = light if ((x // cell) + (y // cell)) % 2 == 0 else dark
    return Image.fromarray(arr, "RGB")


def make_cutout_asset_sheet(output_root: Path, assets: list[dict]) -> None:
    selected = sorted(assets, key=lambda asset: (int(asset["class_index"]), int(asset["asset_id"])))
    selected = selected[: min(80, len(selected))]
    thumb_w, thumb_h = 144, 152
    label_h = 26
    cols = 8
    rows = math.ceil(len(selected) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (245, 245, 242))
    draw_sheet = ImageDraw.Draw(sheet)
    for idx, asset in enumerate(selected):
        x0 = (idx % cols) * thumb_w
        y0 = (idx // cols) * (thumb_h + label_h)
        cell = checkerboard((thumb_w, thumb_h)).convert("RGBA")
        cutout = asset["cutout"].copy().convert("RGBA")
        cutout.thumbnail((thumb_w - 18, thumb_h - 18), Image.Resampling.LANCZOS)
        px = x0 + (thumb_w - cutout.width) // 2
        py = y0 + label_h + (thumb_h - cutout.height) // 2
        cell.alpha_composite(cutout, (px - x0, py - y0 - label_h))
        sheet.paste(cell.convert("RGB"), (x0, y0 + label_h))
        draw_sheet.text(
            (x0 + 3, y0 + 4),
            f"c{asset['class_index']} {asset['product_id']}",
            fill=(0, 0, 0),
        )
    (output_root / "reports" / "audit").mkdir(parents=True, exist_ok=True)
    sheet.save(output_root / "reports" / "audit" / "realcopy_sam2_cutout_assets.jpg", quality=95)


def write_cutout_assets(output_root: Path, assets: list[dict]) -> None:
    cutout_dir = output_root / "assets" / "cutouts"
    cutout_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for asset in sorted(assets, key=lambda item: int(item["asset_id"])):
        file_name = (
            f"realcopy_sam2_asset_{int(asset['asset_id']):04d}"
            f"_class{int(asset['class_index']):02d}_cat{int(asset['category_id'])}.png"
        )
        path = cutout_dir / file_name
        asset["cutout"].save(path, format="PNG", optimize=True)
        asset["asset_path"] = str(path.relative_to(output_root))
        rows.append(
            {
                "asset_id": int(asset["asset_id"]),
                "asset_path": asset["asset_path"],
                "source_file": asset["file"],
                "category_id": int(asset["category_id"]),
                "class_index": int(asset["class_index"]),
                "product_id": asset["product_id"],
                "source_bbox": json.dumps(asset["bbox_px"], separators=(",", ":")),
                "source_instance_index": int(asset["source_instance_index"]),
                "cutout_width": asset["cutout"].width,
                "cutout_height": asset["cutout"].height,
                "quality_method": asset["quality"].get("method", ""),
                "quality_score": asset["quality"].get("score", ""),
                "quality": json.dumps(asset["quality"], ensure_ascii=False, separators=(",", ":")),
            }
        )
    with (output_root / "assets" / "cutout_assets_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "asset_id",
            "asset_path",
            "source_file",
            "category_id",
            "class_index",
            "product_id",
            "source_bbox",
            "source_instance_index",
            "cutout_width",
            "cutout_height",
            "quality_method",
            "quality_score",
            "quality",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(
    output_root: Path,
    categories: list[dict],
    manifest: dict,
    assets: list[dict],
    backgrounds: list[dict],
    dropped_source: list[dict],
    failed_cutouts: list[dict],
    coco: dict,
    shard_rows: list[dict],
    background_contamination_count: int,
) -> dict:
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    generated_counts = collections.Counter(int(ann["category_id"]) for ann in coco["annotations"])
    asset_counts = collections.Counter(int(asset["category_id"]) for asset in assets)
    cutout_method_counts = collections.Counter(str(asset["quality"].get("method", "unknown")) for asset in assets)
    manifest_source_files = {image["file"] for image in manifest["images"]}
    asset_source_files = {asset["file"] for asset in assets}
    asset_source_outside_manifest = sorted(asset_source_files - manifest_source_files)
    asset_source_missing_from_assets = sorted(manifest_source_files - asset_source_files)
    background_source_files = {record.get("source_file", "") for record in backgrounds if record.get("source_file")}
    background_source_outside_manifest = sorted(background_source_files - manifest_source_files)
    background_source_pool_count = 0
    chip_manifest = output_root / "reports" / "background_chip_manifest.csv"
    if chip_manifest.exists():
        with chip_manifest.open(newline="", encoding="utf-8") as handle:
            background_source_pool_count = len(
                {
                    row["source_file"]
                    for row in csv.DictReader(handle)
                    if row.get("status") == "ok" and row.get("source_file")
                }
            )
    source_counts = {int(k): int(v) for k, v in manifest["class_count_natural"].items()}
    active_categories = sorted(source_counts)
    source_rank = rank({cat: source_counts.get(cat, 0) for cat in active_categories})
    gen_rank = rank({cat: generated_counts.get(cat, 0) for cat in active_categories})
    spearman = pearson([source_rank[c] for c in active_categories], [gen_rank[c] for c in active_categories])
    source_values = [source_counts.get(c, 0) for c in active_categories]
    gen_values = [generated_counts.get(c, 0) for c in active_categories]
    pearson_count = pearson(source_values, gen_values)

    with (reports / "class_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "class_index",
            "category_id",
            "product_id",
            "source_count_manifest",
            "asset_count_after_quality_gate",
            "generated_instances",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cat in categories:
            category_id = int(cat["id"])
            writer.writerow(
                {
                    "class_index": int(cat["class_index"]),
                    "category_id": category_id,
                    "product_id": cat["product_id"],
                    "source_count_manifest": source_counts.get(category_id, 0),
                    "asset_count_after_quality_gate": asset_counts.get(category_id, 0),
                    "generated_instances": generated_counts.get(category_id, 0),
                }
            )

    pill_count = collections.Counter(len([ann for ann in coco["annotations"] if ann["image_id"] == image["id"]]) for image in coco["images"])
    areas = [(ann["bbox"][2] * ann["bbox"][3]) / (976 * 1280) for ann in coco["annotations"]]
    quantiles = np.quantile(np.asarray(areas, dtype=np.float32), [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).round(5).tolist()
    bbox_failures = 0
    category_failures = 0
    allowed_categories = {int(cat["id"]) for cat in categories}
    for ann in coco["annotations"]:
        x, y, w, h = [float(v) for v in ann["bbox"]]
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > 976 or y + h > 1280:
            bbox_failures += 1
        if int(ann["category_id"]) not in allowed_categories:
            category_failures += 1

    summary = {
        "output_root": str(output_root),
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "categories_total": len(categories),
        "classes_used": len(generated_counts),
        "source_images": len(manifest["images"]),
        "source_manifest_path": manifest.get("_source_manifest_path", ""),
        "source_images_manifest_count": len(manifest_source_files),
        "cutout_asset_source_files_count": len(asset_source_files),
        "cutout_asset_source_outside_manifest_count": len(asset_source_outside_manifest),
        "cutout_asset_source_outside_manifest_samples": asset_source_outside_manifest[:20],
        "cutout_asset_source_missing_from_assets_count": len(asset_source_missing_from_assets),
        "cutout_asset_source_missing_from_assets_samples": asset_source_missing_from_assets[:20],
        "background_source_files_count": len(background_source_files),
        "background_source_pool_image_count": background_source_pool_count,
        "background_source_outside_manifest_count": len(background_source_outside_manifest),
        "background_source_outside_manifest_samples": background_source_outside_manifest[:20],
        "source_pill_instances_manifest": int(manifest["source_pill_instances"]),
        "source_records_after_basic_gate": len(assets) + len(failed_cutouts),
        "cutout_assets": len(assets),
        "cutout_method_counts": {str(k): int(v) for k, v in sorted(cutout_method_counts.items())},
        "dropped_source_records": len(dropped_source),
        "failed_cutouts": len(failed_cutouts),
        "webdataset_shards": len(shard_rows),
        "pills_per_image_counts": {str(k): int(v) for k, v in sorted(pill_count.items())},
        "pills_per_image_mean": round(len(coco["annotations"]) / max(1, len(coco["images"])), 4),
        "bbox_norm_area_quantiles": quantiles,
        "class_count_spearman_vs_manifest": round(spearman, 4),
        "class_count_pearson_vs_manifest": round(pearson_count, 4),
        "background_pill_mixed_count": background_contamination_count,
        "bbox_failures": bbox_failures,
        "category_failures": category_failures,
        "category_id_1_count": int(generated_counts.get(1, 0)),
    }
    (reports / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (reports / "validation_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (reports / "dropped_source_records.json").write_text(json.dumps(dropped_source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Avoid embedding RGBA images in JSON.
    serializable_failed = [{k: v for k, v in row.items() if k != "cutout"} for row in failed_cutouts]
    (reports / "failed_cutouts.json").write_text(json.dumps(serializable_failed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_dataset(args) -> dict:
    rng = random.Random(args.seed)
    categories, manifest, by_category = load_inputs(args.handoff_dir, args.source_manifest, args.categories_path)
    output_root = args.output_root
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_root}. Pass --overwrite.")
        shutil.rmtree(output_root)
    (output_root / "coco" / "images").mkdir(parents=True, exist_ok=True)
    (output_root / "webdataset" / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)
    (output_root / "spec").mkdir(parents=True, exist_ok=True)
    guide_path = args.handoff_dir / "REALCOPY_GUIDE.md"
    categories_path = args.categories_path or (args.handoff_dir / "target_categories_schema.json")
    source_manifest_path = args.source_manifest or (args.handoff_dir / "realcopy_source_manifest_fold0train.json")
    if guide_path.exists():
        shutil.copy2(guide_path, output_root / "spec" / "REALCOPY_GUIDE.md")
    shutil.copy2(categories_path, output_root / "spec" / "target_categories_schema.json")
    shutil.copy2(source_manifest_path, output_root / "spec" / source_manifest_path.name)

    source_records, dropped_source = build_source_records(manifest, by_category)
    mask_provider = None
    if args.mask_provider == "sam2":
        mask_provider = Sam2MaskProvider(
            checkpoint=args.sam2_checkpoint,
            config=args.sam2_config,
            device=args.sam2_device,
            multimask=True,
            logit_threshold=args.sam2_logit_threshold,
            box_expansion_ratio=args.sam2_box_expansion_ratio,
        )
    assets, failed_cutouts = build_cutout_bank(
        args.data_root,
        manifest,
        source_records,
        by_category,
        mask_provider,
        args.mask_provider,
        args.min_mask_score,
        args.margin_ratio,
    )
    if len(assets) < 100:
        raise RuntimeError(f"Too few cutout assets: {len(assets)}")
    write_cutout_assets(output_root, assets)
    make_cutout_asset_sheet(output_root, assets)
    if args.background_mode == "clean64_preprocessed":
        backgrounds, background_contamination_count = make_clean64_preprocessed_backgrounds(
            args.clean64_background_dir, manifest
        )
    elif args.background_mode == "fold_train_bg64":
        backgrounds, background_contamination_count = make_fold_train_bg64_backgrounds(
            args.data_root,
            manifest,
            output_root,
            args.seed,
            args.fold_bg_count,
            args.fold_bg_chip_size,
        )
    elif args.background_mode == "validated_v2_clean":
        backgrounds, background_contamination_count = make_validated_v2_clean_backgrounds(
            args.validated_background_dir, manifest, args.seed
        )
    else:
        backgrounds, background_contamination_count = make_inpaint_backgrounds(args.data_root, manifest)
    if not backgrounds:
        raise RuntimeError("No usable backgrounds")

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    ann_id = 1
    sample_retry_count = 0
    for sample_index in range(1, args.num_images + 1):
        last_error: Exception | None = None
        for render_attempt in range(30):
            try:
                image, annotations, meta = render_sample(
                    sample_index,
                    assets,
                    backgrounds,
                    manifest,
                    rng,
                    args.min_edge_margin,
                )
                sample_retry_count += render_attempt
                break
            except RuntimeError as exc:
                last_error = exc
        else:
            raise RuntimeError(f"Sample {sample_index}: failed after retries: {last_error}") from last_error
        file_name = f"{args.file_prefix}_{sample_index:06d}.jpg"
        image.save(output_root / "coco" / "images" / file_name, format="JPEG", quality=95, subsampling=0, optimize=True)
        image_id = sample_index
        coco_images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": 976,
                "height": 1280,
                "realcopy": True,
                "combo_size": meta["combo_size"],
                "background_source": meta["background_source"],
            }
        )
        for annotation in annotations:
            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    **annotation,
                }
            )
            ann_id += 1

    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
        "info": {
            "name": output_root.name,
            "source": f"real Copy-Paste from {source_manifest_path.name}; no manifest-external images",
            "augmentation": f"realcopy_{args.mask_provider}_alpha_cutout_{args.background_mode}_background_grid_jitter",
            "num_images": args.num_images,
            "file_prefix": args.file_prefix,
            "sample_retry_count": sample_retry_count,
            "mask_provider": args.mask_provider,
            "min_mask_score": args.min_mask_score,
            "background_mode": args.background_mode,
            "source_manifest": source_manifest_path.name,
            "min_edge_margin": args.min_edge_margin,
        },
    }
    (output_root / "coco" / "annotations_coco.json").write_text(
        json.dumps(coco, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shard_rows = write_webdataset(output_root, coco, args.shard_size)
    summary = write_reports(
        output_root,
        categories,
        manifest,
        assets,
        backgrounds,
        dropped_source,
        failed_cutouts,
        coco,
        shard_rows,
        background_contamination_count,
    )
    make_contact_sheet(output_root, coco, rng)
    readme = f"""# {output_root.name}

Kaggle per-fold train real Copy-Paste augmentation.

## Contents
- `coco/images/*.jpg`: {len(coco_images)} realcopy images
- `coco/annotations_coco.json`: COCO labels, categories copied from `target_categories_schema.json`
- `webdataset/train/*.tar`: self-contained WebDataset shards with `.jpg` + `.json`
- `assets/cutouts/*.png`: accepted {args.mask_provider.upper()} cutout assets for audit/repro checks
- `spec/`: handoff guide, target category schema, and fold0-train source manifest
- `reports/`: validation, class distribution, dropped source records, and audit sheet

## Counts
- images: {len(coco_images)}
- annotations: {len(coco_annotations)}
- classes used: {summary["classes_used"]}
- cutout assets: {summary["cutout_assets"]}
- pills/image mean: {summary["pills_per_image_mean"]}
- bbox failures: {summary["bbox_failures"]}
- category failures: {summary["category_failures"]}
- background pill mixed count: {summary["background_pill_mixed_count"]}
- WebDataset shards: {len(shard_rows)}
- mask provider: {args.mask_provider}
- background mode: {args.background_mode}
- source manifest: {source_manifest_path.name}

This dataset uses only the provided source manifest. It does not use manifest-external val/test images.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    (output_root / "dataset_info.json").write_text(
        json.dumps(
            {
                "dataset_name": output_root.name,
                "format": ["COCO + images", "WebDataset"],
                "images": len(coco_images),
                "annotations": len(coco_annotations),
                "classes": len(categories),
                "classes_used": summary["classes_used"],
                "source": f"real Copy-Paste from {source_manifest_path.name} only",
                "mask_provider": args.mask_provider,
                "background_mode": args.background_mode,
                "source_manifest": f"spec/{source_manifest_path.name}",
                "coco_annotations": "coco/annotations_coco.json",
                "webdataset": "webdataset/train/*.tar",
                "summary": "reports/summary.json",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(build_dataset(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
