#!/usr/bin/env python3
"""Shared helpers for v2 manifest, split, synthetic, and shard generation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from aihub_common import (
    extract_annotations,
    extract_primary_image,
    iter_members,
    load_json_bytes,
    manifest_value_to_member_list,
    member_list_to_manifest_value,
    member_stem,
    read_manifest_rows,
    zip_map,
)


DATA_ROOT = Path(
    "/Volumes/SSD 4T/166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터/01.데이터"
)
TRAIN_SINGLE_SOURCE_DIR = DATA_ROOT / "1.Training_jpeg_q95" / "원천데이터" / "단일경구약제 5000종"
TRAIN_SINGLE_LABEL_DIR = DATA_ROOT / "1.Training_jpeg_q95" / "라벨링데이터" / "단일경구약제 5000종"
VAL_SINGLE_SOURCE_DIR = DATA_ROOT / "2.Validation_jpeg_q95" / "원천데이터" / "단일경구약제 5000종"
VAL_SINGLE_LABEL_DIR = DATA_ROOT / "2.Validation_jpeg_q95" / "라벨링데이터" / "단일경구약제 5000종"
VAL_COMBO_SOURCE_DIR = DATA_ROOT / "2.Validation_jpeg_q95" / "원천데이터" / "경구약제조합 5000종"
VAL_COMBO_LABEL_DIR = DATA_ROOT / "2.Validation_jpeg_q95" / "라벨링데이터" / "경구약제조합 5000종"
DEFAULT_PROCESSED_ROOT = DATA_ROOT / "processed" / "v2"

TRAIN_MANIFEST_DIR = Path("reports/manifests")
VAL_SINGLE_MANIFEST_DIR = Path("reports/validation_single_q95/manifests")
VAL_COMBO_MANIFEST_DIR = Path("reports/validation_combo_q95/manifests")

CANONICAL_FIELDS = [
    "sample_id",
    "dataset_kind",
    "split_source",
    "split",
    "image_path",
    "image_member",
    "label_path",
    "label_members",
    "product_id",
    "combo_product_ids",
    "combo_size",
    "bbox",
    "annotations",
    "width",
    "height",
    "shape",
    "color",
    "back_color",
    "light_color",
    "camera_la",
    "camera_lo",
    "drug_dir",
    "source_zip",
    "label_zip",
    "source_zip_name",
    "label_zip_name",
    "set_id",
    "original_stem",
    "sha256",
    "phash",
    "synthetic",
    "source_refs",
    "transform",
    "mask_quality",
    "ignore_for_id",
    "class_index",
]

K_ID_RE = re.compile(r"K-\d{6}")


@dataclass(frozen=True)
class ManifestInput:
    split_source: str
    dataset_kind: str
    paired_manifest_dir: Path
    image_zip_dir: Path
    label_zip_dir: Path
    image_prefix: str
    label_prefix: str
    zip_kind: str


def json_compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def json_loads_or(value: str | None, default):
    if value:
        return json.loads(value)
    return default


def extract_product_id(*texts: str) -> str:
    for text in texts:
        if not text:
            continue
        match = K_ID_RE.search(text)
        if match:
            return match.group(0)
    return ""


def parse_combo_product_ids(stem_or_member: str) -> list[str]:
    key = PurePosixPath(stem_or_member).stem.split("_", 1)[0]
    parts = key.split("-")
    if parts and parts[0] == "K":
        ids = [f"K-{part}" for part in parts[1:] if part.isdigit()]
        if ids:
            return ids
    ids = K_ID_RE.findall(key)
    return ids


def stable_sample_id(split_source: str, source_zip_name: str, stem: str) -> str:
    raw = f"{split_source}|{source_zip_name}|{stem}".encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:16]
    return f"{split_source}_{digest}"


def sanitize_tar_key(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)


def bbox_in_bounds(bbox: list[float], width: int, height: int) -> bool:
    if len(bbox) != 4:
        return False
    x, y, w, h = bbox
    return x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= width and y + h <= height


def iou_xywh(a: list[float], b: list[float]) -> float:
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


def phash_bytes(image_data: bytes) -> str:
    with Image.open(BytesIO(image_data)) as image:
        gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    dct = _dct_2d(pixels)
    low = dct[:8, :8].flatten()
    median = np.median(low[1:])
    bits = low > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return f"{value:016x}"


def _dct_2d(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    coeff = np.empty((n, n), dtype=np.float32)
    factor = math.pi / (2.0 * n)
    scale0 = math.sqrt(1.0 / n)
    scale = math.sqrt(2.0 / n)
    xs = np.arange(n, dtype=np.float32)
    for k in range(n):
        coeff[k, :] = (scale0 if k == 0 else scale) * np.cos((2 * xs + 1) * k * factor)
    return coeff @ values @ coeff.T


def optional_write_parquet(csv_path: Path, parquet_path: Path) -> bool:
    try:
        import pyarrow.csv as pv
        import pyarrow.parquet as pq
    except Exception:
        return False
    table = pv.read_csv(csv_path)
    pq.write_table(table, parquet_path, compression="zstd")
    return True


def write_csv(path: Path, fields: list[str], rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
            count += 1
    return count


def read_csv_rows(path: Path, limit: int | None = None) -> Iterator[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            yield row


def zip_members_by_stem(zip_file: ZipFile, suffix: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for member in iter_members(zip_file.namelist(), suffix):
        result.setdefault(member_stem(member), []).append(member)
    return result


def canonical_annotations_from_labels(label_docs: list[dict], product_ids: list[str]) -> list[dict]:
    annotations = []
    for index, label_doc in enumerate(label_docs):
        image = extract_primary_image(label_doc, "label")
        annotation = extract_annotations(label_doc, "label")[0]
        product_id = product_ids[index] if index < len(product_ids) else image.get("drug_N", "")
        annotations.append(
            {
                "id": index + 1,
                "bbox": annotation.get("bbox"),
                "area": annotation.get("area"),
                "category_id": annotation.get("category_id", 1),
                "product_id": product_id,
                "ignore_for_id": False,
                "source_image_id": image.get("id"),
                "source_label_member": "",
            }
        )
    return annotations


def label_metadata(label_doc: dict) -> dict:
    image = extract_primary_image(label_doc, "label")
    return {
        "shape": image.get("drug_shape", ""),
        "color": "|".join(part for part in [image.get("color_class1", ""), image.get("color_class2", "")] if part),
        "back_color": image.get("back_color", ""),
        "light_color": image.get("light_color", ""),
        "camera_la": image.get("camera_la", ""),
        "camera_lo": image.get("camera_lo", ""),
        "drug_dir": image.get("drug_dir", ""),
        "width": image.get("width", ""),
        "height": image.get("height", ""),
    }


def canonical_json(row: dict) -> dict:
    return {
        "sample_id": row["sample_id"],
        "image": {
            "file_name": f"{row['sample_id']}.jpg",
            "width": int(row["width"]),
            "height": int(row["height"]),
            "source_zip": row.get("source_zip", ""),
            "image_member": row.get("image_member", ""),
        },
        "annotations": json_loads_or(row.get("annotations"), []),
        "product_id": row.get("product_id", ""),
        "combo_product_ids": json_loads_or(row.get("combo_product_ids"), []),
        "split": row.get("split", ""),
        "split_source": row.get("split_source", ""),
        "dataset_kind": row.get("dataset_kind", ""),
        "synthetic": row.get("synthetic", "false") == "true",
        "source_refs": json_loads_or(row.get("source_refs"), []),
        "transform": json_loads_or(row.get("transform"), {}),
        "mask_quality": json_loads_or(row.get("mask_quality"), {}),
        "ignore_for_id": row.get("ignore_for_id", "false") == "true",
    }


def add_tar_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, BytesIO(data))


def make_soft_shape_mask(size: tuple[int, int], shape: str, blur_radius: float = 1.4) -> Image.Image:
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    inset_x = max(1, int(width * 0.03))
    inset_y = max(1, int(height * 0.03))
    box = [inset_x, inset_y, width - inset_x, height - inset_y]
    shape_norm = shape or ""
    aspect = width / max(1, height)
    if "원형" in shape_norm or 0.78 <= aspect <= 1.22:
        draw.ellipse(box, fill=255)
    elif "타원" in shape_norm or "장방" in shape_norm or "캡슐" in shape_norm or aspect > 1.55 or aspect < 0.65:
        radius = min(width, height) // 2
        draw.rounded_rectangle(box, radius=radius, fill=255)
    else:
        draw.rounded_rectangle(box, radius=max(8, min(width, height) // 5), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(blur_radius))


def alpha_bbox(alpha: Image.Image, threshold: int = 16) -> list[int] | None:
    arr = np.asarray(alpha)
    ys, xs = np.where(arr > threshold)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return [x1, y1, x2 - x1, y2 - y1]


def trim_transparent(image: Image.Image, padding: int = 8) -> Image.Image:
    bbox = alpha_bbox(image.getchannel("A"))
    if bbox is None:
        return image
    x, y, w, h = bbox
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(image.width, x + w + padding)
    bottom = min(image.height, y + h + padding)
    return image.crop((left, top, right, bottom))


def fallback_shape_cutout(crop: Image.Image, local_box: list[int], shape: str) -> tuple[Image.Image, dict]:
    mask = Image.new("L", crop.size, 0)
    shape_mask = make_soft_shape_mask((local_box[2] - local_box[0], local_box[3] - local_box[1]), shape)
    mask.paste(shape_mask, (local_box[0], local_box[1]))
    output = crop.convert("RGBA")
    output.putalpha(mask)
    return trim_transparent(output), {
        "method": "bbox_shape_soft_mask_fallback",
        "score": 0.50,
        "sam_used": False,
        "opencv_grabcut": False,
    }


def _expanded_box(local_box: list[int], width: int, height: int, ratio: float = 0.07) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(value)) for value in local_box]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad = max(2, int(round(max(box_w, box_h) * ratio)))
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


def _mask_box(binary: np.ndarray) -> list[int] | None:
    ys, xs = np.where(binary)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return [x1, y1, x2 - x1, y2 - y1]


def _xyxy_intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _keep_best_component(binary: np.ndarray, local_box: list[int]) -> np.ndarray:
    try:
        import cv2
    except Exception:
        return binary

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if labels_count <= 1:
        return binary

    box_tuple = tuple(int(round(value)) for value in local_box)
    box_area = max(1, (box_tuple[2] - box_tuple[0]) * (box_tuple[3] - box_tuple[1]))
    min_area = max(24, int(round(box_area * 0.08)))
    best_label = 0
    best_score = -1.0
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        component_box = (x, y, x + w, y + h)
        overlap = _xyxy_intersection_area(component_box, box_tuple)
        overlap_ratio = overlap / max(1, w * h)
        area_ratio = min(area / box_area, 1.5)
        score = overlap_ratio * 2.0 + area_ratio
        if score > best_score:
            best_score = score
            best_label = label

    if best_label <= 0:
        return np.zeros_like(binary, dtype=bool)
    return labels == best_label


def _fill_binary_holes(binary: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except Exception:
        return binary

    mask = binary.astype(np.uint8)
    height, width = mask.shape[:2]
    flood = mask.copy()
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = flood == 0
    return binary | holes


def _smooth_binary_alpha(binary: np.ndarray, local_box: list[int], dilate_iterations: int = 1) -> np.ndarray:
    try:
        import cv2
    except Exception:
        return (binary.astype(np.uint8) * 255)

    x1, y1, x2, y2 = [int(round(value)) for value in local_box]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    kernel_size = max(3, int(round(min(box_w, box_h) * 0.012)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    alpha = binary.astype(np.uint8) * 255
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=1)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=1)
    if dilate_iterations > 0:
        alpha = cv2.dilate(alpha, kernel, iterations=dilate_iterations)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=0.65)
    return alpha


def _alpha_quality(alpha: np.ndarray, local_box: list[int], method: str, score: float, extra: dict | None = None) -> tuple[bool, dict]:
    alpha_image = Image.fromarray(alpha.astype(np.uint8), mode="L")
    bbox = alpha_bbox(alpha_image, threshold=24)
    x1, y1, x2, y2 = [int(round(value)) for value in local_box]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    bbox_area = box_w * box_h
    fg_area = int((alpha > 24).sum())
    mask_bbox_area = bbox[2] * bbox[3] if bbox else 0
    fill_ratio = fg_area / mask_bbox_area if mask_bbox_area else 0.0
    area_ratio = fg_area / bbox_area
    width_ratio = bbox[2] / box_w if bbox else 0.0
    height_ratio = bbox[3] / box_h if bbox else 0.0
    touches_edge = bool(
        (alpha[0, :] > 24).any()
        or (alpha[-1, :] > 24).any()
        or (alpha[:, 0] > 24).any()
        or (alpha[:, -1] > 24).any()
    )
    quality = {
        "method": method,
        "score": round(float(score), 4),
        "foreground_area": fg_area,
        "bbox_area": bbox_area,
        "fill_ratio": round(fill_ratio, 4),
        "area_ratio": round(area_ratio, 4),
        "width_ratio": round(width_ratio, 4),
        "height_ratio": round(height_ratio, 4),
        "touches_crop_edge": touches_edge,
    }
    if extra:
        quality.update(extra)
    passed = not (
        bbox is None
        or fg_area < bbox_area * 0.20
        or fg_area > bbox_area * 1.28
        or fill_ratio > 0.93
        or width_ratio < 0.48
        or height_ratio < 0.48
        or width_ratio > 1.24
        or height_ratio > 1.24
        or touches_edge
    )
    if not passed:
        quality["failure"] = "quality_gate_failed"
    return passed, quality


class Sam2MaskProvider:
    """BBox-prompt SAM2 provider with strict geometry cleanup for pill cutouts."""

    def __init__(
        self,
        checkpoint: Path,
        config: str,
        device: str = "auto",
        multimask: bool = True,
        logit_threshold: float = 0.8,
        box_expansion_ratio: float = 0.035,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.config = config
        self.multimask = multimask
        self.logit_threshold = logit_threshold
        self.box_expansion_ratio = box_expansion_ratio
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.checkpoint}")
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = device
        self.model = build_sam2(config, str(self.checkpoint), device=device)
        self.predictor = SAM2ImagePredictor(self.model)

    def cutout(self, crop_rgb: Image.Image, local_box: list[int]) -> tuple[Image.Image | None, dict]:
        crop_arr = np.asarray(crop_rgb.convert("RGB"))
        height, width = crop_arr.shape[:2]
        x1, y1, x2, y2 = [float(value) for value in local_box]
        prompt_box = np.array([x1, y1, x2, y2], dtype=np.float32)
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        pad = max(4.0, min(box_w, box_h) * 0.08)
        point_coords = [[cx, cy]]
        point_labels = [1]
        for px, py in ((x1 - pad, cy), (x2 + pad, cy), (cx, y1 - pad), (cx, y2 + pad)):
            if 0.0 <= px < width and 0.0 <= py < height:
                point_coords.append([px, py])
                point_labels.append(0)
        try:
            with self.torch.inference_mode():
                self.predictor.set_image(crop_arr)
                masks, scores, _ = self.predictor.predict(
                    point_coords=np.asarray(point_coords, dtype=np.float32),
                    point_labels=np.asarray(point_labels, dtype=np.int32),
                    box=prompt_box,
                    multimask_output=self.multimask,
                    return_logits=True,
                )
        except Exception as exc:
            return None, {
                "method": "sam2_bbox",
                "score": 0.0,
                "sam_used": True,
                "sam2_device": self.device,
                "failure": f"sam2_predict_failed:{exc}",
            }

        if masks.ndim == 2:
            masks = masks[None, :, :]
        prior = np.zeros((height, width), dtype=bool)
        px1, py1, px2, py2 = _expanded_box(local_box, width, height, ratio=self.box_expansion_ratio)
        prior[py1:py2, px1:px2] = True

        best: tuple[float, np.ndarray, dict] | None = None
        failures = []
        for index, (mask, raw_score) in enumerate(zip(masks, scores)):
            binary_pre = np.asarray(mask > self.logit_threshold, dtype=bool)
            pre_area = int(binary_pre.sum())
            binary = _keep_best_component(binary_pre & prior, local_box)
            binary = _fill_binary_holes(binary)
            alpha = _smooth_binary_alpha(binary, local_box, dilate_iterations=0)
            alpha = np.where(prior, alpha, 0).astype(np.uint8)
            leakage_removed = (pre_area - int((binary_pre & prior).sum())) / max(1, pre_area)
            passed, quality = _alpha_quality(
                alpha,
                local_box,
                "sam2_bbox",
                float(raw_score),
                {
                    "sam_used": True,
                    "sam2_device": self.device,
                    "sam2_config": self.config,
                    "sam2_checkpoint": self.checkpoint.name,
                    "sam2_mask_index": index,
                    "sam2_logit_threshold": round(self.logit_threshold, 4),
                    "sam2_box_expansion_ratio": round(self.box_expansion_ratio, 4),
                    "sam2_prompt_points": len(point_coords),
                    "leakage_removed_ratio": round(leakage_removed, 4),
                },
            )
            if not passed:
                failures.append(quality)
                continue
            quality_score = float(raw_score)
            quality_score -= max(0.0, quality["fill_ratio"] - 0.86) * 0.8
            quality_score -= max(0.0, quality["area_ratio"] - 1.00) * 0.5
            quality_score -= max(0.0, 0.42 - quality["area_ratio"]) * 0.5
            if best is None or quality_score > best[0]:
                best = (quality_score, alpha, quality)

        if best is None:
            return None, {
                "method": "sam2_bbox",
                "score": 0.0,
                "sam_used": True,
                "sam2_device": self.device,
                "failure": "no_sam2_mask_passed_quality_gate",
                "sam2_failures": failures[:3],
            }

        _, alpha, quality = best
        output = crop_rgb.convert("RGBA")
        output.putalpha(Image.fromarray(alpha, mode="L"))
        return trim_transparent(output, padding=5), quality


def grabcut_alpha(crop_rgb: Image.Image, local_box: list[int], shape: str) -> tuple[Image.Image | None, dict]:
    try:
        import cv2
    except Exception:
        return None, {"opencv_grabcut": False, "failure": "cv2_unavailable"}

    original_width, original_height = crop_rgb.size
    max_side = max(original_width, original_height)
    scale = min(1.0, 240.0 / max(1, max_side))
    if scale < 1.0:
        work_size = (max(2, int(round(original_width * scale))), max(2, int(round(original_height * scale))))
        work_image = crop_rgb.resize(work_size, Image.Resampling.BILINEAR)
        work_box = [int(round(value * scale)) for value in local_box]
    else:
        work_image = crop_rgb
        work_box = list(local_box)
    crop_arr = np.asarray(work_image.convert("RGB"))
    height, width = crop_arr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in local_box]
    x1, y1, x2, y2 = [int(v) for v in work_box]
    x1 = max(1, min(width - 2, x1))
    y1 = max(1, min(height - 2, y1))
    x2 = max(x1 + 2, min(width - 1, x2))
    y2 = max(y1 + 2, min(height - 1, y2))
    box_w = x2 - x1
    box_h = y2 - y1

    mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
    margin_x = max(4, int(round(box_w * 0.08)))
    margin_y = max(4, int(round(box_h * 0.08)))
    rx1 = max(1, x1 - margin_x)
    ry1 = max(1, y1 - margin_y)
    rx2 = min(width - 1, x2 + margin_x)
    ry2 = min(height - 1, y2 + margin_y)
    mask[ry1:ry2, rx1:rx2] = cv2.GC_PR_FGD
    mask[:1, :] = cv2.GC_BGD
    mask[-1:, :] = cv2.GC_BGD
    mask[:, :1] = cv2.GC_BGD
    mask[:, -1:] = cv2.GC_BGD

    inner_x1 = x1 + int(round(box_w * 0.18))
    inner_y1 = y1 + int(round(box_h * 0.18))
    inner_x2 = x2 - int(round(box_w * 0.18))
    inner_y2 = y2 - int(round(box_h * 0.18))
    if inner_x2 > inner_x1 and inner_y2 > inner_y1:
        mask[inner_y1:inner_y2, inner_x1:inner_x2] = cv2.GC_FGD

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop_arr, mask, None, bg_model, fg_model, 2, cv2.GC_INIT_WITH_MASK)
    except Exception as exc:
        return None, {"opencv_grabcut": True, "failure": f"grabcut_failed:{exc}"}

    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel_size = max(3, int(round(min(box_w, box_h) * 0.025)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=1)
    alpha = cv2.dilate(alpha, kernel, iterations=1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.2)
    if scale < 1.0:
        alpha = cv2.resize(alpha, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

    bbox = alpha_bbox(Image.fromarray(alpha, mode="L"), threshold=24)
    original_box_w = max(1, int(round((local_box[2] - local_box[0]))))
    original_box_h = max(1, int(round((local_box[3] - local_box[1]))))
    bbox_area = original_box_w * original_box_h
    fg_area = int((alpha > 24).sum())
    mask_bbox_area = bbox[2] * bbox[3] if bbox else 0
    fill_ratio = fg_area / mask_bbox_area if mask_bbox_area else 0.0
    touches_edge = bool(
        (alpha[0, :] > 24).any()
        or (alpha[-1, :] > 24).any()
        or (alpha[:, 0] > 24).any()
        or (alpha[:, -1] > 24).any()
    )
    if (
        bbox is None
        or fg_area < bbox_area * 0.18
        or fg_area > bbox_area * 1.65
        or fill_ratio > 0.92
        or touches_edge
    ):
        return None, {
            "opencv_grabcut": True,
            "failure": "quality_gate_failed",
            "foreground_area": fg_area,
            "bbox_area": bbox_area,
            "fill_ratio": round(fill_ratio, 4),
            "touches_crop_edge": touches_edge,
        }

    output = crop_rgb.convert("RGBA")
    output.putalpha(Image.fromarray(alpha, mode="L"))
    output = trim_transparent(output)
    quality = {
        "method": "opencv_grabcut_bbox",
        "score": 0.82,
        "sam_used": False,
        "opencv_grabcut": True,
        "foreground_area": fg_area,
        "bbox_area": bbox_area,
        "fill_ratio": round(fill_ratio, 4),
        "touches_crop_edge": False,
        "work_scale": round(scale, 4),
    }
    return output, quality


def crop_cutout(
    image: Image.Image,
    bbox: list[float],
    shape: str,
    margin_ratio: float,
    mask_provider: Sam2MaskProvider | None = None,
) -> tuple[Image.Image, dict]:
    width, height = image.size
    x, y, box_w, box_h = bbox
    margin = int(round(max(box_w, box_h) * margin_ratio))
    left = max(0, int(math.floor(x - margin)))
    top = max(0, int(math.floor(y - margin)))
    right = min(width, int(math.ceil(x + box_w + margin)))
    bottom = min(height, int(math.ceil(y + box_h + margin)))
    crop = image.crop((left, top, right, bottom)).convert("RGBA")
    mask = Image.new("L", crop.size, 0)
    local_box = [
        int(round(x - left)),
        int(round(y - top)),
        int(round(x + box_w - left)),
        int(round(y + box_h - top)),
    ]
    if mask_provider is not None:
        sam2_cutout, sam2_quality = mask_provider.cutout(crop.convert("RGB"), local_box)
        if sam2_cutout is not None:
            sam2_quality["margin_ratio"] = round(margin_ratio, 4)
            return sam2_cutout, sam2_quality

    grabcut_cutout, quality = grabcut_alpha(crop.convert("RGB"), local_box, shape)
    if grabcut_cutout is not None:
        quality["margin_ratio"] = round(margin_ratio, 4)
        return grabcut_cutout, quality
    fallback, fallback_quality = fallback_shape_cutout(crop, local_box, shape)
    fallback_quality.update({key: value for key, value in quality.items() if key not in fallback_quality})
    fallback_quality["margin_ratio"] = round(margin_ratio, 4)
    return fallback, fallback_quality


def synthetic_background(width: int, height: int, rng: random.Random, profile_image: Image.Image | None = None) -> Image.Image:
    if profile_image is not None:
        arr = np.asarray(profile_image.resize((160, 210)).convert("RGB"), dtype=np.float32)
        flat = arr.reshape(-1, 3)
        mean = np.percentile(flat, 50, axis=0)
        std = np.clip(np.percentile(np.abs(flat - mean), 65, axis=0), 4, 28)
    else:
        palettes = np.array(
            [
                [188, 196, 212],
                [208, 199, 178],
                [72, 69, 86],
                [116, 87, 78],
                [186, 176, 136],
                [222, 222, 214],
            ],
            dtype=np.float32,
        )
        mean = palettes[rng.randrange(len(palettes))]
        std = np.array([9, 9, 9], dtype=np.float32)
    small = rng.normalvariate
    noise = np.zeros((max(16, height // 16), max(16, width // 16), 3), dtype=np.float32)
    for c in range(3):
        noise[:, :, c] = mean[c] + np.random.default_rng(rng.randrange(2**32)).normal(0, std[c], noise.shape[:2])
    small_image = Image.fromarray(np.uint8(np.clip(noise, 0, 255)), "RGB").filter(
        ImageFilter.GaussianBlur(radius=1.1)
    )
    image = small_image.resize((width, height), Image.Resampling.BICUBIC)
    return Image.blend(image, ImageEnhance.Brightness(image).enhance(0.94), 0.18)


def paste_with_shadow(canvas: Image.Image, cutout: Image.Image, xy: tuple[int, int], rng: random.Random) -> None:
    alpha = cutout.getchannel("A")
    shadow = Image.new("RGBA", cutout.size, (0, 0, 0, 0))
    shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=rng.uniform(5.0, 10.0)))
    shadow.putalpha(shadow_alpha.point(lambda v: int(v * rng.uniform(0.20, 0.34))))
    sx = xy[0] + rng.randint(3, 8)
    sy = xy[1] + rng.randint(4, 10)
    canvas.alpha_composite(shadow, (sx, sy))
    canvas.alpha_composite(cutout, xy)


def apply_common_photometric(image: Image.Image, rng: random.Random) -> Image.Image:
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.94, 1.08))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.94, 1.10))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.95, 1.06))
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(0.8, 2.2), arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")
