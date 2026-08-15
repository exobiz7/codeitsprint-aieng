#!/usr/bin/env python3
"""Reconstruct the Kaggle v2 spec folder when the handoff zip is unavailable."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


DEFAULT_DATA_ROOT = Path("/Volumes/SSD 4T/01_sprint_ai_project1_data")
DEFAULT_V1_COCO = DEFAULT_DATA_ROOT / "processed" / "kaggle_sam2_synth_v1" / "runs" / "kaggle_30k" / "annotations_coco.json"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "processed" / "kaggle_sam2_synth_v2"
TARGET_RGB = np.array([112.0, 130.0, 154.0], dtype=np.float32)
CLASS_FIELDS = [
    "class_index",
    "category_id",
    "K_code",
    "product_name",
    "ingredient",
    "di_class_no",
    "otc",
    "shape",
    "color",
    "imprint_front",
    "item_seq",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--v1-coco", type=Path, default=DEFAULT_V1_COCO)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--background-count", type=int, default=12)
    parser.add_argument("--patch-size", type=int, default=384)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def read_product_metadata(data_root: Path) -> tuple[dict[str, dict], dict[str, list[list[float]]]]:
    metadata: dict[str, dict] = {}
    bboxes_by_file: dict[str, list[list[float]]] = collections.defaultdict(list)
    for json_path in sorted((data_root / "train_annotations").rglob("*.json")):
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        image_meta = doc["images"][0]
        ann = doc["annotations"][0]
        product_id = image_meta.get("drug_N") or image_meta.get("dl_mapping_code")
        if product_id and product_id not in metadata:
            metadata[product_id] = image_meta
        bboxes_by_file[image_meta["file_name"]].append([float(v) for v in ann["bbox"]])
    return metadata, bboxes_by_file


def write_class_map(spec_dir: Path, v1_coco: Path, metadata: dict[str, dict]) -> list[dict]:
    coco = json.loads(v1_coco.read_text(encoding="utf-8"))
    categories = coco.get("categories", [])
    if len(categories) != 56:
        raise RuntimeError(f"Expected 56 categories in {v1_coco}, got {len(categories)}")
    rows = []
    for class_index, category in enumerate(categories):
        product_id = category["product_id"]
        meta = metadata.get(product_id, {})
        color = meta.get("color_class1", "")
        if meta.get("color_class2"):
            color = f"{color}|{meta['color_class2']}" if color else meta["color_class2"]
        rows.append(
            {
                "class_index": class_index,
                "category_id": int(category["id"]),
                "K_code": product_id,
                "product_name": category.get("name") or meta.get("dl_name", ""),
                "ingredient": meta.get("dl_material", ""),
                "di_class_no": meta.get("di_class_no", ""),
                "otc": meta.get("di_etc_otc_code", ""),
                "shape": meta.get("drug_shape", ""),
                "color": color,
                "imprint_front": meta.get("print_front", ""),
                "item_seq": meta.get("item_seq", ""),
            }
        )
    with (spec_dir / "class_map_56.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLASS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (spec_dir / "class_map_56.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return rows


def write_domain_profile(spec_dir: Path) -> None:
    profile = {
        "resolution_WxH": [976, 1280],
        "background_rgb_target_RGB": [112, 130, 154],
        "background_note": "Reconstructed blue-gray profile; backgrounds/*.png are sampled from real train background regions with pill bboxes masked out.",
        "bbox_rel_area_ratio_pct": {
            "0.05": 0.027,
            "0.25": 0.029,
            "0.5": 0.043,
            "0.75": 0.075,
            "0.95": 0.124,
        },
        "bbox_side_px_pct": {
            "0.05": 182.99,
            "0.25": 191.499,
            "0.5": 231.517,
            "0.75": 305.765,
            "0.95": 394.235,
        },
        "objects_per_image": {"2": 7, "3": 151, "4": 74},
        "camera_la_seen": [70, 75, 90],
        "camera_lo_seen": [0],
        "orientation": "앞면 위주",
        "lighting": "주백색(neutral)",
    }
    (spec_dir / "domain_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def color_transfer_patch(patch: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(patch.convert("RGB"), dtype=np.float32).copy()
    target = TARGET_RGB + np.asarray([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-3, 3)], dtype=np.float32)
    arr += target - arr.reshape(-1, 3).mean(axis=0)
    arr = np.clip(arr, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB").filter(ImageFilter.GaussianBlur(radius=0.25))


def crop_background_patch(
    image: Image.Image,
    bboxes: list[list[float]],
    patch_size: int,
    rng: random.Random,
) -> Image.Image | None:
    width, height = image.size
    blocked = np.zeros((height, width), dtype=bool)
    for x, y, w, h in bboxes:
        pad = 52
        x1 = max(0, int(round(x - pad)))
        y1 = max(0, int(round(y - pad)))
        x2 = min(width, int(round(x + w + pad)))
        y2 = min(height, int(round(y + h + pad)))
        blocked[y1:y2, x1:x2] = True
    if width < patch_size or height < patch_size:
        return None
    for _ in range(250):
        x = rng.randint(0, width - patch_size)
        y = rng.randint(0, height - patch_size)
        patch_blocked = blocked[y : y + patch_size, x : x + patch_size].mean()
        if patch_blocked <= 0.005:
            return image.crop((x, y, x + patch_size, y + patch_size))
    return None


def write_backgrounds(
    spec_dir: Path,
    data_root: Path,
    bboxes_by_file: dict[str, list[list[float]]],
    count: int,
    patch_size: int,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    background_dir = spec_dir / "backgrounds"
    background_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted((data_root / "train_images").glob("*.png"))
    rng.shuffle(image_files)
    outputs = []
    for image_path in image_files:
        if len(outputs) >= count:
            break
        bboxes = bboxes_by_file.get(image_path.name, [])
        if not bboxes:
            continue
        with Image.open(image_path) as image:
            patch = crop_background_patch(image.convert("RGB"), bboxes, patch_size, rng)
        if patch is None:
            continue
        patch = color_transfer_patch(patch, rng)
        output_path = background_dir / f"background_{len(outputs):02d}.png"
        patch.save(output_path, format="PNG", optimize=True)
        outputs.append(str(output_path))
    if len(outputs) < count:
        raise RuntimeError(f"Could only reconstruct {len(outputs)}/{count} background patches")
    return outputs


def write_notes(spec_dir: Path, rows: list[dict], background_paths: list[str], args) -> None:
    refs_dir = spec_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "note": "The original codex_handoff.zip was unavailable in the pasteboard cache, so this spec was reconstructed from local Kaggle v1 category metadata and Kaggle train background-only patches. Cutout sources for v2 generation still remain AI Hub single images only.",
        "v1_coco": str(args.v1_coco),
        "data_root": str(args.data_root),
        "class_rows": len(rows),
        "backgrounds": background_paths,
        "target_rgb": TARGET_RGB.astype(int).tolist(),
    }
    (refs_dir / "reconstruction_sources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (spec_dir / "SPEC.md").write_text(
        "# Kaggle SAM2 Synth v2 Spec\n\n"
        "- Classes: 56 K-code products, reconstructed in the existing Kaggle category order.\n"
        "- Synthetic cutouts: AI Hub single-pill sources only; no Kaggle train combo cutouts.\n"
        "- Backgrounds: blue-gray patches sampled from real train background regions with labeled pill boxes excluded.\n"
        "- Output: COCO images + annotations with category_id, class_index, and product_id.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = build_parser().parse_args()
    spec_dir = args.output_root / "spec" / "codex_handoff"
    if spec_dir.exists() and not args.overwrite:
        print(f"spec exists: {spec_dir}")
        return 0
    spec_dir.mkdir(parents=True, exist_ok=True)
    metadata, bboxes_by_file = read_product_metadata(args.data_root)
    rows = write_class_map(spec_dir, args.v1_coco, metadata)
    write_domain_profile(spec_dir)
    background_paths = write_backgrounds(
        spec_dir,
        args.data_root,
        bboxes_by_file,
        args.background_count,
        args.patch_size,
        args.seed,
    )
    write_notes(spec_dir, rows, background_paths, args)
    print(json.dumps({"spec_dir": str(spec_dir), "classes": len(rows), "backgrounds": len(background_paths)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
