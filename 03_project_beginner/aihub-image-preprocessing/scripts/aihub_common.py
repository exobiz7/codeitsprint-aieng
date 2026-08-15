#!/usr/bin/env python3
"""Shared helpers for the AI Hub pill image compression pipeline."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator


DEFAULT_TRAINING_ROOT = Path(
    "/Volumes/SSD 4T/166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터/01.데이터/1.Training"
)
DEFAULT_OUTPUT_TRAINING_ROOT = Path(
    "/Volumes/SSD 4T/166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터/01.데이터/1.Training_jpeg_q95"
)
SOURCE_RELATIVE_DIR = Path("원천데이터") / "단일경구약제 5000종"
LABEL_RELATIVE_DIR = Path("라벨링데이터") / "단일경구약제 5000종"

IMAGE_PREFIX = "TS"
LABEL_PREFIX = "TL"
ZIP_KIND = "단일"

MANIFEST_FIELDS = [
    "set_id",
    "stem",
    "image_member",
    "label_member",
    "output_image_member",
    "output_label_member",
    "label_members",
    "width",
    "height",
    "bbox",
]

UNMATCHED_FIELDS = ["set_id", "side", "stem", "member"]


@dataclass(frozen=True)
class DatasetPaths:
    source_dir: Path
    label_dir: Path
    output_source_dir: Path
    output_label_dir: Path


def resolve_dataset_paths(args) -> DatasetPaths:
    training_root = Path(args.training_root).expanduser()
    output_training_root = Path(args.output_training_root).expanduser()
    source_dir = Path(args.source_dir).expanduser() if args.source_dir else training_root / SOURCE_RELATIVE_DIR
    label_dir = Path(args.label_dir).expanduser() if args.label_dir else training_root / LABEL_RELATIVE_DIR
    output_source_dir = (
        Path(args.output_source_dir).expanduser()
        if getattr(args, "output_source_dir", None)
        else output_training_root / SOURCE_RELATIVE_DIR
    )
    output_label_dir = (
        Path(args.output_label_dir).expanduser()
        if getattr(args, "output_label_dir", None)
        else output_training_root / LABEL_RELATIVE_DIR
    )
    return DatasetPaths(
        source_dir=source_dir,
        label_dir=label_dir,
        output_source_dir=output_source_dir,
        output_label_dir=output_label_dir,
    )


def add_path_args(parser) -> None:
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--label-dir", type=Path)
    parser.add_argument("--output-training-root", type=Path, default=DEFAULT_OUTPUT_TRAINING_ROOT)
    parser.add_argument("--output-source-dir", type=Path)
    parser.add_argument("--output-label-dir", type=Path)


def add_zip_args(parser) -> None:
    parser.add_argument("--image-prefix", default=IMAGE_PREFIX)
    parser.add_argument("--label-prefix", default=LABEL_PREFIX)
    parser.add_argument("--zip-kind", default=ZIP_KIND)


def parse_set_id(path_or_name: str | Path, prefix: str, zip_kind: str = ZIP_KIND) -> int | None:
    name = unicodedata.normalize("NFC", Path(path_or_name).name)
    match = re.match(rf"^{re.escape(prefix)}_(\d+)_{re.escape(zip_kind)}\.zip$", name)
    return int(match.group(1)) if match else None


def zip_map(directory: Path, prefix: str, zip_kind: str = ZIP_KIND) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted(directory.glob(f"{prefix}_*.zip"), key=lambda item: unicodedata.normalize("NFC", item.name)):
        set_id = parse_set_id(path, prefix, zip_kind)
        if set_id is not None:
            if set_id in result and result[set_id] != path:
                raise ValueError(f"Duplicate {prefix}_{set_id} zip files: {result[set_id]} / {path}")
            result[set_id] = path
    return dict(sorted(result.items()))


def dataset_zip_name(prefix: str, set_id: int, zip_kind: str = ZIP_KIND) -> str:
    return f"{prefix}_{set_id}_{zip_kind}.zip"


def image_zip_name(set_id: int, zip_kind: str = ZIP_KIND, image_prefix: str = IMAGE_PREFIX) -> str:
    return dataset_zip_name(image_prefix, set_id, zip_kind)


def label_zip_name(set_id: int, zip_kind: str = ZIP_KIND, label_prefix: str = LABEL_PREFIX) -> str:
    return dataset_zip_name(label_prefix, set_id, zip_kind)


def zip_path_for(directory: Path, prefix: str, set_id: int, zip_kind: str = ZIP_KIND) -> Path:
    return zip_map(directory, prefix, zip_kind).get(set_id, directory / dataset_zip_name(prefix, set_id, zip_kind))


def iter_members(names: Iterable[str], suffix: str) -> Iterator[str]:
    lowered_suffix = suffix.lower()
    for name in names:
        if name.lower().endswith(lowered_suffix) and not name.endswith("/"):
            yield name


def member_stem(member: str) -> str:
    return PurePosixPath(member).stem


def replace_member_suffix(member: str, suffix: str) -> str:
    posix = PurePosixPath(member)
    return str(posix.with_suffix(suffix))


def change_filename_suffix(value: object, suffix: str) -> object:
    if not isinstance(value, str):
        return value
    path = PurePosixPath(value)
    if path.suffix.lower() != ".png":
        return value
    return str(path.with_suffix(suffix))


def update_label_image_filenames(label_doc: dict, suffix: str = ".jpg") -> dict:
    for image in label_doc.get("images", []):
        if isinstance(image, dict):
            for key in ("file_name", "imgfile"):
                if key in image:
                    image[key] = change_filename_suffix(image[key], suffix)
    return label_doc


def manifest_dir_for(reports_dir: Path) -> Path:
    return reports_dir / "manifests"


def manifest_path_for(manifest_dir: Path, set_id: int, image_prefix: str = IMAGE_PREFIX) -> Path:
    return manifest_dir / f"{image_prefix}_{set_id}_paired.tsv"


def write_tsv(path: Path, fields: list[str], rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def read_manifest_rows(path: Path, limit: int | None = None) -> Iterator[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            yield row


def member_list_to_manifest_value(members: list[str]) -> str:
    return json.dumps(members, ensure_ascii=False, separators=(",", ":"))


def manifest_value_to_member_list(value: str | None, fallback: str | None = None) -> list[str]:
    if value:
        members = json.loads(value)
        if not isinstance(members, list) or not all(isinstance(member, str) for member in members):
            raise ValueError(f"Invalid label_members manifest value: {value}")
        return members
    return [fallback] if fallback else []


def json_dumps_bytes(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def load_json_bytes(data: bytes, source: str) -> dict:
    try:
        value = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object in {source}")
    return value


def bbox_to_manifest_value(bbox: object) -> str:
    return json.dumps(bbox, ensure_ascii=True, separators=(",", ":"))


def parse_bbox_value(value: str) -> object:
    if value == "":
        return None
    return json.loads(value)


def extract_primary_image(label_doc: dict, source: str) -> dict:
    images = label_doc.get("images")
    if not isinstance(images, list) or not images or not isinstance(images[0], dict):
        raise ValueError(f"Missing images[0] in {source}")
    return images[0]


def extract_annotations(label_doc: dict, source: str) -> list[dict]:
    annotations = label_doc.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        raise ValueError(f"Missing annotations in {source}")
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError(f"Annotation is not an object in {source}")
    return annotations


def validate_bbox(label_doc: dict, source: str) -> tuple[int, int, object]:
    image = extract_primary_image(label_doc, source)
    width = image.get("width")
    height = image.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions in {source}: {width}x{height}")
    annotations = extract_annotations(label_doc, source)
    primary_bbox = None
    for annotation in annotations:
        bbox = annotation.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(item, (int, float)) for item in bbox)
        ):
            raise ValueError(f"Invalid bbox in {source}: {bbox}")
        x, y, box_width, box_height = bbox
        if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
            raise ValueError(f"Non-positive bbox in {source}: {bbox}")
        if x + box_width > width or y + box_height > height:
            raise ValueError(f"Bbox outside image bounds in {source}: {bbox} vs {width}x{height}")
        if primary_bbox is None:
            primary_bbox = bbox
    return width, height, primary_bbox


def require_set_selection(set_ids: list[int] | None, all_sets: bool) -> list[int] | None:
    if all_sets:
        return None
    if not set_ids:
        raise SystemExit("Pass --set-ids ... for a pilot/single set run, or --all for every manifest set.")
    return sorted(set(set_ids))


def existing_manifest_set_ids(manifest_dir: Path, image_prefix: str = IMAGE_PREFIX) -> list[int]:
    ids: list[int] = []
    for path in manifest_dir.glob(f"{image_prefix}_*_paired.tsv"):
        match = re.match(rf"^{re.escape(image_prefix)}_(\d+)_paired\.tsv$", path.name)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)
