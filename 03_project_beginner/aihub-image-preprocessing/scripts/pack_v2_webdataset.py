#!/usr/bin/env python3
"""Pack real v2 splits into WebDataset-compatible tar shards."""

from __future__ import annotations

import argparse
import collections
import json
import tarfile
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from v2_common import (
    DEFAULT_PROCESSED_ROOT,
    add_tar_bytes,
    canonical_json,
    read_csv_rows,
    sanitize_tar_key,
)


REAL_SPLITS = [
    "train_seen",
    "val_seen_id",
    "val_official_single_ood",
    "val_official_combo_real",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--splits", nargs="+", default=REAL_SPLITS)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


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


def open_shard(output_dir: Path, split: str, shard_index: int, overwrite: bool) -> tuple[Path, tarfile.TarFile]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{split}-{shard_index:06d}.tar"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Shard exists: {path}. Pass --overwrite to replace.")
    return path, tarfile.open(path, "w")


def validate_image_bytes(image_data: bytes, expected_width: int, expected_height: int) -> None:
    from io import BytesIO

    with Image.open(BytesIO(image_data)) as image:
        image.load()
        if image.size != (expected_width, expected_height):
            raise ValueError(f"Image size {image.size} != {(expected_width, expected_height)}")


def pack_split(rows, split: str, output_dir: Path, shard_size: int, overwrite: bool) -> dict:
    zip_cache = ZipCache()
    shard_index = 0
    count_in_shard = 0
    total = 0
    shard_paths = []
    shard_path, tar = open_shard(output_dir, split, shard_index, overwrite)
    shard_paths.append(str(shard_path))
    try:
        for row in rows:
            image_zip = zip_cache.get(row["source_zip"])
            image_data = image_zip.read(row["image_member"])
            validate_image_bytes(image_data, int(row["width"]), int(row["height"]))
            key = sanitize_tar_key(row["sample_id"])
            add_tar_bytes(tar, f"{key}.jpg", image_data)
            add_tar_bytes(
                tar,
                f"{key}.json",
                json.dumps(canonical_json(row), ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            )
            count_in_shard += 1
            total += 1
            if count_in_shard >= shard_size:
                tar.close()
                shard_index += 1
                count_in_shard = 0
                shard_path, tar = open_shard(output_dir, split, shard_index, overwrite)
                shard_paths.append(str(shard_path))
        tar.close()
        tar = None
        if count_in_shard == 0 and total > 0:
            empty_path = Path(shard_paths.pop())
            empty_path.unlink(missing_ok=True)
    finally:
        if tar is not None:
            tar.close()
        zip_cache.close()
    return {"split": split, "samples": total, "shards": len(shard_paths), "paths": shard_paths}


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = args.processed_root / "manifests" / "split_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing split manifest: {manifest_path}. Run build_v2_splits.py first.")
    rows_by_split: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in read_csv_rows(manifest_path):
        split = row["split"]
        if split in args.splits and (args.limit_per_split is None or len(rows_by_split[split]) < args.limit_per_split):
            rows_by_split[split].append(row)
    summaries = []
    for split in args.splits:
        rows = rows_by_split.get(split, [])
        if not rows:
            continue
        output_dir = args.processed_root / "webdataset" / split
        summaries.append(pack_split(rows, split, output_dir, args.shard_size, args.overwrite))
        print(f"{split}: packed {summaries[-1]['samples']} samples into {summaries[-1]['shards']} shards")
    summary_path = args.processed_root / "webdataset" / "real_shards_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
