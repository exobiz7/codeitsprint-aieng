#!/usr/bin/env python3
"""Convert paired AI Hub PNG images to JPEG q95 4:4:4 and derive matching label zips."""

from __future__ import annotations

import argparse
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from aihub_common import (
    DEFAULT_OUTPUT_TRAINING_ROOT,
    DEFAULT_TRAINING_ROOT,
    add_path_args,
    add_zip_args,
    existing_manifest_set_ids,
    image_zip_name,
    json_dumps_bytes,
    label_zip_name,
    load_json_bytes,
    manifest_value_to_member_list,
    manifest_dir_for,
    manifest_path_for,
    read_manifest_rows,
    require_set_selection,
    resolve_dataset_paths,
    update_label_image_filenames,
    zip_path_for,
)


def import_pillow():
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required. Install it with: python -m pip install -r requirements.txt") from exc
    return Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_args(parser)
    add_zip_args(parser)
    parser.set_defaults(training_root=DEFAULT_TRAINING_ROOT, output_training_root=DEFAULT_OUTPUT_TRAINING_ROOT)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--set-ids", type=int, nargs="+")
    parser.add_argument("--all", action="store_true", help="Process every manifest set.")
    parser.add_argument("--limit", type=int, help="Process at most N rows per set for pilot runs.")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=1, help="Parallel JPEG encoder workers per zip.")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-partial", action="store_true")
    return parser


def jpeg_bytes(image_data: bytes, Image, quality: int) -> tuple[bytes, tuple[int, int]]:
    with Image.open(BytesIO(image_data)) as image:
        image.load()
        if image.mode != "RGB":
            image = image.convert("RGB")
        size = image.size
        output = BytesIO()
        image.save(
            output,
            format="JPEG",
            quality=quality,
            subsampling=0,
            optimize=True,
            progressive=False,
        )
    return output.getvalue(), size


def jpeg_bytes_worker(image_data: bytes, quality: int) -> tuple[bytes, tuple[int, int]]:
    from PIL import Image

    return jpeg_bytes(image_data, Image, quality)


def write_converted_pair(row, encoded_image, size, label_zip, label_zip_path, output_image_zip, output_label_zip) -> None:
    if row.get("width") and row.get("height"):
        expected_size = (int(row["width"]), int(row["height"]))
        if size != expected_size:
            raise ValueError(f"Image size mismatch for {row['image_member']}: {size} != {expected_size}")
    output_image_zip.writestr(row["output_image_member"], encoded_image, compress_type=ZIP_STORED)

    for label_member in manifest_value_to_member_list(row.get("label_members"), row.get("label_member")):
        label_doc = load_json_bytes(label_zip.read(label_member), f"{label_zip_path.name}:{label_member}")
        update_label_image_filenames(label_doc, ".jpg")
        output_label_zip.writestr(label_member, json_dumps_bytes(label_doc), compress_type=ZIP_DEFLATED)


def maybe_print_progress(set_id: int, converted: int, total: int, progress_every: int) -> None:
    if progress_every > 0 and converted % progress_every == 0:
        print(f"converted set {set_id}: {converted}/{total}", flush=True)


def compress_set(args, paths, set_id: int, Image) -> dict:
    manifest_dir = args.manifest_dir or manifest_dir_for(args.reports_dir)
    manifest_path = manifest_path_for(manifest_dir, set_id, args.image_prefix)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}. Run inventory_dataset.py first.")

    rows = list(read_manifest_rows(manifest_path, limit=args.limit))
    if not rows:
        raise ValueError(f"No manifest rows for {args.image_prefix}_{set_id}")

    source_zip_path = zip_path_for(paths.source_dir, args.image_prefix, set_id, args.zip_kind)
    label_zip_path = zip_path_for(paths.label_dir, args.label_prefix, set_id, args.zip_kind)
    output_image_zip_path = paths.output_source_dir / image_zip_name(set_id, args.zip_kind, args.image_prefix)
    output_label_zip_path = paths.output_label_dir / label_zip_name(set_id, args.zip_kind, args.label_prefix)
    output_image_zip_path.parent.mkdir(parents=True, exist_ok=True)
    output_label_zip_path.parent.mkdir(parents=True, exist_ok=True)

    if (output_image_zip_path.exists() or output_label_zip_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Output exists for {args.image_prefix}_{set_id}. Pass --overwrite to replace: "
            f"{output_image_zip_path} / {output_label_zip_path}"
        )

    tmp_image_zip_path = output_image_zip_path.with_suffix(output_image_zip_path.suffix + ".tmp")
    tmp_label_zip_path = output_label_zip_path.with_suffix(output_label_zip_path.suffix + ".tmp")
    for tmp_path in (tmp_image_zip_path, tmp_label_zip_path):
        if tmp_path.exists():
            tmp_path.unlink()

    converted = 0
    try:
        with (
            ZipFile(source_zip_path) as source_zip,
            ZipFile(label_zip_path) as label_zip,
            ZipFile(tmp_image_zip_path, "w", compression=ZIP_STORED, allowZip64=True) as output_image_zip,
            ZipFile(tmp_label_zip_path, "w", compression=ZIP_DEFLATED, allowZip64=True) as output_label_zip,
        ):
            if args.workers <= 1:
                for row in rows:
                    image_data = source_zip.read(row["image_member"])
                    encoded_image, size = jpeg_bytes(image_data, Image, args.quality)
                    write_converted_pair(
                        row, encoded_image, size, label_zip, label_zip_path, output_image_zip, output_label_zip
                    )
                    converted += 1
                    maybe_print_progress(set_id, converted, len(rows), args.progress_every)
            else:
                pending = deque()
                row_iter = iter(rows)

                def submit_next(executor: ProcessPoolExecutor) -> bool:
                    try:
                        next_row = next(row_iter)
                    except StopIteration:
                        return False
                    image_data = source_zip.read(next_row["image_member"])
                    pending.append((next_row, executor.submit(jpeg_bytes_worker, image_data, args.quality)))
                    return True

                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    for _ in range(max(1, args.workers * 2)):
                        if not submit_next(executor):
                            break
                    while pending:
                        row, future = pending.popleft()
                        encoded_image, size = future.result()
                        write_converted_pair(
                            row, encoded_image, size, label_zip, label_zip_path, output_image_zip, output_label_zip
                        )
                        converted += 1
                        maybe_print_progress(set_id, converted, len(rows), args.progress_every)
                        submit_next(executor)

        os.replace(tmp_image_zip_path, output_image_zip_path)
        os.replace(tmp_label_zip_path, output_label_zip_path)
    except Exception:
        if not args.keep_partial:
            for tmp_path in (tmp_image_zip_path, tmp_label_zip_path):
                if tmp_path.exists():
                    tmp_path.unlink()
        raise

    return {
        "set_id": set_id,
        "converted": converted,
        "output_image_zip": str(output_image_zip_path),
        "output_label_zip": str(output_label_zip_path),
        "output_image_size_bytes": output_image_zip_path.stat().st_size,
        "output_label_size_bytes": output_label_zip_path.stat().st_size,
    }


def main() -> int:
    args = build_parser().parse_args()
    Image = import_pillow()
    paths = resolve_dataset_paths(args)
    manifest_dir = args.manifest_dir or manifest_dir_for(args.reports_dir)
    selected_set_ids = require_set_selection(args.set_ids, args.all)
    if selected_set_ids is None:
        selected_set_ids = existing_manifest_set_ids(manifest_dir, args.image_prefix)
    if not selected_set_ids:
        raise SystemExit(f"No manifest sets found under {manifest_dir}")

    for set_id in selected_set_ids:
        result = compress_set(args, paths, set_id, Image)
        print(
            f"{args.image_prefix}_{set_id}: converted={result['converted']} "
            f"image_zip={result['output_image_size_bytes']} bytes "
            f"label_zip={result['output_label_size_bytes']} bytes",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
