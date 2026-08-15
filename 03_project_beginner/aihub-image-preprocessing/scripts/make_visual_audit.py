#!/usr/bin/env python3
"""Create original-vs-JPEG bbox overlay contact sheets for visual QA."""

from __future__ import annotations

import argparse
import json
import random
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from aihub_common import (
    DEFAULT_OUTPUT_TRAINING_ROOT,
    DEFAULT_TRAINING_ROOT,
    add_path_args,
    add_zip_args,
    extract_annotations,
    iter_members,
    label_zip_name,
    load_json_bytes,
    manifest_value_to_member_list,
    manifest_dir_for,
    manifest_path_for,
    member_stem,
    read_manifest_rows,
    resolve_dataset_paths,
    zip_path_for,
)


def import_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise SystemExit("Pillow is required. Install it with: python -m pip install -r requirements.txt") from exc
    return Image, ImageDraw, ImageFont


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_args(parser)
    add_zip_args(parser)
    parser.set_defaults(training_root=DEFAULT_TRAINING_ROOT, output_training_root=DEFAULT_OUTPUT_TRAINING_ROOT)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--set-ids", type=int, nargs="+", default=[7, 16, 24, 43, 67, 81])
    parser.add_argument("--samples-per-set", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--thumb-width", type=int, default=360)
    return parser


def read_image(zip_file: ZipFile, member: str, Image):
    with Image.open(BytesIO(zip_file.read(member))) as image:
        image.load()
        return image.convert("RGB")


def thumbnail_with_bboxes(image, bboxes, thumb_width: int, ImageDraw):
    scale = thumb_width / image.width
    thumb_height = int(round(image.height * scale))
    thumb = image.resize((thumb_width, thumb_height))
    draw = ImageDraw.Draw(thumb)
    palette = [(255, 20, 20), (20, 120, 255), (20, 180, 80), (255, 160, 20), (180, 40, 220)]
    for index, bbox in enumerate(bboxes):
        x, y, width, height = bbox
        box = [x * scale, y * scale, (x + width) * scale, (y + height) * scale]
        for offset in range(3):
            draw.rectangle(
                [box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset],
                outline=palette[index % len(palette)],
            )
    return thumb


def choose_rows(rows: list[dict], count: int, seed: int, set_id: int) -> list[dict]:
    if len(rows) <= count:
        return rows
    rng = random.Random(seed + set_id)
    return [rows[index] for index in sorted(rng.sample(range(len(rows)), count))]


def available_jpeg_stems(zip_file: ZipFile) -> set[str]:
    return {member_stem(member) for member in iter_members(zip_file.namelist(), ".jpg")}


def draw_title(draw, xy, text: str, font, fill=(20, 20, 20)) -> None:
    draw.text(xy, text, fill=fill, font=font)


def audit_set(args, paths, set_id: int, Image, ImageDraw, ImageFont) -> Path | None:
    manifest_dir = args.manifest_dir or manifest_dir_for(args.reports_dir)
    manifest_path = manifest_path_for(manifest_dir, set_id, args.image_prefix)
    if not manifest_path.exists():
        print(f"{args.image_prefix}_{set_id}: skip, missing manifest {manifest_path}")
        return None

    output_image_zip_path = zip_path_for(paths.output_source_dir, args.image_prefix, set_id, args.zip_kind)
    if not output_image_zip_path.exists():
        print(f"{args.image_prefix}_{set_id}: skip, missing compressed image zip {output_image_zip_path}")
        return None

    with ZipFile(output_image_zip_path) as output_image_zip:
        converted_stems = available_jpeg_stems(output_image_zip)
    candidate_rows = [row for row in read_manifest_rows(manifest_path) if row["stem"] in converted_stems]
    rows = choose_rows(candidate_rows, args.samples_per_set, args.seed, set_id)
    if not rows:
        print(f"{args.image_prefix}_{set_id}: skip, no manifest rows are present in compressed image zip")
        return None

    cell_gap = 18
    title_height = 30
    pair_width = args.thumb_width * 2 + cell_gap
    sample_height = int(round(1280 * (args.thumb_width / 976))) + title_height + cell_gap
    canvas_width = pair_width
    canvas_height = sample_height * len(rows) + title_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), (245, 245, 242))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw_title(draw, (8, 8), f"{args.image_prefix}_{set_id}_{args.zip_kind} original PNG vs JPEG q95 bbox audit", font)

    source_zip_path = zip_path_for(paths.source_dir, args.image_prefix, set_id, args.zip_kind)
    label_zip_path = zip_path_for(paths.label_dir, args.label_prefix, set_id, args.zip_kind)
    with (
        ZipFile(source_zip_path) as source_zip,
        ZipFile(label_zip_path) as label_zip,
        ZipFile(output_image_zip_path) as output_image_zip,
    ):
        y_cursor = title_height
        for row in rows:
            bboxes = []
            for label_member in manifest_value_to_member_list(row.get("label_members"), row.get("label_member")):
                label_doc = load_json_bytes(
                    label_zip.read(label_member),
                    f"{label_zip_name(set_id, args.zip_kind, args.label_prefix)}:{label_member}",
                )
                bboxes.append(extract_annotations(label_doc, row["stem"])[0]["bbox"])
            original = thumbnail_with_bboxes(
                read_image(source_zip, row["image_member"], Image), bboxes, args.thumb_width, ImageDraw
            )
            compressed = thumbnail_with_bboxes(
                read_image(output_image_zip, row["output_image_member"], Image), bboxes, args.thumb_width, ImageDraw
            )
            draw_title(draw, (8, y_cursor), f"{row['stem']}  original", font)
            draw_title(draw, (args.thumb_width + cell_gap + 8, y_cursor), "JPEG q95 4:4:4", font)
            canvas.paste(original, (0, y_cursor + title_height))
            canvas.paste(compressed, (args.thumb_width + cell_gap, y_cursor + title_height))
            y_cursor += sample_height

    audit_dir = args.reports_dir / "visual_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    output_path = audit_dir / f"audit_{args.image_prefix}_{set_id}_{args.zip_kind}.jpg"
    canvas.save(output_path, format="JPEG", quality=95, subsampling=0, optimize=True)
    return output_path


def main() -> int:
    args = build_parser().parse_args()
    Image, ImageDraw, ImageFont = import_pillow()
    paths = resolve_dataset_paths(args)
    outputs = []
    for set_id in sorted(set(args.set_ids)):
        path = audit_set(args, paths, set_id, Image, ImageDraw, ImageFont)
        if path:
            outputs.append(str(path))
            print(f"{args.image_prefix}_{set_id}: wrote {path}")
    summary_path = args.reports_dir / "visual_audit" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"outputs": outputs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
