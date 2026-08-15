#!/usr/bin/env python3
"""Run one-by-one q95 conversion, validation, audit, and source zip deletion."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from aihub_common import (
    DEFAULT_OUTPUT_TRAINING_ROOT,
    DEFAULT_TRAINING_ROOT,
    IMAGE_PREFIX,
    add_path_args,
    add_zip_args,
    image_zip_name,
    resolve_dataset_paths,
    zip_map,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_args(parser)
    add_zip_args(parser)
    parser.set_defaults(training_root=DEFAULT_TRAINING_ROOT, output_training_root=DEFAULT_OUTPUT_TRAINING_ROOT)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--set-ids", type=int, nargs="+")
    parser.add_argument("--start-after", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--audit-samples", type=int, default=8)
    parser.add_argument("--state-file", type=Path, default=Path("reports/sequential_q95_state.jsonl"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def append_state(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts": now_iso(), **event}, ensure_ascii=False, sort_keys=True) + "\n")


def run_command(argv: list[str], event_name: str, state_file: Path, set_id: int, image_prefix: str) -> None:
    append_state(state_file, {"set_id": set_id, "event": f"{event_name}_start", "argv": argv})
    print(f"[{image_prefix}_{set_id}] {event_name} start", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(argv, env=env, check=False)
    append_state(state_file, {"set_id": set_id, "event": f"{event_name}_end", "returncode": completed.returncode})
    if completed.returncode != 0:
        raise RuntimeError(f"{event_name} failed for {image_prefix}_{set_id} with exit code {completed.returncode}")
    print(f"[{image_prefix}_{set_id}] {event_name} done", flush=True)


def selected_source_ids(paths, requested: list[int] | None, start_after: int | None, image_prefix: str, zip_kind: str) -> list[int]:
    source_zips = zip_map(paths.source_dir, image_prefix, zip_kind)
    ids = sorted(source_zips)
    if requested:
        requested_set = set(requested)
        ids = [set_id for set_id in ids if set_id in requested_set]
    if start_after is not None:
        ids = [set_id for set_id in ids if set_id > start_after]
    return ids


def selected_source_zips(
    paths, requested: list[int] | None, start_after: int | None, image_prefix: str, zip_kind: str
) -> dict[int, Path]:
    source_zips = zip_map(paths.source_dir, image_prefix, zip_kind)
    ids = selected_source_ids(paths, requested, start_after, image_prefix, zip_kind)
    return {set_id: source_zips[set_id] for set_id in ids}


def forwarded_args(args) -> list[str]:
    forwarded: list[str] = []
    for option, attr in (
        ("--training-root", "training_root"),
        ("--source-dir", "source_dir"),
        ("--label-dir", "label_dir"),
        ("--output-training-root", "output_training_root"),
        ("--output-source-dir", "output_source_dir"),
        ("--output-label-dir", "output_label_dir"),
    ):
        value = getattr(args, attr)
        if value is not None:
            forwarded.extend([option, str(value)])
    forwarded.extend(["--image-prefix", args.image_prefix, "--label-prefix", args.label_prefix, "--zip-kind", args.zip_kind])
    return forwarded


def main() -> int:
    args = build_parser().parse_args()
    paths = resolve_dataset_paths(args)
    source_zips = selected_source_zips(paths, args.set_ids, args.start_after, args.image_prefix, args.zip_kind)
    set_ids = sorted(source_zips)
    if not set_ids:
        print("No source zip sets selected.", flush=True)
        return 0

    print(f"Selected sets: {set_ids}", flush=True)
    append_state(args.state_file, {"event": "run_start", "set_ids": set_ids, "workers": args.workers})
    script_dir = Path(__file__).resolve().parent
    python = sys.executable

    for set_id in set_ids:
        source_zip = source_zips[set_id]
        output_zip = paths.output_source_dir / image_zip_name(set_id, args.zip_kind, args.image_prefix)
        source_size = source_zip.stat().st_size
        append_state(
            args.state_file,
            {
                "set_id": set_id,
                "event": "set_start",
                "source_zip": str(source_zip),
                "source_size_bytes": source_size,
            },
        )
        print(f"[{args.image_prefix}_{set_id}] set start source_size={source_size}", flush=True)
        if args.dry_run:
            continue

        common_args = forwarded_args(args)
        run_command(
            [
                python,
                str(script_dir / "inventory_dataset.py"),
                *common_args,
                "--set-ids",
                str(set_id),
                "--reports-dir",
                str(args.reports_dir),
            ],
            "inventory",
            args.state_file,
            set_id,
            args.image_prefix,
        )
        run_command(
            [
                python,
                str(script_dir / "compress_aihub_images.py"),
                *common_args,
                "--set-ids",
                str(set_id),
                "--reports-dir",
                str(args.reports_dir),
                "--workers",
                str(args.workers),
                "--progress-every",
                str(args.progress_every),
                "--overwrite",
            ],
            "compress",
            args.state_file,
            set_id,
            args.image_prefix,
        )
        run_command(
            [
                python,
                str(script_dir / "validate_compressed_dataset.py"),
                *common_args,
                "--set-ids",
                str(set_id),
                "--reports-dir",
                str(args.reports_dir),
            ],
            "validate",
            args.state_file,
            set_id,
            args.image_prefix,
        )
        run_command(
            [
                python,
                str(script_dir / "make_visual_audit.py"),
                *common_args,
                "--set-ids",
                str(set_id),
                "--samples-per-set",
                str(args.audit_samples),
                "--reports-dir",
                str(args.reports_dir),
            ],
            "visual_audit",
            args.state_file,
            set_id,
            args.image_prefix,
        )

        output_size = output_zip.stat().st_size if output_zip.exists() else None
        source_zip.unlink()
        append_state(
            args.state_file,
            {
                "set_id": set_id,
                "event": "source_deleted",
                "deleted_source_zip": str(source_zip),
                "deleted_source_size_bytes": source_size,
                "output_zip": str(output_zip),
                "output_size_bytes": output_size,
            },
        )
        print(f"[{args.image_prefix}_{set_id}] deleted source zip, reclaimed {source_size} bytes", flush=True)
        run_command(["df", "-h", str(paths.source_dir)], "df", args.state_file, set_id, args.image_prefix)
        append_state(args.state_file, {"set_id": set_id, "event": "set_done"})
        print(f"[{args.image_prefix}_{set_id}] set done", flush=True)

    append_state(args.state_file, {"event": "run_done", "set_ids": set_ids})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
