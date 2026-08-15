#!/usr/bin/env python3
"""Run the v2 manifest, split, real-shard, synthetic, and validation pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from v2_common import DEFAULT_PROCESSED_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--synthetic-images", type=int, default=600_000)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--skip-hashes", action="store_true")
    parser.add_argument("--skip-parquet", action="store_true")
    parser.add_argument("--skip-real-shards", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="Quick run: no hashes, 50 rows/source, 200 synthetic images.")
    return parser


def run(argv: list[str]) -> None:
    print("+", " ".join(argv), flush=True)
    subprocess.run(argv, check=True)


def main() -> int:
    args = build_parser().parse_args()
    script_dir = Path(__file__).resolve().parent
    python = sys.executable
    processed = str(args.processed_root)
    skip_hashes = args.skip_hashes or args.pilot
    skip_parquet = args.skip_parquet
    manifest_args = [python, str(script_dir / "build_v2_manifest.py"), "--processed-root", processed]
    if skip_hashes:
        manifest_args.append("--no-hashes")
    if skip_parquet:
        manifest_args.append("--skip-parquet")
    if args.pilot:
        manifest_args.extend(["--limit-per-source", "50"])
    run(manifest_args)
    split_args = [python, str(script_dir / "build_v2_splits.py"), "--processed-root", processed]
    if skip_parquet:
        split_args.append("--skip-parquet")
    run(split_args)
    if not args.skip_real_shards:
        real_args = [
            python,
            str(script_dir / "pack_v2_webdataset.py"),
            "--processed-root",
            processed,
            "--shard-size",
            str(args.shard_size),
        ]
        if args.pilot:
            real_args.extend(["--limit-per-split", "50"])
        if args.overwrite:
            real_args.append("--overwrite")
        run(real_args)
    if not args.skip_synthetic:
        synth_args = [
            python,
            str(script_dir / "build_combo_synthetic.py"),
            "--processed-root",
            processed,
            "--num-images",
            str(200 if args.pilot else args.synthetic_images),
            "--shard-size",
            str(args.shard_size),
        ]
        if skip_parquet:
            synth_args.append("--skip-parquet")
        if args.overwrite:
            synth_args.append("--overwrite")
        run(synth_args)
    validate_args = [python, str(script_dir / "validate_v2_dataset.py"), "--processed-root", processed, "--validate-shards"]
    if args.pilot:
        validate_args.extend(["--max-shard-samples", "20"])
    run(validate_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
