#!/usr/bin/env python3
"""Run asset-bank combo synthesis in parallel parts and merge them into one run."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from v2_common import CANONICAL_FIELDS, DEFAULT_PROCESSED_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--bank-name", default="sam2_large_a6")
    parser.add_argument("--run-name", default="combo_synth_600k")
    parser.add_argument("--num-images", type=int, default=600_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--remove-part-runs", action="store_true")
    return parser


def split_counts(total: int, workers: int) -> list[int]:
    base, remainder = divmod(total, workers)
    return [base + (1 if idx < remainder else 0) for idx in range(workers)]


def stream_output(prefix: str, pipe) -> None:
    try:
        for line in iter(pipe.readline, ""):
            if line:
                print(f"[{prefix}] {line}", end="", flush=True)
    finally:
        pipe.close()


def run_parts(args) -> list[Path]:
    script = Path(__file__).with_name("build_v2_asset_combo_synthetic.py")
    env = os.environ.copy()
    external_tmp = Path("/Volumes/SSD 4T/codex_tmp")
    env["TMPDIR"] = str(external_tmp / "pytorch_tmp")
    env["TEMP"] = str(external_tmp / "pytorch_tmp")
    env["TMP"] = str(external_tmp / "pytorch_tmp")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    part_roots = []
    processes = []
    threads = []
    for idx, count in enumerate(split_counts(args.num_images, args.workers)):
        part_run_name = f"{args.run_name}_part{idx:02d}"
        part_roots.append(args.processed_root / "runs" / part_run_name)
        cmd = [
            sys.executable,
            str(script),
            "--processed-root",
            str(args.processed_root),
            "--bank-name",
            args.bank_name,
            "--run-name",
            part_run_name,
            "--num-images",
            str(count),
            "--shard-size",
            str(args.shard_size),
            "--seed",
            str(args.seed + idx * 104_729),
            "--overwrite",
            "--skip-parquet",
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes.append((idx, proc))
        thread = threading.Thread(target=stream_output, args=(f"part{idx:02d}", proc.stdout), daemon=True)
        thread.start()
        threads.append(thread)

    failed = []
    for idx, proc in processes:
        return_code = proc.wait()
        if return_code != 0:
            failed.append((idx, return_code))
    for thread in threads:
        thread.join(timeout=1)
    if failed:
        raise SystemExit(f"Part generation failed: {failed}")
    return part_roots


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def merge_parts(args, part_roots: list[Path]) -> dict:
    final_root = args.processed_root / "runs" / args.run_name
    if final_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Run exists: {final_root}. Pass --overwrite to replace.")
        shutil.rmtree(final_root)
    final_manifest_dir = final_root / "manifests"
    final_shard_dir = final_root / "webdataset" / "train_combo_synth"
    final_manifest_dir.mkdir(parents=True, exist_ok=True)
    final_shard_dir.mkdir(parents=True, exist_ok=True)

    shard_map: dict[str, str] = {}
    part_summaries = []
    global_shard_index = 0
    for part_root in part_roots:
        summary_path = part_root / "manifests" / "combo_synth_v1_summary.json"
        part_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        part_summaries.append(part_summary)
        for old_shard in sorted((part_root / "webdataset" / "train_combo_synth").glob("*.tar")):
            new_shard = final_shard_dir / f"train_combo_synth-{global_shard_index:06d}.tar"
            hardlink_or_copy(old_shard, new_shard)
            shard_map[str(old_shard)] = str(new_shard)
            global_shard_index += 1

    final_csv = final_manifest_dir / "combo_synth_v1_manifest.csv"
    rows = 0
    annotation_total = 0
    combo_size_counts = collections.Counter()
    product_counts = collections.Counter()
    with final_csv.open("w", newline="", encoding="utf-8") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        for part_root in part_roots:
            part_csv = part_root / "manifests" / "combo_synth_v1_manifest.csv"
            with part_csv.open(newline="", encoding="utf-8") as in_handle:
                for row in csv.DictReader(in_handle):
                    new_source_zip = shard_map.get(row.get("source_zip", ""), row.get("source_zip", ""))
                    row["source_zip"] = new_source_zip
                    row["source_zip_name"] = Path(new_source_zip).name if new_source_zip else ""
                    writer.writerow({field: row.get(field, "") for field in CANONICAL_FIELDS})
                    rows += 1
                    combo_size = int(row["combo_size"])
                    annotation_total += combo_size
                    combo_size_counts[combo_size] += 1
                    for product_id in json.loads(row["combo_product_ids"]):
                        product_counts[product_id] += 1

    summary = {
        "run_name": args.run_name,
        "num_images": rows,
        "annotations": annotation_total,
        "asset_bank": args.bank_name,
        "asset_products": len(product_counts),
        "products_used": len(product_counts),
        "product_instance_count_min": min(product_counts.values()) if product_counts else 0,
        "product_instance_count_max": max(product_counts.values()) if product_counts else 0,
        "combo_size_counts": dict(combo_size_counts),
        "attempts": sum(int(item.get("attempts", 0)) for item in part_summaries),
        "placement_failures": sum(int(item.get("placement_failures", 0)) for item in part_summaries),
        "shards": global_shard_index,
        "shard_paths": [str(path) for path in sorted(final_shard_dir.glob("*.tar"))],
        "manifest_csv": str(final_csv),
        "part_runs": [str(path) for path in part_roots],
        "part_summaries": part_summaries,
        "merge_mode": "hardlink_or_copy",
    }
    (final_manifest_dir / "combo_synth_v1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.remove_part_runs:
        for part_root in part_roots:
            shutil.rmtree(part_root)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    part_roots = run_parts(args)
    summary = merge_parts(args, part_roots)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
