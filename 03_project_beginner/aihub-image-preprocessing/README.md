# AI Hub Pill Image JPEG q95 Pipeline

Utilities for creating a storage-efficient derived copy of the AI Hub single oral pill image dataset without changing geometry, crop, resize, or bbox coordinates.

## Workflow

1. Install dependencies.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

2. Build inventory and paired manifests.

```bash
python scripts/inventory_dataset.py
```

3. Run a small pilot conversion.

```bash
python scripts/compress_aihub_images.py --set-ids 7 16 24 --limit 200 --overwrite
python scripts/validate_compressed_dataset.py --set-ids 7 16 24 --limit 200
python scripts/make_visual_audit.py --set-ids 7 16 24 --samples-per-set 8
```

4. Convert complete zip sets one by one after the pilot is accepted.

```bash
python scripts/compress_aihub_images.py --set-ids 7 --overwrite
python scripts/validate_compressed_dataset.py --set-ids 7
```

Use `--all` only after confirming there is enough free disk space and the pilot reports look good.

## Safety Defaults

- Source PNG zip files are never modified or deleted.
- Output is written under `1.Training_jpeg_q95`.
- Output source and label zip names match the original `TS_*_단일.zip` and `TL_*_단일.zip` names, but image entries are `.jpg`.
- Derived labels only change `images[*].file_name` and `images[*].imgfile` from `.png` to `.jpg`.
- `TS_67/TL_67` mismatches are handled by using only image/JSON stem intersections from the inventory manifest.

## v2 Training Dataset Pipeline

The v2 pipeline builds a Detection + product-ID training dataset from the JPEG q95
zips. It writes canonical manifests and WebDataset-compatible tar shards under:

```text
/Volumes/SSD 4T/166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터/01.데이터/processed/v2
```

Run a small pilot first:

```bash
python scripts/run_v2_pipeline.py --pilot --overwrite --skip-parquet
```

Run the full v2 build:

```bash
python scripts/run_v2_pipeline.py --overwrite
```

Outputs:

- `processed/v2/manifests/train_single_raw.csv`
- `processed/v2/manifests/val_official_single_ood.csv`
- `processed/v2/manifests/val_official_combo_real.csv`
- `processed/v2/manifests/split_manifest.csv`
- `processed/v2/manifests/sampler_plan.json`
- `processed/v2/manifests/combo_synth_v1_manifest.csv`
- `processed/v2/webdataset/<split>/*.tar`

The synthetic combo builder uses only `train_seen` single-pill samples. Official
validation images are never used as synthetic sources. If `pyarrow` is installed,
matching `.parquet` files are also written; CSV is always written.

### SAM2 synthetic cutouts

For precision-sensitive synthetic combos, install the SAM2 dependencies and keep
the checkpoint under `models/sam2`. Use the large checkpoint for quality pilots;
the tiny checkpoint is faster but less conservative around difficult edges.

```bash
python -m pip install -r requirements.txt
mkdir -p models/sam2
curl -L -o models/sam2/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

Generate a SAM2-only synthetic pilot and inspect both the cutouts and final
compositions:

```bash
python scripts/build_combo_synthetic.py \
  --processed-root "/Volumes/SSD 4T/166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터/01.데이터/processed/v2_pilot" \
  --num-images 40 \
  --overwrite \
  --skip-parquet \
  --mask-provider sam2 \
  --sam2-checkpoint models/sam2/sam2.1_hiera_large.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --min-mask-score 0.86 \
  --sam2-logit-threshold 0.8 \
  --sam2-box-expansion-ratio 0.035

python scripts/make_v2_cutout_audit.py \
  --processed-root "/Volumes/SSD 4T/166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터/01.데이터/processed/v2_pilot" \
  --sam2-checkpoint models/sam2/sam2.1_hiera_large.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml

python scripts/make_v2_synthetic_audit.py \
  --processed-root "/Volumes/SSD 4T/166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터/01.데이터/processed/v2_pilot" \
  --samples-per-size 8
```

Transparent, semi-transparent, and soft-gel-like source rows are excluded from
synthetic combo generation by default because their pixels already contain the
original background seen through the pill. Pass `--include-transparent-sources`
only for a separate transparent-pill matting experiment.
