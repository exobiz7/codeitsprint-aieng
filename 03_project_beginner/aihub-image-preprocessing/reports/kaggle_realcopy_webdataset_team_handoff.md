# Kaggle Real-Copy WebDataset 팀 전달 가이드

작성일: 2026-07-06

## 1. 전달해야 할 최종 데이터 폴더

아래 3개 폴더를 팀원에게 전달하면 된다.

- `/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg64_perfold0_nogrid_edge16`
- `/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg64_perfold1_nogrid_edge16`
- `/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg64_perfold2_nogrid_edge16`

주의: 기존 폴더인 `kaggle_realcopy_bg64_perfold0/1/2`는 격자 배경 문제가 있던 이전 산출물이므로 학습에 사용하지 않는다.

## 2. 폴드별 사용 원칙

k-fold 학습에서 누수를 막기 위해 반드시 같은 fold의 train 증강만 사용한다.

| 학습 fold | 함께 사용할 real-copy 증강 |
|---|---|
| fold0 train | `kaggle_realcopy_bg64_perfold0_nogrid_edge16` |
| fold1 train | `kaggle_realcopy_bg64_perfold1_nogrid_edge16` |
| fold2 train | `kaggle_realcopy_bg64_perfold2_nogrid_edge16` |

validation/test에는 이 증강 데이터를 섞지 않는다.

## 3. 각 폴더 안에서 중요한 파일

각 fold 폴더는 같은 구조다.

- `webdataset/train/*.tar`: 팀 공유와 학습 로딩용 WebDataset shard
- `coco/images/*.jpg`: 같은 이미지를 일반 파일로 풀어 둔 버전
- `coco/annotations_coco.json`: COCO 형식 annotation
- `spec/target_categories_schema.json`: 56개 클래스 스키마
- `spec/realcopy_src_fold{K}train.json`: 해당 fold train source manifest
- `reports/summary.json`: 생성/검증 요약
- `reports/validation_report.json`: 검증 리포트
- `reports/audit/realcopy_v1_contact_sheet.png`: 시각 audit sheet
- `assets/cutout_assets_manifest.csv`: 사용된 cutout asset source 추적용

학습에는 보통 `webdataset/train/*.tar` 또는 `coco/annotations_coco.json + coco/images` 중 하나만 쓰면 된다.

## 4. 라벨 구조

WebDataset shard 내부는 다음 구조다.

- `sample_key.jpg`
- `sample_key.json`

JSON 주요 필드:

- `image.width = 976`
- `image.height = 1280`
- `image.combo_size = 2|3|4`
- `annotations[]`
- `annotations[].bbox`: `[x, y, w, h]` pixel 좌표
- `annotations[].category_id`: Kaggle/COCO용 `dl_idx`
- `annotations[].class_index`: `0..55`
- `annotations[].product_id`: `K-xxxxxx`
- `annotations[].source_ref`: 원본 fold train 이미지와 bbox 추적용

중요: `category_id`와 `class_index`는 서로 다르다. 학습 코드가 `0..55` contiguous class id를 기대하면 `class_index`를 사용한다. COCO evaluator가 `category_id`를 기대하면 `category_id=dl_idx`를 그대로 사용한다.

## 5. WebDataset 로딩 예시

필요 패키지:

```bash
pip install webdataset pillow torch torchvision
```

PyTorch/WebDataset 예시:

```python
import json
from pathlib import Path

import torch
import webdataset as wds
from PIL import Image


def parse_json(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, bytes):
        return json.loads(obj.decode("utf-8"))
    return obj


def make_target(meta):
    anns = meta["annotations"]
    boxes = torch.tensor([ann["bbox"] for ann in anns], dtype=torch.float32)  # xywh
    labels = torch.tensor([ann["class_index"] for ann in anns], dtype=torch.long)  # 0..55
    category_ids = torch.tensor([ann["category_id"] for ann in anns], dtype=torch.long)
    product_ids = [ann["product_id"] for ann in anns]
    return {
        "boxes_xywh": boxes,
        "labels": labels,
        "category_ids": category_ids,
        "product_ids": product_ids,
        "image_id": meta["image"]["id"],
        "combo_size": meta["image"]["combo_size"],
    }


def preprocess(sample):
    image = sample["jpg"]
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    meta = parse_json(sample["json"])
    return image, make_target(meta)


root = Path("/path/to/kaggle_realcopy_bg64_perfold0_nogrid_edge16")
shards = sorted(str(p) for p in (root / "webdataset" / "train").glob("*.tar"))

dataset = (
    wds.WebDataset(shards, shardshuffle=True)
    .decode("pil")
    .map(preprocess)
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,
    collate_fn=lambda batch: tuple(zip(*batch)),
)

for images, targets in loader:
    # images: tuple[PIL.Image]
    # targets: tuple[dict]
    break
```

Detection 모델이 `xyxy` bbox를 기대하면 변환해서 사용한다.

```python
boxes_xywh = target["boxes_xywh"]
boxes_xyxy = boxes_xywh.clone()
boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]
boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]
```

## 6. COCO 로딩 예시

COCO loader를 쓰는 팀원은 다음 파일을 사용한다.

```text
images:      coco/images/*.jpg
annotation:  coco/annotations_coco.json
```

`pycocotools` 예시:

```bash
pip install pycocotools
```

```python
from pathlib import Path
from pycocotools.coco import COCO

root = Path("/path/to/kaggle_realcopy_bg64_perfold0_nogrid_edge16")
coco = COCO(str(root / "coco" / "annotations_coco.json"))
image_dir = root / "coco" / "images"

img_id = coco.getImgIds()[0]
img_info = coco.loadImgs([img_id])[0]
ann_ids = coco.getAnnIds(imgIds=[img_id])
anns = coco.loadAnns(ann_ids)
image_path = image_dir / img_info["file_name"]
```

## 7. 팀원에게 같이 전달하면 좋은 문서

데이터 폴더 3개와 함께 아래 repo 파일도 같이 전달하면 된다.

- `/Users/macstudio/dev/learning/codeitsprint/aieng/03_project_beginner/aihub-image-preprocessing/reports/kaggle_realcopy_perfold_nogrid_edge16_report.md`
- `/Users/macstudio/dev/learning/codeitsprint/aieng/03_project_beginner/aihub-image-preprocessing/reports/kaggle_realcopy_webdataset_team_handoff.md`
- `/Users/macstudio/dev/learning/codeitsprint/aieng/03_project_beginner/aihub-image-preprocessing/scripts/build_kaggle_realcopy_v1.py`
- `/Users/macstudio/dev/learning/codeitsprint/aieng/03_project_beginner/aihub-image-preprocessing/codex-handoff/handoff_realcopy/realcopy_src_fold0train.json`
- `/Users/macstudio/dev/learning/codeitsprint/aieng/03_project_beginner/aihub-image-preprocessing/codex-handoff/handoff_realcopy/realcopy_src_fold1train.json`
- `/Users/macstudio/dev/learning/codeitsprint/aieng/03_project_beginner/aihub-image-preprocessing/codex-handoff/handoff_realcopy/realcopy_src_fold2train.json`

`build_kaggle_realcopy_v1.py`는 학습 로딩에는 필요 없고, 재생성/감사/회귀 확인이 필요할 때만 필요하다.

## 8. 최종 검증 요약

| fold | images | annotations | shards | jpg/json pair | decode issue | source leakage | bg leakage | bbox/category issue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fold0 | 1500 | 4943 | 6 | 0 | 0 | 0 | 0 | 0 |
| fold1 | 1500 | 4922 | 6 | 0 | 0 | 0 | 0 | 0 |
| fold2 | 1500 | 4892 | 6 | 0 | 0 | 0 | 0 | 0 |

최종 사용 권장 데이터셋은 `*_nogrid_edge16` 세 폴더다.
