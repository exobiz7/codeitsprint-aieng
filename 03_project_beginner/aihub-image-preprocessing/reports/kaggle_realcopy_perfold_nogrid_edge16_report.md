# Kaggle Real-Copy Per-Fold bg64 재생성 리포트

작성일: 2026-07-06

## 1. 목적

캐글 대회용 real-copy 증강 데이터셋에서 fold별 누수를 막기 위해 `realcopy_src_fold{K}train.json`만 사용하는 per-fold 버전을 생성했으나, 생성 이미지 대부분에서 배경이 격자형으로 반복되는 문제가 확인되었다.

이번 작업의 목표는 다음과 같았다.

- fold0/fold1/fold2 각각 train manifest 밖 이미지를 절대 사용하지 않는다.
- 알약 cutout은 각 fold train manifest에 포함된 이미지의 `bbox_px`만 사용한다.
- 배경도 같은 fold train 이미지의 알약 없는 영역에서만 추출한다.
- 기존 격자형 배경 문제를 제거한다.
- 먼저 fold별 100장 audit으로 품질을 확인한 뒤, 최종 1500장씩 재생성한다.
- 기존 문제 있는 결과물과 섞이지 않도록 새 폴더명과 파일명에 `nogrid_edge16`을 명시한다.

## 2. 문제 현상

사용자가 확인한 기존 산출물:

- `/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg64_perfold0`
- `/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg64_perfold1`
- `/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg64_perfold2`

주요 문제:

- 384x384 배경칩이 976x1280 canvas에 반복 배치되면서 수평/수직 격자선이 보였다.
- 일부 배경칩의 코너 밝은 부분이 반복되어, 실제 촬영 배경이 아닌 타일 패턴처럼 보였다.
- 누수 검증은 통과했지만, 시각 품질 기준에서 학습 데이터로 사용하기 어렵다고 판단했다.

## 3. 원인 분석

원인은 배경 생성 방식이었다.

기존 fold-safe bg64 방식은 각 fold train 이미지에서 알약 없는 384x384 배경칩을 추출한 뒤, 이 작은 칩을 976x1280 canvas에 확장하는 구조였다. 이때 칩을 반복/미러링하는 과정에서 다음 문제가 생겼다.

- 같은 질감과 밝기 변화가 일정 간격으로 반복됨.
- 칩 경계가 canvas 내부에 선처럼 드러남.
- 밝은 코너/배경지 끝부분이 반복되어 인공적인 패턴이 됨.

즉, 데이터 누수 방지는 잘 됐지만, 작은 배경칩을 직접 타일링한 것이 시각 품질 실패의 핵심 원인이었다.

## 4. 수정 방향

`scripts/build_kaggle_realcopy_v1.py`를 수정했다.

핵심 변경:

- 384x384 배경칩을 직접 타일링하지 않는다.
- fold train 배경칩은 색상/노이즈/채널 편향 profile 추정에만 사용한다.
- canvas는 976x1280 전체 크기로 새로 생성한다.
- target RGB는 캐글 도메인에 맞춘 blue-gray `[112, 130, 154]` 근방으로 유지한다.
- multi-scale smooth noise, 약한 gradient, Gaussian blur를 적용해 자연스러운 배경 변화를 만든다.
- fold 격리는 유지한다. 배경 profile 추정용 chip은 여전히 해당 fold train 이미지에서만 추출한다.

추가 품질 게이트:

- `--min-edge-margin` 옵션을 추가했다.
- 최종 full 생성은 `--min-edge-margin 16`으로 실행했다.
- bbox가 이미지 경계에서 16px 미만으로 가까우면 배치를 실패로 보고 재시도한다.

수정된 주요 파일:

- `/Users/macstudio/dev/learning/codeitsprint/aieng/03_project_beginner/aihub-image-preprocessing/scripts/build_kaggle_realcopy_v1.py`

## 5. 시행착오 및 단계별 결과

### 5.1 1차 audit: `*_audit100_nogrid`

먼저 타일링 제거 배경만 적용해 fold별 100장 audit을 생성했다.

출력:

- `/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg64_perfold0_audit100_nogrid`
- `/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg64_perfold1_audit100_nogrid`
- `/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg64_perfold2_audit100_nogrid`

결과:

| fold | images | annotations | classes used | source images | source outside | bg outside | bg mixed | bbox/category issue | shards |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fold0 | 100 | 329 | 48 | 181 | 0 | 0 | 0 | 0 | 1 |
| fold1 | 100 | 328 | 48 | 185 | 0 | 0 | 0 | 0 | 1 |
| fold2 | 100 | 323 | 52 | 184 | 0 | 0 | 0 | 0 | 1 |

판단:

- 격자형 배경 문제는 contact sheet 기준으로 제거됐다.
- 다만 fold2에서 bbox가 이미지 경계에 3px까지 가까운 샘플이 1개 확인됐다.
- 잘림은 아니었지만, 의료/의약품 식별 데이터 기준에서는 더 엄격한 여백 조건이 필요하다고 판단했다.

### 5.2 2차 audit: `*_audit100_nogrid_edge16`

`--min-edge-margin 16`을 추가해 fold별 100장 audit을 다시 생성했다.

출력:

- `/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg64_perfold0_audit100_nogrid_edge16`
- `/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg64_perfold1_audit100_nogrid_edge16`
- `/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg64_perfold2_audit100_nogrid_edge16`

결과:

| fold | images | annotations | classes used | source images | source outside | bg outside | bg mixed | min edge | edge < 16 | decode issue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fold0 | 100 | 329 | 47 | 181 | 0 | 0 | 0 | 17px | 0 | 0 |
| fold1 | 100 | 328 | 48 | 185 | 0 | 0 | 0 | 20px | 0 | 0 |
| fold2 | 100 | 322 | 51 | 184 | 0 | 0 | 0 | 19px | 0 | 0 |

WebDataset 검증:

| fold | shards | jpg | json | pair mismatch | decode issue | json issue |
|---|---:|---:|---:|---:|---:|---:|
| fold0 | 1 | 100 | 100 | 0 | 0 | 0 |
| fold1 | 1 | 100 | 100 | 0 | 0 | 0 |
| fold2 | 1 | 100 | 100 | 0 | 0 | 0 |

판단:

- 격자형 배경은 제거됐다.
- bbox가 canvas 경계에 너무 붙는 문제도 제거됐다.
- 누수, 라벨, decode, WebDataset pairing 모두 통과했다.
- 이 설정을 full 1500장 생성 기준으로 확정했다.

## 6. 최종 full 재생성 결과

기존 문제 폴더는 덮어쓰지 않았다. 새 full 결과물은 아래 별도 폴더에 생성했다.

- `/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg64_perfold0_nogrid_edge16`
- `/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg64_perfold1_nogrid_edge16`
- `/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg64_perfold2_nogrid_edge16`

최종 검증 요약:

| fold | images | annotations | classes used | source images | cutout source outside | bg source outside | bg mixed | bbox issue | category issue | category_id=1 | shards |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fold0 | 1500 | 4943 | 49 | 181 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| fold1 | 1500 | 4922 | 50 | 185 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| fold2 | 1500 | 4892 | 54 | 184 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |

경계 여백 재계산:

| fold | min edge | edge < 16 | bbox fail | category_id=1 |
|---|---:|---:|---:|---:|
| fold0 | 16px | 0 | 0 | 0 |
| fold1 | 17px | 0 | 0 | 0 |
| fold2 | 16px | 0 | 0 | 0 |

WebDataset 검증:

| fold | shards | jpg | json | pair mismatch | decode issue | json issue |
|---|---:|---:|---:|---:|---:|---:|
| fold0 | 6 | 1500 | 1500 | 0 | 0 | 0 |
| fold1 | 6 | 1500 | 1500 | 0 | 0 | 0 |
| fold2 | 6 | 1500 | 1500 | 0 | 0 | 0 |

조합 수 분포:

| fold | 2 pills | 3 pills | 4 pills | mean pills/image |
|---|---:|---:|---:|---:|
| fold0 | 15 | 1027 | 458 | 3.2953 |
| fold1 | 42 | 994 | 464 | 3.2813 |
| fold2 | 58 | 992 | 450 | 3.2613 |

class distribution 유지도:

| fold | Spearman vs manifest | Pearson vs manifest |
|---|---:|---:|
| fold0 | 0.9722 | 0.9933 |
| fold1 | 0.9768 | 0.9923 |
| fold2 | 0.9752 | 0.9901 |

## 7. Audit 이미지

최종 full contact sheet:

- fold0: `/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg64_perfold0_nogrid_edge16/reports/audit/realcopy_v1_contact_sheet.png`
- fold1: `/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg64_perfold1_nogrid_edge16/reports/audit/realcopy_v1_contact_sheet.png`
- fold2: `/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg64_perfold2_nogrid_edge16/reports/audit/realcopy_v1_contact_sheet.png`

최종 full cutout asset sheet:

- fold0: `/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg64_perfold0_nogrid_edge16/reports/audit/realcopy_sam2_cutout_assets.jpg`
- fold1: `/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg64_perfold1_nogrid_edge16/reports/audit/realcopy_sam2_cutout_assets.jpg`
- fold2: `/Volumes/SSD 4T/01_sprint_ai_project1_data/processed/kaggle_realcopy_bg64_perfold2_nogrid_edge16/reports/audit/realcopy_sam2_cutout_assets.jpg`

## 8. 최종 데이터셋 구조

각 full 폴더는 COCO + WebDataset 형식이다.

주요 파일:

- `coco/annotations_coco.json`
- `coco/images/*.jpg`
- `webdataset/train/*.tar`
- `reports/summary.json`
- `reports/validation_report.json`
- `reports/audit/realcopy_v1_contact_sheet.png`
- `reports/audit/realcopy_sam2_cutout_assets.jpg`
- `spec/target_categories_schema.json`
- `spec/realcopy_src_fold{K}train.json`

학습 시에는 기존 `kaggle_realcopy_bg64_perfold0/1/2`가 아니라 새 `kaggle_realcopy_bg64_perfold{K}_nogrid_edge16` 폴더를 사용해야 한다.

## 9. 남은 주의사항

- 기존 문제 있는 per-fold 폴더는 삭제하지 않았다. 비교와 회귀 확인용으로 남아 있다.
- 새 데이터셋은 natural distribution을 유지하도록 만들었다. rare-first, 균등화, class floor 같은 추가 분포 보정은 이번 작업 범위에 포함하지 않았다.
- 이번 작업의 핵심 변수는 배경 생성 방식과 edge margin이다. 이후 성능 비교 시에는 기존 bad per-fold, `nogrid`, `nogrid_edge16`을 혼동하지 않아야 한다.
- 최종 사용 권장 경로는 `*_nogrid_edge16` 세 폴더다.

## 10. 결론

이번 세션에서는 per-fold real-copy 데이터셋의 가장 큰 품질 문제였던 격자형 배경을 해결했다. 원인은 fold train 이미지에서 얻은 384x384 배경칩을 canvas에 반복 확장한 방식이었다. 이를 fold별 배경 profile 기반의 비타일 procedural blue-gray 배경 생성으로 바꾸고, `min_edge_margin=16` 배치 게이트를 추가했다.

최종적으로 fold0/fold1/fold2 각각 1500장, 총 4500장의 누수 없는 real-copy 증강 데이터셋을 새 폴더에 생성했고, COCO/WebDataset 정합성, decode, jpg/json pairing, source leakage, bbox/category 검증을 모두 통과했다.
