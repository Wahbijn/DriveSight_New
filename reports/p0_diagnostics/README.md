# P0 diagnostics — Baseline vs Oversampling vs Fine-tuning

Read-only diagnostic pass. **No model was retrained, no checkpoint was overwritten, no
dataset file was modified.** Scope: the four P0 items from the DriveSight detection
audit:

1. Error audit (FP/FN/class-confusion) on all three checkpoints, identical settings.
2. One consistent `yolo val(save_json=True)` pass on all three checkpoints.
3. Provenance check of `data/bdd100k/labels/100k/test`.
4. Everything saved here, under `reports/p0_diagnostics/`.

All commands below were run from the project root with:
`ultralytics 8.4.189`, `torch 2.10.0+cu130`, CUDA available (RTX 2050, 4GB VRAM).

## Directory layout

```
reports/p0_diagnostics/
├── README.md                          <- this file
├── test_label_provenance.md           <- item 3 writeup
├── validation/                        <- item 2: consistent save_json=True val pass
│   ├── run_config.json                <- exact settings used
│   ├── full_run_log.txt               <- raw console output, all 3 models
│   ├── summary_global_metrics.csv     <- P/R/mAP50/mAP50-95 per model
│   ├── summary_per_class_metrics.csv  <- AP50/AP50-95/P/R per class per model
│   ├── summary_ap50_by_class_pivot.csv<- same, wide format for quick reading
│   └── {baseline,oversampling,finetune}/
│       ├── predictions.json           <- save_json=True COCO-format predictions
│       ├── confusion_matrix(_normalized).png
│       └── Box{P,R,F1,PR}_curve.png
└── error_audit/                       <- item 1: custom FP/FN/confusion audit
    ├── comparison_missed_objects.csv
    ├── comparison_unmatched_predictions.csv
    ├── comparison_audit_recall.csv
    ├── comparison_class_confusions.csv
    ├── {baseline,oversampling,finetune}_log.txt
    └── {baseline,oversampling,finetune}/
        ├── run_config.json
        ├── missed_objects.csv / unmatched_predictions.csv / class_confusions.csv
        ├── correct_by_class.csv / summary_totals.csv
        └── examples/ (up to 30 annotated error images)
```

## Scripts (new, configurable, kept outside `notebooks/` so the team's existing
notebooks/outputs are untouched)

- `scripts/run_consistent_validation.py` — wraps `YOLO(...).val()`.
- `scripts/audit_validation_errors.py` — configurable version of
  `notebooks/audit_validation_errors.py` (identical matching logic, CLI args instead
  of hardcoded constants).
- `scripts/compare_error_audits.py` — merges the three audits into comparison CSVs.

## Exact commands run

```bash
# Item 2 — consistent validation, save_json=True, imgsz=960, same val split, all 3 models
python scripts/run_consistent_validation.py \
  --model baseline=runs/detect/runs/bdd100k_yolo11n_baseline/weights/best.pt \
  --model oversampling=runs/detect/runs/bdd100k_yolo11n_oversampling/weights/best.pt \
  --model finetune=runs/detect/runs/bdd100k_yolo11n_finetune_960/weights/best.pt \
  --imgsz 960 --split val --batch 8 --device 0
# (conf omitted -> Ultralytics default 0.001 for mAP/PR-curve construction;
#  iou=0.70 NMS, data=data/bdd100k/yolo_dataset/data.yaml)

# Item 1 — error audit, identical settings, each of the 3 checkpoints
python scripts/audit_validation_errors.py \
  --model runs/detect/runs/bdd100k_yolo11n_baseline/weights/best.pt \
  --output-dir reports/p0_diagnostics/error_audit/baseline --run-label baseline \
  --conf 0.25 --iou 0.70 --match-iou 0.50 --imgsz 960 --device 0

python scripts/audit_validation_errors.py \
  --model runs/detect/runs/bdd100k_yolo11n_oversampling/weights/best.pt \
  --output-dir reports/p0_diagnostics/error_audit/oversampling --run-label oversampling \
  --conf 0.25 --iou 0.70 --match-iou 0.50 --imgsz 960 --device 0

python scripts/audit_validation_errors.py \
  --model runs/detect/runs/bdd100k_yolo11n_finetune_960/weights/best.pt \
  --output-dir reports/p0_diagnostics/error_audit/finetune --run-label finetune \
  --conf 0.25 --iou 0.70 --match-iou 0.50 --imgsz 960 --device 0

# merge into comparison tables
python scripts/compare_error_audits.py --audit-dir reports/p0_diagnostics/error_audit \
  --model baseline --model oversampling --model finetune
```

## Headline results (item 2 — `validation/summary_global_metrics.csv`)

| model | precision | recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| baseline | 53.8% | 43.3% | 45.3% | 24.4% |
| oversampling | 54.1% | 44.5% | 45.9% | 24.9% |
| finetune | 54.4% | 40.8% | 42.5% | 22.6% |

Matches the previously-reported table to within rounding (confirms the three
checkpoints and this validation protocol reproduce the original headline numbers, and
confirms the Fine-tuning checkpoint's `best.pt` is indeed the epoch-1 weights — see the
main diagnosis for why).

Per-class AP50 (`validation/summary_ap50_by_class_pivot.csv`):

| class | baseline | oversampling | finetune |
|---|---:|---:|---:|
| pedestrian | 55.7% | 56.0% | 53.7% |
| rider | 34.8% | 26.1% | 27.1% |
| car | 74.8% | 74.8% | 72.5% |
| truck | 43.6% | 42.0% | 41.4% |
| bus | 48.1% | 47.7% | 45.6% |
| motorcycle | 19.9% | 33.7% | 18.3% |
| bicycle | 27.1% | 26.4% | 21.0% |
| traffic light | 51.1% | 52.8% | 51.1% |
| traffic sign | 52.9% | 53.8% | 52.1% |

COCO-style AP/AR by object size, from `predictions.json` (pycocotools eval, printed in
`full_run_log.txt`), baseline model: AP@[.5:.95]=0.246 overall but only **0.104 for
small objects** vs **area=large 0.655** on AR@[.5:.95] — confirms a real,
size-correlated localization gap on top of the class-imbalance gap (oversampling and
finetune show the same pattern: small-object AR stays in the 0.24–0.29 range across all
three models).

## Error-audit comparison (item 1 — `error_audit/comparison_*.csv`)

This is the audit that had never been run on anything but the Baseline checkpoint
before. Rider-related class confusions, baseline → oversampling → finetune:

| true → predicted | baseline | oversampling | finetune |
|---|---:|---:|---:|
| rider → pedestrian | 4 | **9** | **13** |
| rider → bicycle | 2 | 1 | 1 |
| rider → motorcycle | 1 | 0 | 0 |
| rider → car | 0 | 2 | 1 |
| pedestrian → rider | 3 | 7 | 3 |

See the main chat response for what this means — it overturns the leading hypothesis
from the earlier research-only pass.
