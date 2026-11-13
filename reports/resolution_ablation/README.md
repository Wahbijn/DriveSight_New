# Inference-resolution ablation (960 / 1152 / 1280 px)

Read-only ablation. **No model was retrained, no checkpoint or dataset file was
modified.** Tests whether evaluating the three existing checkpoints at a higher
inference resolution recovers some of the recall lost to small-object detection
(identified in `reports/p0_diagnostics/error_cause_analysis.md`).

Full narrative + embedded figures: `notebooks/07_inference_resolution_ablation.ipynb`.
This file is the quick-reference index of raw outputs.

## Directory layout

```
reports/resolution_ablation/
├── README.md
├── validation/
│   ├── run_config.json
│   ├── full_run_log.txt
│   ├── summary_global_metrics.csv      <- P/R/mAP50/mAP50-95 + speed + peak GPU mem, all 9 runs
│   ├── summary_per_class_ap.csv        <- AP50/AP50-95/P/R per class, all 9 runs
│   └── {model}_{imgsz}/                <- raw Ultralytics val() output per run
│       ├── predictions.json            <- save_json=True COCO-format predictions
│       ├── confusion_matrix(_normalized).png
│       └── Box{P,R,F1,PR}_curve.png
├── size_breakdown/
│   ├── coco_gt_val.json                <- COCO-format GT built from the YOLO val labels
│   └── size_breakdown_by_class.csv     <- AP/AR by COCO small/medium/large, all 9 runs x 5 classes
├── visual_comparison/
│   └── <image_stem>/
│       ├── gt_only.jpg
│       ├── {model}_{imgsz}.jpg         <- full-frame panel
│       ├── grid.jpg                    <- 3x3 (model x imgsz) composite
│       └── zoom_*.jpg / zoom_grid.jpg  <- cropped+upscaled panels for 2 key objects
└── charts/
    ├── global_metrics_vs_imgsz.png
    ├── ar_small_vs_imgsz_by_class.png
    ├── inference_time_vs_imgsz.png
    └── gpu_memory_vs_imgsz.png
```

## Scripts (new, under `scripts/`, configurable; nothing under `notebooks/` or
`runs/` was touched)

- `scripts/run_resolution_ablation.py` — the 9 val() runs, batch=1, save_json=True,
  with CUDA-OOM try/except per run and GPU-memory/speed instrumentation.
- `scripts/coco_size_analysis.py` — reused as-is from the P0 pass (per-class,
  per-size AP/AR via `faster_coco_eval`), pointed at the 9 new `predictions.json` files.
- `scripts/visualize_resolution_grid.py` — GT-vs-prediction grids, 6 representative
  images x 3 models x 3 resolutions.
- `scripts/zoom_region_resolution_grid.py` — cropped/zoomed version for 2 small/
  borderline objects where the full-frame panel is too small to read labels.
- `scripts/plot_resolution_ablation_charts.py` — the 4 PNG charts above.

## Exact commands run

```bash
python scripts/run_resolution_ablation.py \
  --model baseline=runs/detect/runs/bdd100k_yolo11n_baseline/weights/best.pt \
  --model oversampling=runs/detect/runs/bdd100k_yolo11n_oversampling/weights/best.pt \
  --model finetune=runs/detect/runs/bdd100k_yolo11n_finetune_960/weights/best.pt \
  --imgsz 960 1152 1280 --batch 1 --device 0
# (conf omitted -> Ultralytics default 0.001; iou=0.70 NMS; same val split/data.yaml
#  as every previous round)

python scripts/coco_size_analysis.py \
  --model baseline_960=reports/resolution_ablation/validation/baseline_960/predictions.json \
  --model baseline_1152=reports/resolution_ablation/validation/baseline_1152/predictions.json \
  --model baseline_1280=reports/resolution_ablation/validation/baseline_1280/predictions.json \
  --model oversampling_960=... [... all 9 ...] \
  --classes pedestrian rider car motorcycle bicycle \
  --output-dir reports/resolution_ablation/size_breakdown

python scripts/visualize_resolution_grid.py \
  --model baseline=... --model oversampling=... --model finetune=... \
  --imgsz 960 1152 1280 \
  --images c96e0b6a-f3d8aa78 4c1d2408-79034cda a48909f7-54e4bcff \
           7a0f932b-47be6c4b 79f06acb-fa30703f 04180105-d7aca785 \
  --conf 0.25 --iou 0.70 --device 0 \
  --output-dir reports/resolution_ablation/visual_comparison

python scripts/zoom_region_resolution_grid.py --image 4c1d2408-79034cda --region 132 346 222 479 --pad 50 --zoom 3 [... same models/imgsz ...]
python scripts/zoom_region_resolution_grid.py --image c96e0b6a-f3d8aa78 --region 894 18 1122 403 --pad 50 --zoom 2 [... same models/imgsz ...]

python scripts/plot_resolution_ablation_charts.py \
  --global-csv reports/resolution_ablation/validation/summary_global_metrics.csv \
  --size-csv reports/resolution_ablation/size_breakdown/size_breakdown_by_class.csv \
  --output-dir reports/resolution_ablation/charts
```

## Headline results (`validation/summary_global_metrics.csv` — all 9 runs completed, 0 OOM/errors)

| model | imgsz | precision | recall | mAP50 | mAP50-95 | inference ms/img | peak GPU mem (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 960 | 53.8% | 43.3% | 45.3% | 24.4% | 19.2 | 93.5 |
| baseline | **1152** | 56.1% | **43.7%** | **45.6%** | **24.4%** | 18.9 | 130.1 |
| baseline | 1280 | 52.7% | 41.7% | 43.7% | 23.2% | 18.5 | 146.6 |
| oversampling | 960 | 54.1% | 44.6% | 45.9% | 24.9% | 18.5 | 104.5 |
| oversampling | **1152** | **58.5%** | 43.2% | **47.1%** | **25.0%** | 19.4 | 129.4 |
| oversampling | 1280 | 55.1% | 42.1% | 45.8% | 23.8% | 18.3 | 146.6 |
| finetune | 960 | 54.3% | 41.0% | 42.6% | 22.6% | 18.6 | 104.5 |
| finetune | **1152** | 51.8% | **42.5%** | **44.1%** | **23.2%** | 18.5 | 129.4 |
| finetune | 1280 | 53.0% | 40.4% | 43.0% | 22.8% | 18.7 | 146.6 |

**1152 is a peak, not a monotonic improvement** — every model's mAP50/mAP50-95 rises
960→1152 then falls 1152→1280. Inference time per image is flat (~18.3–19.4ms)
across all three resolutions at batch=1 on this GPU — at this model size (2.6M
params), the bottleneck is fixed per-call overhead, not resolution-dependent compute.
GPU memory scales clearly with resolution (93–105MB at 960 → 147MB at 1280), still
far under the 4GB budget.

## Small-object AR by class (`size_breakdown/size_breakdown_by_class.csv`)

| class | model | AR_small @960 | AR_small @1152 | AR_small @1280 |
|---|---|---:|---:|---:|
| rider | baseline | 17.1% | **21.2%** | 18.8% |
| rider | oversampling | 15.9% | **23.5%** | 19.4% |
| motorcycle | baseline | 13.3% | **23.3%** | 19.3% |
| motorcycle | oversampling | 20.7% | **24.7%** | 18.0% |
| bicycle | baseline | 11.7% | 17.2% | **20.0%** |
| bicycle | oversampling | 12.8% | 21.7% | **24.4%** |
| pedestrian | baseline | 29.9% | 33.3% | **35.0%** |
| car | baseline | 34.2% | 37.3% | **38.0%** |

Rider/motorcycle peak at 1152 and give some of it back at 1280; pedestrian/car/bicycle
keep improving to 1280. `rider`/`motorcycle` small-object GT counts are tiny (15–17
instances) — treat any single-point delta as indicative, not precise.
