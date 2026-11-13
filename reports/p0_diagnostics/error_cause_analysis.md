# Error-cause analysis: small objects, rider↔pedestrian confusion

Read-only follow-up to the P0 diagnostics. No files modified, no training run.

## Commands run

```bash
# Per-class, per-object-size AP/AR using the already-saved predictions.json files
python scripts/coco_size_analysis.py \
  --model baseline=reports/p0_diagnostics/validation/baseline/predictions.json \
  --model oversampling=reports/p0_diagnostics/validation/oversampling/predictions.json \
  --model finetune=reports/p0_diagnostics/validation/finetune/predictions.json \
  --classes pedestrian rider car motorcycle bicycle

# Locate + render every rider -> pedestrian confusion instance, Baseline and Oversampling
python scripts/find_specific_confusions.py \
  --model baseline=runs/detect/runs/bdd100k_yolo11n_baseline/weights/best.pt \
  --model oversampling=runs/detect/runs/bdd100k_yolo11n_oversampling/weights/best.pt \
  --true-class rider --pred-class pedestrian \
  --output-dir reports/p0_diagnostics/error_audit/rider_pedestrian_confusions \
  --conf 0.25 --iou 0.70 --match-iou 0.50 --imgsz 960 --device 0
```

Outputs: `reports/p0_diagnostics/validation/size_breakdown_by_class.csv`,
`reports/p0_diagnostics/error_audit/rider_pedestrian_confusions/` (per-model `*_full.jpg` /
`*_crop.jpg` images + `rider_to_pedestrian_instances.csv`).

## 1. Size-stratified AP/AR (COCO small <32², medium 32²-96², large >96² px, on
original-resolution boxes)

| class | model | AR_small | AR_medium | AR_large | n_small / n_medium / n_large |
|---|---|---:|---:|---:|---|
| car | baseline | 0.342 | 0.666 | 0.872 | 3410 / 2632 / 1359 |
| pedestrian | baseline | 0.299 | 0.543 | 0.652 | 495 / 513 / 75 |
| rider | baseline | 0.171 | 0.403 | 0.590 | 17 / 32 / 10 |
| rider | oversampling | 0.159 | 0.372 | 0.470 | 17 / 32 / 10 |
| motorcycle | baseline | 0.133 | 0.383 | 0.433 | 15 / 18 / 12 |
| motorcycle | oversampling | **0.207** | 0.400 | 0.458 | 15 / 18 / 12 |
| motorcycle | finetune | **0.040** | 0.356 | 0.442 | 15 / 18 / 12 |
| bicycle | baseline | 0.117 | 0.448 | 0.338 | 18 / 48 / 8 |
| bicycle | oversampling | 0.128 | 0.446 | 0.400 | 18 / 48 / 8 |
| bicycle | finetune | **0.033** | 0.406 | 0.325 | 18 / 48 / 8 |

**Finding 1 — the small-object gap is universal, not rare-class-specific.** Even `car`
(3,410 small-GT instances, the best-resourced class) only reaches 34.2% recall on small
instances vs 87.2% on large ones. The same ~2-3x small-vs-large recall gap shows up for
every class checked. This rules out "it's only bad because rider/motorcycle are rare"
as the full story — there's a real, size-correlated detection ceiling that resolution
(imgsz) and architecture (no small-object head) impose on every class, rare or not.

**Finding 2 — oversampling's real, confirmed effect is concentrated on *small*
motorcycles specifically** (AR_small +7.4pt, vs +1.7pt medium / +2.5pt large) — the
single cleanest, most size-localized signal in this whole investigation.

**Finding 3 — Fine-tuning's checkpoint is catastrophically bad on small rare-class
objects specifically** (motorcycle AR_small 0.133→0.040, bicycle 0.117→0.033 — far
worse than its generic degradation elsewhere), reinforcing that this checkpoint is
broken, not just "a bit worse."

## 2. Rider → pedestrian confusion — every instance found, Baseline (4) + Oversampling (9)

For every one of the 13 instances: **zero competing rider-class prediction existed
anywhere near the ground-truth box**, even below the matching IoU threshold — this
rules out "weak correct signal lost to a confident wrong one" and confirms these are
genuine, confident misclassifications, not localization/competition artifacts.

| cause bucket | evidence (see full writeup in chat) |
|---|---|
| **Small/distant object** | 3 instances, GT box area 608–1,425 px² (0.065–0.15% of frame) — genuinely tiny/low-contrast even to a human eye. |
| **Ground-truth annotation ambiguity** | 1 instance (`a48909f7-54e4bcff.jpg`): a person plainly walking across a crosswalk, no bicycle/motorcycle visible anywhere in frame, labeled `rider`. The model's "pedestrian" call is arguably correct. |
| **Confident misclassification on a large, unoccluded, easy instance — viewpoint-driven** | `c96e0b6a-f3d8aa78.jpg`: a large (9.5% of frame), confirmed unoccluded/untruncated rear-view motorcyclist, called "pedestrian" at 0.86 (baseline) and **0.89** (oversampling) confidence. Viewing the rider from directly behind visually hides the motorcycle body behind the rider's own torso/legs. |
| **Confident misclassification on a large, unoccluded, easy instance — oversampling-introduced** | `4c1d2408-79034cda.jpg`, **oversampling only**: a large, clearly pedaling cyclist in daylight, called "pedestrian 0.68". Not a baseline error — this is a regression specifically attributable to oversampling. |

Oversampling did not fix any of Baseline's 4 original mistakes (all 4 recur, and the
rear-view-motorcyclist case got *more* confidently wrong: 0.86→0.89) and added 5 new
ones, including one previously-correct, visually unambiguous case.
