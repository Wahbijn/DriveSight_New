# v2_cls_pw vs v2_baseline — Evaluation Report

Checkpoints: `runs/drivesight_v2/v2_baseline/weights/best.pt` (epoch 91, untouched)
vs `runs/drivesight_v2/v2_cls_pw/weights/best.pt` (epoch 46, `cls_pw=0.5`, training
completed naturally — EarlyStopping at epoch 76, no improvement in 30 epochs,
301 min / ~5h, zero kills this run). Both evaluated at imgsz=1152, same val split,
same matching criteria (iou=0.70 NMS, match-iou=0.50) as every prior V2 comparison.
Raw outputs: `reports/drivesight_v2/evaluation/cls_pw_vs_baseline/`.

**Headline: this is a real trade-off, not a clean win.** cls_pw=0.5 produces a
strong motorcycle improvement and the first measured reduction in the
rider→pedestrian confusion anywhere in this project — but at the cost of broad
regressions on most other classes, including a 7.7pt recall drop on pedestrian.
Reported plainly below, not minimized.

## Overall Precision/Recall (threshold-dependent, Ultralytics' own max-F1 point)

| | v2_baseline | v2_cls_pw | Δ |
|---|---:|---:|---:|
| Precision | 60.5% | **63.9%** | +3.4pt |
| Recall | **49.3%** | 45.8% | -3.6pt |
| mAP50 | **52.6%** | 51.9% | -0.8pt |
| mAP50-95 | **28.9%** | 28.4% | -0.5pt |

Precision rose, but Recall, mAP50 and mAP50-95 all fell. cls_pw traded recall for
precision overall — the opposite of what the project needs most.

## Per-class AP50 / AP50-95 (threshold-independent)

| class | AP50 baseline | AP50 cls_pw | Δ | AP50-95 baseline | AP50-95 cls_pw |
|---|---:|---:|---:|---:|---:|
| pedestrian | 61.2% | 60.6% | -0.6pt | 30.2% | 29.1% |
| **rider** | 41.0% | 40.4% | -0.6pt | 20.3% | 18.0% |
| car | 78.6% | 77.2% | -1.5pt | 49.2% | 48.1% |
| truck | 48.5% | 49.3% | +0.9pt | 33.3% | 34.5% |
| bus | 56.5% | 52.6% | **-3.9pt** | 42.7% | 39.2% |
| **motorcycle** | 29.7% | **39.9%** | **+10.2pt** | 14.7% | 20.4% |
| **bicycle** | 37.7% | 33.4% | **-4.4pt** | 13.5% | 13.4% |
| traffic light | 60.7% | 57.0% | **-3.7pt** | 24.5% | 23.3% |
| traffic sign | 59.8% | 56.4% | **-3.4pt** | 31.7% | 29.2% |

Only motorcycle and truck improved. Six of nine classes regressed, three
(bus, traffic light, traffic sign) by more than the 2pt threshold this project has
used throughout to flag a "material" regression.

## Rider → pedestrian confusion — the one clear, specific win

| | v2_baseline | v2_cls_pw |
|---|---:|---:|
| **rider → pedestrian** | **10** | **5** |

**Cut in half.** This is the first thing in this entire project — not resolution
(960/1152/1280), not the earlier naive oversampling, not class-specific confidence
thresholds — that has actually moved this specific confusion. Consistent with the
pre-training mechanism explanation: rider's classification loss weight (1.91x mean)
pulls harder toward activating the rider channel on true rider instances, and this
shows up as a real, measured drop in rider being mistaken for pedestrian. The
reverse direction (pedestrian→rider) moved slightly the other way (4→6), a small
and expected side effect of the same pull.

## False positives / false negatives (`error_audit/comparison_*.csv`, conf=0.25)

| class | FN baseline | FN cls_pw | FP baseline | FP cls_pw |
|---|---:|---:|---:|---:|
| pedestrian | 469 | 546 (+77) | 248 | 137 (-111) |
| rider | 30 | 34 (+4) | 12 | 16 (+4) |
| car | 1,923 | 2,005 (+82) | 1,745 | **2,122 (+377, +21.6%)** |
| truck | 114 | 124 (+10) | 99 | 100 (+1) |
| bus | 30 | 48 (+18) | 29 | 45 (+16) |
| **motorcycle** | 27 | 23 (-4) | **22** | **3 (-19, -86%)** |
| bicycle | 48 | 46 (-2) | 15 | 24 (+9) |
| traffic light | 689 | 808 (+119) | 682 | 536 (-146) |
| traffic sign | 1,058 | 1,205 (+147) | 705 | 641 (-64) |

**Car's false positives jumped 21.6%** (+377) — this is exactly the risk flagged
*before* training (item 5 of the pre-training analysis): de-weighting car's
classification loss (weight 0.17x mean) reduces the penalty for incorrectly firing
the car channel, and the data confirms it. **Motorcycle's false positives nearly
vanished** (22→3) alongside its AP50 gain — a clean, consistent improvement on every
motorcycle metric simultaneously.

## Bicycle / motorcycle / rider — the three priority classes, in detail

- **Motorcycle: a clean, unambiguous win.** AP50 +10.2pt, AP50-95 +5.7pt, FN -4,
  FP -19 (-86%). Every metric moved the same direction. This is the single strongest
  result in the whole V2 project so far for this class.
- **Rider: a wash, with one specific bright spot.** AP50 essentially flat (-0.6pt),
  AP50-95 down (-2.3pt), FN +4, FP +4 — but the rider→pedestrian confusion
  specifically dropped from 10 to 5. The overall number doesn't move much because
  the gain from fewer rider→pedestrian misclassifications is offset by a small rise
  in plain misses and false positives elsewhere.
- **Bicycle: genuinely mixed, not simply worse.** AP50 (threshold-independent, -4.4pt)
  and AP50-95 (flat) say this got worse. But the fixed-threshold audit (conf=0.25)
  shows recall actually *improved* (31.1%→35.1%) and FN dropped slightly (48→46) —
  the cost shows up as FP (+9, +60%) and, more importantly, in the shape of the
  precision-recall curve: more low-confidence true positives are now present (which
  genuinely helps recall at this one fixed threshold) but the curve's higher-
  confidence region is apparently less precise, which is what drags the full-range
  AP50 down. Both observations are real and correctly measured; they are not
  contradictory, they are describing different parts of the same curve.

## Does this pass the project's own regression bar?

The V2 plan's model-selection rule (declared before this round, for V1-vs-V2) was:
*no AP50 regression >2pt on car/truck/bus/traffic light/traffic sign*. Applying that
same bar one level deeper, v2_cls_pw **fails it**: bus (-3.9pt), traffic light
(-3.7pt), and traffic sign (-3.4pt) all exceed 2pt, and car (-1.5pt) is close.
Pedestrian recall dropping 7.7pt at the fixed operating point is also a real cost to
a class the project needs to keep, not just the four explicitly-named priority
classes.

## Conclusion

`cls_pw=0.5` should **not** simply replace `v2_baseline` — it fails this project's
own "no material regression" standard on three non-priority classes and reduces
pedestrian recall substantially. But it is not a failed experiment either: the
motorcycle result and the rider→pedestrian confusion reduction are real, specific,
measured wins that validate the underlying mechanism (classification-loss
reweighting can move a confusion pattern that resolution and thresholding provably
cannot). The most informative next step is not to abandon class weighting but to
tune it less aggressively — e.g. `cls_pw=0.25` (weight ratio 3.7:1 vs 0.5's 13.4:1,
computed in the pre-training analysis) — to test whether a gentler reweighting keeps
most of the motorcycle/rider-confusion gain while costing less on bus/traffic-light/
traffic-sign/pedestrian. That, or combining a lighter `cls_pw` with `mixup` (the
originally-planned `v2_combined` run), are reasonable candidates for the next
experiment — pending your approval, not started.

All V2 baseline and v2_cls_pw files preserved; nothing else was touched.
