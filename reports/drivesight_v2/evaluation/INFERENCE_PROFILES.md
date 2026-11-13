# DriveSight V2 — Frozen Inference Profiles

Canonical definitions: `scripts/drivesight_v2/inference_profiles.py` (single
source of truth — import `PROFILES` from there; do not redefine these
thresholds elsewhere). Both calibrated exclusively on the **validation** split
against `runs/drivesight_v2/v2_cls_pw025/weights/best.pt` (epoch 91). Evaluation
settings both require: imgsz=1152, NMS iou=0.70, match-iou=0.50.

## `balanced_max_f1`

| pedestrian | rider | bicycle | motorcycle | car | bus | truck | traffic light | traffic sign |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.20 | 0.25 | 0.25 | 0.35 | 0.35 | 0.25 | 0.30 | 0.30 |

Each class's own empirically-best-F1 threshold. Best overall precision of any
profile evaluated; best rider→pedestrian confusion count (8, vs 9 at a flat
0.25). **Use when overall precision/balance matters more than maximizing any
one class's recall.**

## `vru_recall_bicycle`

| pedestrian | rider | bicycle | motorcycle | car | bus | truck | traffic light | traffic sign |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.20 | **0.15** | 0.25 | 0.35 | 0.35 | 0.25 | 0.30 | 0.30 |

Identical to `balanced_max_f1` except bicycle, lowered to 0.15. On validation:
+6 bicycle TP (+8.1pt recall) for +27 bicycle FP (4.5 FP per recovered
bicycle), proven isolated — every other class and the rider→pedestrian
confusion count (still 8) are unaffected. **Use when bicycle recall is
prioritized over bicycle precision** — e.g. a risk-estimation objective where a
missed cyclist is judged costlier than an extra false alarm.

## Known limitations of both profiles

- **Calibrated for `v2_cls_pw025` only.** Re-run the calibration
  (`scripts/drivesight_v2/threshold_strategy_eval.py`) if the reference
  checkpoint changes — the earlier rounds already showed per-class optimal
  thresholds shift meaningfully between checkpoints (e.g. rider's best-F1 point
  moved from 0.15 under `v2_baseline` to 0.20 under `v2_cls_pw025`).
- **Neither profile fixes the rider→pedestrian confusion** — both carry it
  forward at whatever rate the underlying model produces (8 on validation);
  thresholding can only filter predictions, not correct a classification error.
  Only a training-time change (e.g. MixUp, still pending approval) can address
  that directly.
- **Small validation sample sizes for rider/motorcycle/bicycle** (59/45/74
  instances) mean single-digit TP/FP swings move percentages by several points
  — treat any one class's exact number as indicative, not precise, especially
  rider and motorcycle.
- **Neither profile has been re-optimized jointly** — each class's threshold
  was chosen independently (or, for bicycle in `vru_recall_bicycle`, changed in
  isolation specifically to avoid the cross-class interaction found in the
  rejected hybrid policy). A true joint optimization across all 9 classes
  simultaneously was not attempted and could in principle find a better point,
  at the cost of the clean single-variable attribution these two profiles
  currently have.
- **conf=0.05 floor**: both profiles are derived by filtering a single
  low-confidence (0.05) inference pass. Any class whose true useful signal
  lives below 0.05 (unlikely, but not verified) would be invisible to this
  calibration regardless of threshold choice.
