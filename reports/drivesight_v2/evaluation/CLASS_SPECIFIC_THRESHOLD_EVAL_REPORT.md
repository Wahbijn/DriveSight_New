# Class-Specific Confidence Threshold Evaluation (Action 1)

Checkpoint: `runs/drivesight_v2/v2_baseline/weights/best.pt` (epoch 91) — **not
retrained, not modified**; mtime verified unchanged before and after this run
(2026-11-08 03:09:54). **Val split only** — test was never touched. Separate
experiment, saved entirely under
`reports/drivesight_v2/evaluation/class_specific_thresholds/`, no existing results
overwritten. Tooling: `scripts/drivesight_v2/class_specific_threshold_eval.py` (new).

Method: inference ran **once** per image at conf=0.05 (requirement 1); both policies
below were then derived by filtering that single candidate list per-class in Python
— exact for standard per-class greedy NMS, not an approximation, so both policies are
evaluated against literally identical underlying detections (requirement 2-3, same
IoU/match criteria — iou=0.70 NMS, match-iou=0.50 — as every prior V2 evaluation).

- **`global_025`**: conf≥0.25 for all 9 classes (the existing baseline policy).
- **`mixed_vru_015`**: conf≥0.25 for pedestrian/car/truck/bus/traffic light/traffic
  sign, conf≥0.15 for rider/motorcycle/bicycle (the proposed Action 1 policy).

## AP50 / AP50-95 — threshold-independent, unchanged, reported once

mAP50/mAP50-95 integrate over the full confidence range and do not depend on a
deployment cutoff — they are **identical for both policies** because both come from
the same underlying model and predictions. Already measured for v2_baseline (no need
to recompute): **mAP50 = 52.6%, mAP50-95 = 28.9%** (overall Precision 60.5%, Recall
49.3% at Ultralytics' own max-F1 point — `reports/drivesight_v2/evaluation/V2_BASELINE_EVALUATION_REPORT.md`).
Everything below is the threshold-*dependent* comparison these two fixed policies
produce, which is a different, complementary question.

## 4-5. Measured overall Precision/Recall/F1 (`policy_comparison.csv`)

| | precision | recall | f1 | fp | fn |
|---|---:|---:|---:|---:|---:|
| macro, global_025 | 63.0% | 48.4% | 0.548 | 3,557 | 4,612 |
| macro, mixed_vru_015 | 59.9% | **51.0%** | 0.551 | 3,599 | 4,598 |
| micro, global_025 | 71.2% | 65.6% | 0.683 | 3,557 | 4,612 |
| micro, mixed_vru_015 | 71.0% | 65.7% | 0.683 | 3,599 | 4,598 |

Macro recall (+2.6pt) moves because the three reweighted classes are 3 of 9 in that
average; micro recall barely moves (+0.1pt) because rider/motorcycle/bicycle are a
tiny fraction of total instances (178 of 13,420) — `car` dominates the
instance-weighted view regardless of what happens to the rare classes. Both are real,
correctly computed, and answer different questions; neither is "more correct."

## Per-class, the three reweighted classes (full table: `policy_comparison.csv`)

| class | metric | global_025 | mixed_vru_015 | Δ |
|---|---|---:|---:|---:|
| rider | precision | 58.6% | 50.0% | -8.6pt |
| rider | recall | 28.8% | **35.6%** | **+6.8pt** |
| rider | f1 | 0.386 | 0.416 | +0.030 |
| rider | fp | 12 | 21 | +9 |
| motorcycle | precision | 37.1% | 29.1% | -8.0pt |
| motorcycle | recall | 28.9% | **35.6%** | **+6.7pt** |
| motorcycle | f1 | 0.325 | 0.320 | -0.005 |
| motorcycle | fp | 22 | 39 | +17 |
| bicycle | precision | 60.5% | 49.2% | -11.3pt |
| bicycle | recall | 31.1% | **40.5%** | **+9.4pt** |
| bicycle | f1 | 0.411 | 0.444 | +0.033 |
| bicycle | fp | 15 | 31 | +16 |

**Correction to my earlier prediction**: in the recall-diagnosis report I estimated
rider's recall gain at +10.2pt (to ~39.0%). The actual measured gain, from an
independently-written evaluation script against the real predictions, is **+6.8pt
(to 35.6%)** — I misread the threshold-0.15 row of the earlier sweep. Motorcycle
(+6.7pt, predicted +6.7pt) and bicycle (+9.4pt, predicted +9.4pt) matched the earlier
estimate closely. Flagging this discrepancy explicitly, per the instruction to report
actual results rather than predictions.

car/truck/bus/pedestrian/traffic light/traffic sign are **byte-for-byte identical**
between the two policies (their threshold didn't change) — confirmed in
`policy_comparison.csv`.

## Confusion matrix, focus on rider → pedestrian

Full matrices: `confusion_matrix_global_025.csv`, `confusion_matrix_mixed_vru_015.csv`.

| true class | predicted | global_025 | mixed_vru_015 |
|---|---|---:|---:|
| rider | pedestrian | **10** | **10** |
| rider | bicycle | 1 | 1 |
| rider | motorcycle | 1 | 0 |
| rider | correct (rider) | 17 | 21 |
| rider | missed (background) | 30 | 27 |

**rider→pedestrian confusion is exactly unchanged (10 in both policies)** — expected
and mechanistically clean: that confusion is governed by `pedestrian`'s own
threshold (unchanged at 0.25), not rider's. Lowering rider's threshold cannot fix a
classification error made by a confident wrong-class prediction; it only recovers
previously-missed *correct* rider detections. The 4 additional correct riders (17→21)
come entirely from recovered misses (30→27) and one resolved rider→motorcycle mix-up
(1→0) — not from reducing rider→pedestrian confusion. **Action 1 does not, and
structurally cannot, address the rider/pedestrian classification problem** — that
remains squarely a job for the `cls_pw`/`mixup` experiments (or a dedicated fix for
that specific confusion).

The same clean pattern holds for motorcycle and bicycle: every one of their
pre-existing confusion pairs (motorcycle→bicycle, bicycle→motorcycle/car) is
unchanged between policies — all the recall gain comes from recovering true misses,
none from resolving misclassifications.

## 6. Does class-specific filtering introduce unacceptable false positives?

| class | FP global_025 | FP mixed_vru_015 | absolute Δ | relative Δ |
|---|---:|---:|---:|---:|
| rider | 12 | 21 | +9 | +75% |
| motorcycle | 22 | 39 | +17 | +77% |
| bicycle | 15 | 31 | +16 | +107% |

Relative increases are large (75-107%), but two mitigating facts, both directly
measured above, not assumed:
1. **Every new false positive is an unmatched/background detection, not a new
   misclassification** — the confusion-matrix comparison shows zero change in any
   wrong-class confusion pair for these three classes. The cost is phantom boxes
   with no real object there, not a real object being mislabeled as something it
   isn't.
2. **Absolute volume is small**: 42 extra false-positive boxes total, across 686
   validation images, split across 3 classes.

Whether this is "acceptable" is a judgment call that depends on the downstream use:
for a *risk-estimation* system (this project's stated objective) where a spurious
VRU flag costs a slightly-elevated risk score and a missed one costs a real blind
spot, trading precision for recall on exactly these three classes is a defensible
choice — but it is a real precision cost (8.0-11.3pt per class) and should be stated
as such, not minimized.

## Summary

- Actual measured gains: rider +6.8pt recall, motorcycle +6.7pt, bicycle +9.4pt —
  real, but smaller for rider than my earlier (incorrect) prediction.
- Zero effect on AP50/AP50-95 (expected, threshold-independent) and zero effect on
  the other 6 classes (their threshold didn't change).
- The rider→pedestrian confusion is completely unaffected — confirms this policy
  change is not a substitute for the `cls_pw`/`mixup` experiments.
- New false positives are real (75-107% relative increase on the 3 classes) but are
  background noise, not new misclassifications, and small in absolute count.
- **Recommendation**: adopt `mixed_vru_015` as the deployment policy — it is a free
  (no retraining), real, measured recall improvement on the three priority classes
  with a quantified and bounded precision cost — while treating it as a complement
  to, not a replacement for, the still-pending `cls_pw`/`mixup` training experiments,
  which remain the only levers that can touch the rider/pedestrian confusion itself.

No training was started. Awaiting approval before `cls_pw` or `mixup`.
