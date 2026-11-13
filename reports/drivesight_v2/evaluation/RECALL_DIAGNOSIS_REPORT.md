# V2 Baseline Recall Diagnosis

Checkpoint: `runs/drivesight_v2/v2_baseline/weights/best.pt` (epoch 91, untouched,
verified unchanged throughout: mtime 2026-11-08 03:09:54). Read-only: no training, no
dataset change. Tooling: `scripts/drivesight_v2/recall_diagnosis.py` (new) +
`scripts/compare_error_audits.py` (reused). Raw outputs:
`reports/drivesight_v2/evaluation/recall_diagnosis/`.

Methodology note: Ultralytics' own reported P/R (the "49.3% Recall" headline) is
measured at the confidence that maximizes F1 on the PR curve, not at a literal fixed
threshold. Everything below instead runs inference once at conf=0.05 and derives
every higher threshold's result by filtering that same candidate list (exact for
standard per-class greedy NMS, not an approximation) -- so these are the real P/R/F1
you'd get by actually deploying at each threshold.

## 4-5. Confidence threshold sweep (overall + per-class P/R/F1)

Full table: `recall_diagnosis/threshold_sweep_precision_recall_f1.csv`.

| threshold | macro P | macro R | macro F1 | micro P | micro R | micro F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 28.8% | 61.6% | 0.392 | 34.2% | 78.9% | 0.477 |
| 0.10 | 39.4% | 57.4% | 0.467 | 46.9% | 75.1% | 0.578 |
| 0.15 | 48.1% | 54.6% | 0.512 | 56.4% | 71.8% | 0.632 |
| 0.20 | 53.4% | 51.5% | 0.524 | 63.7% | 68.8% | 0.662 |
| 0.25 | 58.2% | 48.4% | 0.529 | 70.0% | 65.6% | 0.677 |

Macro (unweighted mean over 9 classes, matches the project's "49.3%" convention) and
micro (instance-weighted, dominated by `car`'s 7,401 val instances) tell different
stories: micro recall is far higher at every threshold because `car` alone is doing
well; macro is what actually reflects the rare-class problem.

**VRU classes individually** (full data in the CSV):

| class | best-F1 threshold | F1 @ 0.05 | F1 @ 0.25 | Recall @ 0.05 | Recall @ 0.25 |
|---|---:|---:|---:|---:|---:|
| pedestrian | 0.25 | 0.421 | **0.618** | 70.9% | 55.5% |
| rider | **0.15** | 0.323 | 0.366 | 42.4% | 28.8% |
| motorcycle | **0.15** | 0.213 | 0.306 | 40.0% | 28.9% |
| bicycle | **0.15-0.20** | 0.262 | 0.393 | 48.6% | 31.1% |

**Pedestrian's F1 keeps improving all the way to 0.25 — but rider/motorcycle/bicycle
all peak at a *lower* threshold (0.15) than the project's standard 0.25.** A single
universal operating point is not well-suited to this class mix: 0.25 is right for
pedestrian, wrong for the three rarest VRU classes.

## 1-2. Missed detections by class (`recall_diagnosis/missed_object_causes.csv`, at conf=0.25)

| class | missed (FN) | % of all misses |
|---|---:|---:|
| car | 1,923 | 43.8% |
| traffic sign | 1,058 | 24.1% |
| traffic light | 689 | 15.7% |
| pedestrian | 469 | 10.7% |
| truck | 114 | 2.6% |
| bicycle | 48 | 1.1% |
| bus | 30 | 0.7% |
| rider | 30 | 0.7% |
| motorcycle | 27 | 0.6% |

By raw volume, **car + traffic sign + traffic light account for 83.6% of all missed
detections** — not the VRU classes. This matters for the "overall Recall" headline
number (which is macro-averaged and gives rider the same weight as car), but the VRU
classes' *relative* miss rate (missed / total instances) is far worse: rider 50.8%
(30/59), motorcycle 60.0% (27/45), bicycle 64.9% (48/74) vs car 26.0% (1923/7401).

## 3 & 6. Cause of miss, by class and size (reference threshold = 0.25)

| class | low_confidence | poor_localization | true_miss | dominant size bucket |
|---|---:|---:|---:|---|
| car | 45.7% | 39.9% | 14.4% | 80.7% small |
| traffic sign | 38.9% | 19.5% | 41.6% | 84.4% small |
| traffic light | 37.9% | 32.4% | 29.8% | 93.9% small |
| pedestrian | 36.2% | 24.1% | 39.7% | 69.1% small |
| rider | 20.0% | 10.0% | **70.0%** | 46.7% small / 46.7% medium |
| motorcycle | 14.8% | 18.5% | **66.7%** | 51.9% small |
| bicycle | 27.1% | 20.8% | **52.1%** | 33.3% small / 58.3% medium |

**Two distinct failure regimes, cleanly separated by the data:**
- **car / traffic sign / traffic light / pedestrian**: dominated by `low_confidence`
  + `poor_localization` (63.6-85.6% combined) and are mostly small objects. The model
  *is* finding these objects, just under-confident or imprecisely boxed — a
  calibration/localization problem, not a detection-capacity problem.
- **rider / motorcycle / bicycle**: dominated by `true_miss` (52-70%) and are **not**
  predominantly small (rider/bicycle are split small/medium, motorcycle just over
  half small) — confirms the earlier project finding that these classes' problem is
  NOT primarily small-object detection. No right-class prediction exists at *any*
  confidence down to 0.05 for most of their misses — lowering the deployment
  threshold (the pedestrian/car-style fix) has limited headroom here, matching the
  threshold-sweep table above.

**Occlusion**: 59.7% of all missed objects (with recoverable attribute data, 84.7%
coverage) were occluded in the original BDD100K annotation. For VRU classes this
looks dramatic (rider 100%, bicycle 91.9%, motorcycle 87.0%, pedestrian 72.2%
occluded) -- **but I checked this against the baseline occlusion rate for ALL val
instances of these classes, not just missed ones, and it's a confound, not a clean
signal**: rider is 100% occluded in val *overall* (53/53 instances), bicycle 91.2%,
motorcycle 82.9% -- essentially identical to the missed-instance rates. Occlusion is
a pervasive characteristic of how these classes appear in BDD100K generally, not a
differentiator of which specific instances get missed. **Pedestrian is the one
exception**: baseline occlusion is 58.8% but missed pedestrians run 72.2% occluded
(+13.4pt) -- a real, if modest, signal specific to that class.

**Dataset annotation problems**: the one quantified issue remains the
rider-without-visible-vehicle pattern from notebook 02 (6.5% of rider-containing
images). Visual spot-checks of two large (hence visually legible), occluded true-miss
cases (`160d9039-7f5b9f08` motorcycle, `1d981a62-27584b3b` rider,
`recall_diagnosis/<stem>/zoom_grid.jpg`) found no annotation-placement errors -- both
are genuine, visually-confirmable hard cases (a motorcycle substantially hidden
behind another object; a rider partly hidden behind a street pole at dusk, which the
model confidently mislabels `pedestrian 0.80` -- the same rider/pedestrian confusion
documented throughout this project, now also linked to occlusion-driven visual
ambiguity).

## Biggest causes of low Recall, ranked

1. **True detection failure (not threshold, not box precision) on rider/motorcycle/
   bicycle** -- 52-70% of their misses have zero model signal at any confidence down
   to 0.05. This is the dominant, specific driver of why "49.3% overall Recall" is
   still insufficient for the VRU-priority objective, and it's a capacity/
   representation problem rooted in the known data scarcity (166-399 train
   instances), not something a threshold or NMS tweak can fix.
2. **A single global confidence threshold (0.25) is mismatched to the class mix.**
   Pedestrian benefits from 0.25; rider/motorcycle/bicycle all peak at 0.15. This
   alone is costing real, recoverable recall on exactly the classes the objective
   prioritizes.
3. **Calibration/localization issues dominate the highest-volume misses** (car,
   traffic sign, traffic light -- 83.6% of all missed objects by count), but these
   are the "easier" failure mode (right class, just low-confidence or imprecisely
   boxed) and already the least-bad part of the picture.

## Three most promising actions

1. **Switch to per-class confidence thresholds, biased toward recall for VRU
   classes** -- zero retraining, testable immediately. Concretely: keep ~0.25 for
   pedestrian/car/truck/bus/traffic-light/traffic-sign, drop to ~0.10-0.15 for
   rider/motorcycle/bicycle. For a risk-*estimation* system (not a final bounding-box
   product), trading some VRU precision for materially better VRU recall is squarely
   justified. Expected effect, read directly off the sweep table: rider recall
   28.8%→39.0% (+10.2pt), motorcycle 28.9%→35.6% (+6.7pt), bicycle 31.1%→40.5%
   (+9.4pt) by moving those three classes to conf=0.15, with no change to the other
   six classes.
2. **Run the already-approved `cls_pw` / `mixup` ablations next.** These are
   exactly the right levers for a `true_miss`-dominated, data-scarcity-rooted
   problem (class-imbalance loss reweighting and augmentation diversity both target
   detection capacity, not confidence calibration) -- this diagnosis makes the case
   for them stronger than before, not weaker.
3. **Investigate further training / stronger localization weighting for the
   high-volume, calibration-dominated classes** (car/traffic-sign/traffic-light,
   83.6% of all misses by count): the V2 baseline was paused at epoch 91 with
   patience=30 not yet exhausted (best was epoch 91, 9 epochs since improvement when
   paused) -- it may not have fully converged. Since these classes' problem is
   low_confidence/poor_localization rather than true_miss, more training and/or a
   higher `box=` loss weight are plausible, low-risk levers, and any localization
   improvement benefits every class including the VRU ones.

Action 1 requires no new experiment and I'd recommend validating it immediately
(re-run the audit at conf=0.15 for the three VRU classes to confirm the sweep table's
numbers hold up end-to-end). Actions 2-3 are new training experiments and, per your
standing instruction, await your explicit go-ahead before I launch anything.
