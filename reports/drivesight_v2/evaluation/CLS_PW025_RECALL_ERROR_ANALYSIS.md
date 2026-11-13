# v2_cls_pw025 — Recall Error Analysis (VALIDATION split only)

**Analysis-only.** No training launched, no checkpoint/dataset/label modified, test
split never touched (checked: zero references to `test` in any command this round).
Checkpoint `runs/drivesight_v2/v2_cls_pw025/weights/best.pt` verified unchanged
throughout. Tooling: `scripts/drivesight_v2/recall_diagnosis.py`, extended this
round to add an explicit "incorrect classification" cause category (previously,
GT instances matched to the *wrong* class were silently excluded from the
cause-of-miss breakdown — they are now counted, since a wrong-class match is still
a false negative for the true class). Raw outputs:
`reports/drivesight_v2/evaluation/cls_pw025_recall_diagnosis/`.

## 1. Per-class Recall and biggest contributors to missed detections

Official per-class Recall (Ultralytics, max-F1 point, from the already-saved
`cls_pw025_vs_others/summary_per_class_metrics.csv` — reused, not recomputed):

| class | Recall | missed (FN) @conf=0.25 | % of all misses |
|---|---:|---:|---:|
| car | 70.1% | 1,975 | 43.3% |
| traffic sign | 52.8% | 1,086 | 23.8% |
| traffic light | 55.6% | 706 | 15.5% |
| pedestrian | 56.6% | 439 | 9.6% |
| truck | 50.3% | 169 | 3.7% |
| bus | 51.7% | 72 | 1.6% |
| bicycle | 28.4% | 47 | 1.0% |
| rider | **39.0%** | 38 | 0.8% |
| motorcycle | **31.1%** | 30 | 0.7% |

Two different, both-correct answers depending on what "contributes most" means:
**by raw count**, car/traffic-sign/traffic-light/pedestrian dominate (91.2% of all
missed detections combined) — this is what drives the project's low *macro*
recall number the most in absolute terms. **By relative miss rate**, motorcycle
(68.9% of its instances missed) and bicycle (71.6%) are far worse than car
(29.9%) — this is what the stated VRU-priority objective cares about.

## 2. False-negative cause categorization (conf=0.25 reference, n=4,562 misses)

| cause | count | % | meaning |
|---|---:|---:|---|
| low_confidence | 1,805 | 39.6% | right class detected, IoU≥0.5, just below 0.25 |
| poor_localization | 1,293 | 28.3% | right class nearby, IoU too low to count |
| true_miss | 1,262 | 27.7% | no right-class signal at any confidence ≥0.05 |
| **incorrect_classification** | 202 | 4.4% | matched to the **wrong** class at IoU≥0.5 (new category) |

Per-class breakdown (%, full table: `missed_object_causes.csv`):

| class | incorrect_class. | low_conf. | poor_loc. | true_miss |
|---|---:|---:|---:|---:|
| **bus** | **47.2%** | 15.3% | 6.9% | 30.6% |
| **truck** | **39.1%** | 14.2% | 5.9% | 40.8% |
| **rider** | **31.6%** | 15.8% | 2.6% | 50.0% |
| motorcycle | 20.0% | 3.3% | 10.0% | 66.7% |
| bicycle | 4.3% | 29.8% | 14.9% | 51.1% |
| pedestrian | 2.3% | 41.2% | 19.6% | 36.9% |
| car | 1.8% | 44.9% | 38.4% | 14.9% |
| traffic sign | 1.7% | 38.1% | 19.4% | 40.7% |
| traffic light | 2.4% | 38.0% | 30.0% | 29.6% |

**Two distinct failure regimes, now sharper than before**: car/traffic-sign/
traffic-light/pedestrian are overwhelmingly `low_confidence`+`poor_localization`
(63.6-83.3% combined) — detected, just miscalibrated or imprecisely boxed.
Rider/motorcycle/bicycle are dominated by `true_miss` (50-67%) — genuine
non-detection. **New finding this round**: bus and truck have the *highest*
`incorrect_classification` rates of any class (47.2%, 39.1%) — almost entirely
within the car/truck/bus visual-similarity cluster (truck→car 59, bus→truck 19,
bus→car 15) — a real, separate problem from the VRU confusion, consistent with
this cluster dominating every confusion matrix produced in this whole project.

**Small/distant vs occlusion** (full table in the script output): rider/bicycle
misses are *not* predominantly small (rider: 42% small/50% medium/8% large;
bicycle: 32% small/60% medium/9% large) — confirms, again, that the rare-VRU
problem is not primarily a small-object problem, unlike pedestrian (70.6% small)
and traffic sign/light (84-93% small). Occlusion among missed VRU remains very
high (rider 100%, bicycle 94%, motorcycle 81%) but — as established in the
equivalent v2_baseline analysis — this closely tracks these classes' *overall*
occlusion rate in BDD100K (rider is ~100% occluded project-wide), so it is a
pervasive dataset characteristic, not a specific differentiator of which
instances get missed.

## 3. VRU focus, and rider → pedestrian specifically

- **Rider**: 50.0% true_miss, **31.6% incorrect_classification — and 9 of rider's
  12 misclassifications (75%) are specifically rider→pedestrian.** This confirms
  rider→pedestrian is not just *a* confusion pattern, it is *the* dominant specific
  classification error for this class, now precisely quantified rather than
  inferred from the confusion matrix alone.
- **Motorcycle**: 66.7% true_miss (the highest of any class) — this is overwhelmingly
  a non-detection problem, barely a confidence problem (only 3.3% low_confidence).
- **Bicycle**: 51.1% true_miss, but a notably higher low_confidence share (29.8%)
  than motorcycle — bicycle's remaining gap has more recoverable-by-threshold
  headroom than motorcycle's does.

## 4. Confidence threshold sweep — a genuine shift from the earlier analysis

Extended sweep (0.05-0.45) to find each class's true best-F1 point, not just
check the 0.05-0.25 range used previously:

| class | best-F1 threshold | F1 at that point | (earlier v2_baseline best-F1 threshold) |
|---|---:|---:|---|
| pedestrian | 0.30 | 0.624 | 0.25 |
| **rider** | **0.20** | 0.505 | 0.15 |
| motorcycle | 0.25 | 0.455 | 0.15 |
| bicycle | 0.25 | 0.432 | 0.15-0.20 |

**This is an important, actionable finding**: cls_pw025's own training has
already absorbed much of what the earlier "Action 1" class-specific threshold
policy (0.25 standard / 0.15 for VRU) was compensating for. Applying that same
*old* policy on top of this *new* model would now push rider/motorcycle/bicycle
**past** their own best-F1 points — the right-sized class-specific policy for
this model is milder (rider ~0.20, motorcycle/bicycle ~0.25, i.e. close to the
project's standard threshold) or arguably not needed at all for motorcycle/
bicycle. **Any class-specific threshold policy must be re-derived per checkpoint,
not reused from a previous one** — confirmed here with real numbers, not assumed.

Full per-class P/R/F1/FP/FN at every threshold: `extended_sweep/
threshold_sweep_precision_recall_f1.csv`.

## 5. Visual evidence — three representative false negatives

All three images and exact boxes: `visual_evidence/<stem>/zoom_grid.jpg`.

1. **`2c9d86cc-0910b237`, motorcycle, large (15,859px²), *not* occluded, true_miss.**
   A parked/crashed motorcycle wedged against a car, visually merged with the
   vehicle's wheel well — genuinely hard even for a human to parse as "motorcycle"
   at first glance. The model predicts `car 0.83` for the scene and nothing else.
   **This rules out occlusion and size as the excuse** — it's a visually atypical
   pose/state the model has never learned to recognize, pointing at training-data
   diversity (BDD100K likely has very few wrecked/non-upright motorcycles), not a
   resolution or architecture limitation.

2. **`4e721b4b-7d171227`, rider, large (10,454px²), incorrect_classification
   (→pedestrian).** A person with a bicycle at dusk, dim red-tinted ambient light.
   Model confidently predicts `pedestrian 0.76`, no rider hypothesis at all. Same
   failure signature found for v2_baseline in the earlier analysis — **low-light
   conditions are a recurring, specific contributor to this exact confusion**, on
   top of the general viewpoint/pose ambiguity already documented.

3. **`1d981a62-27584b3b`, bicycle, large (12,783px²), low_confidence (best IoU
   match conf=0.05) — the same scene flagged as a rider true-miss for
   v2_baseline.** A person next to a bicycle, partly behind a street pole, dusk
   lighting. Now shows multiple fragmented, overlapping `pedestrian` predictions
   (0.31, 0.49, 0.77) and one weak `bicycle 0.22` (different, lower-IoU box than
   the one scored in the CSV, itself evidence of poor localization alongside low
   confidence). **This specific hard scene has now defeated every model variant
   tried in this project** — a recurring, identifiable hard-case family (dusk +
   partial occlusion by street furniture + rider/bicycle visual entanglement), not
   a one-off.

## 6. Improvement recommendations, ranked by evidence strength

| # | Recommendation | Lever | Why it should help (evidence) | How to measure |
|---|---|---|---|---|
| 1 | **Re-derive class-specific confidence thresholds for v2_cls_pw025 specifically** (rider≈0.20, pedestrian≈0.30, leave motorcycle/bicycle near 0.25) | Threshold calibration | Directly measured in this report (§4) — the old policy is now miscalibrated for this checkpoint | Re-run `class_specific_threshold_eval.py` with the new per-class values on val; confirm F1 gains before any deployment change |
| 2 | **Targeted data collection/augmentation for low-light and atypical-pose VRU instances** (dusk/night scenes, non-upright or partially-dismantled vehicles) | Dataset / augmentation | Two of three visual-evidence cases are specifically low-light; case 1 shows a non-standard motorcycle pose with zero model signal — these are diversity gaps, not volume or resolution gaps | Build a held-out slice of night/dusk VRU instances (already have weather/timeofday attributes from notebook 01); track recall on that slice specifically before/after |
| 3 | **A dedicated fix for the car/truck/bus visual-similarity cluster** (e.g. `cls_pw` applied more selectively, or additional training signal for this cluster specifically) | Dataset / loss design | Newly quantified this round: bus (47.2%) and truck (39.1%) have the *highest* incorrect-classification rates of any class — larger in relative terms than rider's — and this was not previously a stated priority | Track the truck↔car↔bus confusion-matrix block specifically across any future experiment, not just AP50 |
| 4 | **MixUp ablation (already approved in principle, not yet run)** | Augmentation | Targets `true_miss` directly by adding compositional diversity without oversampling's duplication artifacts — the dominant failure mode for rider/motorcycle/bicycle (50-67% of their misses) is exactly what more visual diversity is meant to address | Compare true_miss rate specifically (not just AP50) for VRU classes, pre/post |
| 5 | **Architecture change (P2 small-object head / YOLO26)** — *downgraded from earlier rounds* | Architecture | The visual evidence this round argues against this as the next lever: the motorcycle miss case was **large**, not small, and the dominant VRU problem (true_miss) is about visual diversity/representation, not resolution. A P2 head targets small objects specifically, which is not what's failing here | If still pursued later: track AR_small specifically (already shown to be fine for VRU — they're not small-object-dominated) vs AR_medium/large (where the actual gap is) |

## Central question — answered

**What is preventing `v2_cls_pw025` from achieving higher Recall?** Two separable
problems, not one: (a) a large-volume, low-severity calibration/localization gap
concentrated in car/traffic-sign/traffic-light/pedestrian (83.6-91.2% of all raw
misses, but the "easier" failure mode — these objects are mostly being found, just
under-confident or imprecisely boxed), and (b) a smaller-volume but much more
severe representation gap concentrated in rider/motorcycle/bicycle, where 50-67%
of misses are genuine non-detections driven by visual diversity gaps (low-light,
atypical poses) rather than confidence, localization, or object size.

**Most effective next step**: of the five ranked above, **#1 (re-derive
per-checkpoint thresholds)** is the only zero-cost, immediately-verifiable action
and should be done first, purely as validation — it is not expected to move the
true_miss-dominated VRU problem much on its own. **#4 (MixUp)** is the most
promising *training* lever already in scope, since it targets `true_miss`
directly, which is what both the cause breakdown and the visual evidence point to
as the real ceiling. #2 and #3 are evidence-backed but require new data/labeling
work beyond a single training run.

No training started, no files modified beyond the new analysis outputs listed
above. Waiting for your approval before implementing anything.
