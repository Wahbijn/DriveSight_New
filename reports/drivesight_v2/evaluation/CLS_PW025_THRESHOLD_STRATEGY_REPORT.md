# v2_cls_pw025 — Threshold Recalibration Report

**VALIDATION split only** (confirmed: `"split": "val"` in `run_config.json`, test
never referenced). No training, no checkpoint/dataset modification. Checkpoint
`runs/drivesight_v2/v2_cls_pw025/weights/best.pt` unchanged throughout (verified).
Tooling: `scripts/drivesight_v2/threshold_strategy_eval.py` (new, reuses the
established single-low-conf-pass-then-filter approach). Raw outputs:
`reports/drivesight_v2/evaluation/cls_pw025_threshold_strategy/`, built on the
extended sweep from `cls_pw025_recall_diagnosis/extended_sweep/`.

## 3. Two strategies, and exactly how each threshold was chosen

**Strategy A — Max F1 per class**: each class's own empirically-best-F1
threshold from the extended sweep (0.05-0.45, 9 points tested):

| class | threshold | class | threshold |
|---|---:|---|---:|
| pedestrian | 0.30 | motorcycle | 0.25 (= global) |
| rider | 0.20 | bicycle | 0.25 (= global) |
| car | 0.35 | traffic light | 0.30 |
| truck | 0.25 (= global) | traffic sign | 0.30 |
| bus | 0.35 | | |

**Strategy B — Recall-priority for pedestrian/rider/motorcycle/bicycle**: *not*
simply "lower all four" — each choice is justified against the sweep, including
two deliberately *not* lowered:

| class | threshold | justification |
|---|---:|---|
| pedestrian | **0.15** | real recoverable headroom: TP 611→724 (+113) moving 0.30→0.15 |
| rider | **0.10** | modest gain only: TP 24→26 (+2); **0.15 is strictly dominated by 0.20** (identical TP=24, but 0.15 has 21 FP vs 0.20's 12) — confirmed directly in the sweep data, not assumed |
| motorcycle | **0.25 (unchanged)** | TP is **flat at 15** across 0.15/0.20/0.25 — lowering recovers *zero* additional true positives, only adds FP (16/9/6) — there is no recall to buy here |
| bicycle | **0.15** | real headroom: TP 27→33 (+6) moving 0.25→0.15 |
| (other 5 classes) | same as Strategy A | not re-optimized for "recall priority" — out of scope for the VRU focus |

This is the direct answer to requirement 6 (separating causes): pedestrian and
bicycle have genuine low-confidence recoverable headroom (41.2% and 29.8% of
their misses are `low_confidence`, per the cause-of-miss breakdown); rider has
much less (15.8%) and motorcycle almost none (3.3%, dominated instead by 66.7%
`true_miss`) — **the cause breakdown computed last round directly predicts, and
this round's sweep confirms, which classes a threshold change can and cannot fix.**

## 1-2 & 4. Full comparison: global 0.25 vs the two strategies

| | global_025 | max_f1_per_class | recall_priority |
|---|---:|---:|---:|
| macro Precision | 67.9% | **72.0%** | 64.6% |
| macro Recall | 51.0% | 49.5% | **52.0%** |
| macro F1 | 0.583 | **0.587** | 0.576 |
| micro Precision | 70.7% | **78.7%** | 74.9% |
| micro Recall | 66.0% | 61.3% | **62.2%** |
| Total FP | 3,666 | **2,225** | 2,795 |

Per-class (full table: `policy_comparison.csv`):

| class | metric | global_025 | max_f1 | recall_priority |
|---|---|---:|---:|---:|
| pedestrian | R / F1 / FP | 59.5% / 0.614 / 371 | 56.4% / **0.629** / 250 | **66.8%** / 0.563 / 767 |
| rider | R / F1 / FP | 35.6% / 0.477 / 8 | 40.7% / **0.511** / 11 | **44.1%** / 0.444 / 32 |
| motorcycle | R / F1 / FP | 33.3% / 0.476 / 3 | 33.3% / 0.476 / 3 | 33.3% / 0.469 / 4 |
| bicycle | R / F1 / FP | 36.5% / **0.443** / 21 | 36.5% / **0.443** / 21 | **44.6%** / 0.423 / 49 |

Note the common-class thresholds (car/bus/traffic light/sign raised to 0.30-0.35
in both new policies) drive much of the overall FP reduction and precision gain
seen in `max_f1_per_class` and `recall_priority` alike — this is a separate,
independently-justified effect from the VRU-specific tuning, not something to
credit to "recall priority" specifically.

## 5. The low-confidence-recovery vs false-positive trade-off, with real numbers

Moving pedestrian 0.30→0.15: **+113 TP for +526 FP (4.7 FP per recovered
pedestrian)**. Moving bicycle 0.25→0.15: **+6 TP for +30 FP (5.0 FP per
recovered bicycle)**. Moving rider 0.20→0.10: **+2 TP for +21 FP (10.5 FP per
recovered rider)** — a much worse ratio, consistent with rider having little
low-confidence headroom to begin with. Moving motorcycle below 0.25 recovers
**zero TP at any tested point down to 0.15** — pure FP cost, no recall benefit.

**The general pattern**: the trade-off ratio directly reflects each class's
cause-of-miss mix. Classes with a large `low_confidence` share (pedestrian,
bicycle) convert threshold-lowering into real recall efficiently. Classes
dominated by `true_miss` (motorcycle, and to a lesser extent rider) do not —
lowering the threshold there mostly manufactures false positives on background
clutter, not recovers real detections.

## 3b. Rider → pedestrian confusion under each policy — a genuine cost to disclose

| policy | rider → pedestrian |
|---|---:|
| global_025 | 9 |
| max_f1_per_class (rider @0.20) | **8** (slightly better) |
| recall_priority (rider @0.10) | **13** (worse) |

**This is an important, not-to-be-hidden finding**: lowering rider's threshold
to 0.10 increases the specific rider→pedestrian confusion count, because more
low-confidence rider boxes enter the matching pool and some of them land close
enough to an existing confident pedestrian prediction to be scored as a
cross-class match rather than a clean miss. `max_f1_per_class`'s milder
threshold (0.20) does not have this problem — it is strictly better than both
alternatives on this specific metric the project has tracked since the P0 round.

## Recommended practical policy (synthesis, not implemented)

Not simply "adopt Strategy A" or "adopt Strategy B" wholesale — the evidence
supports different choices per class:

| class | recommended threshold | rationale |
|---|---:|---|
| car, truck, bus, traffic light, traffic sign | **Strategy A values** (0.35/0.25/0.35/0.30/0.30) | Clear win: better precision, F1 flat-to-better, no VRU trade-off involved |
| pedestrian | **0.15** (Strategy B) | Explicitly a priority class per the stated objective; real recoverable headroom; FP cost (767 vs 371) is real and should be monitored, not hidden |
| rider | **0.20 (Strategy A)**, not 0.10 | Better F1 *and* fewer rider→pedestrian confusions than the recall-priority option — the "catch 5 more riders" trade from going to 0.10 costs more confusion errors on this exact class, which should be a deliberate decision, not a default |
| motorcycle | **0.25 (unchanged)** | No recall to recover below this point at any tested threshold — lowering only adds false positives |
| bicycle | **0.15** (Strategy B) | Priority class, modest F1 cost (0.443→0.423) for a real recall gain (+8.1pt) |

This is a recommendation for your review, not an implemented change — no
threshold was applied to any pipeline, this is read-only analysis.

## Central question — MixUp recommendation

**Is MixUp still worth testing after this calibration? Yes — more so, not
less.** This round's own data reinforces the prior conclusion rather than
weakening it: motorcycle's recall is **mathematically unmovable by any threshold
in the tested range** (flat at 33.3% from 0.15 to 0.25, confirmed with exact TP
counts, not inferred), and rider's threshold headroom is small and partly
self-defeating (the recall-priority option actively worsens the
rider→pedestrian confusion this project has tracked from the start).
Calibration has now visibly done everything it can do for free — the remaining
gap for motorcycle and most of rider's gap is structurally a `true_miss`
problem (66.7% and 50.0% of their misses respectively), which only more
training-time visual diversity (MixUp's actual mechanism) can address, not a
post-hoc confidence cutoff.

No training started, no threshold applied, no checkpoint modified. Waiting for
your approval before implementing the recommended policy or starting MixUp.
