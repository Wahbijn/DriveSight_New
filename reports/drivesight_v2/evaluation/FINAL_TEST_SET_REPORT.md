# Final Test-Set Evaluation — Does v2_cls_pw025 Generalize?

**All three checkpoints preserved and verified untouched** (mtimes unchanged
throughout this evaluation). This is the test split's **first and only use** in
this project for model performance — see `TEST_SET_AUDIT.md` for the full,
evidence-based contamination audit (verdict: clean). Evaluated at imgsz=1152,
same matching criteria (iou=0.70, match-iou=0.50) and same class mapping as every
val-based comparison. No threshold tuning was performed on test — the conf=0.25
reference threshold used for the error audit is the same one used throughout the
project, carried over unchanged, not fitted to these results.
Raw outputs: `reports/drivesight_v2/evaluation/final_test_set/`.

## 2. Test-set audit — summary (full detail: `TEST_SET_AUDIT.md`)

Grepped every saved `run_config.json`/`args.yaml` in the entire repository: **every
training run and every evaluation this project has ever run used `split: val`,
with zero exceptions.** The only touches of `test` were (a) the original dataset-
construction scripts defining the split, and (b) two purely descriptive uses
(test-label provenance verification, EDA instance counts) — neither ran a model or
influenced any checkpoint/threshold/hyperparameter decision. **This evaluation is
a genuine, unbiased first use of held-out data.**

## 1. Overall Precision / Recall / mAP50 / mAP50-95 (test, n=830 images)

| | v2_baseline | v2_cls_pw (0.5) | v2_cls_pw025 (0.25) |
|---|---:|---:|---:|
| Precision | **63.2%** | 63.0% | 61.4% |
| Recall | 48.1% | 44.6% | **50.7%** |
| mAP50 | 51.9% | 51.6% | **53.7%** |
| mAP50-95 | 27.6% | 27.4% | **28.4%** |

**The core result replicates on held-out data**: cls_pw025 still wins on Recall,
mAP50, and mAP50-95. **One reversal from val, reported plainly**: on val,
cls_pw025 had the *highest* precision of the three (70.5%); on test it has the
*lowest* (61.4%, vs baseline's 63.2%). The recall/mAP advantage is real and
generalizes; the precision advantage seen on val does not.

## 2. Per-class AP50 / AP50-95 (test)

| class | AP50 baseline | AP50 cls_pw | **AP50 cls_pw025** | Δ vs baseline |
|---|---:|---:|---:|---:|
| pedestrian | 59.7% | 57.4% | 59.6% | -0.1pt |
| **rider** | 34.5% | 35.5% | **39.6%** | **+5.1pt** |
| car | 77.8% | 76.2% | 77.6% | -0.2pt |
| truck | 50.7% | 52.3% | 51.6% | +1.0pt |
| bus | 41.3% | **46.1%** | 43.5% | +2.3pt |
| **motorcycle** | 38.3% | **44.9%** | 42.7% | **+4.4pt** |
| **bicycle** | 42.9% | 36.9% | **46.8%** | **+3.9pt** |
| traffic light | 61.9% | 59.0% | 61.6% | -0.4pt |
| traffic sign | 60.4% | 56.5% | 60.6% | +0.1pt |

**Every directional finding from val replicates on test**: rider/motorcycle/
bicycle/bus all improve over baseline (and by larger margins than on val for
bicycle specifically: +3.9pt here vs +1.2pt on val); pedestrian/car/traffic light/
traffic sign are flat (within ±0.4pt, noise-level); cls_pw=0.5's bicycle
regression (-6.0pt vs baseline) also replicates, confirming it's a real effect of
that configuration, not a val-specific artifact. Full AP50-95 and official
per-class Recall: `summary_per_class_metrics.csv`.

## 3. Rider → pedestrian confusion (test)

| | v2_baseline | v2_cls_pw (0.5) | v2_cls_pw025 (0.25) |
|---|---:|---:|---:|
| rider → pedestrian | 12 | **5** | 11 |

**Replicates almost exactly** the val pattern (10→5→9): cls_pw=0.5 roughly halves
it, cls_pw=0.25 gives most of that back. The mechanism-based prediction holds up
on genuinely unseen data, not just the validation set it was tuned against.

## False positives / false negatives (test, conf=0.25)

| class | FN baseline→pw→pw025 | FP baseline→pw→pw025 |
|---|---|---|
| pedestrian | 561→688→515 | 261→131→**371** |
| rider | 30→37→34 | 18→33→**15** |
| car | 2445→2620→2471 | 2207→**2568**→2151 |
| motorcycle | 19→20→19 | 14→10→**5** |
| bicycle | 65→67→62 | 23→36→**12** |

Three key generalization checks, all confirmed:
- **Car's FP spike under cls_pw=0.5 (+16.4% vs baseline) is, again, fully resolved
  at 0.25** (2151, actually *below* baseline's 2207).
- **Motorcycle's FP reduction is retained and extended** (14→10→5 — cls_pw025 is
  best here on test, better even than cls_pw=0.5).
- **Pedestrian's FP increase under cls_pw025 also replicates** (261→371, +42%) —
  this is a real, consistent cost, not val noise.

## 4. Motorcycle and bicycle on test

- **Motorcycle**: AP50 +4.4pt vs baseline (smaller margin than val's +8.0pt, but
  same direction), and now has the *fewest* false positives of all three models
  (5, vs baseline's 14) — the best motorcycle result on test by precision.
- **Bicycle**: AP50 +3.9pt vs baseline — **a larger margin than on val** (+1.2pt),
  and FP nearly halved (23→12). Bicycle generalizes better than val suggested.

## Visual inspection (test images, `visual_comparison/`)

Selected by object density, not cherry-picked for a predetermined outcome: the
most VRU-dense test image (`f2dff522-f0a38141`, 10 rider/motorcycle/bicycle
instances, a protected bike-lane scene), two dense small-object street scenes
(`f3c442f2-781b3df1`, `e81a0375-2115baff`), one pure small-object-dense scene
(`ced813cc-480d1c0c`, 56 small instances), and one more mixed scene
(`ea244d67-61de2bfd`).

All three models produce sane, consistent detections with no qualitative failures
or bizarre artifacts on this unseen data — the bike-lane scene shows v2_baseline
producing somewhat more total boxes (32) than cls_pw/cls_pw025 (27/26), consistent
with baseline's generally higher raw detection volume found quantitatively across
this whole evaluation. Full-resolution grids and per-model panels available in
`visual_comparison/<stem>/` for direct inspection.

## 6. Conclusion

**Yes — `v2_cls_pw025` genuinely generalizes better than `v2_baseline`, on a test
set verified to be free of any prior influence on model selection.** The headline
gains (Recall, mAP50, mAP50-95, rider/motorcycle/bicycle AP50, the car-FP-spike
fix, the motorcycle-FP reduction) all replicate in the same direction on held-out
data, in some cases (bicycle) with an even larger margin than validation
suggested. This is a materially stronger result than a typical val-only claim,
precisely because the audit confirmed nothing about this checkpoint's selection
depended on test-set feedback.

**Remaining weaknesses, unchanged by this round and worth carrying forward:**
- **Precision does not generalize as cleanly as recall** — cls_pw025 loses its
  val-time precision lead on test (61.4% vs baseline's 63.2%), driven by the
  pedestrian/traffic-sign FP increases that also replicate here.
- **The rider→pedestrian confusion is still present** (11 of ~45 rider instances
  on test) — cls_pw025 only captures a fraction of cls_pw=0.5's reduction on this
  specific error; it has not been solved, only partially improved.
- **Rider and bicycle AP50-95 remain low in absolute terms** (14.4% and 19.0%
  respectively) even though AP50 improved — localization quality on these classes,
  not just classification, is still weak.
- Car/pedestrian/traffic-light/traffic-sign AP50 are essentially unchanged from
  baseline — this experiment did not move the classes that drive the largest raw
  volume of missed/false detections (per the earlier recall-diagnosis report).

No training was started. Awaiting your approval before any further experiment.
