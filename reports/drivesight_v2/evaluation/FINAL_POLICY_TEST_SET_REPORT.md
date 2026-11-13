# Final Policy Test-Set Comparison

**One-time test-split check** (830 images). Thresholds imported unchanged from
the frozen `scripts/drivesight_v2/inference_profiles.py` — **nothing was tuned
against these results.** Checkpoint `v2_cls_pw025/weights/best.pt` unchanged
throughout (verified before and after). Same evaluation settings as every
validation-based round (imgsz=1152, NMS iou=0.70, match-iou=0.50). Raw outputs:
`reports/drivesight_v2/evaluation/final_policy_test_set/`.

## 3. Comparison: global_025 vs the two frozen profiles

| | global_025 | balanced_max_f1 | vru_recall_bicycle |
|---|---:|---:|---:|
| macro Precision | 69.7% | **72.6%** | 70.1% |
| macro Recall | **51.4%** | 49.6% | 50.3% |
| macro F1 | **0.592** | 0.590 | 0.586 |
| micro Precision | 71.3% | **79.0%** | 78.8% |
| micro Recall | **65.8%** | 60.8% | 60.8% |
| Total FP | 4,382 | **2,667** | 2,696 |
| Total TP | **10,875** | 10,035 | 10,042 |

Both frozen profiles replicate the general pattern seen on validation: large FP
reduction (-39%) and precision gain from raising car/bus/traffic-light/sign
thresholds, at a recall cost spread across those same classes (their own
thresholds moved from 0.25 to 0.30-0.35). `vru_recall_bicycle` recovers a sliver
of that recall back (50.3% vs 49.6%) via bicycle alone, exactly as designed.

### Bicycle — the one class that differs between the two frozen profiles

| | global_025 / balanced_max_f1 (identical) | **vru_recall_bicycle** | Δ |
|---|---:|---:|---:|
| Precision | 80.6% | 58.8% | -21.8pt |
| Recall | 43.1% | **49.1%** | **+6.0pt** |
| F1 | 0.562 | 0.535 | -0.027 |
| TP / FP / FN | 50 / 12 / 66 | **57 / 40 / 59** | +7 / +28 / -7 |

**This replicates the validation finding directionally** (val: +6 TP/+27 FP;
test: +7 TP/+28 FP — a very similar 4.0-4.5 FP-per-recovered-bicycle ratio on
both splits). **One honest difference to flag**: the *relative* FP increase is
steeper on test — bicycle FP more than triples (12→40, +233%) versus val's
roughly doubling (21→48, +129%), because the baseline FP count on test (12) was
lower to begin with. The per-TP cost generalizes well; the relative-FP-increase
framing looks more dramatic on this split and should be read alongside the
absolute numbers, not instead of them.

## rider → pedestrian confusion on test

| global_025 | balanced_max_f1 | vru_recall_bicycle |
|---:|---:|---:|
| 11 | 11 | 11 |

**Identical across all three policies on test.** The small improvement seen on
validation (9→8 moving from global_025 to balanced_max_f1) does **not**
replicate as an improvement here — it's flat, not worse. Net assessment: both
frozen profiles are confirmed **not harmful** to this specific, closely-tracked
confusion on held-out data, even though the modest validation-time gain didn't
carry over. (A trivial, mechanistically-expected 1-count artifact was also
found: rider's own FP count differs by 1 between the two profiles, and
bicycle→rider confusion differs by 1 between global_025 and balanced_max_f1 —
both are the same kind of minor cross-class-threshold interaction documented in
earlier rounds, not a new concern, and far too small to change any conclusion.)

## 4. Do the additional bicycle detections justify the false-positive cost?

**Yes, for the stated risk-estimation objective, with the cost stated plainly
rather than minimized.** On test: 7 additional real cyclists detected per 830
images, at a cost of 28 additional phantom bicycle boxes (≈1 extra false
bicycle flag per 30 images) and bicycle precision dropping from 80.6% to 58.8%.
For a system whose stated purpose is *estimating risk*, not issuing final
automated actions, a missed cyclist is a safety-relevant blind spot while an
extra flagged phantom is a comparatively low-cost nuisance (an elevated risk
score that review or downstream logic can discount). The per-recovered-bicycle
cost (4.0 FP/TP) is consistent between val and test, which is the more
important generalization signal than the relative-increase framing above.

This is a judgment call about acceptable cost, not a purely statistical one —
the numbers support adopting `vru_recall_bicycle` for that objective, but
`balanced_max_f1` remains the better choice if bicycle precision (e.g. avoiding
nuisance alerts) matters more in your downstream use of the risk score.

## 5. Final profile documentation and limitations

Full documentation: `reports/drivesight_v2/evaluation/INFERENCE_PROFILES.md`,
canonical definitions: `scripts/drivesight_v2/inference_profiles.py`. Key
limitations, confirmed or added by this test-set round:

- Both profiles are calibrated for **`v2_cls_pw025` specifically** — not
  validated for any other checkpoint.
- **Neither profile improves the rider→pedestrian confusion on held-out data**
  (flat at 11 on test) — the earlier validation-time improvement for
  `balanced_max_f1` (9→8) did not generalize as an improvement, though it also
  didn't regress. Confusion-matrix problems need a training-time fix, not a
  threshold.
- **The bicycle recall/precision trade-off generalizes in per-TP cost
  (~4-4.5 FP per recovered bicycle) but looks more dramatic in relative terms
  on test** (FP more than tripling vs roughly doubling on val) — both framings
  are correct and should be considered together when judging "acceptability."
- Small instance counts for rider/motorcycle/bicycle on both splits (test:
  72/39/116 GT instances) mean single-digit changes move percentages
  substantially — treat exact values as indicative.

Both checkpoints and all prior evaluation results preserved and verified
untouched. No training started, no threshold applied to any pipeline, test set
used exactly once for this final check as instructed. Waiting for your
approval.
