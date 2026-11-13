# v2_cls_pw025 — Hybrid Threshold Policy Evaluation

**VALIDATION split only** (`"split": "val"` in `run_config.json`, test never
referenced). No training, no checkpoint/dataset/threshold change applied anywhere
— this is a read-only comparison. Checkpoint `v2_cls_pw025/weights/best.pt`
unchanged throughout. All four policies evaluated from the **same single
low-confidence (0.05) inference pass**, filtered per-policy in Python — exact,
not approximate, and guarantees all four are compared on identical underlying
detections. Tooling: `scripts/drivesight_v2/threshold_strategy_eval.py` (now with
the exact hybrid config added as a fourth named policy). Raw outputs:
`reports/drivesight_v2/evaluation/cls_pw025_hybrid_policy/`.

**Hybrid policy evaluated** (exactly as specified): pedestrian 0.15, bicycle
0.15, rider 0.20, motorcycle 0.25, car 0.35, bus 0.35, truck 0.25, traffic light
0.30, traffic sign 0.30.

## Overall Precision / Recall / F1 / FP, all four policies

| | global_025 | max_f1_per_class | recall_priority | **hybrid_recommended** |
|---|---:|---:|---:|---:|
| macro Precision | 67.9% | **72.0%** | 64.6% | 67.5% |
| macro Recall | 51.0% | 49.5% | **52.0%** | 51.6% |
| macro F1 | 0.583 | **0.587** | 0.576 | 0.585 |
| micro Precision | 70.7% | **78.7%** | 74.9% | 75.1% |
| micro Recall | 66.0% | 61.3% | **62.2%** | 62.2% |
| Total FP | 3,666 | **2,225** | 2,795 | 2,773 |
| Total TP | 8,858 | 8,229 | 8,350 | **8,348** |
| Total FN | 4,562 | 5,191 | 5,070 | 5,072 |

**Hybrid sits almost exactly between `max_f1_per_class` and `recall_priority`**
on every overall metric — which is expected, since per-class it borrows the
"safer" rider/motorcycle choices from Strategy A and the "recall" pedestrian/
bicycle choices from Strategy B. It is not the best on any single overall metric,
but it is the only policy that is simultaneously close to best on macro Recall
(51.6%, vs best 52.0%) and close to best on macro Precision (67.5%, vs best
72.0%) — a genuine middle ground, not a compromise that loses on everything.

## Per-class, all four policies (full table: `policy_comparison.csv`)

| class | metric | global_025 | max_f1 | recall_priority | **hybrid** |
|---|---|---:|---:|---:|---:|
| pedestrian | P/R/F1 | 63.4/59.5/0.614 | 71.0/56.4/**0.629** | 48.6/**66.8**/0.562 | 48.6/**66.8**/0.563 |
| pedestrian | TP/FP/FN | 644/371/439 | 611/250/472 | 724/767/359 | 724/766/359 |
| **rider** | P/R/F1 | 72.4/35.6/0.477 | 68.6/40.7/**0.511** | 44.8/44.1/0.444 | **66.7/40.7/0.505** |
| **rider** | TP/FP/FN | 21/8/38 | 24/11/35 | 26/32/33 | 24/12/35 |
| motorcycle | P/R/F1 | 83.3/33.3/0.476 | 83.3/33.3/**0.476** | 78.9/33.3/0.469 | 83.3/33.3/**0.476** |
| motorcycle | TP/FP/FN | 15/3/30 | 15/3/30 | 15/4/30 | 15/3/30 |
| bicycle | P/R/F1 | 56.2/36.5/**0.443** | 56.2/36.5/**0.443** | 40.2/44.6/0.423 | 40.2/44.6/0.423 |
| bicycle | TP/FP/FN | 27/21/47 | 27/21/47 | 33/49/41 | 33/49/41 |
| car | P/R/F1 | 75.7/73.3/0.745 | 86.1/67.1/**0.754** | 86.0/67.1/**0.754** | 86.0/67.1/**0.754** |
| truck | P/R/F1 | 63.0/52.5/0.573 | 64.9/52.5/**0.581** | 64.9/52.5/**0.581** | 64.9/52.5/**0.581** |
| bus | P/R/F1 | 69.3/52.3/0.596 | 78.9/49.7/**0.610** | 78.9/49.7/**0.610** | 78.9/49.7/**0.610** |
| traffic light | P/R/F1 | 63.3/59.9/0.615 | 69.5/55.7/**0.619** | 69.5/55.7/**0.619** | 69.5/55.7/**0.619** |
| traffic sign | P/R/F1 | 64.8/56.4/0.603 | 69.7/53.9/**0.608** | 69.6/53.9/**0.608** | 69.6/53.9/**0.608** |

As designed: hybrid reproduces `max_f1_per_class` exactly for motorcycle and the
five common classes (same thresholds), reproduces `recall_priority` exactly for
pedestrian and bicycle (same thresholds), and lands at its own distinct point for
rider (TP=24, matching `max_f1_per_class`'s rider outcome — same rider threshold).

## Rider → pedestrian confusion — the critical, non-obvious finding

| policy | rider→pedestrian | rider's own threshold | pedestrian's own threshold |
|---|---:|---:|---:|
| global_025 | 9 | 0.25 | 0.25 |
| max_f1_per_class | **8 (best)** | 0.20 | 0.30 |
| recall_priority | 13 | 0.10 | 0.15 |
| **hybrid_recommended** | **14 (worst)** | 0.20 | 0.15 |

**Hybrid has the worst rider→pedestrian confusion of all four policies — worse
even than `recall_priority`, despite using the "safer" rider threshold (0.20,
not 0.10).** I isolated the mechanism directly: rider's own TP/FN is *identical*
between `max_f1_per_class` and `hybrid_recommended` (24/35 in both — same rider
threshold, same rider-side outcome). The only difference between those two
policies is pedestrian's threshold (0.30 vs 0.15), and that alone moves the
confusion count from 8 to 14. **The mechanism**: confusion is scored when an
unmatched rider GT gets cross-matched to a nearby pedestrian *prediction* —
lowering pedestrian's threshold puts far more candidate pedestrian boxes into
play, which increases the chance of catching a stray rider GT, regardless of
what rider's own threshold is doing. This was not visible in the single-class
sweep from the previous round (which only varied one class's threshold at a
time) — it only emerged once rider and pedestrian were both adjusted together,
which is exactly what the hybrid policy does.

## Recommendation: do not adopt the hybrid policy as-is

The hybrid policy is a reasonable *overall* middle ground (second-best macro
Recall, third-best macro Precision, close to both leaders) — but it fails the
one thing this project has tracked most closely since the P0 round: it produces
**more** rider→pedestrian confusion than even the recall-priority policy, and
55% more than the simple `max_f1_per_class` policy (8→14), for essentially no
rider-recall benefit over `max_f1_per_class` (both give rider TP=24). Adopting it
as-is would mean accepting a real regression on this specific, previously-flagged
error pattern in exchange for pedestrian/bicycle recall gains that could be
captured without that cost.

**Suggested adjustment for your consideration, not yet evaluated**: keep
pedestrian and bicycle at their recall-priority thresholds (0.15 each) but
re-examine whether pairing pedestrian's lower threshold with rider specifically
needs compensation (e.g. a stricter match-IoU for rider-vs-pedestrian crossing,
or simply accepting `max_f1_per_class`'s rider=0.20/pedestrian=0.30 pairing and
treating pedestrian's recall gain as a separate, not-yet-resolved trade-off)
rather than combining the two independently-chosen thresholds as this version
did. This would need a new evaluation round, not implied by what's been run here.

No threshold was applied, no checkpoint modified, test set untouched. Waiting for
your approval before any further action.
