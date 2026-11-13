# v2_cls_pw025 — Final Threshold Candidate (Max-F1 + bicycle=0.15 only)

**VALIDATION split only** (`"split": "val"`, test never referenced). No
training, no checkpoint/dataset/threshold change applied anywhere. Checkpoint
`v2_cls_pw025/weights/best.pt` unchanged throughout. All four policies evaluated
from the same single low-confidence (0.05) inference pass, filtered per-policy —
exact, not approximate. Tooling: `scripts/drivesight_v2/threshold_strategy_eval.py`.
Raw outputs: `reports/drivesight_v2/evaluation/cls_pw025_bicycle_candidate/`.

**Candidate evaluated**: pedestrian 0.30, rider 0.20, bicycle **0.15**,
motorcycle 0.25, car 0.35, bus 0.35, truck 0.25, traffic light 0.30, traffic
sign 0.30 — i.e. `max_f1_per_class` with exactly one value changed.

## Overall macro Precision / Recall / F1 / FP, all four policies

| | global_025 | max_f1_per_class | recall_priority | **candidate** |
|---|---:|---:|---:|---:|
| macro Precision | 67.9% | **72.0%** | 64.6% | 70.3% |
| macro Recall | 51.0% | 49.5% | **52.0%** | 50.4% |
| macro F1 | 0.583 | 0.587 | 0.576 | **0.587** |
| Total FP | 3,666 | **2,225** | 2,795 | 2,252 |
| Total TP | 8,858 | 8,229 | 8,350 | 8,235 |

The candidate improves on `max_f1_per_class`'s macro Recall (+0.9pt, 49.54%→50.44%)
and macro F1 (0.5870→0.5874, a small but genuine improvement, now the best of
all four policies) for a small, isolated FP cost (+27, all of it attributable to
bicycle), while staying far closer to `max_f1_per_class`'s precision than
`recall_priority` does.

## Per-class — confirms perfect isolation

**Every class except bicycle is byte-identical to `max_f1_per_class`** (same
TP/FP/FN/P/R/F1 for pedestrian, rider, car, truck, bus, motorcycle, traffic
light, traffic sign — verified directly in `policy_comparison.csv`, not just
visually). This is exactly what a correctly-isolated single-variable change
should produce, and confirms bicycle's threshold has no detectable interaction
with any other class's matching outcome.

| class | metric | max_f1_per_class | **candidate** | Δ |
|---|---|---:|---:|---:|
| bicycle | Precision | 56.2% | 40.7% | -15.5pt |
| bicycle | Recall | 36.5% | **44.6%** | **+8.1pt** |
| bicycle | F1 | 0.443 | 0.426 | -0.017 |
| bicycle | TP / FP / FN | 27 / 21 / 47 | **33 / 48 / 41** | +6 / +27 / -6 |
| *(all other 8 classes)* | *(all metrics)* | *(unchanged)* | *(unchanged)* | 0 |

## Pay special attention to — point by point

**Rider → pedestrian confusion**: **8, identical to `max_f1_per_class`** (the
best score of any policy tested across both threshold rounds) — confirmed via
the confusion matrix, not inferred. Bicycle's threshold has zero pathway to
affect rider/pedestrian matching; this is the whole reason pedestrian and rider
were deliberately left untouched in this candidate, and it worked exactly as
intended.

**Bicycle Recall improvement**: 36.5% → **44.6%** (+8.1pt), recovering 6 of the
47 previously-missed bicycle instances (now 41 missed). This matches the
single-class sweep's earlier prediction almost exactly.

**Any new cross-class confusion?** One trivial change: rider→bicycle ticks up
by 1 (1→2 instances) — a single additional rider GT now lands close enough to a
bicycle prediction to register as a cross-class match. At this scale (1 of 59
rider instances) this is noise, not a pattern. No other new confusion pairs
appeared anywhere in either the rider or bicycle confusion rows. Notably,
`recall_priority` (bicycle=0.15 also, but combined with pedestrian=0.15)
introduced a **bicycle→pedestrian** confusion (0→1) that this candidate does
**not** show — further evidence that keeping pedestrian's threshold untouched
avoids the cross-class side effects found in the rejected hybrid policy.

**Does the TP increase justify the FP increase?** Bicycle: +6 TP for +27 FP
(**4.5 FP per recovered bicycle**), precision drops 56.2%→40.7%. This is a real,
non-trivial cost, not free — but it is now the *only* cost anywhere in the
policy: every other class, and critically the rider→pedestrian confusion
metric, is completely unaffected. Framed against the project's stated
objective (bicycle is an explicitly named VRU-priority class, and this project
has repeatedly treated missed VRUs as more costly than an extra false
detection in a risk-*estimation*, not final-decision, context), recovering 6
real bicycles at a cost of 27 phantom detections across 686 images (~1 extra
false bicycle box per 25 images) is a defensible, bounded trade — and unlike
every other threshold combination tested across both rounds, it comes with no
side effect on any other class or the rider/pedestrian confusion.

## Recommendation

**Adopt this candidate** (`max_f1_per_class` with bicycle lowered to 0.15) as
the default inference policy: pedestrian 0.30, rider 0.20, bicycle 0.15,
motorcycle 0.25, car 0.35, bus 0.35, truck 0.25, traffic light 0.30, traffic
sign 0.30. It strictly dominates `max_f1_per_class` on bicycle recall with a
contained, isolated cost, and avoids every failure mode found in
`recall_priority` and the rejected hybrid policy (worse rider confusion, larger
pedestrian FP blowup, bicycle→pedestrian cross-confusion). This is the best
policy produced across both threshold-calibration rounds on the evidence
gathered.

No threshold was applied, no checkpoint modified, test set untouched. This was
the final threshold-policy investigation, as requested — waiting for your
approval before implementing anything.
