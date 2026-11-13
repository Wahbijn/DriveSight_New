# v2_cls_pw025 — Comparison Report (vs v2_baseline and v2_cls_pw)

Checkpoints: `v2_baseline/best.pt` (epoch 91), `v2_cls_pw/best.pt` (epoch 46,
`cls_pw=0.5`), `v2_cls_pw025/best.pt` (epoch 91, `cls_pw=0.25`, this experiment).
All other config identical (transfer learning from `yolo11n.pt`, imgsz=1152,
batch=4, AdamW lr0=0.001, freeze=None, same dataset/split). Evaluated at imgsz=1152,
same val split, same matching criteria (iou=0.70, match-iou=0.50) as every prior V2
comparison. Raw outputs: `reports/drivesight_v2/evaluation/cls_pw025_vs_others/`.
All three experiments' checkpoints/results preserved and verified untouched
throughout.

## 6. Training history and early stopping

Three interruptions (system low-memory kills, same pattern as prior rounds), all
recovered via the proven `resume=True` mechanism with **zero progress lost** —
confirmed by `results.csv` and checkpoint epoch fields after every resume:
- Epoch 1→40 (uninterrupted), killed mid-epoch 41 → resumed at epoch 41.
- Epoch 41→75, killed mid-epoch 76 → resumed at epoch 76 (died again within the
  first minute, zero new epochs lost, resumed a second time at epoch 76).
- Epoch 76→121, **EarlyStopping triggered**: "no improvement observed in last 30
  epochs. Best results observed at epoch 91."

**121 total epochs, best = epoch 91** (mAP50-95=0.2930) — more epochs than either
prior run (v2_baseline: paused at 91/100 by user instruction; v2_cls_pw: 76 epochs,
best at 46).

## 1. Overall Precision/Recall/mAP

| | v2_baseline | v2_cls_pw (0.5) | **v2_cls_pw025 (0.25)** |
|---|---:|---:|---:|
| Precision | 60.5% | 63.9% | **70.5%** |
| Recall | **49.3%** | 45.8% | 48.4% |
| mAP50 | 52.6% | 51.9% | **54.7%** |
| mAP50-95 | 28.9% | 28.4% | **29.4%** |

**cls_pw025 has the best mAP50, mAP50-95, and Precision of all three runs**, and
Recall is close to baseline (-0.9pt) — a much better balance than cls_pw=0.5
(-3.6pt recall for +3.4pt precision).

## 2. Per-class AP50 / AP50-95, and the objective's five named classes

| class | AP50 baseline | AP50 cls_pw | **AP50 cls_pw025** | Δ vs baseline |
|---|---:|---:|---:|---:|
| pedestrian | 61.2% | 60.6% | **63.0%** | **+1.8pt** |
| rider | 41.0% | 40.4% | **44.9%** | **+3.9pt** |
| car | 78.6% | 77.2% | 78.4% | -0.2pt |
| truck | 48.5% | 49.3% | **49.6%** | +1.1pt |
| bus | 56.5% | 52.6% | **59.6%** | **+3.1pt** |
| motorcycle | 29.7% | **39.9%** | 37.7% | **+8.0pt** |
| bicycle | 37.7% | 33.4% | **38.9%** | +1.2pt |
| traffic light | 60.7% | 57.0% | **61.4%** | **+0.7pt** |
| traffic sign | 59.8% | 56.4% | 59.1% | -0.7pt |

**The objective is met**: all five named "must not regress" classes
(pedestrian/car/bus/traffic light/traffic sign) either **improved** or moved by
less than 1pt (car -0.2, traffic sign -0.7 — both noise-level, not material
regressions by this project's own 2pt bar). Motorcycle (+8.0pt) and rider
(+3.9pt) both improved substantially over baseline. **cls_pw025 beats v2_baseline
on 7 of 9 classes outright**, with the other two (car, traffic sign) essentially
flat.

Full per-class AP50-95: `summary_per_class_metrics.csv`. Pattern matches AP50 —
cls_pw025 leads or ties on nearly every class.

## 3. Rider → pedestrian confusion

| | v2_baseline | v2_cls_pw (0.5) | v2_cls_pw025 (0.25) |
|---|---:|---:|---:|
| rider → pedestrian | 10 | 5 | **9** |
| pedestrian → rider | 4 | 6 | **1** |

Exactly the predicted trade-off from the pre-training mechanism explanation:
**weaker class weighting → the confusion reduction partially reverses** (5→9,
most of the way back to baseline's 10, since rider's pull toward its own channel is
weaker at 0.25 than at 0.5). The reverse-direction confusion (pedestrian→rider)
actually improved to its best value of all three (1, down from baseline's 4) — a
real, if small, favorable asymmetry: pedestrian-as-rider errors are rarer than
before, even though rider-as-pedestrian errors mostly came back.

## 4. Motorcycle and bicycle performance

- **Motorcycle**: AP50 37.7% (vs cls_pw=0.5's 39.9% and baseline's 29.7%) — **most
  of the cls_pw=0.5 gain is retained** (+8.0pt over baseline, vs +10.2pt at 0.5).
  FP: 3, identical to cls_pw=0.5 (both down from baseline's 22, -86%) — **the false
  positive reduction is fully retained**, not just partially.
- **Bicycle**: AP50 38.9%, now the **best of all three runs**, including beating
  baseline (+1.2pt) — this reverses cls_pw=0.5's regression (-4.4pt) entirely.
  Audit-recall also best of the three (36.5%).

## 5. Car false positives

| | v2_baseline | v2_cls_pw (0.5) | v2_cls_pw025 (0.25) |
|---|---:|---:|---:|
| car FP | 1,745 | 2,122 (+21.6%) | **1,742 (-3, essentially identical to baseline)** |

**The car false-positive regression from cls_pw=0.5 is fully resolved** — confirms
the pre-training prediction that a gentler weight (car=0.449 vs 0.173) would
suppress car's negative-anchor loss far less, removing the pressure that caused the
FP spike.

**One new, real trade-off worth flagging plainly**: pedestrian FP rose to 371 (vs
baseline's 248, cls_pw=0.5's 137) and traffic sign FP rose to 765 (vs 705/641) —
both classes got *more* detections overall (more TPs, which is why their AP50
improved, but also more FPs). Bus and truck FP also ticked up slightly (35 vs 29,
110 vs 99). None of these are large relative to each class's instance count, and
none dragged the corresponding AP50 down, but they are a real, measured cost, not
zero — reported here rather than omitted because the AP50 numbers alone look clean.

## Conclusion

`v2_cls_pw025` is, on the full evidence, **the best checkpoint produced in this
project so far** — it satisfies the stated objective (retain the motorcycle/rider
gains without unacceptable regression on pedestrian/car/bus/traffic light/traffic
sign) more cleanly than I expected going in: 7 of 9 classes beat baseline outright,
the car FP spike from `cls_pw=0.5` is fully resolved, and most of motorcycle's FP
reduction is retained. The two costs are real and specific, not hand-waved away:
(1) the rider→pedestrian confusion reduction is mostly lost going from 0.5→0.25
(5→9), and (2) pedestrian and traffic-sign false positives rose somewhat even as
their AP50 improved.

No MixUp experiment started. Awaiting your direction on next steps (e.g. whether
`v2_cls_pw025` becomes the new reference checkpoint, whether to proceed to the
`v2_combined`/MixUp experiment from here, or something else).
