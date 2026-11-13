# reports/drivesight_v2

Reports and raw outputs for the V2 pipeline (clean restart on `yolo11n.pt`,
`cls_pw` class-weighting experiments, threshold calibration, final test-set
evaluation). The full narrative is in
`notebooks/drivesight_v2/DriveSight_Object_Detection_Journey.ipynb`; this
folder holds the underlying evidence it cites.

```
dataset_audit/       EDA figures/CSVs: class counts, box sizes, co-occurrence,
                      scene attributes (notebook 01)
preprocessing/        annotation-ambiguity flagging pass (notebook 02)
training/              raw training logs, v2_{baseline,cls_pw,cls_pw025}
evaluation/            the full evaluation trail -- see below
resolution_ablation/   (reserved; V1's resolution ablation lives under
                       reports/resolution_ablation/ instead)
```

## `evaluation/` — one report per round, chronological

| report | what it covers |
|---|---|
| `V2_BASELINE_EVALUATION_REPORT.md` | v2_baseline vs both V1 variants |
| `CLASS_SPECIFIC_THRESHOLD_EVAL_REPORT.md` | "Action 1" threshold policy, v2_baseline |
| `RECALL_DIAGNOSIS_REPORT.md` | v2_baseline cause-of-miss analysis |
| `CLS_PW_VS_BASELINE_REPORT.md` | v2_cls_pw (0.5) vs v2_baseline |
| `CLS_PW025_VS_OTHERS_REPORT.md` | v2_cls_pw025 (0.25) vs the other two — the winner |
| `CLS_PW025_RECALL_ERROR_ANALYSIS.md` | v2_cls_pw025 cause-of-miss + visual evidence |
| `CLS_PW025_THRESHOLD_STRATEGY_REPORT.md` | per-class threshold re-calibration for v2_cls_pw025 |
| `CLS_PW025_HYBRID_POLICY_REPORT.md` | a rejected hybrid policy (cross-class interaction finding) |
| `CLS_PW025_BICYCLE_CANDIDATE_REPORT.md` | the adopted bicycle-recall refinement |
| `INFERENCE_PROFILES.md` | the two frozen, deployable threshold profiles |
| `TEST_SET_AUDIT.md` | contamination audit, performed before any test-split use |
| `FINAL_TEST_SET_REPORT.md` | one-time, 3-checkpoint test-split evaluation |
| `FINAL_POLICY_TEST_SET_REPORT.md` | one-time test-split check of the two frozen profiles |

Canonical thresholds/checkpoint path: `scripts/drivesight_v2/inference_profiles.py`
(import `PROFILES`/`get_profile()` — don't redefine these values elsewhere).
