# Test-Set Contamination Audit

Performed before any test-split evaluation, as requested. Method: exhaustively
grepped every saved `run_config.json` and `args.yaml` in the entire repository
(training runs, validation runs, error audits — every artifact that records which
split a command actually used), rather than relying on memory of past commands.

## Finding: clean. Zero model-performance use of the test split anywhere.

```
$ grep -rH "^split:" runs/*/*/args.yaml runs/*/*/*/args.yaml
... (9 files, every training run: baseline, oversampling, finetune, v2_baseline,
     v2_cls_pw, v2_cls_pw025, + 3 smoke tests)
split: val      <- every single one, no exceptions

$ grep -rH "\"split\"" reports/**/run_config.json
... (17 files: every P0 validation/audit, every resolution-ablation run, every
     V1-vs-V2 and V2-vs-V2 comparison, every class-specific-threshold run)
"split": "val"  <- every single one, no exceptions
```

- **Training**: Ultralytics validates against `data.yaml`'s `val:` key during
  training by default; there is no mechanism by which training touches `test`
  unless explicitly configured to, and it never was (confirmed structurally and
  via every `args.yaml`).
- **Model/checkpoint selection**: every "best epoch" decision (early stopping,
  `best.pt`) is driven by the in-training validation metric, which is always `val`.
- **Every comparison report in this project** (P0 diagnostics, resolution ablation,
  V1-vs-V2, Action-1 threshold calibration, cls_pw, cls_pw025) ran exclusively
  against `val`.
- **cls_pw value selection (0.5 → 0.25)** was decided from `val`-only comparisons.
- **The confidence-threshold policy (Action 1)** was calibrated and validated on
  `val` only.

## The only two touches of `test` found, and why they don't bias evaluation

1. **`notebooks/convert_to_yolo.py` / `create_rare_class_dataset.py`** (original
   V1 scripts) — these *define* the test split as part of dataset construction.
   Defining a split is not the same as using it for a decision.
2. **Descriptive statistics only**, in two places:
   - P0 diagnostics' test-label-provenance investigation
     (`reports/p0_diagnostics/test_label_provenance.md`) — read the original BDD100K
     JSON attributes/timestamps to verify the test labels are genuine, not
     fabricated. This never ran a model or touched predictions.
   - V2 notebook 01 (`scripts/drivesight_v2/dataset_eda.py`) — reports class
     instance counts and weather/attribute distributions for train/val/**test**
     side by side, purely descriptive (e.g. "test: rider 72 instances"). This is
     knowledge of the test set's *composition*, not its *performance under any
     model* — it did not inform which checkpoint was selected, which threshold was
     chosen, or any hyperparameter. I'm reporting it explicitly rather than
     omitting it, per your instruction, but assess it as not biasing — knowing a
     test set has "72 rider instances" before evaluating is standard practice and
     does not let a model's predictions leak into training decisions.

## Verdict

**The test split is genuinely untouched for any model-performance purpose.** No
checkpoint, no threshold, no hyperparameter choice anywhere in this project was
selected using test-split predictions or metrics. The evaluation below is a
legitimate, unbiased first use of this held-out data for all three models.

One residual caveat, stated for completeness rather than as a real concern: I
cannot audit what happened in this repository before this project's diagnostic work
began (i.e. whatever process originally produced the three V1 checkpoints baseline/
oversampling/finetune, before I was engaged). Everything *verifiable from saved
artifacts* — which is everything that exists — shows val-only usage throughout.
