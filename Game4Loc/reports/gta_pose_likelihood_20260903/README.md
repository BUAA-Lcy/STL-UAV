# GTA-UAV Multi-Tile Pose-Likelihood Pilot

## Decision

Initial pilot: `REJECT` under the heuristic 70%/15% rule.
Statistical and frozen full-test follow-up: `KEEP`.

The bounded multi-tile candidate pool has substantial oracle headroom, but the
pre-registered reliability gate was not met after the single allowed nonlinear
follow-up. The plan therefore stops before full same-area and cross-area runs.

## Protocol

- Dataset: GTA-UAV same-area only.
- Train cache: deterministic stratified 2,000-query subset.
- Test cache: deterministic stratified 345-query subset.
- Seed: `20260903`.
- Candidates: retrieval top-5 × VOP top-4 (20/query).
- Retrieval: frozen same-area checkpoint.
- Orientation: frozen full-teacher Exp C VOP checkpoint.
- Matcher: frozen sparse defaults.
- Hardware/runtime: one RTX 5070, `gtauav`, `--num_workers 0`, W&B disabled.
- Target: candidate localization error below 20 m.
- Model sequence: L2 logistic regression with `C={0.1,1,10}` and temperature
  scaling; one fixed shallow HistGradientBoosting follow-up if required.

## Results

| Variant | Dis@1 (m) | MA@20 (%) | Worse than coarse (%) | Catastrophic +50m (%) |
|---|---:|---:|---:|---:|
| Coarse top-1 | 140.02 | 7.54 | 0.00 | 0.00 |
| Legacy top-1 VOP | 86.36 | 42.61 | 11.88 | 2.90 |
| Raw top-5×top-4 | 158.98 | 45.22 | 16.23 | 7.83 |
| Oracle top-5×top-4 | 26.59 | 65.51 | 2.90 | 0.58 |
| Calibrated adaptive | 74.72 | 47.25 | 5.51 | 0.58 |

The oracle reduces `Dis@1` by 69.21% and increases `MA@20` by 22.90 points
relative to legacy top-1 VOP, passing the oracle gate.

The initial logistic model had AUROC 0.8532, ECE 0.2129, and 7.45% selective
risk improvement. It triggered the one fixed HGB follow-up. HGB achieved:

- AUROC: 0.8682 (passes 0.75 threshold)
- AUPRC: 0.5087
- NLL: 0.2833
- Brier: 0.0872
- 15-bin equal-mass ECE: 0.0206 (passes 0.05 threshold)
- 70%-coverage error: 22.94 m versus 26.68 m for raw inlier confidence
- 70%-coverage risk improvement: 13.99% (fails 15% threshold)

The learned adaptive policy evaluated 7.08 hypotheses/query on average. Its
headline metrics are diagnostic only because the calibration gate failed.

## Interpretation

The result separates candidate generation from candidate selection:

1. Expanding beyond retrieval top-1 exposes many good latent positions.
2. Selecting those positions by raw inlier count is unsafe and severely harms
   the mean due to catastrophic false geometry.
3. The fixed HGB model ranks and calibrates candidates reasonably well, but the
   improvement at the specified 70% coverage is insufficient under the locked
   decision rule.

The user re-opened the line after questioning the heuristic single-point gate.
A 10,000-replicate paired bootstrap and tie-aware AURC audit supported the
direction of improvement, so a frozen-calibrator full 3,443-query test cache
was completed.

## Frozen Full-Test Follow-up

| Cohort | Variant | Dis@1 (m) | MA@20 (%) | Worse than coarse (%) | Catastrophic +50m (%) |
|---|---|---:|---:|---:|---:|
| Full 3,443 | Legacy top-1 VOP | 58.16 | 46.09 | 11.88 | 3.19 |
| Full 3,443 | Adaptive calibrated | 48.99 | 49.75 | 6.62 | 0.90 |
| Non-pilot 3,098 | Legacy top-1 VOP | 54.74 | 46.71 | 11.59 | 2.97 |
| Non-pilot 3,098 | Adaptive calibrated | 45.90 | 50.26 | 6.52 | 0.90 |

On the primary 3,098-query holdout, paired bootstrap gives `8.84m`
Dis@1 improvement (95% CI `[7.12,10.64]`), `+3.55pp` MA@20 (95% CI
`[2.58,4.55]`), and `2.07pp` catastrophic-rate reduction (95% CI
`[1.45,2.71]`). Tie-aware AURC improves by `9.77m` (95% CI
`[7.63,12.41]`).

The full same-area thresholds are therefore met by the frozen 2,000-query
calibrator. The next active stage is generating the complete 13,851-query
train cache and fitting a final train-only calibrator.

## Artifacts

- Executed notebook: `Game4Loc/notebooks/gta_pose_likelihood_results.ipynb`
- Append-only chronology: `Process.md`
- Ignored runtime artifacts:
  - `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/pilot_train2000.jsonl`
  - `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/pilot_test345.jsonl`
  - `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/pilot_calibrator.json`
  - `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/pilot_summary.json`

## Validation Notes

- Cache integrity: 2,000/345 unique queries, exactly 20 candidates/query, all
  feature values finite.
- Unit tests: 7/7 passed after adding the risk-confidence tie regression.
- Official-evaluator smoke: 64 queries completed in calibrated mode.
- Official inference feature schema contains no GT/error labels.
- A smoke fitter bug that used error as an implicit confidence tie-break and
  compared models on test metrics was found before formal fitting, documented,
  fixed in `b0fce4c`, and covered by regression testing.
