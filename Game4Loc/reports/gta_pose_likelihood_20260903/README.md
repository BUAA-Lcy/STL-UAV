# GTA-UAV Multi-Tile Pose-Likelihood Experiments

## Decision

Initial pilot: `REJECT` under the heuristic 70%/15% rule.
Statistical and frozen full-test follow-up: `KEEP`.
Final full-train offline audit: `KEEP`.
Final matched official same-area evaluation: `KEEP` (3,443 queries).

The bounded multi-tile candidate pool has substantial oracle headroom, but the
initial pre-registered reliability gate was not met after the single allowed
nonlinear follow-up. That initial experiment stopped. The user subsequently
reopened the line; later stages are recorded separately below.

## Historical Pilot Protocol

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

## Historical Pilot Results

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

The full same-area thresholds are met in this offline audit by the frozen
2,000-query calibrator. These are cached results, not an official headline table.

## Final Full-Train Offline Audit

### Experiment Name

- Refit the pre-fixed HGB family using the complete same-area training cache.

### Change Compared to Baseline

- Frozen retrieval, full-teacher Exp C VOP and matcher; only the calibrated
  selector uses the expanded 13,851-query training cache.
- HGB settings remain depth 3, 100 maximum iterations, learning rate 0.05,
  L2 regularization 1.0 and seed 20260903. Fit uses 11,773 queries;
  temperature/policy calibration uses the remaining 2,078 queries.
- Temperature: 0.937807. Expansion thresholds: R1 0.221187, R3 0.411756;
  abstention threshold: 0.256296. These are selected on train only.

### Quantitative Results

| Variant (same 3,443-query cache) | Dis@1 (m) | MA@20 (%) | Worse than coarse (%) | Catastrophic +50m (%) | Mean hypotheses |
|---|---:|---:|---:|---:|---:|
| Legacy top-1 VOP | 58.157 | 46.094 | 11.879 | 3.195 | 4 |
| Final full-train adaptive | 49.467 | 49.259 | 5.896 | 0.668 | 6.788 |
| Frozen 2,000-query adaptive (historical follow-up) | 48.990 | 49.753 | 6.622 | 0.900 | 7.369 |

- Dis@1 reduction: 14.94%; MA@20 gain: 3.166pp; both robustness rates decrease.
- Paired bootstrap (10,000 replicates): error reduction 8.691m, 95% CI
  [7.136, 10.284]; MA@20 gain 3.166pp, [2.295, 4.008]; catastrophic reduction
  2.527pp, [1.975, 3.108].
- Candidate AUROC 0.891212, AUPRC 0.598288, NLL 0.278925, Brier 0.086758,
  15-bin equal-mass ECE 0.012620.
- Tie-aware AURC reduction: 10.390m, 95% CI [8.168, 13.057]. The 90%-coverage
  point remains inconclusive: improvement CI [-1.577, 26.912]m.
- Exhaustive cache VOP+matcher time: train 1.7050s/query, test 1.7621s/query.
  This is not adaptive runtime. Official matched timing is reported below.

### Interpretation

- Full-train calibration improves ranking/calibration metrics and reduces
  catastrophic corrections versus the earlier 2,000-query calibrator, but its
  mean localization error and MA@20 are slightly worse. No test-driven choice
  between those two artifacts is made.
- The fitter still reports `REJECT` under its historical single-point 15%
  heuristic (observed risk improvement 10.08%). The reopened full-stage rule
  is recorded independently; the historical output is preserved verbatim.
- Query bootstrap assumes exchangeable queries; spatial correlation is not
  modeled. Candidate calibration is not proof of calibrated spatial density.
- Cache/official fallback semantics require care: raw inlier selection picks
  a non-top1 tile's fallback center on 122 test queries, whereas the official
  evaluator returns coarse top-1 in such cases. Final adaptive selection has
  zero such above-threshold cases (494 confidence abstentions). Consequently,
  the raw cache row must not be used as an official baseline.

### Decision

`KEEP` — offline full-stage gates and paired audit pass. The subsequent matched
official confirmation follows; the offline rows above are not headline results.

## Final Matched Official Same-Area Evaluation

### Experiment Name

- Validate final calibrated adaptive multi-tile selection against matched
  top-1 VOP and raw multi-tile selection in the official evaluator.

### Change Compared to Baseline

- Baseline: top-1 retrieved tile, four VOP angles, legacy inlier selection.
- Raw: five retrieved tiles, four angles each, legacy inlier selection.
- Adaptive: the pre-fixed final full-train HGB, train-selected 1→3→5 expansion
  and abstention thresholds. No test-driven refit or threshold adjustment.
- All runs use inference snapshot `4cb181e`, the same 3,443 test queries,
  14,640 gallery tiles, retrieval/VOP checkpoints, sparse defaults, batch size
  64, one RTX 5070, `gtauav`, workers 0 and W&B disabled. Runs are sequential.
- The reporting-only log-source repair was applied after all rows finished;
  it did not change inference or require rerunning any evaluator row.

### Quantitative Results

| Official variant | Dis@1 (m) | MA@3 (%) | MA@5 (%) | MA@10 (%) | MA@20 (%) | Worse than coarse (%) | Catastrophic +50m (%) | Fallback (%) | Mean hypotheses | VOP+matcher (s/query) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Legacy top-1 VOP | 56.9898 | 4.3276 | 10.4850 | 24.4845 | 46.5582 | 12.0825 | 2.5559 | 9.5847 | 4.000 | 0.260941 |
| Raw top-5×4 | 73.8959 | 4.2695 | 10.1946 | 25.5010 | 49.8693 | 12.8086 | 5.7508 | 5.9832 | 20.000 | 1.334793 |
| Calibrated adaptive | 48.8255 | 4.4438 | 10.5141 | 24.5716 | 48.6494 | 5.8089 | 0.6971 | 15.5969 | 6.937 | 0.452635 |

All rows: Recall@1/5/10 = `91.1124/99.3901/99.5353%`, mAP `94.8114%`,
Dis@3/5 = `165.2348/216.9398m`. Per-query coarse errors match exactly.

| Official variant | Worse count | Catastrophe count | Fallback count | Identity-H | Out of bounds | Invalid projection | Mean retained | Mean inliers | Inlier ratio | VOP s/query | Matcher s/query | Overall evaluator s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Legacy | 416 | 88 | 330 | 326 | 4 | 0 | 263.08 | 84.74 | 0.2935 | 0.036453 | 0.224488 | 1272.721 |
| Raw | 441 | 198 | 206 | 206 | 0 | 0 | 237.96 | 88.40 | 0.3391 | 0.180886 | 1.153907 | 5234.727 |
| Adaptive | 200 | 24 | 537 | 537 | 0 | 0 | 174.06 | 53.28 | 0.2858 | 0.063020 | 0.389614 | 1904.061 |

Paired query bootstrap, seed `20260903`, 10,000 replicates (adaptive vs legacy):

| Improvement | Estimate | 95% interval |
|---|---:|---:|
| Mean error reduction (m) | 8.164 | [5.841, 11.137] |
| MA@20 gain (pp) | 2.091 | [0.958, 3.253] |
| Worse-than-coarse reduction (pp) | 6.274 | [5.257, 7.232] |
| Catastrophic-rate reduction (pp) | 1.859 | [1.278, 2.440] |

### Interpretation

- The adaptive method lowers mean error by **14.33%** and catastrophic
  corrections from **88 to 24**, while improving MA@20 by **2.09pp**.
  All four pre-agreed full-stage point-estimate gates pass.
- MA@20 narrowly clears the 2pp engineering gate; its confidence interval
  supports positive improvement but not a guaranteed improvement above 2pp.
  Query-level intervals do not account for spatial correlation or training
  randomness, and this is not a newly untouched test dataset.
- Raw expansion has slightly higher MA@20 than adaptive, but worse mean
  error and 198 catastrophic corrections. Do not claim adaptive wins every
  threshold metric; its benefit is the accuracy/robustness tradeoff.
- Adaptive costs about **1.73×** legacy VOP+matcher time, but about **66.1% less**
  than raw expansion. These timers exclude some retrieval/selection overhead
  and are not end-to-end deployment latency. Overall evaluator time is
  reported separately, also not a per-frame online navigation benchmark.
- Abstention increases fallback from 9.58% to 15.60%; lower catastrophe rate
  is not a free improvement in acceptance coverage. The original 70%/15%
  selective-risk heuristic remains historical, not a new universal criterion.
- Raw cache/official semantics differ: cached fallback candidates can use
  non-top1 tile centers, whereas official fallback uses coarse top1. Use the
  official table above for headline comparisons; do not explain the full
  cached/official discrepancy by that difference alone without decomposition.
- No matched dense rerun or cross-area validation was performed. No SOTA,
  dense superiority, or calibrated continuous pose-density claim is supported.

### Decision

`KEEP` — final calibrated adaptive same-area method. Raw uncalibrated expansion
is `REJECT` as a main method and retained as an ablation. Stop this scheduled
stage here; cross-area needs a separately authorized next stage.

## Artifacts

- Executed notebook: `Game4Loc/notebooks/gta_pose_likelihood_results.ipynb`
- Append-only chronology: `Process.md`
- Committed audit snapshots in this report directory:
  - `full_cache_integrity.json`
  - `final_fulltrain_hgb_summary.json` (retains historical heuristic decision)
  - `final_fulltrain_statistical_audit.json`
  - `official_summary.json` (matched metrics, intervals, source-log hashes)
- Final inference artifact (ignored run directory):
  - `final_fulltrain_hgb_calibrator.json`
- Official runner: `Game4Loc/scripts/run_gta_pose_full_evaluation.sh`
- Ignored runtime artifacts:
  - `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/pilot_train2000.jsonl`
  - `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/pilot_test345.jsonl`
  - `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/pilot_calibrator.json`
  - `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/pilot_summary.json`

## Validation Notes

- Full cache integrity: 13,851/3,443 unique train/test queries, exactly 20
  candidates/query, all features finite, no query-name overlap, manifest
  fingerprints verified and checkpoint size/mtime identities unchanged.
- Unit tests: 15/15 passed, including precise official-log parsing, duplicate
  query rejection, split console/file handlers and ambiguous/missing sources.
- All three official runs completed; each contains 3,443 unique audit queries,
  with matching coarse errors and audit means matching official Dis@1.
- The initial automatic summarizer failed because console logs omit DEBUG
  audit rows. Its traceback is preserved in `official_fulltrain_20260904/runner.log`.
  The repaired summarizer reads only the explicitly named detailed app log;
  original logs were neither overwritten nor concatenated.
- Official-evaluator smoke: 64 queries completed in calibrated mode.
- Official inference feature schema contains no GT/error labels.
- A smoke fitter bug that used error as an implicit confidence tie-break and
  compared models on test metrics was found before formal fitting, documented,
  fixed in `b0fce4c`, and covered by regression testing.
