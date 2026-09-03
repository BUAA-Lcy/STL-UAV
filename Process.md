# Research Process Log

This file is an append-only experiment diary for the GTA-UAV post-retrieval
fine-localization project. Record successful and failed runs alike. Formal
headline results still come from the official evaluator and its logs.

## 2026-09-03 — Calibrated multi-tile pose-likelihood line

### Goal

Test whether the current `retrieval top-1 + VOP top-4` pipeline is limited by
premature commitment to one satellite tile and over-confidence in internally
consistent but geographically wrong homographies.

The new hypothesis is:

> Generate a bounded `retrieval top-R × VOP top-k` discrete pose set, calibrate
> candidate reliability from inference-only retrieval/VOP/geometry evidence,
> and abstain to the coarse top-1 center when no fine hypothesis is reliable.

### Git preparation

- Published `RealDataDemo` at `38002b4` to `origin/RealDataDemo`.
- Pushed `codex/vop-experiment` through `06ff837` to
  `origin/codex/vop-experiment`.
- Remote reported that the repository moved to `BUAA-Lcy/STL-UAV.git`, but both
  pushes through the existing remote completed successfully.
- All subsequent work is restricted to `codex/vop-experiment` and GTA-UAV.

### Implementation in progress

Added the following uncommitted experiment infrastructure:

- inference-only pose-likelihood feature extraction and calibrator loading;
- resumable GTA `top-5 tile × top-4 angle` JSONL cache building;
- logistic calibration, temperature scaling, risk-coverage analysis, bounded
  HGB follow-up, and adaptive `1 → 3 → 5` tile expansion;
- opt-in official-evaluator flags whose defaults preserve legacy top-1 behavior;
- standard-library unit tests.

Ground truth is attached only by the offline cache builder. It is not part of
the feature schema or the official evaluator's candidate-selection interface.

### Test attempt 1 — pytest

- Command:
  - `/home/lcy/miniconda3/envs/gtauav/bin/python -m pytest -q tests/test_gta_pose_likelihood.py`
- Result: **FAILED BEFORE TEST EXECUTION**.
- Cause: the `gtauav` environment does not contain `pytest`.
- Decision: do not alter the comparable environment; convert the tests to
  standard-library `unittest`.

### Test attempt 2 — unittest module invocation

- Command:
  - `/home/lcy/miniconda3/envs/gtauav/bin/python -m unittest -v tests.test_gta_pose_likelihood`
- Result: **FAILED BEFORE TEST EXECUTION**.
- Cause: `tests/` is not a Python package, so module discovery could not import
  `tests.test_gta_pose_likelihood`.
- Decision: invoke the test file directly and make its repository import path
  explicit.

### Test attempt 3 — unittest direct invocation

- Command:
  - `/home/lcy/miniconda3/envs/gtauav/bin/python tests/test_gta_pose_likelihood.py -v`
- Result: **PASSED, 6/6 tests**.
- Covered:
  - supported-center geometry and reprojection residuals;
  - empty matches and singular homographies;
  - calibrator schema/probability loading;
  - deterministic candidate tie-breaking;
  - deterministic stratified sampling and cache resume parsing;
  - absence of GT/error labels from the inference feature schema.
- Decision: `KEEP`.

### Smoke cache — same-area train, 64 queries

- Status: **RUNNING** at the time of this entry.
- Candidate budget: retrieval top-5 × VOP top-4.
- Sampling: deterministic stratified sample, seed `20260903`.
- Retrieval checkpoint:
  - `Game4Loc/pretrained/gta/vit_base_eva_gta_same_area.pth`
- VOP checkpoint:
  - `Game4Loc/work_dir/gta_vop_same_area_runs/gta_samearea_fullteacher_exp_c_20260417_125519/artifacts/gta_samearea_useful5_weight30_e6.pth`
- Output:
  - `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/smoke_train64.jsonl`
- Exact command:

```bash
cd /home/lcy/Workplace/GTA-UAV/Game4Loc
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python \
  build_gta_pose_hypothesis_cache.py \
  --data_root ./data/GTA-UAV-data \
  --pairs_meta_file same-area-drone2sate-train.json \
  --checkpoint_start ./pretrained/gta/vit_base_eva_gta_same_area.pth \
  --orientation_checkpoint ./work_dir/gta_vop_same_area_runs/gta_samearea_fullteacher_exp_c_20260417_125519/artifacts/gta_samearea_useful5_weight30_e6.pth \
  --retrieval_topk 5 --orientation_topk 4 \
  --query_limit 64 --sample_mode stratified --sample_seed 20260903 \
  --batch_size 64 --num_workers 0 \
  --output_path ./work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/smoke_train64.jsonl \
  --overwrite
```

### Confirmatory model-family lock — smoke validation

- Added `--model_type {auto,logistic,hist_gradient_boosting}` to the fitting
  entry point.
- `auto` preserves the original pilot protocol; an explicit family disables
  evaluation-triggered family selection.
- A 64-query smoke fit with
  `--model_type hist_gradient_boosting` completed successfully:
  - requested family: `hist_gradient_boosting`;
  - fitted family: `hist_gradient_boosting`;
  - logistic fit: not executed;
  - HGB follow-up trigger: not used.
- The smoke decision was `REJECT`, as expected for a pipeline-only 64-query
  sample; it is not used as paper evidence.
- Decision: `KEEP` (use the explicit HGB family for the final train-only fit).

### Full same-area train cache — durable execution handoff

- The initial interactive process completed both retrieval feature passes and
  wrote 24/13,851 candidate records.
- It was deliberately interrupted to move the multi-hour run away from the
  interactive PTY. The JSONL is flushed once/query, so all 24 records remained
  valid.
- A first plain `nohup` attempt exited immediately with an empty log. This is
  recorded as an execution-wrapper failure, not a model/data failure; no cache
  row was lost or duplicated.
- A foreground `--resume` check accepted the existing manifest fingerprint and
  began feature extraction normally; it was stopped before candidate writing.
- Durable execution was then started as the user-level transient systemd unit:
  - unit: `codex-gta-full-train-cache-20260904.service`;
  - initial main PID: `53993`;
  - mode: the same command with `--resume` instead of `--overwrite`;
  - log: `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/full_train13851.log`.
- The cache configuration and scientific protocol did not change during the
  wrapper migration.
- Decision: `KEEP` (continue the durable resumed process).

Warnings from deprecated AMP/albumentations argument spellings were observed;
they are pre-existing environment/code warnings and have not stopped the run.

### Smoke cache completion and resume check

- Train cache: **COMPLETED**, 64/64 queries.
- Test cache: **COMPLETED**, 64/64 queries.
- Each query contains at most 20 candidates from retrieval top-5 × VOP top-4.
- Resume validation: rerunning the completed train-cache command without
  `--overwrite` returned `Cache already complete` and appended no records.
- Outputs:
  - `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/smoke_train64.jsonl`
  - `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/smoke_test64.jsonl`
- Decision: `KEEP` (pipeline and cache-resume mechanism only).

### Smoke calibration and oracle audit — 64 train / 64 test

- This is a plumbing audit, not a formal statistical decision.
- Coarse retrieval top-1:
  - `Dis@1 = 97.98m`
  - `MA@20 = 3.12%`
- Legacy retrieval top-1 + VOP top-4:
  - `Dis@1 = 45.95m`
  - `MA@20 = 43.75%`
  - `worse-than-coarse = 6.25%`
  - `>50m catastrophic correction = 3.12%`
- Raw retrieval top-5 × VOP top-4, selected by inlier evidence:
  - `Dis@1 = 38.24m`
  - `MA@20 = 45.31%`
  - `worse-than-coarse = 9.38%`
  - `>50m catastrophic correction = 3.12%`
- Oracle retrieval top-5 × VOP top-4:
  - `Dis@1 = 20.21m`
  - `MA@20 = 60.94%`
  - relative `Dis@1` improvement over legacy: `56.01%`
  - `MA@20` improvement over legacy: `+17.19pp`
- The bounded HGB diagnostic was selected after logistic calibration was weak:
  - candidate positive rate: `15.70%`
  - AUROC: `0.8779`
  - AUPRC: `0.5269`
  - NLL: `0.3291`
  - Brier: `0.0979`
  - 15-bin ECE: `0.0701`
  - calibrated mean error at 70% coverage: `30.39m`
  - raw-inlier mean error at 70% coverage: `24.21m`
  - relative risk improvement: `-25.53%`
- Adaptive smoke result:
  - `Dis@1 = 36.74m`
  - `MA@20 = 45.31%`
  - `worse-than-coarse = 6.25%`
  - `>50m catastrophic correction = 3.12%`
  - mean evaluated hypotheses: `9.75`
- Result: oracle headroom passes its provisional threshold, but calibration
  fails ECE and risk-coverage thresholds on this tiny sample.
- Decision: `REJECT` (smoke calibration only). Proceed to the pre-registered
  2,000/345 pilot before deciding the method line.

### Official evaluator smoke — calibrated mode, 64 queries

- Purpose: validate that the unchanged official GTA evaluator can load the
  versioned calibrator, generate retrieval/VOP candidates, apply adaptive
  expansion and abstention, and report standard metrics.
- Configuration:
  - same-area test, first 64 queries (`--query_limit 64`)
  - retrieval top-5, VOP top-4
  - `--fine_selection_mode calibrated_likelihood`
  - smoke HGB calibrator
  - `--num_workers 0`, one RTX 5070, `WANDB_MODE=disabled`
- Result: **COMPLETED**, exit code 0.
- Official metrics:
  - `Dis@1 = 48.12m`
  - `MA@3/5/10/20 = 4.69/10.94/21.88/35.94%`
  - `fallback/abstain = 14.06%`
  - `worse-than-coarse = 15.62%`
  - mean tiles = `2.50`
  - mean hypotheses = `10.00`
  - mean confidence = `0.5280`
  - mean fine-localization time = `0.6989s/query`
- Log:
  - `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/smoke_official_calibrated64.log`
- The `query_yaw_list=None` warning is expected because this line intentionally
  does not use yaw metadata.
- Decision: `KEEP` (official-evaluator integration only; metrics are not paper
  evidence).

### Implementation snapshot

- Validation before commit:
  - `git diff --check`: passed
  - `py_compile`: passed for cache builder, fitter, helper and both evaluator
    entry points
  - standard-library unit tests: 6/6 passed
- Commit: `93b4586` (`feat(gta): add calibrated multi-tile pose hypotheses`)
- Push: completed to `origin/codex/vop-experiment`.
- Remote again reported the repository-moved notice, but accepted the push.
- Decision: `KEEP`.

### Formal pilot train cache — 2,000 queries

- Status: **RUNNING** at the time of this entry.
- Sampling: deterministic stratified sample, seed `20260903`.
- Candidate budget: retrieval top-5 × VOP top-4.
- Output:
  - `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/pilot_train2000.jsonl`
- Exact command:

```bash
cd /home/lcy/Workplace/GTA-UAV/Game4Loc
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python \
  build_gta_pose_hypothesis_cache.py \
  --data_root ./data/GTA-UAV-data \
  --pairs_meta_file same-area-drone2sate-train.json \
  --checkpoint_start ./pretrained/gta/vit_base_eva_gta_same_area.pth \
  --orientation_checkpoint ./work_dir/gta_vop_same_area_runs/gta_samearea_fullteacher_exp_c_20260417_125519/artifacts/gta_samearea_useful5_weight30_e6.pth \
  --retrieval_topk 5 --orientation_topk 4 \
  --query_limit 2000 --sample_mode stratified --sample_seed 20260903 \
  --batch_size 64 --num_workers 0 \
  --output_path ./work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/pilot_train2000.jsonl \
  --overwrite
```

### Pre-pilot fitter audit — two label-leakage defects

- While preparing the result notebook, static inspection found two defects in
  the smoke-only fitter implementation:
  1. Python tuple sorting in the raw-inlier 70%-coverage baseline implicitly
     used candidate error as a tie-break when inlier counts tied.
  2. After the fixed HGB follow-up ran, logistic versus HGB was selected using
     pilot test metrics.
- Impact:
  - the already reported 64/64 smoke calibration/risk values are invalid as
    calibration evidence (they were already marked smoke-only and `REJECT`);
  - cache contents, candidate features, oracle results and official evaluator
    integration are unaffected.
- Fix:
  - sort risk-coverage rows only by observable confidence, preserving input
    order for ties;
  - when the pre-registered HGB follow-up is triggered, report that fixed model
    directly rather than choosing between models on test labels;
  - retain initial logistic metrics in the summary for auditability.
- Formal 2,000/345 fitting will use only the corrected implementation.
- Decision: `REJECT` (the original smoke fitter logic); corrected fitter must
  pass tests before formal use.

### Leakage-fix validation and snapshot

- Added a regression test proving equal inlier-confidence rows are not ordered
  by ground-truth error in risk-coverage calculation.
- Standard-library tests after the fix: **7/7 passed**.
- `git diff --check` and fitter `py_compile`: passed.
- Commit: `b0fce4c` (`fix(gta): prevent pilot label leakage`).
- Push: completed to `origin/codex/vop-experiment`.
- Decision: `KEEP`.

### Pilot train-cache rolling audit — 25%

- Progress: 500/2,000 queries written; no crash or malformed-line event.
- A read-only audit at 460 completed records found:
  - exactly 20 candidates per query;
  - all 24 candidate features finite;
  - prefix legacy top-1 VOP: `Dis@1 = 53.91m`, `MA@20 = 44.35%`;
  - prefix oracle top-5×top-4: `Dis@1 = 18.51m`, `MA@20 = 65.65%`;
  - mean exhaustive candidate time: `1.713s/query`.
- These prefix numbers are run-health diagnostics only and must not be used for
  a gate or paper claim.
- Decision: `KEEP` (continue cache generation).

### Result notebook preparation

- Added a 13-cell reader-facing notebook with:
  - TL;DR, Context & Methods, Data, Results, and Takeaways sections;
  - oracle/headline table, calibration curve, risk-coverage curve, fixed-cost
    configurations, adaptive policy and validation audit;
  - actual cached timing and candidate-count summaries.
- Notebook:
  - `Game4Loc/notebooks/gta_pose_likelihood_results.ipynb`
- Validation attempt 1: **FAILED BEFORE VALIDATION** because `gtauav` did not
  contain `nbformat`; the system and bundled workspace Python also lacked it.
- Resolution:
  - installed `nbformat`, `nbconvert`, and `ipykernel` only into ignored
    `Game4Loc/work_dir/notebook_runtime`;
  - did not modify the `gtauav` conda environment.
- Strict nbformat schema validation then passed for all 13 cells.
- Execution is intentionally deferred until the formal pilot summary/cache
  artifacts exist.
- Decision: `KEEP` (notebook structure; execution still pending).

### Pilot train-cache rolling audit — 50%

- Progress checkpoint: 1,000/2,000 queries; audit read 1,015 complete JSONL
  records while generation continued.
- Integrity:
  - candidate count min/max: 20/20;
  - all cached feature values finite.
- Prefix health metrics:
  - coarse top-1 mean error: `103.41m`;
  - legacy top-1 VOP: `Dis@1 = 48.13m`, `MA@20 = 42.86%`;
  - oracle top-5×top-4: `Dis@1 = 19.11m`, `MA@20 = 63.55%`;
  - mean exhaustive candidate time: `1.744s/query`.
- As at 25%, these are diagnostic-only prefix values. No fitting, threshold
  selection or gate decision was performed.
- Decision: `KEEP` (continue unchanged).

### Formal pilot train cache — completion

- Result: **COMPLETED**, exit code 0.
- Final integrity audit:
  - 2,000 JSONL records;
  - 2,000 unique query names;
  - exactly 20 candidates per query;
  - all candidate feature values finite;
  - manifest fingerprint:
    `b19cad0b8a39c2f4d9c715bd02aa6db36cf35cb48dffcb475d18f978670c07e4`.
- Candidate-stage elapsed time reported by the builder: `3682.7s`.
- No interruption/resume was needed during this run.
- Decision: `KEEP`.

### Formal pilot test cache — 345 queries

- Status: **RUNNING** at the time of this entry.
- Sampling: deterministic stratified sample, seed `20260903`.
- Candidate budget: retrieval top-5 × VOP top-4.
- Output:
  - `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/pilot_test345.jsonl`
- Exact command:

```bash
cd /home/lcy/Workplace/GTA-UAV/Game4Loc
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python \
  build_gta_pose_hypothesis_cache.py \
  --data_root ./data/GTA-UAV-data \
  --pairs_meta_file same-area-drone2sate-test.json \
  --checkpoint_start ./pretrained/gta/vit_base_eva_gta_same_area.pth \
  --orientation_checkpoint ./work_dir/gta_vop_same_area_runs/gta_samearea_fullteacher_exp_c_20260417_125519/artifacts/gta_samearea_useful5_weight30_e6.pth \
  --retrieval_topk 5 --orientation_topk 4 \
  --query_limit 345 --sample_mode stratified --sample_seed 20260903 \
  --batch_size 64 --num_workers 0 \
  --output_path ./work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/pilot_test345.jsonl \
  --overwrite
```

### Formal pilot test cache — completion

- Result: **COMPLETED**, exit code 0.
- Final integrity audit:
  - 345 JSONL records and 345 unique query names;
  - exactly 20 candidates per query;
  - all candidate feature values finite;
  - manifest fingerprint:
    `b4db3419d6ef9054363651c5e9f63ec74e465877edf2f95545ad90ed49765b2d`.
- Decision: `KEEP` (cache artifact).

### Formal pilot / oracle / calibration result — 2,000 train, 345 test

- Corrected fitter commit: `b0fce4c`.
- Logistic model was fitted first with `C ∈ {0.1, 1, 10}` and temperature
  scaling under the fixed query-group split.
- The logistic calibration gate failed, so the one allowed fixed shallow HGB
  follow-up was run. No further model or threshold sweep was performed.
- Test baselines:
  - coarse top-1: `Dis@1 = 140.02m`, `MA@20 = 7.54%`;
  - legacy top-1 VOP: `Dis@1 = 86.36m`, `MA@20 = 42.61%`,
    `worse-than-coarse = 11.88%`, catastrophic `= 2.90%`;
  - raw top-5×top-4: `Dis@1 = 158.98m`, `MA@20 = 45.22%`,
    `worse-than-coarse = 16.23%`, catastrophic `= 7.83%`;
  - oracle top-5×top-4: `Dis@1 = 26.59m`, `MA@20 = 65.51%`.
- Oracle headroom versus legacy:
  - relative `Dis@1` reduction: `69.21%` (required ≥15%);
  - `MA@20` increase: `+22.90pp` (required ≥8pp);
  - oracle gate: **PASSED**.
- Initial logistic audit:
  - grouped split: 1,400 train / 300 model-selection / 300 calibration
    queries;
  - selected `C = 1.0`, validation NLL `0.4272`, temperature `1.2115`;
  - AUROC `0.8532`, AUPRC `0.4289`, NLL `0.4607`, Brier `0.1476`,
    ECE `0.2129`;
  - 70%-coverage mean error `24.69m` versus raw-inlier `26.68m`;
  - risk improvement `7.45%`.
- Fixed HGB follow-up audit:
  - AUROC `0.8682` (required ≥0.75): passed;
  - AUPRC `0.5087`, NLL `0.2833`, Brier `0.0872`;
  - ECE `0.0206` (required ≤0.05): passed;
  - 70%-coverage mean error `22.94m` versus raw-inlier `26.68m`;
  - risk improvement `13.99%` (required ≥15%): **failed**.
- Learned adaptive policy test result (reported diagnostically despite failed
  gate):
  - `Dis@1 = 74.72m`, `MA@20 = 47.25%`;
  - `worse-than-coarse = 5.51%`, catastrophic `= 0.58%`;
  - mean hypotheses/query `= 7.08`.
- Artifacts:
  - `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/pilot_calibrator.json`
  - `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/pilot_summary.json`
  - `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/pilot_summary.md`
- Interpretation:
  - multi-tile candidate generation has strong oracle headroom;
  - naive inlier selection is unsafe and catastrophically worsens mean error;
  - the fixed nonlinear calibrator is well ranked and calibrated, but narrowly
    misses the pre-registered selective-risk improvement requirement;
  - per the plan, do not generate full 13,851/3,443 caches and do not enter
    cross-area validation.
- Decision: `REJECT`.

### Result notebook execution

- Runtime dependencies were loaded from the isolated ignored directory:
  - `Game4Loc/work_dir/notebook_runtime`
- Execution command used `nbconvert --execute --inplace` with the `gtauav`
  Python kernel and a 600-second cell timeout.
- Result: **COMPLETED**, exit code 0.
- Post-execution validation:
  - 13 total cells;
  - 8/8 code cells have execution counts;
  - 13 output blocks;
  - 0 error outputs;
  - strict nbformat schema validation passed.
- The TCP-without-encryption warning is a local ephemeral kernel transport
  warning; no remote notebook service was used.
- Decision: `KEEP` (executed audit artifact).

### Legacy evaluator compatibility audit

- Compared `Game4Loc/game4loc/evaluate/gta.py` against pre-implementation
  commit `06ff837`.
- The legacy `prior_topk` candidate-selection block is source-identical; the
  new multi-tile branch is inserted before it and is disabled when:
  - `fine_retrieval_topk = 1`;
  - `fine_selection_mode = legacy_inlier`.
- Added a regression test for these three official evaluator defaults,
  including the empty calibrator path.
- This is a source/control-flow equivalence check: every query under the
  default flags executes the unchanged legacy block. The 64-query official
  calibrated smoke separately validates the opt-in branch.
- Decision: `KEEP`.

### Final validation before report commit

- `git diff --check`: passed.
- `py_compile`: passed for cache builder, fitter, likelihood helper, CLI
  evaluator and official GTA evaluator.
- Standard-library unit tests: **8/8 passed**.
- Executed notebook and JSON summary consistency audit:
  - first attempt passed assertions but emitted a `ResourceWarning` because the
    one-off audit script did not explicitly close `pilot_summary.json`;
  - corrected `Path.read_text()` audit passed under `-W error` with no warning;
  - confirmed decision `REJECT`, oracle gate passed, calibration gate failed,
    and the single HGB follow-up was used.
- Decision: `KEEP` (validation), while the research method decision remains
  `REJECT`.

### Logistic-detail reproducibility check

- The final HGB artifact replaces the logistic payload, so the selected
  logistic hyperparameter was not retained in `pilot_summary.json`.
- First read-only recomputation completed, but the command wrapper lost its
  stdout session handle; no result was claimed from that invocation.
- A second deterministic recomputation on the same frozen cache and seed
  completed successfully and reported:
  - `C = 1.0`;
  - validation NLL `0.4271913`;
  - temperature `1.2115207`;
  - split sizes 1,400/300/300 queries.
- No artifact, model choice or experiment decision was changed.
- Decision: `KEEP` (audit detail only).

## 2026-09-03 — Statistical follow-up after the user re-opened the line

### Motivation and pre-registered interpretation

- The user explicitly requested that experiments continue after questioning
  the heuristic `70% coverage / 15% improvement` gate.
- This re-opens the line under a new, statistically interpretable audit; it
  does not retroactively change the recorded result of the first plan.
- Fixed inputs:
  - the existing 2,000-query training cache;
  - the existing 345-query test cache;
  - the already fitted HGB calibrator and adaptive policy;
  - no refitting, feature change, threshold tuning or new model sweep.
- Statistical audit to run before any new GPU cache:
  - paired query bootstrap with 10,000 replicates and seed `20260903`;
  - effect sizes and percentile 95% confidence intervals for adaptive versus
    legacy `Dis@1`, `MA@20`, worse-than-coarse and catastrophic rates;
  - tie-aware selective risk at 50%, 70% and 90% coverage;
  - tie-aware AURC over coverage 10%–100%.
- Tie-aware means that when a coverage cutoff falls inside a set of equal
  confidence values—common for integer inlier counts—the expected error of the
  whole tie group is fractionally included. Query file order cannot break ties.
- Follow-up decision rule:
  - `KEEP` if the 95% CIs support improvements in adaptive mean error, MA@20,
    AURC, and non-increase in catastrophic rate;
  - `NEEDS ONE FOLLOW-UP` if all central estimates are favorable but one or
    more CIs include zero; the one follow-up is frozen-calibrator evaluation on
    the complete 3,443-query same-area test set;
  - `REJECT` if a central estimate is unfavorable.
- This rule tests direction and uncertainty rather than imposing another
  arbitrary minimum percentage improvement.

### Statistical follow-up result — 10,000 paired bootstraps

- Implementation:
  - `Game4Loc/audit_gta_pose_likelihood_statistics.py`
  - tie-aware risk regression added to the unit-test suite
  - risk curves use one stable sort and fractional expected error within a
    confidence tie group
- Validation: 9/9 standard-library tests passed; script compiled and
  `git diff --check` passed.
- Adaptive versus legacy paired effects (positive favors adaptive):
  - mean-error improvement: `11.64m`, 95% CI `[7.38, 16.53]`;
  - MA@20 improvement: `+4.64pp`, 95% CI `[1.74, 7.83]`;
  - worse-than-coarse reduction: `6.38pp`, 95% CI `[3.77, 9.28]`;
  - catastrophic +50m reduction: `2.32pp`, 95% CI `[0.58, 4.06]`.
- Tie-aware selective-risk improvement:
  - 50% coverage: `5.51m` / `20.56%`, 95% CI in metres
    `[2.17, 8.78]`;
  - 70% coverage: `3.69m` / `13.86%`, 95% CI `[1.53, 6.17]`;
  - 90% coverage: `3.24m` / `10.96%`, 95% CI `[0.80, 53.29]`.
- Tie-aware AURC over 10%–100% coverage:
  - calibrated: `24.77m`;
  - raw inlier: `34.62m`;
  - improvement: `9.86m` / `28.47%`, 95% CI `[3.41, 19.29]`.
- All pre-registered central effects are favorable and all required confidence
  intervals support the direction of improvement.
- Artifacts:
  - `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/statistical_audit.json`
  - `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/statistical_audit.md`
- Decision: `KEEP`.

### Frozen-calibrator full-test follow-up — start

- The fitted 2,000-query HGB calibrator and adaptive thresholds remain frozen.
- Generate all 3,443 same-area test queries at top-5×top-4.
- Report two views after completion:
  - all 3,443 queries for complete-protocol comparability;
  - the 3,098 queries not used in the 345-query pilot as the primary
    confirmatory holdout.
- No model, feature or threshold is selected using the 3,098-query holdout.
- Output:
  - `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/full_test3443_frozen2000.jsonl`
- Exact command:

```bash
cd /home/lcy/Workplace/GTA-UAV/Game4Loc
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python \
  build_gta_pose_hypothesis_cache.py \
  --data_root ./data/GTA-UAV-data \
  --pairs_meta_file same-area-drone2sate-test.json \
  --checkpoint_start ./pretrained/gta/vit_base_eva_gta_same_area.pth \
  --orientation_checkpoint ./work_dir/gta_vop_same_area_runs/gta_samearea_fullteacher_exp_c_20260417_125519/artifacts/gta_samearea_useful5_weight30_e6.pth \
  --retrieval_topk 5 --orientation_topk 4 \
  --query_limit 0 --sample_mode sequential --sample_seed 20260903 \
  --batch_size 64 --num_workers 0 \
  --output_path ./work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/full_test3443_frozen2000.jsonl \
  --overwrite
```

### Frozen-calibrator full-test progress — 500/3,443

- Candidate generation reached 500/3,443 queries in the original process.
- Candidate-stage elapsed time: approximately `942.8s`.
- Existing AMP/albumentations warnings only; no crash, malformed record or
  manual exclusion.
- Long-tail queries, including coarse errors above 500m and cases where every
  fine candidate is worse than coarse, remain in the cache.
- Decision: `KEEP` (continue unchanged).

### Frozen-calibrator full-test progress — 1,000/3,443

- Original process reached 1,000/3,443 queries.
- Candidate-stage elapsed time: approximately `1868.4s`.
- Cache remains resumable and no configuration changed.
- Decision: `KEEP` (continue unchanged).

### Frozen-calibrator full-test progress — halfway

- Original process reached 1,725/3,443 queries.
- Candidate-stage elapsed time: approximately `3230.5s`.
- No interim metric/gate calculation was performed on the growing cache.
- Decision: `KEEP` (continue unchanged).

### Frozen-calibrator full-test progress — 2,000/3,443

- Original process reached 2,000/3,443 queries after crossing midnight into
  2026-09-04 local time.
- Candidate-stage elapsed time: approximately `3746.5s`.
- No restart, interim selection or configuration change.
- Decision: `KEEP` (continue unchanged).

### Frozen-calibrator full-test progress — 2,500/3,443

- Original process reached 2,500/3,443 queries.
- Candidate-stage elapsed time: approximately `4677.1s`.
- No interruption or configuration change.
- Decision: `KEEP` (continue unchanged).

### Frozen-calibrator full-test progress — 3,000/3,443

- Original process reached 3,000/3,443 queries.
- Candidate-stage elapsed time: approximately `5599.4s`.
- No interruption, filtering or configuration change.
- Decision: `KEEP` (continue unchanged).

### Frozen-calibrator full-test completion and confirmatory result

- Cache generation completed in the original process:
  - 3,443 records / 3,443 unique queries;
  - exactly 20 candidates/query;
  - all candidate feature values finite;
  - candidate-stage elapsed time approximately `6385.9s`;
  - manifest fingerprint:
    `418d7d64630c26acb02e547667d3654b1c550cd754395756e29dc71fc5d130f2`.
- Full 3,443-query matched offline results:
  - legacy top-1 VOP: `Dis@1 = 58.16m`, `MA@20 = 46.09%`,
    worse-than-coarse `11.88%`, catastrophic `3.19%`;
  - raw top-5×top-4: `139.41m`, `48.56%`, `16.82%`, `8.86%`;
  - oracle top-5×top-4: `23.04m`, `67.76%`;
  - frozen adaptive: `48.99m`, `49.75%`, `6.62%`, `0.90%`;
  - mean adaptive hypotheses: `7.37/query`.
- Full paired bootstrap:
  - Dis@1 improvement `9.17m`, 95% CI `[7.55,10.84]`;
  - MA@20 improvement `3.66pp`, 95% CI `[2.73,4.59]`;
  - catastrophic reduction `2.29pp`, 95% CI `[1.71,2.90]`;
  - AURC improvement `10.13m`, 95% CI `[7.96,12.81]`.
- Primary non-pilot holdout (3,098 queries):
  - legacy: `Dis@1 = 54.74m`, `MA@20 = 46.71%`, catastrophic `2.97%`;
  - adaptive: `45.90m`, `MA@20 = 50.26%`, catastrophic `0.90%`;
  - Dis@1 improvement `8.84m`, 95% CI `[7.12,10.64]`;
  - MA@20 improvement `3.55pp`, 95% CI `[2.58,4.55]`;
  - catastrophic reduction `2.07pp`, 95% CI `[1.45,2.71]`;
  - AURC improvement `9.77m`, 95% CI `[7.63,12.41]`.
- The 90%-coverage holdout point has an interval crossing zero, but the
  pre-registered full-curve AURC interval is strictly positive.
- Original full same-area gates are met by the frozen 2,000-query calibrator:
  - Dis@1 relative reduction `15.76%` (required ≥5%);
  - adaptive Dis@1 `48.99m` (target ≤54.75m);
  - MA@20 increase `3.66pp` (required ≥2pp);
  - adaptive MA@20 `49.75%` (target ≥47.92%);
  - worse-than-coarse and catastrophic rates both decrease.
- Decision: `KEEP`.

### Full same-area train cache — start

- Next stage: generate all 13,851 same-area train queries at top-5×top-4.
- The retrieval checkpoint, full-teacher Exp C VOP and matcher remain frozen.
- After completion, fit the final calibrator only on this train cache; the full
  test cache will not participate in fitting or policy selection.
- Final model family is fixed before the full-train fit to the single pilot
  follow-up model, `hist_gradient_boosting`. The fitting CLI now requires this
  explicit confirmatory choice via
  `--model_type hist_gradient_boosting`; the test cache may report metrics but
  cannot trigger or select a model family.
- Output:
  - `Game4Loc/work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/full_train13851.jsonl`
- Exact command:

```bash
cd /home/lcy/Workplace/GTA-UAV/Game4Loc
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python \
  build_gta_pose_hypothesis_cache.py \
  --data_root ./data/GTA-UAV-data \
  --pairs_meta_file same-area-drone2sate-train.json \
  --checkpoint_start ./pretrained/gta/vit_base_eva_gta_same_area.pth \
  --orientation_checkpoint ./work_dir/gta_vop_same_area_runs/gta_samearea_fullteacher_exp_c_20260417_125519/artifacts/gta_samearea_useful5_weight30_e6.pth \
  --retrieval_topk 5 --orientation_topk 4 \
  --query_limit 0 --sample_mode sequential --sample_seed 20260903 \
  --batch_size 64 --num_workers 0 \
  --output_path ./work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/full_train13851.jsonl \
  --overwrite
```
