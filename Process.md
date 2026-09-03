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
