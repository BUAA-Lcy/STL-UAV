#!/usr/bin/env bash
# Sequential same-snapshot official evaluation; no training or policy tuning.
set -euo pipefail
cd "$(dirname "$0")/.."

run_dir="./work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/official_fulltrain_20260904"
calibrator="./work_dir/gta_pose_likelihood_runs/gta_multitile_20260903/final_fulltrain_hgb_calibrator.json"
task_python="/home/lcy/miniconda3/envs/gtauav/bin/python"
test -f "$calibrator"
mkdir -p "$run_dir"
export WANDB_MODE=disabled

common=(
  --data_root ./data/GTA-UAV-data
  --test_pairs_meta_file same-area-drone2sate-test.json
  --test_mode pos --query_mode D2S
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k
  --checkpoint_start ./pretrained/gta/vit_base_eva_gta_same_area.pth
  --orientation_checkpoint ./work_dir/gta_vop_same_area_runs/gta_samearea_fullteacher_exp_c_20260417_125519/artifacts/gta_samearea_useful5_weight30_e6.pth
  --orientation_mode prior_topk --orientation_topk 4
  --with_match --sparse --num_workers 0 --batch_size 64 --gpu_ids 0
)
snapshot="$(git rev-parse HEAD)"
code_fingerprint="$(git ls-files -z -- '*.py' | xargs -0 sha256sum | sha256sum)"
calibrator_fingerprint="$(sha256sum "$calibrator")"
echo "Official full same-area evaluation snapshot=$snapshot"
echo "Tracked Python code fingerprint=$code_fingerprint"
sha256sum "$calibrator"
for variant in legacy_top1 raw_top5x4 adaptive_calibrated; do
  test "$(sha256sum "$calibrator")" = "$calibrator_fingerprint" || {
    echo "Calibrator changed during matched evaluation" >&2; exit 1
  }
  test "$(git ls-files -z -- '*.py' | xargs -0 sha256sum | sha256sum)" = "$code_fingerprint" || {
    echo "Python code changed during matched evaluation; aborting before $variant" >&2
    exit 1
  }
  log="$run_dir/$variant.log"
  test ! -e "$log" || { echo "Refusing to overwrite $log" >&2; exit 1; }
  case "$variant" in
    legacy_top1) extra=(--fine_retrieval_topk 1 --fine_selection_mode legacy_inlier) ;;
    raw_top5x4) extra=(--fine_retrieval_topk 5 --fine_selection_mode legacy_inlier) ;;
    adaptive_calibrated) extra=(--fine_retrieval_topk 5 --fine_selection_mode calibrated_likelihood --fine_calibrator_path "$calibrator") ;;
  esac
  echo "START $variant $(date -Is)"
  "$task_python" -u eval_gta.py "${common[@]}" "${extra[@]}" > "$log" 2>&1
  echo "DONE $variant $(date -Is)"
done
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$task_python" summarize_gta_pose_official.py --run_dir "$run_dir"
echo "ALL_OFFICIAL_RUNS_COMPLETE $(date -Is)"
