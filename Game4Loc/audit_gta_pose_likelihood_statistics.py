"""Paired bootstrap and tie-aware selective-risk audit for GTA pose likelihood."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from fit_gta_pose_likelihood import candidate_pool, load_jsonl, raw_choice
from game4loc.evaluate.gta_pose_likelihood import PoseLikelihoodCalibrator, serializable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_cache", required=True)
    parser.add_argument("--calibrator_path", required=True)
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def tie_aware_risk(confidence: np.ndarray, errors: np.ndarray, coverage: float) -> float:
    """Expected selective risk when confidence ties straddle the cutoff."""
    return float(tie_aware_risk_curve(confidence, errors, [coverage])[0])


def tie_aware_risk_curve(
    confidence: np.ndarray, errors: np.ndarray, coverages: Sequence[float]
) -> np.ndarray:
    """Compute a tie-aware curve with one stable sort and grouped prefix sums."""
    confidence = np.asarray(confidence, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    coverage_values = np.asarray(coverages, dtype=np.float64)
    targets = float(len(errors)) * coverage_values
    if len(errors) == 0 or np.any(targets <= 0.0) or np.any(targets > len(errors) + 1e-12):
        raise ValueError("coverage and errors must be non-empty and positive")
    order = np.argsort(-confidence, kind="stable")
    ordered_confidence = confidence[order]
    ordered_errors = errors[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_confidence) != 0.0) + 1]
    counts = np.diff(np.r_[starts, len(errors)]).astype(np.float64)
    sums = np.add.reduceat(ordered_errors, starts)
    means = sums / counts
    cumulative_counts = np.cumsum(counts)
    cumulative_sums = np.cumsum(sums)
    group_indices = np.searchsorted(cumulative_counts, targets, side="left")
    previous_counts = np.where(group_indices > 0, cumulative_counts[np.maximum(group_indices - 1, 0)], 0.0)
    previous_sums = np.where(group_indices > 0, cumulative_sums[np.maximum(group_indices - 1, 0)], 0.0)
    expected_sums = previous_sums + (targets - previous_counts) * means[group_indices]
    return expected_sums / targets


def normalized_auc(coverages: np.ndarray, risks: np.ndarray) -> float:
    width = float(coverages[-1] - coverages[0])
    return float(np.trapezoid(risks, coverages) / max(width, 1e-12))


def percentile_interval(values: np.ndarray) -> Tuple[float, float]:
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975])
    return float(low), float(high)


def query_rows(records: Sequence[Mapping], calibrator: PoseLikelihoodCalibrator) -> Dict[str, np.ndarray]:
    policy = calibrator.policy
    orientation_topk = int(policy.get("orientation_topk", 4))
    result = {
        "coarse_error": [],
        "legacy_error": [],
        "adaptive_error": [],
        "full_calibrated_error": [],
        "full_calibrated_confidence": [],
        "raw_error": [],
        "raw_confidence": [],
        "adaptive_hypotheses": [],
    }
    for record in records:
        coarse_error = float(record["coarse_top1_error_m"])
        legacy = raw_choice(candidate_pool(record, 1, 4))
        raw = raw_choice(candidate_pool(record, 5, 4))
        full_selected, full_confidence = calibrator.best_candidate(candidate_pool(record, 5, 4))

        selected, confidence = calibrator.best_candidate(candidate_pool(record, 1, orientation_topk))
        stage = 1
        if confidence < float(policy.get("threshold_r1", 1.1)):
            selected, confidence = calibrator.best_candidate(candidate_pool(record, 3, orientation_topk))
            stage = 3
            if confidence < float(policy.get("threshold_r3", 1.1)):
                selected, confidence = calibrator.best_candidate(candidate_pool(record, 5, orientation_topk))
                stage = 5
        adaptive_error = (
            coarse_error
            if confidence < float(policy.get("abstain_threshold", 0.0))
            else float(selected["error_m"])
        )

        result["coarse_error"].append(coarse_error)
        result["legacy_error"].append(float(legacy["error_m"]))
        result["adaptive_error"].append(adaptive_error)
        result["full_calibrated_error"].append(float(full_selected["error_m"]))
        result["full_calibrated_confidence"].append(float(full_confidence))
        result["raw_error"].append(float(raw["error_m"]))
        result["raw_confidence"].append(float(raw.get("inliers", 0)))
        result["adaptive_hypotheses"].append(float(stage * orientation_topk))
    return {name: np.asarray(values, dtype=np.float64) for name, values in result.items()}


def effect_summary(rows: Mapping[str, np.ndarray], bootstrap_indices: np.ndarray) -> Dict[str, dict]:
    legacy = rows["legacy_error"]
    adaptive = rows["adaptive_error"]
    coarse = rows["coarse_error"]

    def boot_mean(values: np.ndarray) -> np.ndarray:
        return np.mean(values[bootstrap_indices], axis=1)

    arrays = {
        "Dis@1_improvement_m": boot_mean(legacy - adaptive),
        "MA@20_improvement_pp": 100.0
        * boot_mean((adaptive < 20.0).astype(float) - (legacy < 20.0).astype(float)),
        "worse_than_coarse_reduction_pp": 100.0
        * boot_mean((legacy > coarse + 1e-6).astype(float) - (adaptive > coarse + 1e-6).astype(float)),
        "catastrophic_50m_reduction_pp": 100.0
        * boot_mean(
            (legacy > coarse + 50.0).astype(float) - (adaptive > coarse + 50.0).astype(float)
        ),
    }
    central = {
        "Dis@1_improvement_m": float(np.mean(legacy - adaptive)),
        "MA@20_improvement_pp": float(
            100.0 * np.mean((adaptive < 20.0).astype(float) - (legacy < 20.0).astype(float))
        ),
        "worse_than_coarse_reduction_pp": float(
            100.0 * np.mean((legacy > coarse + 1e-6).astype(float) - (adaptive > coarse + 1e-6).astype(float))
        ),
        "catastrophic_50m_reduction_pp": float(
            100.0
            * np.mean(
                (legacy > coarse + 50.0).astype(float) - (adaptive > coarse + 50.0).astype(float)
            )
        ),
    }
    return {
        name: {"estimate": central[name], "ci95": list(percentile_interval(values))}
        for name, values in arrays.items()
    }


def selective_summary(
    rows: Mapping[str, np.ndarray], bootstrap_indices: np.ndarray, coverages: np.ndarray
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    calibrated_curve = tie_aware_risk_curve(
        rows["full_calibrated_confidence"], rows["full_calibrated_error"], coverages
    )
    raw_curve = tie_aware_risk_curve(rows["raw_confidence"], rows["raw_error"], coverages)
    calibrated_boot = np.empty((len(bootstrap_indices), len(coverages)), dtype=np.float64)
    raw_boot = np.empty_like(calibrated_boot)
    for position, indices in enumerate(bootstrap_indices):
        calibrated_boot[position] = tie_aware_risk_curve(
            rows["full_calibrated_confidence"][indices], rows["full_calibrated_error"][indices], coverages
        )
        raw_boot[position] = tie_aware_risk_curve(
            rows["raw_confidence"][indices], rows["raw_error"][indices], coverages
        )

    selected = {}
    for coverage in (0.5, 0.7, 0.9):
        index = int(np.argmin(np.abs(coverages - coverage)))
        difference = raw_boot[:, index] - calibrated_boot[:, index]
        selected[f"coverage_{int(coverage * 100)}"] = {
            "coverage": float(coverages[index]),
            "calibrated_risk_m": float(calibrated_curve[index]),
            "raw_inlier_risk_m": float(raw_curve[index]),
            "improvement_m": float(raw_curve[index] - calibrated_curve[index]),
            "relative_improvement": float(
                (raw_curve[index] - calibrated_curve[index]) / max(raw_curve[index], 1e-12)
            ),
            "improvement_ci95_m": list(percentile_interval(difference)),
        }

    calibrated_auc = normalized_auc(coverages, calibrated_curve)
    raw_auc = normalized_auc(coverages, raw_curve)
    calibrated_auc_boot = np.asarray(
        [normalized_auc(coverages, values) for values in calibrated_boot], dtype=np.float64
    )
    raw_auc_boot = np.asarray([normalized_auc(coverages, values) for values in raw_boot], dtype=np.float64)
    auc_difference = raw_auc_boot - calibrated_auc_boot
    auc = {
        "coverage_min": float(coverages[0]),
        "coverage_max": float(coverages[-1]),
        "calibrated_AURC_m": calibrated_auc,
        "raw_inlier_AURC_m": raw_auc,
        "improvement_m": float(raw_auc - calibrated_auc),
        "relative_improvement": float((raw_auc - calibrated_auc) / max(raw_auc, 1e-12)),
        "improvement_ci95_m": list(percentile_interval(auc_difference)),
    }
    curve = {
        "coverage": coverages.tolist(),
        "calibrated_risk_m": calibrated_curve.tolist(),
        "raw_inlier_risk_m": raw_curve.tolist(),
    }
    return selected, {"summary": auc, "curve": curve}


def write_markdown(path: Path, summary: Mapping) -> None:
    effects = summary["paired_effects"]
    selective = summary["selective_risk"]
    auc = summary["AURC"]["summary"]
    lines = [
        "# GTA Pose-Likelihood Statistical Follow-up",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Queries: {summary['query_count']}",
        f"- Bootstrap replicates: {summary['bootstrap']['replicates']}",
        f"- Seed: `{summary['bootstrap']['seed']}`",
        "",
        "## Paired Adaptive-vs-Legacy Effects",
        "",
        "| Effect (positive favors adaptive) | Estimate | 95% CI |",
        "|---|---:|---:|",
    ]
    for name, row in effects.items():
        lines.append(f"| {name} | {row['estimate']:.4f} | [{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}] |")
    lines.extend(["", "## Tie-aware Selective Risk", "", "| Coverage | Calibrated (m) | Raw inlier (m) | Improvement (m) | 95% CI (m) |", "|---:|---:|---:|---:|---:|"])
    for row in selective.values():
        lines.append(
            f"| {100 * row['coverage']:.0f}% | {row['calibrated_risk_m']:.3f} | "
            f"{row['raw_inlier_risk_m']:.3f} | {row['improvement_m']:.3f} | "
            f"[{row['improvement_ci95_m'][0]:.3f}, {row['improvement_ci95_m'][1]:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## AURC",
            "",
            f"- Calibrated: {auc['calibrated_AURC_m']:.4f}m",
            f"- Raw inlier: {auc['raw_inlier_AURC_m']:.4f}m",
            f"- Improvement: {auc['improvement_m']:.4f}m "
            f"(95% CI [{auc['improvement_ci95_m'][0]:.4f}, {auc['improvement_ci95_m'][1]:.4f}])",
            "",
            "## Decision",
            "",
            f"`{summary['decision']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.eval_cache)
    calibrator = PoseLikelihoodCalibrator.load(args.calibrator_path)
    rows = query_rows(records, calibrator)
    rng = np.random.default_rng(int(args.seed))
    bootstrap_indices = rng.integers(0, len(records), size=(int(args.bootstrap_replicates), len(records)))
    coverages = np.linspace(0.10, 1.00, 19)

    effects = effect_summary(rows, bootstrap_indices)
    selective, auc = selective_summary(rows, bootstrap_indices, coverages)
    central_favorable = bool(
        effects["Dis@1_improvement_m"]["estimate"] > 0.0
        and effects["MA@20_improvement_pp"]["estimate"] > 0.0
        and effects["catastrophic_50m_reduction_pp"]["estimate"] >= 0.0
        and auc["summary"]["improvement_m"] > 0.0
    )
    ci_support = bool(
        effects["Dis@1_improvement_m"]["ci95"][0] > 0.0
        and effects["MA@20_improvement_pp"]["ci95"][0] > 0.0
        and effects["catastrophic_50m_reduction_pp"]["ci95"][0] >= 0.0
        and auc["summary"]["improvement_ci95_m"][0] > 0.0
    )
    decision = "KEEP" if central_favorable and ci_support else "NEEDS ONE FOLLOW-UP" if central_favorable else "REJECT"
    summary = {
        "inputs": {
            "eval_cache": str(Path(args.eval_cache).resolve()),
            "calibrator_path": str(Path(args.calibrator_path).resolve()),
        },
        "query_count": len(records),
        "bootstrap": {"replicates": int(args.bootstrap_replicates), "seed": int(args.seed), "ci": "percentile_95"},
        "tie_policy": "fractional expected error within equal-confidence groups",
        "paired_effects": effects,
        "selective_risk": selective,
        "AURC": auc,
        "mean_adaptive_hypotheses": float(np.mean(rows["adaptive_hypotheses"])),
        "gates": {"central_favorable": central_favorable, "ci_support": ci_support},
        "decision": decision,
    }
    output = Path(args.summary_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(serializable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output.with_suffix(".md"), summary)
    print(json.dumps(serializable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
