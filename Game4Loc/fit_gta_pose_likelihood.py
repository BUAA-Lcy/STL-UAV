"""Fit and audit a calibrated GTA-UAV multi-hypothesis pose selector."""

from __future__ import annotations

import argparse
import base64
import json
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from game4loc.evaluate.gta_pose_likelihood import (
    CALIBRATOR_SCHEMA_VERSION,
    FEATURE_NAMES,
    PoseLikelihoodCalibrator,
    feature_vector,
    prefer_inlier_candidate,
    serializable,
)


HEADROOM_DIS_RELATIVE = 0.15
HEADROOM_MA20_PP = 8.0
CALIBRATION_AUROC = 0.75
CALIBRATION_ECE = 0.05
RISK70_RELATIVE = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--eval_cache", required=True)
    parser.add_argument("--artifact_path", required=True)
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--allow_hgb_followup", action="store_true")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> List[dict]:
    records: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def candidate_pool(record: Mapping, retrieval_topk: int, orientation_topk: int) -> List[dict]:
    return [
        candidate
        for candidate in record.get("candidates", [])
        if int(candidate["retrieval_rank"]) <= int(retrieval_topk)
        and int(candidate["angle_rank"]) <= int(orientation_topk)
    ]


def raw_choice(candidates: Sequence[Mapping]) -> Optional[Mapping]:
    best = None
    for candidate in candidates:
        if prefer_inlier_candidate(candidate, best):
            best = candidate
    return best


def summarize_errors(errors: Sequence[float], coarse_errors: Sequence[float]) -> Dict[str, float]:
    values = np.asarray(errors, dtype=np.float64)
    coarse = np.asarray(coarse_errors, dtype=np.float64)
    return {
        "query_count": int(values.size),
        "Dis@1_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "MA@3_pct": float(np.mean(values < 3.0) * 100.0),
        "MA@5_pct": float(np.mean(values < 5.0) * 100.0),
        "MA@10_pct": float(np.mean(values < 10.0) * 100.0),
        "MA@20_pct": float(np.mean(values < 20.0) * 100.0),
        "worse_than_coarse_pct": float(np.mean(values > coarse + 1e-6) * 100.0),
        "catastrophic_50m_pct": float(np.mean(values > coarse + 50.0) * 100.0),
    }


def summarize_strategy(records: Sequence[Mapping], retrieval_topk: int, orientation_topk: int, strategy: str, calibrator=None):
    errors: List[float] = []
    coarse_errors: List[float] = []
    confidences: List[float] = []
    for record in records:
        pool = candidate_pool(record, retrieval_topk, orientation_topk)
        if not pool:
            continue
        if strategy == "oracle":
            selected = min(pool, key=lambda item: float(item["error_m"]))
            confidence = 1.0
        elif strategy == "raw":
            selected = raw_choice(pool)
            confidence = float(selected.get("inliers", 0))
        elif strategy == "calibrated":
            selected, confidence = calibrator.best_candidate(pool)
        else:
            raise ValueError(strategy)
        errors.append(float(selected["error_m"]))
        coarse_errors.append(float(record["coarse_top1_error_m"]))
        confidences.append(float(confidence))
    summary = summarize_errors(errors, coarse_errors)
    summary["mean_confidence"] = float(np.mean(confidences))
    summary["hypotheses_per_query"] = int(retrieval_topk * orientation_topk)
    return summary


def split_query_names(records: Sequence[Mapping], seed: int):
    names = np.asarray(sorted({str(record["query_name"]) for record in records}), dtype=object)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(names)
    n = len(names)
    train_end = max(1, int(round(n * 0.70)))
    valid_end = max(train_end + 1, int(round(n * 0.85)))
    valid_end = min(valid_end, n - 1) if n >= 3 else n
    return set(names[:train_end]), set(names[train_end:valid_end]), set(names[valid_end:])


def flatten(records: Sequence[Mapping], names: Optional[set[str]] = None):
    candidates = []
    for record in records:
        if names is not None and str(record["query_name"]) not in names:
            continue
        candidates.extend(candidate_pool(record, 5, 4))
    matrix = np.stack([feature_vector(candidate) for candidate in candidates])
    labels = np.asarray([bool(candidate["success_20m"]) for candidate in candidates], dtype=np.int32)
    return matrix, labels


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(np.asarray(logits, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int32)

    def objective(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        probabilities = sigmoid(logits / temperature)
        return float(log_loss(labels, np.column_stack([1.0 - probabilities, probabilities]), labels=[0, 1]))

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    return float(math.exp(float(result.x)))


def equal_mass_ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    order = np.argsort(probabilities)
    chunks = np.array_split(order, min(int(bins), len(order)))
    total = float(max(len(order), 1))
    return float(
        sum(
            len(chunk) / total * abs(float(np.mean(probabilities[chunk])) - float(np.mean(labels[chunk])))
            for chunk in chunks
            if len(chunk)
        )
    )


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    metrics = {
        "positive_rate": float(np.mean(labels)),
        "AUPRC": float(average_precision_score(labels, probabilities)),
        "NLL": float(log_loss(labels, np.column_stack([1.0 - probabilities, probabilities]), labels=[0, 1])),
        "Brier": float(brier_score_loss(labels, probabilities)),
        "ECE_15_equal_mass": equal_mass_ece(labels, probabilities, bins=15),
    }
    metrics["AUROC"] = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan")
    return metrics


def _logistic_payload(train_records: Sequence[Mapping], seed: int):
    train_names, valid_names, calibration_names = split_query_names(train_records, seed)
    x_train, y_train = flatten(train_records, train_names)
    x_valid, y_valid = flatten(train_records, valid_names)
    x_cal, y_cal = flatten(train_records, calibration_names)

    best_c = None
    best_loss = float("inf")
    for c_value in (0.1, 1.0, 10.0):
        scaler = StandardScaler().fit(x_train)
        model = LogisticRegression(
            C=float(c_value), class_weight="balanced", max_iter=1000, random_state=int(seed)
        ).fit(scaler.transform(x_train), y_train)
        probabilities = model.predict_proba(scaler.transform(x_valid))[:, 1]
        value = float(log_loss(y_valid, np.column_stack([1.0 - probabilities, probabilities]), labels=[0, 1]))
        if value < best_loss:
            best_loss = value
            best_c = float(c_value)

    fit_names = train_names | valid_names
    x_fit, y_fit = flatten(train_records, fit_names)
    scaler = StandardScaler().fit(x_fit)
    model = LogisticRegression(
        C=float(best_c), class_weight="balanced", max_iter=1000, random_state=int(seed)
    ).fit(scaler.transform(x_fit), y_fit)
    calibration_logits = model.decision_function(scaler.transform(x_cal))
    temperature = fit_temperature(calibration_logits, y_cal)
    payload = {
        "schema_version": CALIBRATOR_SCHEMA_VERSION,
        "model_type": "logistic",
        "feature_names": list(FEATURE_NAMES),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "temperature": float(temperature),
        "training": {
            "seed": int(seed),
            "C": float(best_c),
            "validation_nll": float(best_loss),
            "train_queries": int(len(train_names)),
            "validation_queries": int(len(valid_names)),
            "calibration_queries": int(len(calibration_names)),
        },
        "policy": {},
    }
    return payload, calibration_names


def _hgb_payload(train_records: Sequence[Mapping], seed: int):
    train_names, valid_names, calibration_names = split_query_names(train_records, seed)
    fit_names = train_names | valid_names
    x_fit, y_fit = flatten(train_records, fit_names)
    x_cal, y_cal = flatten(train_records, calibration_names)
    model = HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=100,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=int(seed),
    ).fit(x_fit, y_fit)
    logits = model.decision_function(x_cal)
    temperature = fit_temperature(logits, y_cal)
    payload = {
        "schema_version": CALIBRATOR_SCHEMA_VERSION,
        "model_type": "hist_gradient_boosting",
        "feature_names": list(FEATURE_NAMES),
        "model_pickle_b64": base64.b64encode(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii"),
        "temperature": float(temperature),
        "training": {
            "seed": int(seed),
            "max_depth": 3,
            "max_iter": 100,
            "learning_rate": 0.05,
            "l2_regularization": 1.0,
            "train_validation_queries": int(len(fit_names)),
            "calibration_queries": int(len(calibration_names)),
        },
        "policy": {},
    }
    return payload, calibration_names


def attach_probabilities(records: Sequence[Mapping], calibrator: PoseLikelihoodCalibrator):
    result = {}
    for record in records:
        probabilities = calibrator.predict_candidates(record["candidates"])
        result[str(record["query_name"])] = probabilities
    return result


def risk70(records: Sequence[Mapping], calibrator: PoseLikelihoodCalibrator) -> Dict[str, float]:
    calibrated_rows = []
    raw_rows = []
    for record in records:
        pool = candidate_pool(record, 5, 4)
        selected, confidence = calibrator.best_candidate(pool)
        raw = raw_choice(pool)
        calibrated_rows.append((float(confidence), float(selected["error_m"])))
        raw_rows.append((float(raw.get("inliers", 0)), float(raw["error_m"])))
    keep = max(1, int(math.ceil(len(calibrated_rows) * 0.70)))
    calibrated_error = float(np.mean([row[1] for row in sorted(calibrated_rows, reverse=True)[:keep]]))
    raw_error = float(np.mean([row[1] for row in sorted(raw_rows, reverse=True)[:keep]]))
    relative = (raw_error - calibrated_error) / max(raw_error, 1e-9)
    return {
        "coverage": 0.70,
        "calibrated_mean_error_m": calibrated_error,
        "raw_inlier_mean_error_m": raw_error,
        "relative_improvement": float(relative),
    }


def choose_fixed_configuration(records: Sequence[Mapping], calibrator: PoseLikelihoodCalibrator):
    rows = []
    for retrieval_topk in (1, 3, 5):
        for orientation_topk in (2, 4):
            metrics = summarize_strategy(records, retrieval_topk, orientation_topk, "calibrated", calibrator)
            rows.append({"retrieval_topk": retrieval_topk, "orientation_topk": orientation_topk, **metrics})
    best_error = min(float(row["Dis@1_m"]) for row in rows)
    best_ma20 = max(float(row["MA@20_pct"]) for row in rows if float(row["Dis@1_m"]) <= best_error + 2.0)
    eligible = [
        row
        for row in rows
        if float(row["Dis@1_m"]) <= best_error + 2.0 and float(row["MA@20_pct"]) >= best_ma20 - 1.0
    ]
    selected = min(eligible, key=lambda row: (int(row["hypotheses_per_query"]), float(row["Dis@1_m"])))
    return rows, selected


def choose_policy(records: Sequence[Mapping], calibrator: PoseLikelihoodCalibrator, orientation_topk: int = 4):
    # Select abstention first on fixed R=5, then minimize expansion cost subject
    # to being within 2m / 1pp of that calibrated fixed-pool reference.
    confidence_values = []
    for record in records:
        _, confidence = calibrator.best_candidate(candidate_pool(record, 5, orientation_topk))
        confidence_values.append(confidence)
    quantiles = sorted(set([0.0] + [float(np.quantile(confidence_values, q)) for q in np.linspace(0.05, 0.80, 16)]))

    best_abs = None
    for threshold in quantiles:
        errors, coarse = [], []
        for record in records:
            selected, confidence = calibrator.best_candidate(candidate_pool(record, 5, orientation_topk))
            error = float(record["coarse_top1_error_m"]) if confidence < threshold else float(selected["error_m"])
            errors.append(error)
            coarse.append(float(record["coarse_top1_error_m"]))
        metrics = summarize_errors(errors, coarse)
        if best_abs is None or (metrics["Dis@1_m"], -metrics["MA@20_pct"]) < (
            best_abs["metrics"]["Dis@1_m"], -best_abs["metrics"]["MA@20_pct"]
        ):
            best_abs = {"threshold": float(threshold), "metrics": metrics}

    fixed_metrics = best_abs["metrics"]
    stage_confidences = {1: [], 3: []}
    for record in records:
        for stage in stage_confidences:
            _, confidence = calibrator.best_candidate(candidate_pool(record, stage, orientation_topk))
            stage_confidences[stage].append(confidence)
    stage_thresholds = {
        stage: sorted(set([float(np.quantile(values, q)) for q in np.linspace(0.20, 0.90, 8)]))
        for stage, values in stage_confidences.items()
    }
    candidates = []
    for threshold1 in stage_thresholds[1]:
        for threshold3 in stage_thresholds[3]:
            errors, coarse, costs = [], [], []
            for record in records:
                selected1, confidence1 = calibrator.best_candidate(candidate_pool(record, 1, orientation_topk))
                selected, confidence, stage = selected1, confidence1, 1
                if confidence1 < threshold1:
                    selected3, confidence3 = calibrator.best_candidate(candidate_pool(record, 3, orientation_topk))
                    selected, confidence, stage = selected3, confidence3, 3
                    if confidence3 < threshold3:
                        selected, confidence = calibrator.best_candidate(candidate_pool(record, 5, orientation_topk))
                        stage = 5
                error = (
                    float(record["coarse_top1_error_m"])
                    if confidence < float(best_abs["threshold"])
                    else float(selected["error_m"])
                )
                errors.append(error)
                coarse.append(float(record["coarse_top1_error_m"]))
                costs.append(stage * orientation_topk)
            metrics = summarize_errors(errors, coarse)
            if metrics["Dis@1_m"] <= fixed_metrics["Dis@1_m"] + 2.0 and metrics["MA@20_pct"] >= fixed_metrics["MA@20_pct"] - 1.0:
                candidates.append(
                    {
                        "threshold_r1": float(threshold1),
                        "threshold_r3": float(threshold3),
                        "abstain_threshold": float(best_abs["threshold"]),
                        "mean_hypotheses": float(np.mean(costs)),
                        "metrics": metrics,
                    }
                )
    if candidates:
        chosen = min(candidates, key=lambda item: (item["mean_hypotheses"], item["metrics"]["Dis@1_m"]))
    else:
        chosen = {
            "threshold_r1": 1.1,
            "threshold_r3": 1.1,
            "abstain_threshold": float(best_abs["threshold"]),
            "mean_hypotheses": float(5 * orientation_topk),
            "metrics": fixed_metrics,
        }
    return {
        "expansion_stages": [1, 3, 5],
        "orientation_topk": int(orientation_topk),
        **chosen,
    }


def evaluate_adaptive(records: Sequence[Mapping], calibrator: PoseLikelihoodCalibrator):
    policy = calibrator.policy
    orientation_topk = int(policy.get("orientation_topk", 4))
    errors, coarse, costs = [], [], []
    for record in records:
        selected, confidence = calibrator.best_candidate(candidate_pool(record, 1, orientation_topk))
        stage = 1
        if confidence < float(policy.get("threshold_r1", 1.1)):
            selected, confidence = calibrator.best_candidate(candidate_pool(record, 3, orientation_topk))
            stage = 3
            if confidence < float(policy.get("threshold_r3", 1.1)):
                selected, confidence = calibrator.best_candidate(candidate_pool(record, 5, orientation_topk))
                stage = 5
        error = (
            float(record["coarse_top1_error_m"])
            if confidence < float(policy.get("abstain_threshold", 0.0))
            else float(selected["error_m"])
        )
        errors.append(error)
        coarse.append(float(record["coarse_top1_error_m"]))
        costs.append(stage * orientation_topk)
    metrics = summarize_errors(errors, coarse)
    metrics["mean_hypotheses"] = float(np.mean(costs))
    return metrics


def write_markdown(path: Path, summary: Mapping) -> None:
    baselines = summary["pilot"]
    calibration = summary["calibration"]
    adaptive = summary["adaptive_eval"]
    lines = [
        "# GTA Multi-Tile Pose Likelihood Pilot",
        "",
        "## Experiment Name",
        "- Calibrated retrieval-top-R × VOP-top-k pose hypotheses on GTA-UAV.",
        "",
        "## Change Compared to Baseline",
        "- Replaces top-1/inlier-only candidate selection with an inference-only calibrated pose likelihood and coarse-center abstention.",
        "",
        "## Quantitative Results",
        "",
        "| Variant | Dis@1 (m) | MA@20 (%) | Worse than coarse (%) | Catastrophic +50m (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("legacy_top1_vop", "raw_top5x4", "oracle_top5x4"):
        row = baselines[name]
        lines.append(
            f"| {name} | {row['Dis@1_m']:.2f} | {row['MA@20_pct']:.2f} | "
            f"{row['worse_than_coarse_pct']:.2f} | {row['catastrophic_50m_pct']:.2f} |"
        )
    lines.extend(
        [
            f"| calibrated adaptive | {adaptive['Dis@1_m']:.2f} | {adaptive['MA@20_pct']:.2f} | {adaptive['worse_than_coarse_pct']:.2f} | {adaptive['catastrophic_50m_pct']:.2f} |",
            "",
            f"- AUROC: {calibration['metrics']['AUROC']:.4f}",
            f"- AUPRC: {calibration['metrics']['AUPRC']:.4f}",
            f"- ECE: {calibration['metrics']['ECE_15_equal_mass']:.4f}",
            f"- 70% coverage relative risk improvement: {calibration['risk70']['relative_improvement']:.4f}",
            "",
            "## Interpretation",
            f"- Oracle headroom gate: {'passed' if summary['gates']['oracle_headroom_pass'] else 'failed'}.",
            f"- Calibration gate: {'passed' if summary['gates']['calibration_pass'] else 'failed'}.",
            "",
            "## Decision",
            f"- `{summary['decision']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_records = load_jsonl(args.train_cache)
    eval_records = load_jsonl(args.eval_cache)

    pilot = {
        "coarse_top1": summarize_errors(
            [float(record["coarse_top1_error_m"]) for record in eval_records],
            [float(record["coarse_top1_error_m"]) for record in eval_records],
        ),
        "legacy_top1_vop": summarize_strategy(eval_records, 1, 4, "raw"),
        "raw_top5x4": summarize_strategy(eval_records, 5, 4, "raw"),
        "oracle_top5x4": summarize_strategy(eval_records, 5, 4, "oracle"),
    }
    legacy = pilot["legacy_top1_vop"]
    oracle = pilot["oracle_top5x4"]
    oracle_dis_gain = (float(legacy["Dis@1_m"]) - float(oracle["Dis@1_m"])) / max(float(legacy["Dis@1_m"]), 1e-9)
    oracle_ma20_gain = float(oracle["MA@20_pct"]) - float(legacy["MA@20_pct"])
    oracle_pass = oracle_dis_gain >= HEADROOM_DIS_RELATIVE and oracle_ma20_gain >= HEADROOM_MA20_PP

    payload, calibration_names = _logistic_payload(train_records, args.seed)
    calibrator = PoseLikelihoodCalibrator(payload)
    x_eval, y_eval = flatten(eval_records)
    probabilities = calibrator.predict_proba_matrix(x_eval)
    metrics = classification_metrics(y_eval, probabilities)
    risk = risk70(eval_records, calibrator)
    calibration_pass = (
        metrics["AUROC"] >= CALIBRATION_AUROC
        and metrics["ECE_15_equal_mass"] <= CALIBRATION_ECE
        and risk["relative_improvement"] >= RISK70_RELATIVE
    )
    followup_used = False
    if oracle_pass and not calibration_pass and args.allow_hgb_followup:
        hgb_payload, calibration_names = _hgb_payload(train_records, args.seed)
        hgb_calibrator = PoseLikelihoodCalibrator(hgb_payload)
        hgb_probabilities = hgb_calibrator.predict_proba_matrix(x_eval)
        hgb_metrics = classification_metrics(y_eval, hgb_probabilities)
        hgb_risk = risk70(eval_records, hgb_calibrator)
        if (hgb_metrics["AUROC"], -hgb_metrics["ECE_15_equal_mass"], hgb_risk["relative_improvement"]) > (
            metrics["AUROC"], -metrics["ECE_15_equal_mass"], risk["relative_improvement"]
        ):
            payload, calibrator, metrics, risk = hgb_payload, hgb_calibrator, hgb_metrics, hgb_risk
        followup_used = True
        calibration_pass = (
            metrics["AUROC"] >= CALIBRATION_AUROC
            and metrics["ECE_15_equal_mass"] <= CALIBRATION_ECE
            and risk["relative_improvement"] >= RISK70_RELATIVE
        )

    calibration_records = [record for record in train_records if str(record["query_name"]) in calibration_names]
    fixed_rows, fixed_selected = choose_fixed_configuration(calibration_records, calibrator)
    policy = choose_policy(calibration_records, calibrator, orientation_topk=int(fixed_selected["orientation_topk"]))
    payload["policy"] = policy
    calibrator = PoseLikelihoodCalibrator(payload)
    adaptive_eval = evaluate_adaptive(eval_records, calibrator)

    decision = "KEEP" if oracle_pass and calibration_pass else "REJECT"
    if oracle_pass and not calibration_pass and not followup_used:
        decision = "NEEDS ONE FOLLOW-UP"
    summary = {
        "inputs": {"train_cache": str(Path(args.train_cache).resolve()), "eval_cache": str(Path(args.eval_cache).resolve())},
        "pilot": pilot,
        "headroom": {"Dis@1_relative_gain": oracle_dis_gain, "MA@20_pp_gain": oracle_ma20_gain},
        "calibration": {
            "model_type": payload["model_type"],
            "metrics": metrics,
            "risk70": risk,
            "followup_used": followup_used,
        },
        "fixed_configuration_rows": fixed_rows,
        "selected_fixed_configuration": fixed_selected,
        "adaptive_policy": policy,
        "adaptive_eval": adaptive_eval,
        "gates": {"oracle_headroom_pass": oracle_pass, "calibration_pass": calibration_pass},
        "decision": decision,
    }

    artifact_path = Path(args.artifact_path)
    summary_path = Path(args.summary_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(serializable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(serializable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary_path.with_suffix(".md"), summary)
    print(json.dumps(serializable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

