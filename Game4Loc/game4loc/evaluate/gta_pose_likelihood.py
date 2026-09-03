"""Utilities for calibrated multi-tile GTA-UAV fine-localization hypotheses.

The helpers in this module intentionally consume only inference-time signals.
Ground-truth coordinates are added by offline cache builders, never by the
candidate generator or calibrator used in the official evaluator.
"""

from __future__ import annotations

import base64
import json
import math
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


CALIBRATOR_SCHEMA_VERSION = 1

FEATURE_NAMES: Tuple[str, ...] = (
    "retrieval_rank",
    "retrieval_score",
    "retrieval_gap_top1",
    "retrieval_margin_next",
    "vop_angle_rank",
    "vop_prob",
    "vop_top_prob",
    "vop_entropy",
    "vop_concentration",
    "log_retained_matches",
    "log_inliers",
    "inlier_ratio",
    "symmetric_reproj_median_norm",
    "symmetric_reproj_q90_norm",
    "gallery_inlier_hull_ratio",
    "query_inlier_hull_ratio",
    "gallery_inlier_grid_coverage",
    "query_inlier_grid_coverage",
    "query_center_inside_support",
    "query_center_extrapolation_norm",
    "homography_log_condition",
    "homography_perspective_norm",
    "projected_center_displacement_norm",
    "fallback_to_center",
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def rotate_query_tensor(query_tensor: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate a normalized query tensor without expanding its canvas."""
    if abs(float(angle_deg)) < 1e-6:
        return query_tensor
    fill_value = -1.0
    if query_tensor.ndim == 3:
        return TF.rotate(
            query_tensor,
            angle=float(angle_deg),
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
            fill=[fill_value] * int(query_tensor.shape[0]),
        )
    if query_tensor.ndim == 4:
        return torch.stack(
            [
                TF.rotate(
                    sample,
                    angle=float(angle_deg),
                    interpolation=InterpolationMode.BILINEAR,
                    expand=False,
                    fill=[fill_value] * int(sample.shape[0]),
                )
                for sample in query_tensor
            ],
            dim=0,
        )
    raise ValueError(f"Unsupported query tensor shape: {tuple(query_tensor.shape)}")


def _convex_hull_stats(points: np.ndarray, width: int, height: int) -> Tuple[float, float, float]:
    """Return normalized hull area, 4x4 occupancy, and center extrapolation."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    image_area = float(max(width * height, 1))
    diagonal = float(max(math.hypot(width, height), 1.0))
    if points.shape[0] == 0:
        return 0.0, 0.0, 1.0

    x_bins = np.clip((points[:, 0] / max(width, 1) * 4).astype(np.int32), 0, 3)
    y_bins = np.clip((points[:, 1] / max(height, 1) * 4).astype(np.int32), 0, 3)
    occupancy = float(len(set(zip(x_bins.tolist(), y_bins.tolist())))) / 16.0
    center = (float(width) / 2.0, float(height) / 2.0)

    if points.shape[0] < 3:
        nearest = float(np.min(np.linalg.norm(points - np.asarray(center, dtype=np.float32), axis=1)))
        return 0.0, occupancy, nearest / diagonal

    hull = cv2.convexHull(points).reshape(-1, 2)
    area_ratio = float(abs(cv2.contourArea(hull))) / image_area
    signed_distance = float(cv2.pointPolygonTest(hull, center, True))
    extrapolation = 0.0 if signed_distance >= 0.0 else abs(signed_distance) / diagonal
    return float(np.clip(area_ratio, 0.0, 1.0)), occupancy, extrapolation


def _geometry_features(
    match_info: Optional[Mapping[str, Any]],
    match_debug: Optional[Mapping[str, Any]],
    gallery_hw: Tuple[int, int],
    query_hw: Tuple[int, int],
) -> Dict[str, float]:
    info = dict(match_info or {})
    debug = dict(match_debug or {})
    gallery_h, gallery_w = int(gallery_hw[0]), int(gallery_hw[1])
    query_h, query_w = int(query_hw[0]), int(query_hw[1])
    diagonal = float(max(math.hypot(gallery_w, gallery_h), math.hypot(query_w, query_h), 1.0))

    retained = max(0, int(info.get("n_kept", debug.get("n_kept", 0)) or 0))
    inliers = max(0, int(info.get("inliers", debug.get("inliers", 0)) or 0))
    inlier_ratio = _finite(info.get("inlier_ratio"), float(inliers) / float(max(retained, 1)))
    fallback = float(bool(info.get("fallback_to_center", False)))

    mk0 = np.asarray(debug.get("mk0", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
    mk1 = np.asarray(debug.get("mk1", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
    mask = np.asarray(debug.get("h_mask", np.empty((0,))), dtype=bool).reshape(-1)
    if mask.shape[0] != mk0.shape[0] or mk0.shape[0] != mk1.shape[0]:
        mask = np.zeros((mk0.shape[0],), dtype=bool)
    inlier_mk0 = mk0[mask]
    inlier_mk1 = mk1[mask]

    gallery_hull, gallery_grid, _ = _convex_hull_stats(inlier_mk0, gallery_w, gallery_h)
    query_hull, query_grid, query_extrap = _convex_hull_stats(inlier_mk1, query_w, query_h)
    query_center_inside = float(query_extrap <= 1e-12 and inlier_mk1.shape[0] >= 3)

    H_value = info.get("homography", debug.get("homography"))
    H = None
    if H_value is not None:
        candidate = np.asarray(H_value, dtype=np.float64)
        if candidate.shape == (3, 3) and np.all(np.isfinite(candidate)):
            H = candidate

    median_error = 1.0
    q90_error = 1.0
    log_condition = math.log1p(1e6)
    perspective = 1.0
    center_displacement = 1.0
    if H is not None and abs(float(H[2, 2])) > 1e-12:
        H = H / float(H[2, 2])
        condition = float(np.linalg.cond(H))
        log_condition = math.log1p(min(max(condition, 0.0), 1e6)) if math.isfinite(condition) else math.log1p(1e6)
        perspective = float(np.linalg.norm(H[2, :2])) * diagonal

        query_center = np.asarray([query_w / 2.0, query_h / 2.0, 1.0], dtype=np.float64)
        projected_center = H @ query_center
        if abs(float(projected_center[2])) > 1e-12 and np.all(np.isfinite(projected_center)):
            projected_center = projected_center[:2] / projected_center[2]
            gallery_center = np.asarray([gallery_w / 2.0, gallery_h / 2.0], dtype=np.float64)
            center_displacement = float(np.linalg.norm(projected_center - gallery_center)) / diagonal

        if inlier_mk0.shape[0] > 0:
            ones = np.ones((inlier_mk1.shape[0], 1), dtype=np.float64)
            src_h = np.concatenate([inlier_mk1, ones], axis=1)
            forward_h = (H @ src_h.T).T
            valid_forward = np.abs(forward_h[:, 2]) > 1e-12
            errors: List[np.ndarray] = []
            if np.any(valid_forward):
                forward = forward_h[valid_forward, :2] / forward_h[valid_forward, 2:3]
                errors.append(np.linalg.norm(forward - inlier_mk0[valid_forward], axis=1))
            try:
                H_inv = np.linalg.inv(H)
                dst_h = np.concatenate([inlier_mk0, ones], axis=1)
                backward_h = (H_inv @ dst_h.T).T
                valid_backward = np.abs(backward_h[:, 2]) > 1e-12
                if np.any(valid_backward):
                    backward = backward_h[valid_backward, :2] / backward_h[valid_backward, 2:3]
                    errors.append(np.linalg.norm(backward - inlier_mk1[valid_backward], axis=1))
            except np.linalg.LinAlgError:
                pass
            if errors:
                combined = np.concatenate(errors) / diagonal
                combined = combined[np.isfinite(combined)]
                if combined.size:
                    median_error = float(np.median(combined))
                    q90_error = float(np.quantile(combined, 0.90))

    return {
        "log_retained_matches": math.log1p(retained),
        "log_inliers": math.log1p(inliers),
        "inlier_ratio": float(np.clip(inlier_ratio, 0.0, 1.0)),
        "symmetric_reproj_median_norm": _finite(median_error, 1.0),
        "symmetric_reproj_q90_norm": _finite(q90_error, 1.0),
        "gallery_inlier_hull_ratio": _finite(gallery_hull),
        "query_inlier_hull_ratio": _finite(query_hull),
        "gallery_inlier_grid_coverage": _finite(gallery_grid),
        "query_inlier_grid_coverage": _finite(query_grid),
        "query_center_inside_support": _finite(query_center_inside),
        "query_center_extrapolation_norm": _finite(query_extrap, 1.0),
        "homography_log_condition": _finite(log_condition, math.log1p(1e6)),
        "homography_perspective_norm": _finite(perspective, 1.0),
        "projected_center_displacement_norm": _finite(center_displacement, 1.0),
        "fallback_to_center": fallback,
    }


def build_feature_dict(
    *,
    retrieval_rank: int,
    retrieval_score: float,
    retrieval_top1_score: float,
    retrieval_next_score: float,
    angle_rank: int,
    angle_probability: float,
    posterior: Mapping[str, Any],
    match_info: Optional[Mapping[str, Any]],
    match_debug: Optional[Mapping[str, Any]],
    gallery_hw: Tuple[int, int],
    query_hw: Tuple[int, int],
) -> Dict[str, float]:
    features = {
        "retrieval_rank": float(retrieval_rank),
        "retrieval_score": _finite(retrieval_score),
        "retrieval_gap_top1": _finite(retrieval_top1_score) - _finite(retrieval_score),
        "retrieval_margin_next": _finite(retrieval_score) - _finite(retrieval_next_score),
        "vop_angle_rank": float(angle_rank),
        "vop_prob": _finite(angle_probability),
        "vop_top_prob": _finite(posterior.get("top_prob")),
        "vop_entropy": _finite(posterior.get("entropy")),
        "vop_concentration": _finite(posterior.get("concentration")),
    }
    features.update(_geometry_features(match_info, match_debug, gallery_hw, query_hw))
    return {name: _finite(features.get(name)) for name in FEATURE_NAMES}


def feature_vector(candidate: Mapping[str, Any], feature_names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    features = candidate.get("features", {})
    return np.asarray([_finite(features.get(name)) for name in feature_names], dtype=np.float64)


def prefer_inlier_candidate(candidate: Mapping[str, Any], best: Optional[Mapping[str, Any]]) -> bool:
    if best is None:
        return True
    current_key = (
        int(candidate.get("inliers", 0)),
        _finite(candidate.get("inlier_ratio")),
        -int(candidate.get("retrieval_rank", 10**6)),
        -int(candidate.get("angle_rank", 10**6)),
    )
    best_key = (
        int(best.get("inliers", 0)),
        _finite(best.get("inlier_ratio")),
        -int(best.get("retrieval_rank", 10**6)),
        -int(best.get("angle_rank", 10**6)),
    )
    return current_key > best_key


class PoseLikelihoodCalibrator:
    """Versioned inference wrapper for logistic or diagnostic HGB artifacts."""

    def __init__(self, payload: Mapping[str, Any], source_path: Optional[Path] = None):
        if int(payload.get("schema_version", -1)) != CALIBRATOR_SCHEMA_VERSION:
            raise ValueError(f"Unsupported calibrator schema: {payload.get('schema_version')}")
        self.payload = dict(payload)
        self.feature_names = tuple(payload.get("feature_names", ()))
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("Calibrator feature schema does not match this code version")
        self.model_type = str(payload.get("model_type", "logistic"))
        self.temperature = max(_finite(payload.get("temperature"), 1.0), 1e-6)
        self.policy = dict(payload.get("policy", {}))
        self._hgb_model = None
        if self.model_type == "hist_gradient_boosting":
            encoded = str(payload.get("model_pickle_b64", ""))
            if not encoded:
                raise ValueError("Missing diagnostic HGB model payload")
            self._hgb_model = pickle.loads(base64.b64decode(encoded.encode("ascii")))
        elif self.model_type != "logistic":
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    @classmethod
    def load(cls, path: str | Path) -> "PoseLikelihoodCalibrator":
        source = Path(path)
        return cls(json.loads(source.read_text(encoding="utf-8")), source_path=source)

    def raw_logits(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {matrix.shape[1]}")
        if self.model_type == "hist_gradient_boosting":
            return np.asarray(self._hgb_model.decision_function(matrix), dtype=np.float64).reshape(-1)
        mean = np.asarray(self.payload["scaler_mean"], dtype=np.float64)
        scale = np.asarray(self.payload["scaler_scale"], dtype=np.float64)
        coef = np.asarray(self.payload["coef"], dtype=np.float64)
        intercept = _finite(self.payload.get("intercept"))
        standardized = (matrix - mean) / np.where(scale > 1e-12, scale, 1.0)
        return standardized @ coef + intercept

    def predict_proba_matrix(self, matrix: np.ndarray) -> np.ndarray:
        logits = np.clip(self.raw_logits(matrix) / self.temperature, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict_candidates(self, candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if not candidates:
            return np.empty((0,), dtype=np.float64)
        matrix = np.stack([feature_vector(candidate, self.feature_names) for candidate in candidates])
        return self.predict_proba_matrix(matrix)

    def best_candidate(self, candidates: Sequence[Mapping[str, Any]]) -> Tuple[Optional[Mapping[str, Any]], float]:
        probabilities = self.predict_candidates(candidates)
        if probabilities.size == 0:
            return None, 0.0
        best_index = int(np.argmax(probabilities))
        return candidates[best_index], float(probabilities[best_index])


def evaluate_tile_hypotheses(
    *,
    retrieval_model: torch.nn.Module,
    orientation_model: Any,
    matcher: Any,
    query_img: torch.Tensor,
    gallery_img: torch.Tensor,
    gallery_center_xy: Sequence[float],
    gallery_topleft_xy: Sequence[float],
    gallery_index: int,
    gallery_name: str,
    retrieval_rank: int,
    retrieval_score: float,
    retrieval_top1_score: float,
    retrieval_next_score: float,
    orientation_topk: int,
    device: str,
    case_prefix: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    candidate_angles = getattr(orientation_model, "candidate_angles_deg", None) or [0.0]
    t_vop = time.perf_counter()
    posterior = orientation_model.predict_posterior(
        retrieval_model=retrieval_model,
        gallery_img=gallery_img,
        query_img=query_img,
        candidate_angles_deg=candidate_angles,
        device=device,
        gallery_branch="img2",
        query_branch="img1",
    )
    vop_time = time.perf_counter() - t_vop
    probabilities = np.asarray(posterior["probs"], dtype=np.float64)
    topk = max(1, min(int(orientation_topk), len(candidate_angles)))
    angle_indices = np.argsort(probabilities)[::-1][:topk]

    gallery_hw = (int(gallery_img.shape[-2]), int(gallery_img.shape[-1]))
    query_hw = (int(query_img.shape[-2]), int(query_img.shape[-1]))
    candidates: List[Dict[str, Any]] = []
    match_time = 0.0
    for angle_rank, angle_index in enumerate(angle_indices, start=1):
        angle_deg = float(candidate_angles[int(angle_index)])
        rotated_query = rotate_query_tensor(query_img, angle_deg)
        t_match = time.perf_counter()
        predicted_xy = matcher.est_center(
            gallery_img,
            rotated_query,
            gallery_center_xy,
            gallery_topleft_xy,
            yaw0=None,
            yaw1=None,
            rotate=0.0,
            case_name=f"{case_prefix}_r{retrieval_rank}_a{angle_rank}_{angle_deg:.1f}",
            save_final_vis=False,
        )
        match_time += time.perf_counter() - t_match
        match_info = matcher.get_last_match_info() or {}
        match_debug = matcher.get_last_match_debug() or {}
        features = build_feature_dict(
            retrieval_rank=retrieval_rank,
            retrieval_score=retrieval_score,
            retrieval_top1_score=retrieval_top1_score,
            retrieval_next_score=retrieval_next_score,
            angle_rank=angle_rank,
            angle_probability=float(probabilities[int(angle_index)]),
            posterior=posterior,
            match_info=match_info,
            match_debug=match_debug,
            gallery_hw=gallery_hw,
            query_hw=query_hw,
        )
        retained = int(match_info.get("n_kept", 0))
        inliers = int(match_info.get("inliers", 0))
        candidates.append(
            {
                "gallery_index": int(gallery_index),
                "gallery_name": str(gallery_name),
                "retrieval_rank": int(retrieval_rank),
                "retrieval_score": float(retrieval_score),
                "angle_index": int(angle_index),
                "angle_rank": int(angle_rank),
                "angle_deg": angle_deg,
                "angle_probability": float(probabilities[int(angle_index)]),
                "predicted_xy": [float(predicted_xy[0]), float(predicted_xy[1])],
                "tile_center_xy": [float(gallery_center_xy[0]), float(gallery_center_xy[1])],
                "retained_matches": retained,
                "inliers": inliers,
                "inlier_ratio": _finite(match_info.get("inlier_ratio"), float(inliers) / float(max(retained, 1))),
                "fallback_to_center": bool(match_info.get("fallback_to_center", False)),
                "fallback_reason": match_info.get("fallback_reason"),
                "identity_h_fallback": bool(match_info.get("identity_h_fallback", False)),
                "out_of_bounds": bool(match_info.get("out_of_bounds", False)),
                "projection_invalid": bool(match_info.get("projection_invalid", False)),
                "features": features,
            }
        )
    return candidates, {"vop_time_s": float(vop_time), "match_time_s": float(match_time)}


def serializable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value

