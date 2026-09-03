import json
import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_gta_pose_hypothesis_cache import _read_completed, _stratified_indices
from fit_gta_pose_likelihood import risk70
from game4loc.evaluate.gta_pose_likelihood import (
    CALIBRATOR_SCHEMA_VERSION,
    FEATURE_NAMES,
    PoseLikelihoodCalibrator,
    build_feature_dict,
    prefer_inlier_candidate,
)
from game4loc.evaluate.gta import evaluate as evaluate_gta


def _debug_points(homography):
    query = np.asarray([[20, 20], [80, 20], [80, 80], [20, 80], [50, 50]], dtype=np.float32)
    query_h = np.concatenate([query, np.ones((len(query), 1), dtype=np.float32)], axis=1)
    gallery_h = (homography @ query_h.T).T
    gallery = gallery_h[:, :2] / gallery_h[:, 2:3]
    return {
        "mk0": gallery.astype(np.float32),
        "mk1": query,
        "h_mask": np.ones((len(query), 1), dtype=np.uint8),
        "homography": homography,
    }


class GTAPoseLikelihoodTests(unittest.TestCase):
    def test_geometry_features_are_finite_and_detect_supported_center(self):
        homography = np.asarray([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]])
        features = build_feature_dict(
            retrieval_rank=1,
            retrieval_score=0.9,
            retrieval_top1_score=0.9,
            retrieval_next_score=0.8,
            angle_rank=1,
            angle_probability=0.4,
            posterior={"top_prob": 0.4, "entropy": 0.5, "concentration": 0.6},
            match_info={"n_kept": 5, "inliers": 5, "inlier_ratio": 1.0, "homography": homography},
            match_debug=_debug_points(homography),
            gallery_hw=(100, 100),
            query_hw=(100, 100),
        )
        self.assertEqual(tuple(features), FEATURE_NAMES)
        self.assertTrue(all(np.isfinite(list(features.values()))))
        self.assertEqual(features["query_center_inside_support"], 1.0)
        self.assertLess(features["symmetric_reproj_median_norm"], 1e-6)

    def test_invalid_homography_and_empty_matches_are_bounded(self):
        singular = np.zeros((3, 3), dtype=np.float64)
        features = build_feature_dict(
            retrieval_rank=3,
            retrieval_score=0.2,
            retrieval_top1_score=0.8,
            retrieval_next_score=0.1,
            angle_rank=4,
            angle_probability=0.01,
            posterior={},
            match_info={"fallback_to_center": True, "homography": singular},
            match_debug={},
            gallery_hw=(384, 384),
            query_hw=(384, 384),
        )
        self.assertTrue(all(np.isfinite(list(features.values()))))
        self.assertEqual(features["fallback_to_center"], 1.0)
        self.assertEqual(features["query_center_extrapolation_norm"], 1.0)

    def test_logistic_calibrator_schema_and_probabilities(self):
        payload = {
            "schema_version": CALIBRATOR_SCHEMA_VERSION,
            "model_type": "logistic",
            "feature_names": list(FEATURE_NAMES),
            "scaler_mean": [0.0] * len(FEATURE_NAMES),
            "scaler_scale": [1.0] * len(FEATURE_NAMES),
            "coef": [0.0] * len(FEATURE_NAMES),
            "intercept": 0.0,
            "temperature": 1.0,
            "policy": {"abstain_threshold": 0.4},
        }
        calibrator = PoseLikelihoodCalibrator(payload)
        probabilities = calibrator.predict_proba_matrix(np.zeros((2, len(FEATURE_NAMES))))
        self.assertTrue(np.allclose(probabilities, 0.5))
        self.assertEqual(calibrator.policy["abstain_threshold"], 0.4)

    def test_inlier_selection_has_deterministic_rank_tie_break(self):
        rank2 = {"inliers": 20, "inlier_ratio": 0.5, "retrieval_rank": 2, "angle_rank": 1}
        rank1 = {"inliers": 20, "inlier_ratio": 0.5, "retrieval_rank": 1, "angle_rank": 2}
        self.assertTrue(prefer_inlier_candidate(rank1, rank2))
        self.assertFalse(prefer_inlier_candidate(rank2, rank1))

    def test_stratified_sampling_is_deterministic_and_cache_resume_reads_queries(self):
        names = [f"{alt}_0001_{index:010d}.png" for index in range(5) for alt in (200, 300, 400)]
        first = _stratified_indices(names, 20260903)
        second = _stratified_indices(names, 20260903)
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(len(names))))
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.jsonl"
            cache.write_text(json.dumps({"query_name": "a.png"}) + "\n" + json.dumps({"query_name": "b.png"}) + "\n")
            self.assertEqual(_read_completed(cache), {"a.png", "b.png"})

    def test_feature_schema_contains_no_ground_truth_labels(self):
        forbidden = ("error", "ground_truth", "success", "improves", "catastrophic")
        self.assertFalse(any(token in name for name in FEATURE_NAMES for token in forbidden))

    def test_risk_coverage_does_not_break_confidence_ties_with_error(self):
        class ConstantCalibrator:
            @staticmethod
            def best_candidate(candidates):
                return candidates[0], 1.0

        records = []
        for error in range(1, 11):
            records.append(
                {
                    "query_name": f"q{error}",
                    "coarse_top1_error_m": 100.0,
                    "candidates": [
                        {
                            "retrieval_rank": 1,
                            "angle_rank": 1,
                            "inliers": 10,
                            "error_m": float(error),
                        }
                    ],
                }
            )
        metrics = risk70(records, ConstantCalibrator())
        # ceil(10 * .70) keeps the first seven tied rows: mean(1..7) = 4.
        self.assertEqual(metrics["raw_inlier_mean_error_m"], 4.0)

    def test_official_evaluator_defaults_preserve_legacy_dispatch(self):
        parameters = inspect.signature(evaluate_gta).parameters
        self.assertEqual(parameters["fine_retrieval_topk"].default, 1)
        self.assertEqual(parameters["fine_selection_mode"].default, "legacy_inlier")
        self.assertEqual(parameters["fine_calibrator_path"].default, "")


if __name__ == "__main__":
    unittest.main()
