"""
Decision Gate (Step 4): Accept/Reject decision for a filtered pair.

Provides both hand-set threshold gating and a learned gate
(logistic regression / decision tree) trained on tuning split.
"""

import math
import numpy as np
import pickle
import os
from typing import Dict, Any, Tuple, Optional, List

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Feature keys used by the gate (in order)
GATE_FEATURE_KEYS = [
    "n_inliers",
    "coverage",
    "grid_occupancy",
    "mean_confidence",
    "ransac_sim_inlier_ratio",
    "ransac_homo_inlier_ratio",
    "median_reproj_residual",
    "ransac_scale",
    "survival_after_spatial",
    "survival_after_texture",
    "survival_after_delaunay",
    "survival_after_global",
]


def features_to_vector(features: Dict[str, Any]) -> np.ndarray:
    """Convert feature dict to a fixed-order numpy vector."""
    vec = []
    for key in GATE_FEATURE_KEYS:
        val = features.get(key, 0.0)
        if val == float("inf") or val == float("-inf") or (isinstance(val, float) and math.isnan(val)):
            val = 0.0
        vec.append(float(val))
    return np.array(vec, dtype=np.float64)


def hand_threshold_gate(features: Dict[str, Any], config: dict) -> Tuple[bool, str]:
    """Apply hand-set threshold gate.

    Args:
        features: decision features from filter_matches
        config: the 'decision_gate' section of config

    Returns:
        accept: True if the pair is accepted
        reason: human-readable rejection reason (empty if accepted)
    """
    gc = config["decision_gate"]

    n = features.get("n_inliers", 0)
    if n < gc["min_inliers"]:
        return False, f"too few inliers ({n} < {gc['min_inliers']})"

    cov = features.get("coverage", 0.0)
    if cov < gc["min_coverage"]:
        return False, f"low coverage ({cov:.3f} < {gc['min_coverage']})"

    occ = features.get("grid_occupancy", 0.0)
    if occ < gc["min_grid_occupancy"]:
        return False, f"low grid occupancy ({occ:.3f} < {gc['min_grid_occupancy']})"

    mc = features.get("mean_confidence", 0.0)
    if mc < gc["min_mean_confidence"]:
        return False, f"low mean confidence ({mc:.3f} < {gc['min_mean_confidence']})"

    rir = features.get("ransac_sim_inlier_ratio", 0.0)
    if rir < gc["min_ransac_inlier_ratio"]:
        return False, f"low RANSAC inlier ratio ({rir:.3f} < {gc['min_ransac_inlier_ratio']})"

    res = features.get("median_reproj_residual", float("inf"))
    if res > gc["max_reprojection_residual"]:
        return False, f"high reproj residual ({res:.1f} > {gc['max_reprojection_residual']})"

    return True, ""


class LearnedGate:
    """Learned accept/reject gate using logistic regression and decision tree."""

    def __init__(self):
        self.lr_pipeline = None
        self.dt_pipeline = None
        self.threshold = 0.5  # classification threshold on LR probability

    def train(self, feature_vectors: np.ndarray, labels: np.ndarray,
              sample_weights: Optional[np.ndarray] = None):
        """Train both logistic regression and decision tree gates.

        Args:
            feature_vectors: (N, D) array of feature vectors
            labels: (N,) binary labels (1=accept/positive, 0=reject/negative)
            sample_weights: optional (N,) sample weights
        """
        # Logistic regression
        self.lr_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            ))
        ])
        self.lr_pipeline.fit(feature_vectors, labels,
                              lr__sample_weight=sample_weights)

        # Decision tree (interpretable)
        self.dt_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("dt", DecisionTreeClassifier(
                max_depth=5,
                class_weight="balanced",
                random_state=42
            ))
        ])
        self.dt_pipeline.fit(feature_vectors, labels,
                              dt__sample_weight=sample_weights)

    def predict_lr(self, feature_vector: np.ndarray) -> Tuple[bool, float]:
        """Predict with logistic regression.

        Returns:
            accept: True/False
            probability: P(accept)
        """
        if self.lr_pipeline is None:
            return False, 0.0
        fv = feature_vector.reshape(1, -1)
        prob = self.lr_pipeline.predict_proba(fv)[0, 1]
        return bool(prob >= self.threshold), float(prob)

    def predict_dt(self, feature_vector: np.ndarray) -> Tuple[bool, float]:
        """Predict with decision tree.

        Returns:
            accept: True/False
            probability: P(accept)
        """
        if self.dt_pipeline is None:
            return False, 0.0
        fv = feature_vector.reshape(1, -1)
        prob = self.dt_pipeline.predict_proba(fv)[0, 1]
        return bool(prob >= self.threshold), float(prob)

    def save(self, path: str):
        """Save trained gate to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "lr_pipeline": self.lr_pipeline,
                "dt_pipeline": self.dt_pipeline,
                "threshold": self.threshold,
            }, f)

    def load(self, path: str):
        """Load trained gate from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.lr_pipeline = data["lr_pipeline"]
        self.dt_pipeline = data["dt_pipeline"]
        self.threshold = data["threshold"]

    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importances from the decision tree."""
        if self.dt_pipeline is None:
            return {}
        importances = self.dt_pipeline.named_steps["dt"].feature_importances_
        return dict(zip(GATE_FEATURE_KEYS, importances))


def compute_metadata_consistency(features: Dict[str, Any],
                                  yaw_metadata: float,
                                  gsd_metadata: float) -> Dict[str, float]:
    """Check consistency of RANSAC-fitted rotation and scale with metadata.

    Args:
        features: decision features
        yaw_metadata: yaw from metadata (degrees)
        gsd_metadata: query GSD from metadata (m/px)

    Returns:
        dict with rotation_error_deg and scale_error_ratio
    """
    rot_fit = features.get("ransac_rotation_deg", 0.0)
    scale_fit = features.get("ransac_scale", 0.0)

    # Rotation consistency
    rot_error = abs(((rot_fit - yaw_metadata + 180) % 360) - 180)

    # Scale consistency: expected scale = query_gsd / ref_gsd
    # But we don't have ref_gsd here, so just report the fitted scale
    return {
        "rotation_error_deg": float(rot_error),
        "ransac_scale_fitted": float(scale_fit),
    }

