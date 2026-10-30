"""
Evaluation Metrics (Step 5).

Match-level: precision/recall of filtered vs raw inliers.
Decision-level: ROC/PR curves, false-accept rate, localization error.
Scoring proxy: competition scoring rule.
Ablation: stage removal analysis.
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def label_matches_with_gt(kpts_q: np.ndarray, kpts_t: np.ndarray,
                           gt_homography: np.ndarray,
                           pixel_threshold: float = 5.0) -> np.ndarray:
    """Label each match as inlier/outlier using ground truth homography.

    Args:
        kpts_q: (N, 2) query keypoints
        kpts_t: (N, 2) tile keypoints
        gt_homography: (3, 3) ground truth homography (query->tile)
        pixel_threshold: reprojection error threshold (pixels)

    Returns:
        labels: (N,) boolean array, True = inlier
    """
    N = len(kpts_q)
    if N == 0:
        return np.array([], dtype=bool)

    # Project query points to tile space using GT homography
    ones = np.ones((N, 1))
    kq_h = np.hstack([kpts_q, ones])  # (N, 3)
    projected = kq_h @ gt_homography.T  # (N, 3)
    projected = projected[:, :2] / projected[:, 2:3]  # normalise

    # Reprojection error
    errors = np.linalg.norm(projected - kpts_t, axis=1)
    return errors < pixel_threshold


def match_level_metrics(raw_labels: np.ndarray,
                         filtered_idx: np.ndarray,
                         pixel_thresholds: List[float] = [3, 5, 10],
                         kpts_q: Optional[np.ndarray] = None,
                         kpts_t: Optional[np.ndarray] = None,
                         gt_homography: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Compute match-level precision and recall.

    Args:
        raw_labels: (N,) boolean labels for ALL raw matches at threshold 5px
        filtered_idx: indices of filtered inliers
        pixel_thresholds: additional thresholds to evaluate at
        kpts_q, kpts_t, gt_homography: for multi-threshold evaluation

    Returns:
        metrics dict with precision, recall at each threshold
    """
    N = len(raw_labels)
    metrics = {}

    if N == 0:
        return {"precision": 0.0, "recall": 0.0, "n_raw": 0, "n_filtered": 0}

    n_filtered = len(filtered_idx)
    if n_filtered == 0:
        return {"precision": 0.0, "recall": 0.0, "n_raw": N, "n_filtered": 0,
                "raw_inlier_count": int(raw_labels.sum()),
                "raw_inlier_rate": float(raw_labels.mean())}

    # Filtered labels at default threshold
    filtered_labels = raw_labels[filtered_idx]
    precision = float(filtered_labels.sum() / n_filtered) if n_filtered > 0 else 0.0
    recall = float(filtered_labels.sum() / raw_labels.sum()) if raw_labels.sum() > 0 else 0.0

    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["n_raw"] = N
    metrics["n_filtered"] = n_filtered
    metrics["raw_inlier_count"] = int(raw_labels.sum())
    metrics["raw_inlier_rate"] = float(raw_labels.mean())
    metrics["filtered_inlier_count"] = int(filtered_labels.sum())

    # Multi-threshold evaluation
    if kpts_q is not None and kpts_t is not None and gt_homography is not None:
        for thresh in pixel_thresholds:
            labels_t = label_matches_with_gt(kpts_q, kpts_t, gt_homography, thresh)
            filtered_labels_t = labels_t[filtered_idx]
            p = float(filtered_labels_t.sum() / n_filtered) if n_filtered > 0 else 0.0
            r = float(filtered_labels_t.sum() / labels_t.sum()) if labels_t.sum() > 0 else 0.0
            metrics[f"precision_at_{thresh}px"] = p
            metrics[f"recall_at_{thresh}px"] = r

    return metrics


def decision_level_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Compute ROC and PR curves for the accept/reject gate.

    Args:
        scores: (N,) gate scores (probability of accept)
        labels: (N,) binary labels (1=true positive pair, 0=negative pair)

    Returns:
        metrics dict with ROC AUC, PR AUC, and curve data
    """
    metrics = {}

    if len(scores) == 0 or len(labels) == 0:
        return {"roc_auc": 0.0, "pr_auc": 0.0}

    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"roc_auc": 0.0, "pr_auc": 0.0, "warning": "single class"}

    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    metrics["roc_auc"] = float(roc_auc)
    metrics["pr_auc"] = float(pr_auc)
    metrics["roc_curve"] = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": roc_thresholds.tolist(),
    }
    metrics["pr_curve"] = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": pr_thresholds.tolist(),
    }

    return metrics


def false_accept_rate(gate_decisions: np.ndarray,
                      true_labels: np.ndarray,
                      pair_types: Optional[List[str]] = None) -> Dict[str, float]:
    """Compute false accept rate overall and by pair type.

    Args:
        gate_decisions: (N,) boolean, True=accepted
        true_labels: (N,) boolean, True=positive pair
        pair_types: optional list of pair types for per-type analysis

    Returns:
        dict with false accept rates
    """
    negatives = ~true_labels
    n_neg = negatives.sum()

    metrics = {}
    if n_neg > 0:
        false_accepts = (gate_decisions & negatives).sum()
        metrics["overall_far"] = float(false_accepts / n_neg)
    else:
        metrics["overall_far"] = 0.0

    if pair_types is not None:
        types = np.array(pair_types)
        for pt in ["hard_negative", "easy_negative"]:
            mask = types == pt
            n = mask.sum()
            if n > 0:
                fa = (gate_decisions[mask] & negatives[mask]).sum()
                metrics[f"far_{pt}"] = float(fa / n)
            else:
                metrics[f"far_{pt}"] = 0.0

    # True positive rate (accepted positives)
    positives = true_labels
    n_pos = positives.sum()
    if n_pos > 0:
        true_accepts = (gate_decisions & positives).sum()
        metrics["true_accept_rate"] = float(true_accepts / n_pos)
    else:
        metrics["true_accept_rate"] = 0.0

    return metrics


def competition_score(error_m: float, gsd: float, accepted: bool,
                      config: dict) -> float:
    """Compute competition scoring proxy for a single pair.

    score = +10 if error < 1*sigma, +6 if < 2*sigma, etc.
    sigma = sqrt(base_sigma^2 + (gsd_mult * GSD)^2)

    Args:
        error_m: localization error in metres
        gsd: query GSD in m/px
        accepted: whether the gate accepted this pair
        config: the 'evaluation.scoring' section

    Returns:
        score: float
    """
    sc = config["evaluation"]["scoring"]

    if not accepted:
        return sc["reject_score"]

    sigma = math.sqrt(sc["base_sigma_m"] ** 2 +
                       (sc["gsd_multiplier"] * gsd) ** 2)

    for band in sc["score_bands"]:
        if error_m <= band["max_sigma"] * sigma:
            return band["score"]

    return sc["beyond_8sigma_penalty"]


def compute_localization_error(ransac_transform: np.ndarray,
                                query_center: Tuple[float, float],
                                gt_center_in_tile: Tuple[float, float],
                                ref_gsd: float) -> float:
    """Compute localization error in metres from RANSAC fit.

    Args:
        ransac_transform: (2, 3) similarity transform
        query_center: (x, y) centre of query image
        gt_center_in_tile: (x, y) true query centre in tile pixels
        ref_gsd: reference GSD (m/px)

    Returns:
        error_m: localization error in metres
    """
    # Project query centre through the RANSAC transform
    qc = np.array([[query_center[0], query_center[1], 1.0]])
    projected = qc @ ransac_transform.T  # (1, 2)
    pred_x, pred_y = projected[0, 0], projected[0, 1]

    # Error in pixels
    dx = pred_x - gt_center_in_tile[0]
    dy = pred_y - gt_center_in_tile[1]
    error_px = math.sqrt(dx ** 2 + dy ** 2)

    # Convert to metres
    return error_px * ref_gsd


def ablation_study(kpts_q: np.ndarray, kpts_t: np.ndarray,
                    conf: np.ndarray,
                    img_q: np.ndarray, img_t: np.ndarray,
                    config: dict,
                    gt_labels: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Run ablation: remove each stage individually and measure impact.

    Args:
        kpts_q, kpts_t, conf: raw matches
        img_q, img_t: images
        config: full config
        gt_labels: (N,) boolean GT labels for raw matches

    Returns:
        dict mapping stage_name -> metrics dict
    """
    from csatsf.filters.pipeline import filter_matches

    results = {}

    # Full pipeline
    idx_full, feat_full = filter_matches(kpts_q, kpts_t, conf, img_q, img_t, config)
    labels_full = gt_labels[idx_full] if len(idx_full) > 0 else np.array([])
    p_full = float(labels_full.mean()) if len(labels_full) > 0 else 0.0
    results["full"] = {
        "n_inliers": len(idx_full),
        "precision": p_full,
        "features": feat_full,
    }

    # Remove each stage one at a time
    stages = [
        ("spatial_equalization", "spatial_equalization"),
        ("texture_saliency", "texture_saliency"),
        ("delaunay_topology", "delaunay_topology"),
        ("global_consistency", "global_consistency"),
    ]

    for stage_name, cfg_key in stages:
        # Create modified config with this stage disabled
        import copy
        cfg_mod = copy.deepcopy(config)
        cfg_mod["filters"][cfg_key]["enabled"] = False

        idx_mod, feat_mod = filter_matches(kpts_q, kpts_t, conf, img_q, img_t, cfg_mod)
        labels_mod = gt_labels[idx_mod] if len(idx_mod) > 0 else np.array([])
        p_mod = float(labels_mod.mean()) if len(labels_mod) > 0 else 0.0

        results[f"without_{stage_name}"] = {
            "n_inliers": len(idx_mod),
            "precision": p_mod,
            "precision_change": p_mod - p_full,
            "features": feat_mod,
        }

    return results

