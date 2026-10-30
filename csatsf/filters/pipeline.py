"""
C-SATSF Filter Pipeline.

Main entry point: filter_matches() — a clean function that takes raw matches
and returns filtered inlier indices + decision features. No GT access inside.

Run order: spatial_equalization -> texture_saliency -> delaunay_topology -> global_consistency
"""

import time
import numpy as np
from typing import Tuple, Dict, Any, Optional

from csatsf.filters.spatial_equalization import spatial_equalization
from csatsf.filters.texture_saliency import texture_saliency
from csatsf.filters.delaunay_topology import delaunay_topology
from csatsf.filters.global_consistency import global_consistency


def filter_matches(kpts_q: np.ndarray, kpts_t: np.ndarray,
                   conf: np.ndarray,
                   img_q: np.ndarray, img_t: np.ndarray,
                   config: dict) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply the full C-SATSF cascaded filter pipeline.

    This is the clean, reusable function that can be dropped into a full
    pipeline later. No ground truth access inside.

    Args:
        kpts_q: (N, 2) query keypoints in pixel coords (x, y)
        kpts_t: (N, 2) tile keypoints in pixel coords (x, y)
        conf: (N,) confidence scores
        img_q: query image (H, W, 3) or (H, W), uint8
        img_t: tile image (H, W, 3) or (H, W), uint8
        config: configuration dict (the 'filters' section)

    Returns:
        inlier_idx: 1D array of indices into the original arrays
        features: dict of decision features for the accept/reject gate
    """
    t0 = time.perf_counter()
    N_raw = len(kpts_q)
    fcfg = config["filters"]

    if N_raw == 0:
        return np.array([], dtype=int), _empty_features()

    # Track surviving indices through the cascade
    active_idx = np.arange(N_raw)
    stage_counts = {"raw": N_raw}

    # Current active keypoints/conf
    kq, kt, cf = kpts_q.copy(), kpts_t.copy(), conf.copy()

    # --- Stage A: Spatial Equalization ---
    if fcfg["spatial_equalization"]["enabled"]:
        img_shape = (img_q.shape[0], img_q.shape[1])
        mask_a = spatial_equalization(
            kq, kt, cf, img_shape,
            grid_size=fcfg["spatial_equalization"]["grid_size"],
            q_base=fcfg["spatial_equalization"]["q_base"],
            q_max_multiplier=fcfg["spatial_equalization"]["q_max_multiplier"],
        )
        active_idx = active_idx[mask_a]
        kq, kt, cf = kpts_q[active_idx], kpts_t[active_idx], conf[active_idx]
    stage_counts["after_spatial"] = len(active_idx)

    # --- Stage B: Texture Saliency ---
    if fcfg["texture_saliency"]["enabled"]:
        mask_b = texture_saliency(
            kq, kt, cf, img_q, img_t,
            gamma=fcfg["texture_saliency"]["gamma"],
            kernel_size=fcfg["texture_saliency"]["kernel_size"],
        )
        active_idx = active_idx[mask_b]
        kq, kt, cf = kpts_q[active_idx], kpts_t[active_idx], conf[active_idx]
    stage_counts["after_texture"] = len(active_idx)

    # --- Stage C: Delaunay Topology ---
    degenerate = False
    if fcfg["delaunay_topology"]["enabled"]:
        mask_c, degenerate = delaunay_topology(
            kq, kt, cf,
            eps_topo=fcfg["delaunay_topology"]["eps_topo"],
            vote_threshold=fcfg["delaunay_topology"]["vote_threshold"],
            min_points=fcfg["delaunay_topology"]["min_points"],
        )
        active_idx = active_idx[mask_c]
        kq, kt, cf = kpts_q[active_idx], kpts_t[active_idx], conf[active_idx]
    stage_counts["after_delaunay"] = len(active_idx)

    # --- Stage D: Global Consistency ---
    if fcfg["global_consistency"]["enabled"]:
        mask_d = global_consistency(
            kq, kt, cf,
            eps_ang=fcfg["global_consistency"]["eps_ang"],
            eps_scale=fcfg["global_consistency"]["eps_scale"],
        )
        active_idx = active_idx[mask_d]
        kq, kt, cf = kpts_q[active_idx], kpts_t[active_idx], conf[active_idx]
    stage_counts["after_global"] = len(active_idx)

    elapsed = time.perf_counter() - t0

    # --- Compute decision features ---
    features = compute_decision_features(
        kpts_q[active_idx], kpts_t[active_idx], conf[active_idx],
        img_q.shape, img_t.shape, stage_counts, degenerate,
        fcfg["spatial_equalization"]["grid_size"],
    )
    features["filter_time_s"] = elapsed

    return active_idx, features


def compute_decision_features(kpts_q: np.ndarray, kpts_t: np.ndarray,
                               conf: np.ndarray,
                               img_q_shape: tuple, img_t_shape: tuple,
                               stage_counts: dict, degenerate: bool,
                               grid_size: int) -> Dict[str, Any]:
    """Compute post-filter features for the accept/reject gate.

    Args:
        kpts_q, kpts_t: filtered keypoints
        conf: filtered confidences
        img_q_shape, img_t_shape: image shapes (H, W, ...)
        stage_counts: dict with counts after each stage
        degenerate: whether Delaunay was degenerate
        grid_size: G for grid occupancy

    Returns:
        features dict
    """
    N = len(kpts_q)
    features = {}

    features["n_inliers"] = N
    features["degenerate_delaunay"] = degenerate
    features["stage_counts"] = stage_counts

    # Stage survival ratios
    raw_n = stage_counts.get("raw", 1)
    if raw_n > 0:
        for key in ["after_spatial", "after_texture", "after_delaunay", "after_global"]:
            features[f"survival_{key}"] = stage_counts.get(key, 0) / raw_n
    else:
        for key in ["after_spatial", "after_texture", "after_delaunay", "after_global"]:
            features[f"survival_{key}"] = 0.0

    if N == 0:
        features.update(_empty_feature_values())
        return features

    # Mean confidence
    features["mean_confidence"] = float(conf.mean())

    # Spatial coverage: convex hull area / image area
    h_q, w_q = img_q_shape[:2]
    img_area = h_q * w_q
    try:
        from scipy.spatial import ConvexHull
        if N >= 3:
            hull = ConvexHull(kpts_q)
            features["coverage"] = float(hull.volume / img_area)
        else:
            features["coverage"] = 0.0
    except Exception:
        features["coverage"] = 0.0

    # Grid cell occupancy
    cell_w = w_q / grid_size
    cell_h = h_q / grid_size
    cell_x = np.clip((kpts_q[:, 0] / cell_w).astype(int), 0, grid_size - 1)
    cell_y = np.clip((kpts_q[:, 1] / cell_h).astype(int), 0, grid_size - 1)
    occupied = len(set(zip(cell_x, cell_y)))
    features["grid_occupancy"] = occupied / (grid_size * grid_size)

    # RANSAC similarity transform fit
    if N >= 3:
        import cv2
        try:
            # Similarity transform (4 DOF: rotation, scale, translation)
            M_sim, inliers_sim = cv2.estimateAffinePartial2D(
                kpts_q.astype(np.float32).reshape(-1, 1, 2),
                kpts_t.astype(np.float32).reshape(-1, 1, 2),
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0,
            )
            if inliers_sim is not None:
                inlier_mask = inliers_sim.ravel().astype(bool)
                features["ransac_sim_inlier_ratio"] = float(inlier_mask.sum() / N)

                # Reprojection residuals
                if M_sim is not None:
                    ones = np.ones((N, 1), dtype=np.float32)
                    kq_h = np.hstack([kpts_q.astype(np.float32), ones])  # (N, 3)
                    projected = kq_h @ M_sim.T  # (N, 2)
                    residuals = np.linalg.norm(projected - kpts_t.astype(np.float32), axis=1)
                    features["median_reproj_residual"] = float(np.median(residuals))
                    features["mean_reproj_residual"] = float(np.mean(residuals))

                    # Extract rotation and scale from similarity matrix
                    a, b = M_sim[0, 0], M_sim[0, 1]
                    scale = float(np.sqrt(a ** 2 + b ** 2))
                    rotation_deg = float(np.degrees(np.arctan2(b, a)))
                    features["ransac_scale"] = scale
                    features["ransac_rotation_deg"] = rotation_deg
                else:
                    features["ransac_sim_inlier_ratio"] = 0.0
                    features["median_reproj_residual"] = float("inf")
                    features["mean_reproj_residual"] = float("inf")
                    features["ransac_scale"] = 0.0
                    features["ransac_rotation_deg"] = 0.0
            else:
                features["ransac_sim_inlier_ratio"] = 0.0
                features["median_reproj_residual"] = float("inf")
                features["mean_reproj_residual"] = float("inf")
                features["ransac_scale"] = 0.0
                features["ransac_rotation_deg"] = 0.0
        except Exception:
            features["ransac_sim_inlier_ratio"] = 0.0
            features["median_reproj_residual"] = float("inf")
            features["mean_reproj_residual"] = float("inf")
            features["ransac_scale"] = 0.0
            features["ransac_rotation_deg"] = 0.0

        # Homography fit
        try:
            H, inliers_h = cv2.findHomography(
                kpts_q.astype(np.float32).reshape(-1, 1, 2),
                kpts_t.astype(np.float32).reshape(-1, 1, 2),
                cv2.RANSAC, 5.0
            )
            if inliers_h is not None:
                features["ransac_homo_inlier_ratio"] = float(
                    inliers_h.ravel().astype(bool).sum() / N
                )
            else:
                features["ransac_homo_inlier_ratio"] = 0.0
        except Exception:
            features["ransac_homo_inlier_ratio"] = 0.0
    else:
        features["ransac_sim_inlier_ratio"] = 0.0
        features["ransac_homo_inlier_ratio"] = 0.0
        features["median_reproj_residual"] = float("inf")
        features["mean_reproj_residual"] = float("inf")
        features["ransac_scale"] = 0.0
        features["ransac_rotation_deg"] = 0.0

    return features


def _empty_features() -> Dict[str, Any]:
    """Return empty features dict for zero-match case."""
    f = _empty_feature_values()
    f["n_inliers"] = 0
    f["degenerate_delaunay"] = True
    f["stage_counts"] = {"raw": 0}
    f["filter_time_s"] = 0.0
    for key in ["after_spatial", "after_texture", "after_delaunay", "after_global"]:
        f[f"survival_{key}"] = 0.0
    return f


def _empty_feature_values() -> Dict[str, Any]:
    """Default feature values when there are no inliers."""
    return {
        "mean_confidence": 0.0,
        "coverage": 0.0,
        "grid_occupancy": 0.0,
        "ransac_sim_inlier_ratio": 0.0,
        "ransac_homo_inlier_ratio": 0.0,
        "median_reproj_residual": float("inf"),
        "mean_reproj_residual": float("inf"),
        "ransac_scale": 0.0,
        "ransac_rotation_deg": 0.0,
    }

