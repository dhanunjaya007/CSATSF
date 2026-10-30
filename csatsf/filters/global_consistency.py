"""
Global Consistency Filter (Stage D).

Checks angular and scale consistency of matches relative to the
centroid of the match set.
"""

import numpy as np
from typing import Tuple


def wrap_angle(angles: np.ndarray) -> np.ndarray:
    """Wrap angles to [-180, 180] degrees."""
    return ((angles + 180) % 360) - 180


def global_consistency(kpts_q: np.ndarray, kpts_t: np.ndarray,
                       conf: np.ndarray,
                       eps_ang: float = 20.0,
                       eps_scale: float = 0.3) -> np.ndarray:
    """Apply global consistency filter.

    For each match, computes the angle and scale of the displacement
    from centroid in tile vs query space. Keeps matches consistent
    with the median rotation and scale.

    Args:
        kpts_q: (N, 2) query keypoints (x, y)
        kpts_t: (N, 2) tile keypoints (x, y)
        conf: (N,) confidence scores (unused, kept for API)
        eps_ang: angular consistency threshold (degrees)
        eps_scale: relative scale consistency threshold

    Returns:
        inlier_mask: boolean array of shape (N,)
    """
    N = len(kpts_q)
    if N == 0:
        return np.array([], dtype=bool)

    if N < 3:
        return np.ones(N, dtype=bool)

    # Centroids
    centroid_q = kpts_q.mean(axis=0)
    centroid_t = kpts_t.mean(axis=0)

    # Vectors from centroid
    v_q = kpts_q - centroid_q  # (N, 2)
    v_t = kpts_t - centroid_t  # (N, 2)

    # Distances from centroid
    dist_q = np.linalg.norm(v_q, axis=1)
    dist_t = np.linalg.norm(v_t, axis=1)

    # Filter out points too close to centroid (degenerate angles)
    valid = (dist_q > 1e-3) & (dist_t > 1e-3)
    if valid.sum() < 3:
        return np.ones(N, dtype=bool)

    # Angles in degrees
    ang_q = np.degrees(np.arctan2(v_q[:, 1], v_q[:, 0]))
    ang_t = np.degrees(np.arctan2(v_t[:, 1], v_t[:, 0]))

    # Rotation: phi_i = angle(v_tile) - angle(v_query), wrapped
    phi = wrap_angle(ang_t - ang_q)

    # Scale ratio
    scale = np.ones(N)
    scale[valid] = dist_t[valid] / dist_q[valid]

    # Median rotation and scale (only from valid points)
    med_phi = np.median(phi[valid])
    med_scale = np.median(scale[valid])

    if med_scale < 1e-8:
        return np.ones(N, dtype=bool)

    # Angular consistency
    ang_diff = np.abs(wrap_angle(phi - med_phi))
    ang_ok = ang_diff < eps_ang

    # Scale consistency
    scale_diff = np.abs(scale - med_scale) / med_scale
    scale_ok = scale_diff < eps_scale

    # Points too close to centroid always pass (can't check angle/scale)
    mask = (ang_ok & scale_ok) | ~valid

    return mask

