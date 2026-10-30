"""
Delaunay Topology Filter (Stage C).

Triangulates query keypoints, computes area ratios of corresponding triangles
in tile space, and removes points with too many topologically inconsistent
incident triangles.
"""

import numpy as np
from scipy.spatial import Delaunay
from typing import Tuple, Optional


def delaunay_topology(kpts_q: np.ndarray, kpts_t: np.ndarray,
                      conf: np.ndarray,
                      eps_topo: float = 0.4,
                      vote_threshold: float = 0.5,
                      min_points: int = 4) -> Tuple[np.ndarray, bool]:
    """Apply Delaunay topology consistency filter.

    Args:
        kpts_q: (N, 2) query keypoints (x, y)
        kpts_t: (N, 2) tile keypoints (x, y)
        conf: (N,) confidence scores (unused, kept for API)
        eps_topo: relative area deviation threshold
        vote_threshold: fraction of bad triangles to drop a point
        min_points: minimum points for triangulation

    Returns:
        inlier_mask: boolean array of shape (N,)
        degenerate: True if the case was degenerate (too few points, collinear)
    """
    N = len(kpts_q)
    if N == 0:
        return np.array([], dtype=bool), True

    if N < min_points:
        return np.ones(N, dtype=bool), True

    # Check for collinearity: if all points lie on a line, Delaunay fails
    try:
        tri = Delaunay(kpts_q)
    except Exception:
        return np.ones(N, dtype=bool), True

    simplices = tri.simplices  # (M, 3) triangle vertex indices

    if len(simplices) == 0:
        return np.ones(N, dtype=bool), True

    # Compute triangle areas in query and tile space
    def triangle_areas(pts, simplices):
        """Compute signed areas of triangles."""
        p0 = pts[simplices[:, 0]]
        p1 = pts[simplices[:, 1]]
        p2 = pts[simplices[:, 2]]
        # Cross product / 2
        areas = 0.5 * np.abs(
            (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) -
            (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
        )
        return areas

    areas_q = triangle_areas(kpts_q, simplices)
    areas_t = triangle_areas(kpts_t, simplices)

    # Avoid division by zero
    valid_areas = areas_q > 1e-8
    rho = np.ones(len(simplices))
    rho[valid_areas] = areas_t[valid_areas] / areas_q[valid_areas]

    median_rho = np.median(rho[valid_areas]) if valid_areas.any() else 1.0
    if median_rho < 1e-8:
        return np.ones(N, dtype=bool), True

    # Relative deviation from median
    delta = np.abs(rho - median_rho) / median_rho

    # Mark bad triangles
    bad_triangles = delta > eps_topo

    # Count negative votes per point
    # Each point's vote rate = fraction of its incident triangles that are bad
    neg_votes = np.zeros(N, dtype=float)
    total_incidents = np.zeros(N, dtype=float)

    for tri_idx in range(len(simplices)):
        for vi in range(3):
            pt_idx = simplices[tri_idx, vi]
            total_incidents[pt_idx] += 1
            if bad_triangles[tri_idx]:
                neg_votes[pt_idx] += 1

    # Compute vote rate (avoid div by zero)
    vote_rate = np.zeros(N)
    has_triangles = total_incidents > 0
    vote_rate[has_triangles] = neg_votes[has_triangles] / total_incidents[has_triangles]

    # Keep points with vote rate <= threshold
    mask = vote_rate <= vote_threshold

    return mask, False

