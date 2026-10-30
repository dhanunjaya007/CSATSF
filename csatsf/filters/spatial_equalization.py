"""
Spatial Equalization Filter (Stage A).

Divides the image into a G×G grid and keeps the top-Q matches
per cell (by confidence), where Q adapts to cell density.
"""

import numpy as np
from typing import Tuple


def spatial_equalization(kpts_q: np.ndarray, kpts_t: np.ndarray,
                         conf: np.ndarray, img_shape: Tuple[int, int],
                         grid_size: int = 8, q_base: int = 3,
                         q_max_multiplier: int = 3) -> np.ndarray:
    """Apply spatial equalization filter.

    Args:
        kpts_q: (N, 2) query keypoints (x, y)
        kpts_t: (N, 2) tile keypoints (x, y)
        conf: (N,) confidence scores
        img_shape: (H, W) of the query image
        grid_size: G, grid divisions per side
        q_base: base quota per cell
        q_max_multiplier: Q_max = q_max_multiplier * q_base

    Returns:
        inlier_mask: boolean array of shape (N,)
    """
    N = len(kpts_q)
    if N == 0:
        return np.array([], dtype=bool)

    H, W = img_shape
    q_max = q_max_multiplier * q_base

    # Assign each keypoint to a grid cell (based on query keypoint position)
    cell_w = W / grid_size
    cell_h = H / grid_size
    cell_x = np.clip((kpts_q[:, 0] / cell_w).astype(int), 0, grid_size - 1)
    cell_y = np.clip((kpts_q[:, 1] / cell_h).astype(int), 0, grid_size - 1)
    cell_ids = cell_y * grid_size + cell_x

    # Count matches per cell
    cell_counts = np.bincount(cell_ids, minlength=grid_size * grid_size)

    # Compute adaptive quota per cell
    # Q(c) = min(q_base + floor(log2(c+1)), q_max)
    keep_mask = np.zeros(N, dtype=bool)

    for cell_id in range(grid_size * grid_size):
        cell_mask = cell_ids == cell_id
        c = cell_counts[cell_id]
        if c == 0:
            continue
        quota = min(q_base + int(np.floor(np.log2(c + 1))), q_max)

        # Get indices for this cell, sorted by confidence descending
        cell_indices = np.where(cell_mask)[0]
        sorted_by_conf = cell_indices[np.argsort(-conf[cell_indices])]

        # Keep top-Q
        keep_mask[sorted_by_conf[:quota]] = True

    return keep_mask

