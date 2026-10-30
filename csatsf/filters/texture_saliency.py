"""
Texture Saliency Filter (Stage B).

Computes local variance (texture measure) around each keypoint in both
query and tile images. Keeps matches where both points lie in textured regions.
"""

import cv2
import numpy as np
from typing import Tuple


def compute_local_variance(img: np.ndarray, kernel_size: int = 11) -> np.ndarray:
    """Compute normalised local variance map.

    Args:
        img: grayscale image (H, W), uint8 or float
        kernel_size: window size for local variance

    Returns:
        variance_map: (H, W) normalised to [0, 1]
    """
    img_f = img.astype(np.float32)
    if img_f.ndim == 3:
        img_f = cv2.cvtColor(img_f.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Local mean and mean of squares
    mean = cv2.blur(img_f, (kernel_size, kernel_size))
    mean_sq = cv2.blur(img_f ** 2, (kernel_size, kernel_size))

    # Variance = E[X^2] - E[X]^2
    variance = np.maximum(mean_sq - mean ** 2, 0)

    # Normalise to [0, 1]
    v_max = variance.max()
    if v_max > 0:
        variance = variance / v_max

    return variance


def texture_saliency(kpts_q: np.ndarray, kpts_t: np.ndarray,
                     conf: np.ndarray,
                     img_q: np.ndarray, img_t: np.ndarray,
                     gamma: float = 0.5,
                     kernel_size: int = 11) -> np.ndarray:
    """Apply texture saliency filter.

    Keeps matches where the local variance at both keypoint locations
    exceeds gamma * mean(V) in the respective image.

    Args:
        kpts_q: (N, 2) query keypoints (x, y)
        kpts_t: (N, 2) tile keypoints (x, y)
        conf: (N,) confidence scores (unused here, kept for API consistency)
        img_q: query image (H, W, 3) or (H, W)
        img_t: tile image (H, W, 3) or (H, W)
        gamma: threshold multiplier
        kernel_size: window for local variance

    Returns:
        inlier_mask: boolean array of shape (N,)
    """
    N = len(kpts_q)
    if N == 0:
        return np.array([], dtype=bool)

    # Compute variance maps
    V_q = compute_local_variance(img_q, kernel_size)
    V_t = compute_local_variance(img_t, kernel_size)

    # Thresholds
    thresh_q = gamma * V_q.mean()
    thresh_t = gamma * V_t.mean()

    # Sample variance at keypoint locations
    h_q, w_q = V_q.shape
    h_t, w_t = V_t.shape

    x_q = np.clip(kpts_q[:, 0].astype(int), 0, w_q - 1)
    y_q = np.clip(kpts_q[:, 1].astype(int), 0, h_q - 1)
    v_at_q = V_q[y_q, x_q]

    x_t = np.clip(kpts_t[:, 0].astype(int), 0, w_t - 1)
    y_t = np.clip(kpts_t[:, 1].astype(int), 0, h_t - 1)
    v_at_t = V_t[y_t, x_t]

    # Keep where both pass
    mask = (v_at_q >= thresh_q) & (v_at_t >= thresh_t)
    return mask

