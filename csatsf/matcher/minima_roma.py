"""
MINIMA-RoMa Matcher Wrapper (Step 2).

Wraps MINIMA-RoMa (thermal/IR-to-optical) matching using the MINIMA
repository's own loading and inference logic.
Runs inference once per pair and saves results to cache/.
All subsequent steps run from cache (no GPU needed).
"""

import os
import sys
import numpy as np

# Compatibility patch for NumPy 2.x with legacy MINIMA/RoMa code
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

import cv2
from typing import Optional, Dict
from argparse import Namespace


def setup_minima_path(project_root: str):
    """Add MINIMA source to Python path."""
    minima_dir = os.path.join(project_root, "MINIMA")
    if os.path.isdir(minima_dir) and minima_dir not in sys.path:
        sys.path.insert(0, minima_dir)


class MINIMARoMaMatcher:
    """MINIMA-RoMa matcher for thermal-to-optical matching."""

    def __init__(self, config: dict, project_root: str, device: str = "cuda"):
        self.config = config
        self.project_root = os.path.abspath(project_root)
        self.cache_dir = os.path.abspath(os.path.join(self.project_root, config["paths"]["cache_dir"]))
        os.makedirs(self.cache_dir, exist_ok=True)
        self.device_str = device
        self.matcher_fn = None  # callable: from_cv_imgs(img0, img1) -> dict

    def load_model(self):
        """Load MINIMA-RoMa model using local checkpoints."""
        import torch
        device = torch.device(self.device_str if torch.cuda.is_available() else "cpu")
        print(f"Loading MINIMA-RoMa on {device}...")

        minima_dir = os.path.join(self.project_root, "MINIMA")
        roma_dir = os.path.join(minima_dir, "third_party", "RoMa_minima")
        if roma_dir not in sys.path:
            sys.path.insert(0, roma_dir)
        if minima_dir not in sys.path:
            sys.path.insert(0, minima_dir)

        ckpt_path = os.path.join(self.project_root, self.config["matcher"]["checkpoint"])
        dinov2_path = os.path.join(self.project_root, self.config["matcher"]["dinov2_checkpoint"])

        original_cwd = os.getcwd()
        os.chdir(minima_dir)
        try:
            from third_party.RoMa_minima.romatch import roma_outdoor
            from src.config.default import get_cfg_defaults
            from src.utils.data_io_roma import DataIOWrapper, lower_config

            print(f"Loading dinov2 from {dinov2_path}...")
            dinov2_weights = torch.load(dinov2_path, map_location=device)
            print(f"Loading MINIMA weights from {ckpt_path}...")
            minima_weights = torch.load(ckpt_path, map_location=device)

            model = roma_outdoor(device=device, weights=minima_weights, dinov2_weights=dinov2_weights)
            model.eval()

            config = get_cfg_defaults(inference=True)
            config = lower_config(config)
            wrapper = DataIOWrapper(model, config=config["test"])
            self.matcher_fn = wrapper.from_cv_imgs
            print("MINIMA-RoMa loaded successfully!")
        finally:
            os.chdir(original_cwd)

    def _cache_path(self, pair_id: str) -> str:
        """Get cache file path for a pair."""
        return os.path.join(self.cache_dir, f"{pair_id}.npz")

    def is_cached(self, pair_id: str) -> bool:
        """Check if results for this pair are already cached."""
        return os.path.exists(self._cache_path(pair_id))

    def load_cached(self, pair_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Load cached match results."""
        path = self._cache_path(pair_id)
        if not os.path.exists(path):
            return None
        data = np.load(path, allow_pickle=True)
        return {
            "kpts_q": data["kpts_q"],
            "kpts_t": data["kpts_t"],
            "conf": data["conf"],
            "img_q_shape": tuple(data["img_q_shape"]),
            "img_t_shape": tuple(data["img_t_shape"]),
        }

    def match(self, img_q: np.ndarray, img_t: np.ndarray,
              pair_id: str, force: bool = False) -> Dict[str, np.ndarray]:
        """Run matching on a pair and cache the results.

        Args:
            img_q: query image (H, W, 3) uint8, RGB or BGR
            img_t: tile image (H, W, 3) uint8, RGB or BGR
            pair_id: unique pair identifier
            force: re-compute even if cached

        Returns:
            dict with kpts_q, kpts_t, conf, img shapes
        """
        if not force and self.is_cached(pair_id):
            return self.load_cached(pair_id)

        if self.matcher_fn is None:
            self.load_model()

        h_q, w_q = img_q.shape[:2]
        h_t, w_t = img_t.shape[:2]

        # MINIMA expects BGR images (as loaded by cv2.imread)
        # If input is RGB, convert
        img_q_bgr = cv2.cvtColor(img_q, cv2.COLOR_RGB2BGR) if img_q.ndim == 3 else img_q
        img_t_bgr = cv2.cvtColor(img_t, cv2.COLOR_RGB2BGR) if img_t.ndim == 3 else img_t

        try:
            data = self.matcher_fn(img_q_bgr, img_t_bgr)
            kpts_q = data["mkpts0"].astype(np.float32)
            kpts_t = data["mkpts1"].astype(np.float32)
            conf = data["mconf"].astype(np.float32)
        except Exception as e:
            print(f"Warning: Matching failed for pair {pair_id}: {e}")
            kpts_q = np.zeros((0, 2), dtype=np.float32)
            kpts_t = np.zeros((0, 2), dtype=np.float32)
            conf = np.zeros(0, dtype=np.float32)

        # Apply confidence threshold
        if len(conf) > 0:
            thresh = self.config["matcher"]["confidence_threshold"]
            mask = conf >= thresh
            kpts_q = kpts_q[mask]
            kpts_t = kpts_t[mask]
            conf = conf[mask]

        # Limit max matches
        max_m = self.config["matcher"]["max_matches"]
        if len(conf) > max_m:
            topk = np.argsort(-conf)[:max_m]
            kpts_q = kpts_q[topk]
            kpts_t = kpts_t[topk]
            conf = conf[topk]

        result = {
            "kpts_q": kpts_q,
            "kpts_t": kpts_t,
            "conf": conf,
            "img_q_shape": np.array([h_q, w_q]),
            "img_t_shape": np.array([h_t, w_t]),
        }

        # Cache
        cache_path = self._cache_path(pair_id)
        np.savez_compressed(cache_path, **result)

        return result

    def batch_match(self, pairs_data: list, pair_builder,
                    progress_every: int = 10) -> int:
        """Run matching on a batch of pairs.

        Args:
            pairs_data: list of PairRecord objects
            pair_builder: PairBuilder instance for loading images
            progress_every: print progress every N pairs

        Returns:
            Number of pairs processed (not from cache)
        """
        processed = 0
        skipped_cache = 0
        for i, pair in enumerate(pairs_data):
            if self.is_cached(pair.pair_id):
                skipped_cache += 1
                continue

            img_q = pair_builder.get_query_image(pair)
            img_t = pair_builder.get_tile_image(pair)

            if img_q is None or img_t is None:
                print(f"  Skipping pair {pair.pair_id}: could not load images")
                continue

            self.match(img_q, img_t, pair.pair_id)
            processed += 1

            if processed % progress_every == 0:
                print(f"  Matched {processed} pairs "
                      f"({skipped_cache} from cache, "
                      f"{i + 1}/{len(pairs_data)} total)")

        print(f"Done: {processed} new, {skipped_cache} cached, "
              f"{len(pairs_data)} total")
        return processed
