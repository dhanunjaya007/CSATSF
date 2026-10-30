"""
Pair Builder: generates (thermal query, satellite tile) pairs from the dataset.

Creates three pair types per query:
  - POSITIVE: tile containing the true footprint (with random offset)
  - HARD_NEGATIVE: visually similar tiles nearby (200m-2km)
  - EASY_NEGATIVE: random far-away tiles (>3km)

Also generates augmented variants of positives.
"""

import json
import math
import os
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol, xy
from rasterio.warp import transform as warp_transform


@dataclass
class PairRecord:
    """Metadata for a single (query, tile) pair."""
    pair_id: str
    query_path: str
    query_idx: int  # index into the metadata list
    pair_type: str  # "positive", "hard_negative", "easy_negative", "augmented"
    split: str  # "train", "valid", "test"
    sequence: str  # flight sequence name

    # Query metadata
    lat: float
    lon: float
    rel_height: float
    yaw: float
    query_gsd: float  # m/px

    # Tile info
    tile_center_utm_x: float
    tile_center_utm_y: float
    tile_window: Tuple[int, int, int, int]  # (col_off, row_off, width, height) in ref.tif
    tile_gsd: float  # m/px (resampled to match query)

    # Ground truth (for positives only)
    gt_query_center_in_tile: Optional[Tuple[float, float]] = None  # (x, y) pixel coords
    gt_homography: Optional[List[float]] = None  # 3x3 flattened
    gt_offset_m: float = 0.0  # distance between tile centre and true position

    # Augmentation params (for augmented pairs)
    aug_params: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return asdict(self)


def load_metadata(metadata_dir: str) -> Dict[str, list]:
    """Load train/valid/test metadata JSON files."""
    splits = {}
    for split in ["train", "valid", "test"]:
        fpath = os.path.join(metadata_dir, f"{split}_Thermal.json")
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                splits[split] = json.load(f)
    return splits


def compute_query_gsd(rec: dict) -> float:
    """Compute query GSD in m/px from metadata record.
    cam_size is diagonal in mm; derive horizontal sensor width.
    """
    diag_px = math.sqrt(rec["width"] ** 2 + rec["height"] ** 2)
    sensor_w = rec["cam_size"] * (rec["width"] / diag_px)
    f_px = rec["focal_len"] * rec["width"] / sensor_w
    return rec["rel_height"] / f_px


def latlon_to_utm(lat: float, lon: float, ref_crs: str) -> Tuple[float, float]:
    """Convert WGS84 lat/lon to the reference map CRS (UTM)."""
    xs, ys = warp_transform("EPSG:4326", ref_crs, [lon], [lat])
    return xs[0], ys[0]


def compute_footprint_pixels(rec: dict, ref_gsd: float) -> Tuple[int, int]:
    """Compute the tile size in ref pixels that matches the query footprint."""
    query_gsd = compute_query_gsd(rec)
    # Query footprint in metres
    w_m = rec["width"] * query_gsd
    h_m = rec["height"] * query_gsd
    # Convert to ref pixels
    w_px = int(round(w_m / ref_gsd))
    h_px = int(round(h_m / ref_gsd))
    return w_px, h_px


def extract_tile(ref_ds: rasterio.DatasetReader, center_x: float, center_y: float,
                 tile_w_px: int, tile_h_px: int,
                 target_w: int, target_h: int) -> Optional[np.ndarray]:
    """Extract and resample a tile from the reference map.

    Args:
        ref_ds: opened rasterio dataset
        center_x, center_y: UTM coordinates of tile centre
        tile_w_px, tile_h_px: size in ref pixels
        target_w, target_h: output size (resampled to match query resolution)

    Returns:
        RGB image as numpy array (H, W, 3) or None if out of bounds
    """
    # Convert UTM centre to pixel coordinates
    row_c, col_c = ref_ds.index(center_x, center_y)

    col_off = int(col_c - tile_w_px // 2)
    row_off = int(row_c - tile_h_px // 2)

    # Check bounds
    if (col_off < 0 or row_off < 0 or
            col_off + tile_w_px > ref_ds.width or
            row_off + tile_h_px > ref_ds.height):
        return None

    window = Window(col_off, row_off, tile_w_px, tile_h_px)
    tile = ref_ds.read([1, 2, 3], window=window)  # (3, H, W)
    tile = np.transpose(tile, (1, 2, 0))  # (H, W, 3)

    # Resample to target size
    if tile.shape[0] != target_h or tile.shape[1] != target_w:
        tile = cv2.resize(tile, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return tile


def compute_gt_homography(rec: dict, tile_center_x: float, tile_center_y: float,
                          query_gsd: float, ref_gsd: float,
                          tile_w_px: int, tile_h_px: int,
                          target_w: int, target_h: int) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Compute ground-truth homography from query pixels to tile pixels.

    Uses GPS position, yaw, and GSD to build the mapping.

    Returns:
        H: 3x3 homography matrix (query_px -> tile_px)
        gt_center: (x, y) query centre in tile pixel coords
    """
    query_x, query_y = latlon_to_utm(rec["lat"], rec["lon"],
                                      "EPSG:32649")

    # Query centre position in tile pixel coordinates
    # Tile pixel (0,0) is top-left
    # tile_center corresponds to (target_w/2, target_h/2) in tile pixels
    scale_x = target_w / tile_w_px
    scale_y = target_h / tile_h_px

    dx_m = query_x - tile_center_x
    dy_m = query_y - tile_center_y  # UTM y increases north, pixel y increases down

    gt_cx = target_w / 2.0 + dx_m / ref_gsd * scale_x
    gt_cy = target_h / 2.0 - dy_m / ref_gsd * scale_y  # flip y

    # Build homography: translate query centre to origin, rotate by yaw,
    # scale by GSD ratio, translate to tile position
    yaw_rad = math.radians(rec["yaw"])
    s = query_gsd / ref_gsd * scale_x  # combined scale

    # Query pixel to metres (centred on query image centre)
    qcx = rec["width"] / 2.0
    qcy = rec["height"] / 2.0

    # H = T_tile @ S @ R @ T_query_inv
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)

    H = np.array([
        [s * cos_y,  -s * sin_y, gt_cx - s * (cos_y * qcx - sin_y * qcy)],
        [s * sin_y,   s * cos_y, gt_cy - s * (sin_y * qcx + cos_y * qcy)],
        [0,           0,         1]
    ], dtype=np.float64)

    return H, (gt_cx, gt_cy)


def compute_tile_similarity_orb(tile1: np.ndarray, tile2: np.ndarray) -> float:
    """Cheap similarity score between two tiles using ORB + grayscale correlation."""
    g1 = cv2.cvtColor(tile1, cv2.COLOR_RGB2GRAY) if tile1.ndim == 3 else tile1
    g2 = cv2.cvtColor(tile2, cv2.COLOR_RGB2GRAY) if tile2.ndim == 3 else tile2

    # Downsample for speed
    sz = (128, 128)
    g1s = cv2.resize(g1, sz)
    g2s = cv2.resize(g2, sz)

    # Normalised cross-correlation
    g1f = g1s.astype(np.float32) - g1s.mean()
    g2f = g2s.astype(np.float32) - g2s.mean()
    denom = max(np.linalg.norm(g1f) * np.linalg.norm(g2f), 1e-8)
    ncc = float(np.sum(g1f * g2f) / denom)

    return ncc


def make_pair_id(query_path: str, pair_type: str, idx: int) -> str:
    """Generate a deterministic pair ID."""
    key = f"{query_path}_{pair_type}_{idx}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def sample_augmentation_params(rec: dict, rng: np.random.Generator,
                               config: dict) -> Tuple[Dict[str, Any], np.ndarray]:
    """Sample random augmentation parameters and compute the query->aug affine matrix."""
    aug_cfg = config["pair_builder"]["augmentation"]
    params = {}

    alt = float(rng.uniform(aug_cfg["altitude_range"][0], aug_cfg["altitude_range"][1]))
    orig_gsd = compute_query_gsd(rec)
    diag = math.sqrt(rec["width"] ** 2 + rec["height"] ** 2)
    sensor_w = rec["cam_size"] * (rec["width"] / diag)
    f_px = rec["focal_len"] * rec["width"] / sensor_w
    new_gsd = alt / f_px
    scale_factor = float(orig_gsd / new_gsd)

    params["altitude"] = alt
    params["scale_factor"] = scale_factor
    params["new_gsd"] = float(new_gsd)

    w, h = int(rec["width"]), int(rec["height"])
    new_w = max(16, int(round(w * scale_factor)))
    new_h = max(16, int(round(h * scale_factor)))
    params["scaled_w"] = new_w
    params["scaled_h"] = new_h

    yaw_aug = float(rng.uniform(aug_cfg["yaw_range"][0], aug_cfg["yaw_range"][1]))
    params["yaw_aug"] = yaw_aug

    tw, th = int(aug_cfg["target_width"]), int(aug_cfg["target_height"])
    params["target_w"] = tw
    params["target_h"] = th

    ksize = int(rng.integers(aug_cfg["blur_kernel_range"][0], aug_cfg["blur_kernel_range"][1] + 1))
    if ksize % 2 == 0:
        ksize += 1
    params["blur_kernel"] = ksize

    noise_std = float(rng.uniform(aug_cfg["noise_std_range"][0], aug_cfg["noise_std_range"][1]))
    params["noise_std"] = noise_std
    params["seed"] = int(rng.integers(0, 1000000))

    # Affine matrix T_aug mapping (x_q, y_q) -> (x_aug, y_aug)
    S1 = np.array([[new_w / w, 0, 0], [0, new_h / h, 0], [0, 0, 1.0]], dtype=np.float64)
    M2 = cv2.getRotationMatrix2D((new_w / 2.0, new_h / 2.0), -yaw_aug, 1.0)
    R = np.vstack([M2, [0, 0, 1.0]])
    S2 = np.array([[tw / new_w, 0, 0], [0, th / new_h, 0], [0, 0, 1.0]], dtype=np.float64)
    T_aug = S2 @ R @ S1

    return params, T_aug


def apply_augmentation(img: np.ndarray, params: dict) -> np.ndarray:
    """Deterministically apply augmentation parameters to a query image."""
    new_w = params["scaled_w"]
    new_h = params["scaled_h"]
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    yaw = params["yaw_aug"]
    M = cv2.getRotationMatrix2D((new_w / 2.0, new_h / 2.0), -yaw, 1.0)
    rotated = cv2.warpAffine(scaled, M, (new_w, new_h), borderMode=cv2.BORDER_REFLECT)

    tw = params["target_w"]
    th = params["target_h"]
    result = cv2.resize(rotated, (tw, th), interpolation=cv2.INTER_AREA)

    ksize = params.get("blur_kernel", 3)
    if ksize > 1:
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)

    noise_std = params.get("noise_std", 0.0)
    if noise_std > 0:
        rng = np.random.default_rng(params.get("seed", 42))
        noise = rng.normal(0, noise_std, result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return result


class PairBuilder:
    """Builds (thermal query, satellite tile) pairs for the C-SATSF pipeline."""

    def __init__(self, config: dict, project_root: str):
        self.config = config
        self.project_root = os.path.abspath(project_root)
        self.ref_path = os.path.abspath(os.path.join(self.project_root, config["paths"]["ref_tif"]))
        self.metadata_dir = os.path.abspath(os.path.join(self.project_root, config["paths"]["metadata_dir"]))
        self.thermal_root = os.path.abspath(os.path.join(self.project_root, config["paths"]["thermal_root"]))
        self.ref_gsd = config["reference"]["gsd"]
        self.ref_crs = config["reference"]["crs"]
        self.rng = np.random.default_rng(config["seed"])

    def build_all_pairs(self, split: str = "train",
                        max_queries: Optional[int] = None) -> List[PairRecord]:
        """Build all pairs for a given split.

        Args:
            split: "train", "valid", or "test"
            max_queries: limit number of queries (for debugging)

        Returns:
            List of PairRecord objects
        """
        metadata = load_metadata(self.metadata_dir)
        records = metadata[split]
        if max_queries is not None:
            records = records[:max_queries]

        all_pairs = []
        ref_ds = rasterio.open(self.ref_path)

        try:
            for idx, rec in enumerate(records):
                query_gsd = compute_query_gsd(rec)
                query_utm_x, query_utm_y = latlon_to_utm(
                    rec["lat"], rec["lon"], self.ref_crs
                )

                # Tile size in ref pixels
                tile_w_px, tile_h_px = compute_footprint_pixels(rec, self.ref_gsd)
                target_w = int(rec["width"])
                target_h = int(rec["height"])

                # Extract sequence name from the path
                parts = rec["name"].replace("\\", "/").split("/")
                sequence = ""
                for p in parts:
                    if "changsha" in parts:
                        ci = parts.index("changsha")
                        if ci + 1 < len(parts):
                            sequence = parts[ci + 1]
                        break

                # --- POSITIVE pair ---
                offset_m = self.config["pair_builder"]["positive"]["center_offset_m"]
                dx = self.rng.uniform(-offset_m, offset_m)
                dy = self.rng.uniform(-offset_m, offset_m)
                pos_cx = query_utm_x + dx
                pos_cy = query_utm_y + dy

                pos_tile = extract_tile(ref_ds, pos_cx, pos_cy,
                                        tile_w_px, tile_h_px, target_w, target_h)
                if pos_tile is not None:
                    H, gt_center = compute_gt_homography(
                        rec, pos_cx, pos_cy, query_gsd, self.ref_gsd,
                        tile_w_px, tile_h_px, target_w, target_h
                    )
                    pair_id = make_pair_id(rec["name"], "positive", 0)
                    pr = PairRecord(
                        pair_id=pair_id,
                        query_path=rec["name"],
                        query_idx=idx,
                        pair_type="positive",
                        split=split,
                        sequence=sequence,
                        lat=rec["lat"],
                        lon=rec["lon"],
                        rel_height=rec["rel_height"],
                        yaw=rec["yaw"],
                        query_gsd=query_gsd,
                        tile_center_utm_x=pos_cx,
                        tile_center_utm_y=pos_cy,
                        tile_window=(
                            int(ref_ds.index(pos_cx, pos_cy)[1] - tile_w_px // 2),
                            int(ref_ds.index(pos_cx, pos_cy)[0] - tile_h_px // 2),
                            tile_w_px, tile_h_px
                        ),
                        tile_gsd=self.ref_gsd,
                        gt_query_center_in_tile=gt_center,
                        gt_homography=H.flatten().tolist(),
                        gt_offset_m=math.sqrt(dx ** 2 + dy ** 2),
                    )
                    all_pairs.append(pr)

                    # --- AUGMENTED variants of positive ---
                    aug_cfg = self.config["pair_builder"]["augmentation"]
                    n_aug = aug_cfg.get("augmented_per_positive", 2)
                    for a_i in range(n_aug):
                        params, T_aug = sample_augmentation_params(rec, self.rng, self.config)
                        try:
                            T_inv = np.linalg.inv(T_aug)
                            H_aug = H @ T_inv
                        except Exception:
                            H_aug = H

                        pair_id_aug = make_pair_id(rec["name"], "augmented", a_i)
                        pr_aug = PairRecord(
                            pair_id=pair_id_aug,
                            query_path=rec["name"],
                            query_idx=idx,
                            pair_type="augmented",
                            split=split,
                            sequence=sequence,
                            lat=rec["lat"],
                            lon=rec["lon"],
                            rel_height=params["altitude"],
                            yaw=rec["yaw"] + params["yaw_aug"],
                            query_gsd=params["new_gsd"],
                            tile_center_utm_x=pos_cx,
                            tile_center_utm_y=pos_cy,
                            tile_window=(
                                int(ref_ds.index(pos_cx, pos_cy)[1] - tile_w_px // 2),
                                int(ref_ds.index(pos_cx, pos_cy)[0] - tile_h_px // 2),
                                tile_w_px, tile_h_px
                            ),
                            tile_gsd=self.ref_gsd,
                            gt_query_center_in_tile=gt_center,
                            gt_homography=H_aug.flatten().tolist(),
                            gt_offset_m=math.sqrt(dx ** 2 + dy ** 2),
                            aug_params=params,
                        )
                        all_pairs.append(pr_aug)

                # --- HARD NEGATIVE pairs ---
                hn_cfg = self.config["pair_builder"]["hard_negative"]
                hard_neg_pairs = self._build_hard_negatives(
                    ref_ds, rec, idx, query_utm_x, query_utm_y,
                    tile_w_px, tile_h_px, target_w, target_h,
                    pos_tile, split, sequence, query_gsd, hn_cfg
                )
                all_pairs.extend(hard_neg_pairs)

                # --- EASY NEGATIVE pairs ---
                en_cfg = self.config["pair_builder"]["easy_negative"]
                easy_neg_pairs = self._build_easy_negatives(
                    ref_ds, rec, idx, query_utm_x, query_utm_y,
                    tile_w_px, tile_h_px, target_w, target_h,
                    split, sequence, query_gsd, en_cfg
                )
                all_pairs.extend(easy_neg_pairs)

                if (idx + 1) % 100 == 0:
                    print(f"  [{split}] Processed {idx + 1}/{len(records)} queries, "
                          f"{len(all_pairs)} pairs so far")
        finally:
            ref_ds.close()

        return all_pairs

    def _build_hard_negatives(self, ref_ds, rec, idx, query_utm_x, query_utm_y,
                              tile_w_px, tile_h_px, target_w, target_h,
                              pos_tile, split, sequence, query_gsd, cfg) -> List[PairRecord]:
        """Build hard negative pairs by finding visually similar nearby tiles."""
        pairs = []
        if pos_tile is None:
            return pairs

        candidates = []
        n_candidates = cfg["similarity_candidates"]

        for ci in range(n_candidates):
            angle = self.rng.uniform(0, 2 * math.pi)
            dist = self.rng.uniform(cfg["min_distance_m"], cfg["max_distance_m"])
            cx = query_utm_x + dist * math.cos(angle)
            cy = query_utm_y + dist * math.sin(angle)

            tile = extract_tile(ref_ds, cx, cy,
                                tile_w_px, tile_h_px, target_w, target_h)
            if tile is not None:
                sim = compute_tile_similarity_orb(pos_tile, tile)
                candidates.append((sim, cx, cy, tile))

        # Sort by similarity (descending) and take top-k
        candidates.sort(key=lambda x: x[0], reverse=True)
        for ki, (sim, cx, cy, tile) in enumerate(candidates[:cfg["count"]]):
            pair_id = make_pair_id(rec["name"], "hard_negative", ki)
            pr = PairRecord(
                pair_id=pair_id,
                query_path=rec["name"],
                query_idx=idx,
                pair_type="hard_negative",
                split=split,
                sequence=sequence,
                lat=rec["lat"],
                lon=rec["lon"],
                rel_height=rec["rel_height"],
                yaw=rec["yaw"],
                query_gsd=query_gsd,
                tile_center_utm_x=cx,
                tile_center_utm_y=cy,
                tile_window=(
                    int(ref_ds.index(cx, cy)[1] - tile_w_px // 2),
                    int(ref_ds.index(cx, cy)[0] - tile_h_px // 2),
                    tile_w_px, tile_h_px
                ),
                tile_gsd=self.ref_gsd,
            )
            pairs.append(pr)

        return pairs

    def _build_easy_negatives(self, ref_ds, rec, idx, query_utm_x, query_utm_y,
                              tile_w_px, tile_h_px, target_w, target_h,
                              split, sequence, query_gsd, cfg) -> List[PairRecord]:
        """Build easy negative pairs from random far-away tiles."""
        pairs = []
        for ki in range(cfg["count"]):
            for attempt in range(20):
                angle = self.rng.uniform(0, 2 * math.pi)
                dist = self.rng.uniform(cfg["min_distance_m"],
                                        cfg["min_distance_m"] * 2)
                cx = query_utm_x + dist * math.cos(angle)
                cy = query_utm_y + dist * math.sin(angle)

                tile = extract_tile(ref_ds, cx, cy,
                                    tile_w_px, tile_h_px, target_w, target_h)
                if tile is not None:
                    pair_id = make_pair_id(rec["name"], "easy_negative", ki)
                    pr = PairRecord(
                        pair_id=pair_id,
                        query_path=rec["name"],
                        query_idx=idx,
                        pair_type="easy_negative",
                        split=split,
                        sequence=sequence,
                        lat=rec["lat"],
                        lon=rec["lon"],
                        rel_height=rec["rel_height"],
                        yaw=rec["yaw"],
                        query_gsd=query_gsd,
                        tile_center_utm_x=cx,
                        tile_center_utm_y=cy,
                        tile_window=(
                            int(ref_ds.index(cx, cy)[1] - tile_w_px // 2),
                            int(ref_ds.index(cx, cy)[0] - tile_h_px // 2),
                            tile_w_px, tile_h_px
                        ),
                        tile_gsd=self.ref_gsd,
                    )
                    pairs.append(pr)
                    break

        return pairs

    def save_pairs(self, pairs: List[PairRecord], output_path: str):
        """Save pair metadata as JSON."""
        data = [p.to_dict() for p in pairs]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(pairs)} pairs to {output_path}")

    def load_pairs(self, input_path: str) -> List[PairRecord]:
        """Load pair metadata from JSON."""
        with open(input_path, "r") as f:
            data = json.load(f)
        pairs = []
        for d in data:
            # Handle optional fields
            d.setdefault("gt_query_center_in_tile", None)
            d.setdefault("gt_homography", None)
            d.setdefault("gt_offset_m", 0.0)
            d.setdefault("aug_params", None)
            if d["gt_query_center_in_tile"] is not None:
                d["gt_query_center_in_tile"] = tuple(d["gt_query_center_in_tile"])
            d["tile_window"] = tuple(d["tile_window"])
            pairs.append(PairRecord(**d))
        return pairs

    def get_tile_image(self, pair: PairRecord) -> Optional[np.ndarray]:
        """Load the tile image for a pair from ref.tif."""
        ref_ds = rasterio.open(self.ref_path)
        try:
            col_off, row_off, w, h = pair.tile_window
            target_w = int(self.config["camera"]["image_width"])
            target_h = int(self.config["camera"]["image_height"])

            if (col_off < 0 or row_off < 0 or
                    col_off + w > ref_ds.width or row_off + h > ref_ds.height):
                return None

            window = Window(col_off, row_off, w, h)
            tile = ref_ds.read([1, 2, 3], window=window)
            tile = np.transpose(tile, (1, 2, 0))

            if tile.shape[0] != target_h or tile.shape[1] != target_w:
                tile = cv2.resize(tile, (target_w, target_h),
                                   interpolation=cv2.INTER_LINEAR)
            return tile
        finally:
            ref_ds.close()

    def get_query_image(self, pair: PairRecord) -> Optional[np.ndarray]:
        """Load the query thermal image for a pair."""
        # The name field uses relative paths starting with ./Data/
        query_rel = pair.query_path
        if query_rel.startswith("./"):
            query_rel = query_rel[2:]
        query_path = os.path.join(self.project_root, query_rel)
        if not os.path.exists(query_path):
            return None
        img = cv2.imread(query_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if pair.pair_type == "augmented" and pair.aug_params is not None:
                img = apply_augmentation(img, pair.aug_params)
        return img

