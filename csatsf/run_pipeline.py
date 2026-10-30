"""
C-SATSF Main Pipeline Runner.

Usage:
    python -m csatsf.run_pipeline --step all         # Run everything
    python -m csatsf.run_pipeline --step pairs        # Step 1: Build pairs
    python -m csatsf.run_pipeline --step match        # Step 2: Run matcher
    python -m csatsf.run_pipeline --step filter       # Step 3-4: Filter + gate
    python -m csatsf.run_pipeline --step evaluate     # Step 5: Evaluate
    python -m csatsf.run_pipeline --step tune         # Step 6: Tune
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import yaml
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    return str(Path(__file__).resolve().parent.parent)


def load_config(project_root: str) -> dict:
    """Load configuration from config.yaml."""
    config_path = os.path.join(project_root, "csatsf", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─── Step 1: Build Pairs ─────────────────────────────────────────────────

def run_pairs(config: dict, project_root: str, max_queries: int = None):
    """Build all pairs for train/valid/test splits."""
    from csatsf.data.pair_builder import PairBuilder

    builder = PairBuilder(config, project_root)

    for split in ["train", "valid", "test"]:
        print(f"\n{'='*60}")
        print(f"Building pairs for split: {split}")
        print(f"{'='*60}")

        split_max = max_queries
        if max_queries is not None and split != "train":
            split_max = max(5, max_queries // 2)

        pairs = builder.build_all_pairs(split=split, max_queries=split_max)

        output_path = os.path.join(project_root, "csatsf", "data",
                                    f"pairs_{split}.json")
        builder.save_pairs(pairs, output_path)

        # Print summary
        types = {}
        for p in pairs:
            types[p.pair_type] = types.get(p.pair_type, 0) + 1
        print(f"  Total pairs: {len(pairs)}")
        for pt, count in sorted(types.items()):
            print(f"    {pt}: {count}")


# ─── Step 2: Run Matcher ─────────────────────────────────────────────────

def run_match(config: dict, project_root: str):
    """Run MINIMA-RoMa matching on all pairs."""
    from csatsf.data.pair_builder import PairBuilder
    from csatsf.matcher.minima_roma import MINIMARoMaMatcher

    builder = PairBuilder(config, project_root)
    matcher = MINIMARoMaMatcher(config, project_root)

    for split in ["train", "valid", "test"]:
        pairs_path = os.path.join(project_root, "csatsf", "data",
                                   f"pairs_{split}.json")
        if not os.path.exists(pairs_path):
            print(f"Skipping {split}: no pairs file found")
            continue

        print(f"\n{'='*60}")
        print(f"Matching pairs for split: {split}")
        print(f"{'='*60}")

        pairs = builder.load_pairs(pairs_path)
        n_processed = matcher.batch_match(pairs, builder)
        print(f"  Processed {n_processed} new pairs")


# ─── Steps 3-4: Filter + Gate ────────────────────────────────────────────

def run_filter(config: dict, project_root: str):
    """Run filter pipeline and decision gate on all cached matches."""
    from csatsf.data.pair_builder import PairBuilder
    from csatsf.matcher.minima_roma import MINIMARoMaMatcher
    from csatsf.filters.pipeline import filter_matches
    from csatsf.eval.decision_gate import (
        hand_threshold_gate, features_to_vector, LearnedGate
    )
    from csatsf.eval.metrics import label_matches_with_gt

    builder = PairBuilder(config, project_root)
    matcher = MINIMARoMaMatcher(config, project_root)

    all_results = {}

    for split in ["train", "valid", "test"]:
        pairs_path = os.path.join(project_root, "csatsf", "data",
                                   f"pairs_{split}.json")
        if not os.path.exists(pairs_path):
            continue

        print(f"\n{'='*60}")
        print(f"Filtering pairs for split: {split}")
        print(f"{'='*60}")

        pairs = builder.load_pairs(pairs_path)
        split_results = []

        for i, pair in enumerate(pairs):
            cached = matcher.load_cached(pair.pair_id)
            if cached is None:
                continue

            img_q = builder.get_query_image(pair)
            img_t = builder.get_tile_image(pair)
            if img_q is None or img_t is None:
                continue

            # Run filter pipeline
            inlier_idx, features = filter_matches(
                cached["kpts_q"], cached["kpts_t"], cached["conf"],
                img_q, img_t, config
            )

            # Hand-set gate
            accept_hand, reason = hand_threshold_gate(features, config)

            # GT labels (for positives)
            gt_labels = None
            if pair.pair_type == "positive" and pair.gt_homography is not None:
                H_gt = np.array(pair.gt_homography).reshape(3, 3)
                gt_labels = label_matches_with_gt(
                    cached["kpts_q"], cached["kpts_t"], H_gt, 5.0
                )

            result = {
                "pair_id": pair.pair_id,
                "pair_type": pair.pair_type,
                "split": split,
                "n_raw": len(cached["kpts_q"]),
                "n_filtered": len(inlier_idx),
                "features": features,
                "accept_hand": accept_hand,
                "reject_reason": reason,
                "feature_vector": features_to_vector(features).tolist(),
            }

            if gt_labels is not None:
                n_gt_inliers_raw = int(gt_labels.sum())
                n_gt_inliers_filtered = int(gt_labels[inlier_idx].sum()) if len(inlier_idx) > 0 else 0
                result["gt_precision_raw"] = float(n_gt_inliers_raw / len(gt_labels)) if len(gt_labels) > 0 else 0.0
                result["gt_precision_filtered"] = float(n_gt_inliers_filtered / len(inlier_idx)) if len(inlier_idx) > 0 else 0.0
                result["gt_recall_filtered"] = float(n_gt_inliers_filtered / n_gt_inliers_raw) if n_gt_inliers_raw > 0 else 0.0

            split_results.append(result)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(pairs)} pairs")

        all_results[split] = split_results
        print(f"  Total filtered: {len(split_results)} pairs")

    # Save results
    results_path = os.path.join(project_root, "csatsf", "reports",
                                 "filter_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Convert numpy values for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\nSaved filter results to {results_path}")

    # Train learned gate on the train split
    if "train" in all_results:
        print("\nTraining learned gate...")
        train_gate(all_results["train"], config, project_root)


def train_gate(train_results: list, config: dict, project_root: str):
    """Train the learned accept/reject gate on the training split."""
    from csatsf.eval.decision_gate import LearnedGate, features_to_vector

    feature_vecs = []
    labels = []

    for r in train_results:
        fv = np.array(r["feature_vector"])
        feature_vecs.append(fv)
        # Label: 1 if positive pair, 0 if negative
        labels.append(1 if r["pair_type"] in ["positive", "augmented"] else 0)

    if len(feature_vecs) == 0:
        print("  No training data for gate")
        return

    X = np.array(feature_vecs)
    y = np.array(labels)

    gate = LearnedGate()
    gate.train(X, y)

    gate_path = os.path.join(project_root, "csatsf", "reports", "learned_gate.pkl")
    gate.save(gate_path)
    print(f"  Saved learned gate to {gate_path}")

    # Print feature importances
    importances = gate.get_feature_importances()
    print("  Feature importances (decision tree):")
    for k, v in sorted(importances.items(), key=lambda x: -x[1]):
        if v > 0.01:
            print(f"    {k}: {v:.3f}")


# ─── Step 5: Evaluate ────────────────────────────────────────────────────

def run_evaluate(config: dict, project_root: str):
    """Run full evaluation on valid/test splits."""
    from csatsf.eval.decision_gate import LearnedGate, features_to_vector
    from csatsf.eval.metrics import (
        decision_level_metrics, false_accept_rate, competition_score
    )

    results_path = os.path.join(project_root, "csatsf", "reports",
                                 "filter_results.json")
    if not os.path.exists(results_path):
        print("No filter results found. Run --step filter first.")
        return

    with open(results_path, "r") as f:
        all_results = json.load(f)

    # Load learned gate
    gate_path = os.path.join(project_root, "csatsf", "reports", "learned_gate.pkl")
    gate = None
    if os.path.exists(gate_path):
        gate = LearnedGate()
        gate.load(gate_path)

    for split in ["valid", "test"]:
        if split not in all_results:
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating split: {split}")
        print(f"{'='*60}")

        results = all_results[split]

        # Decision-level metrics
        pair_types = [r["pair_type"] for r in results]
        true_labels = np.array([1 if r["pair_type"] in ["positive", "augmented"] else 0
                                for r in results])
        hand_decisions = np.array([r["accept_hand"] for r in results])

        # Hand gate metrics
        hand_far = false_accept_rate(hand_decisions, true_labels.astype(bool), pair_types)
        print("\n  Hand-set Gate:")
        print(f"    Overall FAR: {hand_far['overall_far']:.4f}")
        print(f"    Hard-neg FAR: {hand_far.get('far_hard_negative', 0):.4f}")
        print(f"    Easy-neg FAR: {hand_far.get('far_easy_negative', 0):.4f}")
        print(f"    True accept rate: {hand_far['true_accept_rate']:.4f}")

        # Learned gate metrics
        if gate is not None:
            lr_probs = []
            lr_decisions = []
            for r in results:
                fv = np.array(r["feature_vector"])
                accept, prob = gate.predict_lr(fv)
                lr_probs.append(prob)
                lr_decisions.append(accept)

            lr_scores = np.array(lr_probs)
            lr_decs = np.array(lr_decisions)

            dl_metrics = decision_level_metrics(lr_scores, true_labels)
            lr_far = false_accept_rate(lr_decs, true_labels.astype(bool), pair_types)

            print("\n  Learned Gate (Logistic Regression):")
            print(f"    ROC AUC: {dl_metrics['roc_auc']:.4f}")
            print(f"    PR AUC: {dl_metrics['pr_auc']:.4f}")
            print(f"    Overall FAR: {lr_far['overall_far']:.4f}")
            print(f"    Hard-neg FAR: {lr_far.get('far_hard_negative', 0):.4f}")
            print(f"    Easy-neg FAR: {lr_far.get('far_easy_negative', 0):.4f}")
            print(f"    True accept rate: {lr_far['true_accept_rate']:.4f}")

        # Competition score proxy
        print("\n  Competition Scoring Proxy:")
        for gate_name, decisions in [("hand", hand_decisions)]:
            scores_list = []
            for r, dec in zip(results, decisions):
                if r["pair_type"] in ["positive", "augmented"]:
                    # Use median reproj residual as error proxy (in pixels)
                    # Convert to metres using ref GSD
                    error_px = r["features"].get("median_reproj_residual", float("inf"))
                    error_m = error_px * config["reference"]["gsd"]
                else:
                    error_m = float("inf")  # negatives: if accepted, large error

                gsd = r["features"].get("ransac_scale", 1.0) * config["reference"]["gsd"]
                s = competition_score(error_m, gsd, bool(dec), config)
                scores_list.append(s)

            avg_score = np.mean(scores_list)
            print(f"    {gate_name} gate: avg score = {avg_score:.2f}")

        # Match-level metrics (positives only)
        pos_results = [r for r in results if r["pair_type"] == "positive"]
        if pos_results:
            prec_raw = np.mean([r.get("gt_precision_raw", 0) for r in pos_results if "gt_precision_raw" in r])
            prec_filt = np.mean([r.get("gt_precision_filtered", 0) for r in pos_results if "gt_precision_filtered" in r])
            rec_filt = np.mean([r.get("gt_recall_filtered", 0) for r in pos_results if "gt_recall_filtered" in r])
            print(f"\n  Match-Level Metrics (positives):")
            print(f"    Raw inlier rate: {prec_raw:.4f}")
            print(f"    Filtered precision: {prec_filt:.4f}")
            print(f"    Filtered recall: {rec_filt:.4f}")


# ─── Step 6: Tune ────────────────────────────────────────────────────────

def run_tune(config: dict, project_root: str):
    """Tune filter thresholds using Optuna on the training split."""
    try:
        import optuna
    except ImportError:
        print("Optuna not installed. Install with: pip install optuna")
        print("Falling back to grid search...")
        run_tune_grid(config, project_root)
        return

    from csatsf.data.pair_builder import PairBuilder
    from csatsf.matcher.minima_roma import MINIMARoMaMatcher
    from csatsf.filters.pipeline import filter_matches
    from csatsf.eval.decision_gate import hand_threshold_gate, features_to_vector
    from csatsf.eval.metrics import (
        label_matches_with_gt, false_accept_rate, competition_score
    )

    builder = PairBuilder(config, project_root)
    matcher = MINIMARoMaMatcher(config, project_root)

    # Load training pairs and their cached matches
    pairs_path = os.path.join(project_root, "csatsf", "data", "pairs_train.json")
    if not os.path.exists(pairs_path):
        print("No training pairs found. Run --step pairs first.")
        return

    pairs = builder.load_pairs(pairs_path)

    # Pre-load all data
    print("Loading training data...")
    pair_data = []
    for pair in pairs:
        cached = matcher.load_cached(pair.pair_id)
        if cached is None:
            continue
        img_q = builder.get_query_image(pair)
        img_t = builder.get_tile_image(pair)
        if img_q is None or img_t is None:
            continue

        gt_labels = None
        if pair.pair_type == "positive" and pair.gt_homography is not None:
            H_gt = np.array(pair.gt_homography).reshape(3, 3)
            gt_labels = label_matches_with_gt(
                cached["kpts_q"], cached["kpts_t"], H_gt, 5.0
            )

        pair_data.append({
            "pair": pair,
            "cached": cached,
            "img_q": img_q,
            "img_t": img_t,
            "gt_labels": gt_labels,
        })

    print(f"Loaded {len(pair_data)} pairs for tuning")

    def objective(trial):
        # Sample filter parameters
        import copy
        cfg = copy.deepcopy(config)

        cfg["filters"]["spatial_equalization"]["grid_size"] = trial.suggest_int("grid_size", 4, 16)
        cfg["filters"]["spatial_equalization"]["q_base"] = trial.suggest_int("q_base", 1, 8)
        cfg["filters"]["texture_saliency"]["gamma"] = trial.suggest_float("gamma", 0.1, 1.5)
        cfg["filters"]["delaunay_topology"]["eps_topo"] = trial.suggest_float("eps_topo", 0.1, 1.0)
        cfg["filters"]["delaunay_topology"]["vote_threshold"] = trial.suggest_float("vote_thresh", 0.2, 0.8)
        cfg["filters"]["global_consistency"]["eps_ang"] = trial.suggest_float("eps_ang", 5.0, 45.0)
        cfg["filters"]["global_consistency"]["eps_scale"] = trial.suggest_float("eps_scale", 0.1, 0.8)

        # Gate thresholds
        cfg["decision_gate"]["min_inliers"] = trial.suggest_int("min_inliers", 3, 20)
        cfg["decision_gate"]["min_ransac_inlier_ratio"] = trial.suggest_float("min_ransac_ratio", 0.1, 0.8)

        scores = []
        hard_neg_accepts = 0
        hard_neg_total = 0

        for pd in pair_data:
            pair = pd["pair"]
            cached = pd["cached"]

            inlier_idx, features = filter_matches(
                cached["kpts_q"], cached["kpts_t"], cached["conf"],
                pd["img_q"], pd["img_t"], cfg
            )

            accept, _ = hand_threshold_gate(features, cfg)

            if pair.pair_type == "hard_negative":
                hard_neg_total += 1
                if accept:
                    hard_neg_accepts += 1

            if pair.pair_type in ["positive", "augmented"]:
                error_px = features.get("median_reproj_residual", float("inf"))
                error_m = error_px * config["reference"]["gsd"]
                gsd = pair.query_gsd
                s = competition_score(error_m, gsd, accept, cfg)
                scores.append(s)
            else:
                s = competition_score(float("inf"), 0.3, accept, cfg)
                scores.append(s)

        # Hard constraint: FAR on hard negatives
        if hard_neg_total > 0:
            far = hard_neg_accepts / hard_neg_total
            if far > config["tuning"]["hard_neg_far_cap"]:
                return -100.0  # penalty

        return np.mean(scores) if scores else -100.0

    study = optuna.create_study(direction="maximize")
    n_trials = config["tuning"]["n_trials"]
    print(f"\nRunning Optuna optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest trial:")
    print(f"  Value: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    # Update config with best params
    best = study.best_params
    config["filters"]["spatial_equalization"]["grid_size"] = best["grid_size"]
    config["filters"]["spatial_equalization"]["q_base"] = best["q_base"]
    config["filters"]["texture_saliency"]["gamma"] = best["gamma"]
    config["filters"]["delaunay_topology"]["eps_topo"] = best["eps_topo"]
    config["filters"]["delaunay_topology"]["vote_threshold"] = best["vote_thresh"]
    config["filters"]["global_consistency"]["eps_ang"] = best["eps_ang"]
    config["filters"]["global_consistency"]["eps_scale"] = best["eps_scale"]
    config["decision_gate"]["min_inliers"] = best["min_inliers"]
    config["decision_gate"]["min_ransac_inlier_ratio"] = best["min_ransac_ratio"]

    # Save updated config
    config_path = os.path.join(project_root, "csatsf", "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Saved best config to {config_path}")


def run_tune_grid(config: dict, project_root: str):
    """Simple grid search fallback for tuning."""
    print("Grid search tuning not yet implemented. Install optuna:")
    print("  pip install optuna")


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="C-SATSF Pipeline Runner")
    parser.add_argument("--step", type=str, required=True,
                        choices=["all", "pairs", "match", "filter",
                                 "evaluate", "tune"],
                        help="Pipeline step to run")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Limit queries per split (for debugging)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml (defaults to csatsf/config.yaml)")

    args = parser.parse_args()

    project_root = get_project_root()
    config = load_config(project_root)

    np.random.seed(config["seed"])

    print(f"Project root: {project_root}")
    print(f"Step: {args.step}")

    t0 = time.time()

    if args.step in ["all", "pairs"]:
        run_pairs(config, project_root, max_queries=args.max_queries)

    if args.step in ["all", "match"]:
        run_match(config, project_root)

    if args.step in ["all", "filter"]:
        run_filter(config, project_root)

    if args.step in ["all", "evaluate"]:
        run_evaluate(config, project_root)

    if args.step in ["all", "tune"]:
        run_tune(config, project_root)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

