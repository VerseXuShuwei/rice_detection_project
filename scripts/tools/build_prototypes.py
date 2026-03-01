"""
Build Background Prototypes from Negative Pool

Usage:
    python scripts/tools/build_prototypes.py \\
        --config configs/algorithm/train_topk_asymmetric.yaml \\
        --output outputs/prototypes/background_prototypes.pth \\
        --auto-k  # Automatically determine optimal cluster count

        ** simple Command:**
        python scripts/tools/build_prototypes.py \
            --config configs/algorithm/train_topk_asymmetric.yaml \
            --batch-size 128


Features:
    - Uses pre-trained EfficientNetV2 (not task-trained)
    - Samples ALL tiles from Negative Pool (no filtering)
    - Automatic cluster selection via elbow + silhouette
    - Saves prototypes + diagnostic info
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn.functional as F
import timm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config_io import load_config
from src.data.negative_pool import NegativeTilePool
from src.utils.device import get_device


def extract_features_from_negative_pool(neg_pool, backbone, device, batch_size=128):
    """
    Extract features from all tiles in Negative Pool.

    Args:
        neg_pool: NegativeTilePool instance
        backbone: Pre-trained EfficientNetV2 backbone (timm model)
        device: torch.device
        batch_size: Batch size for feature extraction

    Returns:
        features: (N, D) numpy array - features from N tiles
        tile_indices: (N,) list - corresponding tile indices in LMDB
    """
    backbone.eval()
    all_features = []
    all_indices = []

    # Get total number of tiles
    total_tiles = len(neg_pool)
    print(f"[EXTRACT] Extracting features from {total_tiles} negative tiles...")

    with torch.no_grad():
        # Sample all tiles (hard_ratio=0 means random sampling)
        # We process in batches to avoid memory issues
        num_batches = (total_tiles + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Feature extraction"):
            # Sample a batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_tiles)
            batch_size_actual = end_idx - start_idx

            batch_tiles_list, batch_indices = neg_pool.sample(
                n=batch_size_actual,
                hard_ratio=0.0  # Random sampling
            )

            # Convert List[np.ndarray] to Tensor
            # Negative pool returns (H, W, C) uint8, need to convert to (C, H, W) float
            batch_tiles = torch.stack([
                torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0  # (H,W,C) → (C,H,W)
                for tile in batch_tiles_list
            ]).to(device)  # (B, C, H, W)

            # Extract features (before classifier)
            # For EfficientNetV2: forward_features() → (B, C, H, W)
            feature_maps = backbone.forward_features(batch_tiles)  # (B, D, h, w)

            # Global Max Pooling (same as model training)
            features = F.adaptive_max_pool2d(feature_maps, 1)  # (B, D, 1, 1)
            features = features.squeeze(-1).squeeze(-1)  # (B, D)

            # L2 Normalize to unit sphere (CRITICAL for cosine-based clustering)
            # This ensures K-Means works on direction, not magnitude
            features = F.normalize(features, p=2, dim=1)

            all_features.append(features.cpu().numpy())
            all_indices.extend(batch_indices)  # batch_indices is already a list

    all_features = np.concatenate(all_features, axis=0)
    print(f"[EXTRACT] Extracted features: {all_features.shape}")

    return all_features, all_indices


def determine_optimal_clusters(features, k_range=(3, 15)):
    """
    Automatically determine optimal number of clusters using Elbow + Silhouette.

    Method:
        1. For each k in k_range:
           - Run K-Means
           - Compute inertia (within-cluster sum of squares)
           - Compute silhouette score
        2. Find "elbow" in inertia curve (maximum curvature point)
        3. Find k with maximum silhouette score
        4. Return k that balances both metrics

    Args:
        features: (N, D) numpy array of features
        k_range: (min_k, max_k) tuple for search range

    Returns:
        optimal_k: int - Best cluster count
        metrics: Dict with detailed metrics for all k values
            - 'inertias': List of inertia values
            - 'silhouettes': List of silhouette scores
            - 'elbow_k': k selected by elbow method
            - 'silhouette_k': k with max silhouette score
    """
    min_k, max_k = k_range
    k_values = range(min_k, max_k + 1)

    inertias = []
    silhouettes = []

    print(f"[AUTO-K] Searching optimal k in range {k_range}...")

    for k in k_values:
        print(f"[AUTO-K] Testing k={k}...")

        # Run K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(features)

        # Compute metrics
        inertia = kmeans.inertia_
        # Subsample for silhouette (expensive for large datasets)
        if len(features) > 10000:
            indices = np.random.choice(len(features), 10000, replace=False)
            silhouette = silhouette_score(features[indices], labels[indices])
        else:
            silhouette = silhouette_score(features, labels)

        inertias.append(inertia)
        silhouettes.append(silhouette)

        print(f"    Inertia: {inertia:.2f}, Silhouette: {silhouette:.4f}")

    # Convert to numpy for easier manipulation
    inertias = np.array(inertias)
    silhouettes = np.array(silhouettes)

    # Method 1: Elbow detection (maximum curvature in inertia curve)
    # Use second derivative approximation
    if len(k_values) >= 3:
        # Normalize inertia to [0, 1] for comparison
        inertia_norm = (inertias - inertias.min()) / (inertias.max() - inertias.min() + 1e-8)

        # Compute curvature using second finite difference
        curvature = np.zeros(len(k_values))
        for i in range(1, len(k_values) - 1):
            curvature[i] = inertia_norm[i-1] - 2*inertia_norm[i] + inertia_norm[i+1]

        elbow_idx = np.argmax(curvature)
        elbow_k = k_values[elbow_idx]
    else:
        elbow_k = k_values[0]

    # Method 2: Maximum silhouette score
    silhouette_idx = np.argmax(silhouettes)
    silhouette_k = k_values[silhouette_idx]

    print(f"\n[AUTO-K] Elbow method suggests: k={elbow_k}")
    print(f"[AUTO-K] Silhouette method suggests: k={silhouette_k}")

    # Decision: Prefer silhouette score (more stable), but check if elbow is close
    if abs(elbow_k - silhouette_k) <= 1:
        # If elbow and silhouette agree (within 1), use silhouette
        optimal_k = silhouette_k
        reason = "Elbow and Silhouette agree"
    else:
        # If they disagree, use silhouette if score is good (>0.5)
        if silhouettes[silhouette_idx] > 0.5:
            optimal_k = silhouette_k
            reason = "High silhouette score (>0.5)"
        else:
            # Otherwise, use elbow (more conservative)
            optimal_k = elbow_k
            reason = "Low silhouette scores, using elbow"

    print(f"[AUTO-K] Selected k={optimal_k} ({reason})")

    metrics = {
        'inertias': inertias.tolist(),
        'silhouettes': silhouettes.tolist(),
        'k_values': list(k_values),
        'elbow_k': int(elbow_k),
        'silhouette_k': int(silhouette_k),
        'optimal_k': int(optimal_k),
        'selection_reason': reason
    }

    return optimal_k, metrics


def build_prototypes(features, n_clusters, random_state=42):
    """
    Build prototypes using K-Means clustering.

    Args:
        features: (N, D) numpy array
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility

    Returns:
        prototypes: (n_clusters, D) torch.Tensor
        cluster_labels: (N,) numpy array - cluster assignment for each tile
        cluster_info: Dict with diagnostic info
    """
    print(f"[KMEANS] Running K-Means with {n_clusters} clusters...")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,  # Multiple initializations for stability
        max_iter=300,
        verbose=1
    )

    cluster_labels = kmeans.fit_predict(features)

    # Convert to torch tensor and L2 normalize
    # CRITICAL: Prototypes must be unit vectors for cosine similarity in FeatureCritic
    prototypes = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    prototypes = F.normalize(prototypes, p=2, dim=1)

    # Compute diagnostics
    cluster_sizes = np.bincount(cluster_labels)
    silhouette = silhouette_score(features, cluster_labels, sample_size=5000)

    cluster_info = {
        'n_clusters': n_clusters,
        'cluster_sizes': cluster_sizes.tolist(),
        'silhouette_score': float(silhouette),
        'inertia': float(kmeans.inertia_)
    }

    print(f"[KMEANS] Completed. Silhouette score: {silhouette:.4f}")
    print(f"[KMEANS] Cluster distribution:")
    for i, size in enumerate(cluster_sizes):
        pct = size / len(cluster_labels) * 100
        print(f"    Cluster {i}: {size:>6d} tiles ({pct:>5.1f}%)")

    return prototypes, cluster_labels, cluster_info


def compute_recommended_threshold(
    features: np.ndarray,
    prototypes: torch.Tensor,
    percentile: float = 5.0
) -> Tuple[float, Dict]:
    """
    Compute recommended threshold based on negative sample similarity distribution.

    Method:
        1. Compute max similarity between each negative tile and all prototypes
        2. Take the specified percentile as threshold
        3. This ensures (100-percentile)% of background tiles are suppressed

    Args:
        features: (N, D) numpy array - all negative tile features
        prototypes: (K, D) torch tensor - K-Means cluster centers
        percentile: Percentile for threshold selection (default: 5th = cover 95% background)

    Returns:
        threshold: Recommended threshold value
        stats: Dict with distribution statistics
    """
    print(f"\n[THRESHOLD] Computing recommended threshold (P{percentile})...")

    # Convert to torch for consistency
    features_t = torch.from_numpy(features).float()
    prototypes_t = prototypes.float()

    # Normalize for cosine similarity
    features_norm = F.normalize(features_t, p=2, dim=1)
    prototypes_norm = F.normalize(prototypes_t, p=2, dim=1)

    # Compute similarity matrix: (N, K)
    similarity_matrix = torch.mm(features_norm, prototypes_norm.T)

    # Max similarity to any prototype for each tile
    max_similarities = similarity_matrix.max(dim=1)[0].numpy()

    # Compute statistics
    stats = {
        'min': float(np.min(max_similarities)),
        'max': float(np.max(max_similarities)),
        'mean': float(np.mean(max_similarities)),
        'std': float(np.std(max_similarities)),
        'p5': float(np.percentile(max_similarities, 5)),
        'p10': float(np.percentile(max_similarities, 10)),
        'p25': float(np.percentile(max_similarities, 25)),
        'p50': float(np.percentile(max_similarities, 50)),
        'p75': float(np.percentile(max_similarities, 75)),
        'p90': float(np.percentile(max_similarities, 90)),
        'p95': float(np.percentile(max_similarities, 95)),
    }

    # Recommended threshold: cover (100 - percentile)% of background
    threshold = float(np.percentile(max_similarities, percentile))

    print(f"[THRESHOLD] Similarity Distribution:")
    print(f"    Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
    print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"    Percentiles: P5={stats['p5']:.4f}, P25={stats['p25']:.4f}, "
          f"P50={stats['p50']:.4f}, P75={stats['p75']:.4f}, P95={stats['p95']:.4f}")
    print(f"\n[THRESHOLD] Recommended threshold (P{percentile}): {threshold:.4f}")
    print(f"[THRESHOLD] This will suppress {100-percentile:.0f}% of background tiles")

    return threshold, stats


def main():
    parser = argparse.ArgumentParser(description="Build background prototypes from Negative Pool")
    parser.add_argument('--config', type=str, required=True, help="Path to training config")
    parser.add_argument('--batch-size', type=int, default=128, help="Batch size for feature extraction")
    parser.add_argument('--skip-threshold', action='store_true',
                        help="Skip threshold calculation (for debugging)")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    device = get_device()  # Auto-detect device

    # Read feature_critic config (config-driven parameters)
    fc_cfg = config.get('feature_critic', {})
    construction_cfg = fc_cfg.get('construction', {})

    # Extract parameters from config
    num_prototypes = construction_cfg.get('num_prototypes', 'auto')  # "auto" or int
    auto_k_range = construction_cfg.get('auto_k_range', [4, 10])
    output_path = construction_cfg.get('save_path', 'outputs/prototypes/background_prototypes.pth')
    vis_output_path = construction_cfg.get('vis_output_path', 'outputs/prototypes/cluster_vis.html')

    print(f"[CONFIG] num_prototypes: {num_prototypes}")
    print(f"[CONFIG] auto_k_range: {auto_k_range}")
    print(f"[CONFIG] save_path: {output_path}")
    print(f"[CONFIG] vis_output_path: {vis_output_path}")

    # Initialize Negative Pool
    print("[INIT] Loading Negative Pool...")
    neg_pool = NegativeTilePool(config, split='train')

    if not neg_pool.exists():
        raise RuntimeError(
            "Negative Pool LMDB not found. Please run:\n"
            "  python scripts/tools/build_negative_pool.py --config " + args.config
        )

    print(f"[INIT] Negative Pool: {len(neg_pool)} tiles")

    # Load pre-trained backbone
    print("[INIT] Loading pre-trained EfficientNetV2-S...")
    backbone = timm.create_model(
        'efficientnetv2_rw_s',
        pretrained=True,
        num_classes=0,  # Remove classifier
        global_pool=''  # We'll use custom pooling
    )
    backbone = backbone.to(device)
    backbone.eval()

    # Extract features
    features, tile_indices = extract_features_from_negative_pool(
        neg_pool, backbone, device, args.batch_size
    )

    # Determine optimal k (config-driven)
    if num_prototypes == 'auto':
        optimal_k, cluster_metrics = determine_optimal_clusters(features, k_range=tuple(auto_k_range))
        n_clusters = optimal_k
    elif isinstance(num_prototypes, int):
        n_clusters = num_prototypes
        cluster_metrics = None
    else:
        raise ValueError(f"Invalid num_prototypes: {num_prototypes}. Expected 'auto' or int.")

    # Build prototypes
    prototypes, cluster_labels, cluster_info = build_prototypes(features, n_clusters)

    # === Threshold Recommendation ===
    recommended_threshold = None
    threshold_stats = None

    if not args.skip_threshold:
        threshold_percentile = construction_cfg.get('threshold_percentile', 5.0)
        recommended_threshold, threshold_stats = compute_recommended_threshold(
            features, prototypes, percentile=threshold_percentile
        )

    # Save results (use config path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        'prototypes': prototypes,
        'n_clusters': n_clusters,
        'cluster_labels': cluster_labels,
        'tile_indices': tile_indices,
        'cluster_info': cluster_info,
        'cluster_metrics': cluster_metrics,
        # NEW: Auto-threshold support
        'recommended_threshold': recommended_threshold,
        'threshold_stats': threshold_stats,
        'config_snapshot': {
            'negative_pool': config.get('negative_pool', {}),
            'model': config.get('model', {})
        }
    }

    torch.save(save_data, output_path)
    print(f"\n[SAVE] Prototypes saved to: {output_path}")

    # Save JSON summary for easy inspection
    import json
    summary_path = output_path.with_suffix('.json')
    summary = {
        'n_clusters': n_clusters,
        'cluster_info': cluster_info,
        'cluster_metrics': cluster_metrics,
        'recommended_threshold': recommended_threshold,
        'threshold_stats': threshold_stats
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] Summary saved to: {summary_path}")

    # === Phase 0.5: Auto-Visualization ===
    if vis_output_path:
        print(f"\n[PHASE 0.5] Generating cluster visualization...")
        try:
            from src.utils.visualization import visualize_clusters, generate_cluster_report

            # Generate interactive visualization
            vis_stats = visualize_clusters(
                prototypes=prototypes,
                features=features,
                labels=cluster_labels,
                output_path=vis_output_path
            )

            # Generate text report
            report_path = Path(vis_output_path).with_suffix('.md')
            generate_cluster_report(
                cluster_labels=cluster_labels,
                cluster_info=cluster_info,
                output_path=str(report_path)
            )

            print(f"[PHASE 0.5] Visualization completed.")
            print(f"    - Interactive plot: {vis_output_path}")
            print(f"    - Text report: {report_path}")

        except ImportError as e:
            print(f"[PHASE 0.5] WARNING: Visualization skipped. Missing dependencies: {e}")
            print(f"[PHASE 0.5] Install with: pip install scikit-learn plotly")
        except Exception as e:
            print(f"[PHASE 0.5] WARNING: Visualization failed: {e}")

    print("\n" + "="*60)
    print("[SUCCESS] Prototype construction completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()