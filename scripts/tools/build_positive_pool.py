#!/usr/bin/env python
"""
Build Positive Tile Pool.

Offline preprocessing script for positive samples. Generates multi-scale tiles
with coordinate metadata for Spatial NMS support.

Recent Updates:
    - [2026-01-29] Feature: Add offline feature extraction for Feature Critic
        - Uses frozen pretrained EfficientNetV2-S (same as build_prototypes.py)
        - Features stored alongside tiles in LMDB
        - Controlled by config: positive_pool.store_features
    - [2026-01-27] NEW: Initial implementation

Usage:
    # Build with default config (includes feature extraction if enabled)
    python scripts/tools/build_positive_pool.py --config configs/algorithm/train_topk_asymmetric.yaml

    # Force rebuild existing pool
    python scripts/tools/build_positive_pool.py --config configs/algorithm/train_topk_asymmetric.yaml --force-rebuild

    # Custom output path
    python scripts/tools/build_positive_pool.py --config configs/algorithm/train_topk_asymmetric.yaml --output data/my_pool.lmdb

Output:
    - LMDB database at configured path (default: data/positive_pool.lmdb)
    - Train/Val split JSON (default: data/positive_pool_split.json)
    - Statistics summary
    - (Optional) Pre-computed feature vectors for Feature Critic
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import torch
import timm

from src.utils.config_io import load_config
from src.utils.device import get_device
from src.data.positive_pool import PositiveTilePool


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build positive tile pool for offline multi-scale preprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard build
    python scripts/tools/build_positive_pool.py --config configs/algorithm/train_topk_asymmetric.yaml

    # Force rebuild
    python scripts/tools/build_positive_pool.py --config configs/algorithm/train_topk_asymmetric.yaml --force-rebuild
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to config file (YAML)'
    )
    parser.add_argument(
        '--force-rebuild', '-f',
        action='store_true',
        help='Force rebuild even if pool exists'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Override output LMDB path'
    )
    parser.add_argument(
        '--split-file',
        type=str,
        default=None,
        help='Override split JSON file path'
    )
    parser.add_argument(
        '--root-dir',
        type=str,
        default=None,
        help='Override dataset root directory'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Positive Tile Pool Builder")
    print("=" * 60)

    # Load config
    print(f"\n[CONFIG] Loading: {args.config}")
    config = load_config(args.config)

    # Override config if arguments provided
    if args.output:
        config.setdefault('positive_pool', {})['lmdb_path'] = args.output
        print(f"[CONFIG] Override lmdb_path: {args.output}")

    if args.split_file:
        config.setdefault('positive_pool', {})['split_file'] = args.split_file
        print(f"[CONFIG] Override split_file: {args.split_file}")

    if args.force_rebuild:
        config.setdefault('positive_pool', {})['rebuild'] = True
        print("[CONFIG] Force rebuild enabled")

    # Print configuration
    pool_cfg = config.get('positive_pool', {})
    dataset_cfg = config.get('dataset', {})
    classes_cfg = config.get('classes', {})

    print("\n[CONFIG] Positive Pool Settings:")
    print(f"  - LMDB path: {pool_cfg.get('lmdb_path', 'data/positive_pool.lmdb')}")
    print(f"  - Split file: {pool_cfg.get('split_file', 'data/positive_pool_split.json')}")
    print(f"  - Scales: {pool_cfg.get('scales', [512, 1024, 1536, 2048])}")
    print(f"  - Overlaps: {pool_cfg.get('overlaps', [128, 256, 384, 512])}")
    print(f"  - Final tile size: {pool_cfg.get('final_tile_size', 384)}")
    print(f"  - JPEG quality: {pool_cfg.get('jpeg_quality', 95)}")
    print(f"  - Filter invalid: {pool_cfg.get('filter_invalid', True)}")
    print(f"  - Valid pixel threshold: {pool_cfg.get('valid_pixel_threshold', 0.3)}")
    print(f"  - Min short edge: {pool_cfg.get('min_short_edge', 512)}")
    print(f"  - Store features: {pool_cfg.get('store_features', False)}")
    print(f"  - Feature dim: {pool_cfg.get('feature_dim', 1280)}")

    print(f"\n[CONFIG] Dataset Settings:")
    print(f"  - Root: {dataset_cfg.get('root', 'data/total_rice_image')}")
    print(f"  - Train ratio: {dataset_cfg.get('train_ratio', 0.8)}")

    # Get class order
    class_order = classes_cfg.get('class_order', [])
    if not class_order:
        print("\n[ERROR] No class_order found in config!")
        print("[ERROR] Please ensure config has 'classes.class_order' defined")
        sys.exit(1)

    print(f"\n[CONFIG] Classes ({len(class_order)}):")
    for i, cls in enumerate(class_order, 1):
        print(f"  {i:2d}. {cls}")

    # Initialize pool
    pool = PositiveTilePool(config)

    # Check if exists
    if pool.exists() and not args.force_rebuild:
        print(f"\n[INFO] Pool already exists at {pool.lmdb_path}")
        print("[INFO] Use --force-rebuild to rebuild")

        # Print existing stats
        if pool.meta:
            stats = pool.meta.get('stats', {})
            print(f"\n[EXISTING] Total tiles: {stats.get('total_tiles', 'N/A')}")
            print(f"[EXISTING] Train tiles: {stats.get('train_tiles', 'N/A')}")
            print(f"[EXISTING] Val tiles: {stats.get('val_tiles', 'N/A')}")
            print(f"[EXISTING] Filtered tiles: {stats.get('filtered_tiles', 'N/A')}")

        return

    # Build pool
    print("\n" + "=" * 60)
    print("Starting Build...")
    print("=" * 60)

    start_time = time.time()

    root_dir = args.root_dir or dataset_cfg.get('root', 'data/total_rice_image')

    # Load backbone for feature extraction (if store_features enabled)
    # CRITICAL: Use same model as build_prototypes.py to ensure feature space consistency
    backbone = None
    device = None
    store_features = pool_cfg.get('store_features', False)

    if store_features:
        print("\n[BACKBONE] Loading pretrained EfficientNetV2-S for feature extraction...")
        device = get_device()

        # IMPORTANT: Use 'efficientnetv2_rw_s' - same as build_prototypes.py
        # This ensures features are in the same space as background prototypes
        backbone = timm.create_model(
            'efficientnetv2_rw_s',
            pretrained=True,
            num_classes=0,      # Remove classifier head
            global_pool=''      # We use custom GMP pooling
        )
        backbone = backbone.to(device)
        backbone.eval()

        # Freeze all parameters (no gradient computation needed)
        for param in backbone.parameters():
            param.requires_grad = False

        print(f"[BACKBONE] Loaded on {device}")
        print(f"[BACKBONE] Feature dim: {pool_cfg.get('feature_dim', 1280)}")

    def progress_callback(processed, total):
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total - processed) / rate if rate > 0 else 0
        print(f"  Progress: {processed}/{total} ({100*processed/total:.1f}%) "
              f"| Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    pool.build_from_directory(
        root_dir=root_dir,
        class_order=class_order,
        progress_callback=progress_callback,
        backbone=backbone,
        device=device
    )

    elapsed_time = time.time() - start_time

    # Print summary
    print("\n" + "=" * 60)
    print("Build Summary")
    print("=" * 60)

    stats = pool.meta.get('stats', {})
    print(f"\n[RESULT] Total tiles: {stats.get('total_tiles', 0):,}")
    print(f"[RESULT] Train tiles: {stats.get('train_tiles', 0):,}")
    print(f"[RESULT] Val tiles: {stats.get('val_tiles', 0):,}")
    print(f"[RESULT] Filtered (garbage): {stats.get('filtered_tiles', 0):,}")
    if store_features:
        print(f"[RESULT] Features stored: {stats.get('features_stored', 0):,}")

    # Estimate storage
    lmdb_path = Path(pool.lmdb_path)
    if lmdb_path.exists():
        total_size = sum(f.stat().st_size for f in lmdb_path.rglob('*'))
        print(f"\n[STORAGE] LMDB size: {total_size / (1024**3):.2f} GB")
        avg_tile_size = total_size / stats.get('total_tiles', 1)
        print(f"[STORAGE] Avg tile size: {avg_tile_size / 1024:.1f} KB")

    print(f"\n[TIME] Total elapsed: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
    print(f"[TIME] Rate: {stats.get('total_tiles', 0) / elapsed_time:.1f} tiles/sec")

    print("\n[OUTPUT FILES]")
    print(f"  - LMDB: {pool.lmdb_path}")
    print(f"  - Split JSON: {pool_cfg.get('split_file', 'data/positive_pool_split.json')}")

    print("\n" + "=" * 60)
    print("Build Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
