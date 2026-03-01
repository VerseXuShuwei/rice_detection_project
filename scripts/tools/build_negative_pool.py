"""
Offline Preprocessing: Negative Tile Pool Builder (Multi-Scale v2.0).

This CLI script handles the creation and maintenance of the LMDB database for negative
(healthy) tiles. It acts as the bridge between raw image folders and the efficient
`NegativeTilePool` used during training.

Recent Updates:
    - [2026-01-27] Aligned with positive pool scales
        - Scales: [512, 1024, 1536, 2048] (same as positive)
        - Overlaps: 25% for each scale
        - Split file persistence (data/negative_pool_split.json)
    - [2026-01-13] Support multi-scale tile generation (v2.0)
        - Generates tiles at multiple scales
        - Progress display per scale
        - Automatic v1.0 → v2.0 migration

Execution:
    Run this script once before training. It will verify the existing pool or
    build a new one if it's missing or if --force-rebuild is specified.

Key Features:
    1. Multi-Scale Building (v2.0):
       - Generates tiles at all configured scales
       - Per-scale progress tracking
       - Proportional storage across scales

    2. Robustness:
       - Atomic write operations (via LMDB transactions)
       - "Sanity Check" validation step after build
       - Backward compatibility (auto-detects v1.0)

    3. Memory Management:
       - Batch processing (defined in config)
       - Memory-mapped LMDB access

Dependencies (YAML Config):
    - dataset.root: Base data directory
    - dataset.negative_class_name: Source folder name
    - negative_pool.scales: Tile scales (e.g., [768, 1024, 1536, 2048])
    - negative_pool.overlaps: Corresponding overlaps
    - negative_pool.final_tile_size: Target resolution for stored tiles
    - negative_pool.lmdb_path: Output database location

Usage:
    # 1. Normal execution (Builds only if missing)
    python scripts/tools/build_negative_pool.py --config configs/algorithm/train_topk_asymmetric.yaml

    # 2. Force a fresh rebuild (Deletes old v1.0/v2.0 DB)
    python scripts/tools/build_negative_pool.py --config configs/algorithm/train_topk_asymmetric.yaml --force-rebuild
"""


import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.negative_pool import NegativeTilePool
from src.utils.config_io import load_config


def main():
    parser = argparse.ArgumentParser(
        description='Build negative tile pool for asymmetric MIL training'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file (train_topk_asymmetric.yaml)'
    )
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Force rebuild even if pool exists'
    )
    args = parser.parse_args()

    # Load configuration
    print(f"[CONFIG] Loading from: {args.config}")
    config = load_config(args.config)

    # Force rebuild if specified (MUST set before creating NegativeTilePool)
    if args.force_rebuild:
        config['negative_pool']['rebuild'] = True
        print("[CONFIG] Force rebuild enabled")

    # Create negative pool (reads rebuild flag from config)
    pool = NegativeTilePool(config)

    # Check if pool exists and should skip building
    # Logic: Skip only if (exists AND not rebuild)
    if pool.exists() and not pool.rebuild:
        print(f"\n[INFO] Negative pool already exists: {len(pool)} tiles")
        print(f"[INFO] Path: {pool.lmdb_path}")
        print(f"[INFO] Use --force-rebuild to rebuild")
        return

    # Build pool
    print("\n" + "="*60)
    print("Building Multi-Scale Negative Tile Pool (v2.0)")
    print("="*60)

    negative_dir = Path(config['dataset']['root']) / config['dataset']['negative_class_name']

    if not negative_dir.exists():
        raise FileNotFoundError(
            f"Negative class directory not found: {negative_dir}\n"
            f"Please check dataset.root and dataset.negative_class_name in config"
        )

    print(f"[SOURCE] Directory: {negative_dir}")
    print(f"[SCALES] {pool.scales}")
    print(f"[OVERLAPS] {pool.overlaps} (25% overlap)")
    print(f"[FINAL SIZE] {pool.final_tile_size}")
    print(f"[OUTPUT] LMDB path: {pool.lmdb_path}")
    print(f"[OUTPUT] Split file: {pool.split_file}")
    print()

    # Build (v2.0 multi-scale)
    pool.build_from_directory(str(negative_dir))

    # Reload metadata and create split indices after building
    pool._load_metadata()
    pool._create_split_indices()

    print("\n" + "="*60)
    print("Build Complete!")
    print("="*60)
    print(f"[RESULT] Total tiles: {pool.num_tiles}")
    print(f"[RESULT] Scale distribution: {pool.scale_tile_counts}")
    print(f"[RESULT] Source images: {pool.meta.get('source_images', 'N/A')}")
    print(f"[RESULT] Final tile size: {pool.final_tile_size}")
    print(f"[RESULT] Database version: v{pool.meta.get('version', '2.0.0')}")
    print(f"[RESULT] LMDB path: {pool.lmdb_path}")
    print(f"[RESULT] Split file: {pool.split_file}")
    split_size = len(pool.train_indices) if pool.split == 'train' else len(pool.val_indices)
    print(f"[RESULT] Split ({pool.split}): {split_size} tiles ({pool.train_ratio*100:.0f}% train)")
    print()

    # Test sampling
    print("[TEST] Testing random sampling...")
    try:
        test_tiles, _ = pool.sample(5)
        print(f"[TEST] OK Successfully sampled {len(test_tiles)} tiles")
        print(f"[TEST] OK Tile shape: {test_tiles[0].shape}")
    except Exception as e:
        print(f"[TEST] FAILED Sampling failed: {e}")

    print("\n[SUCCESS] Negative pool ready for training!")


if __name__ == '__main__':
    main()
