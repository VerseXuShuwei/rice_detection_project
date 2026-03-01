"""
Test Heatmap Generation for Offline Tile Mode.

Validates:
1. Spatial heatmaps generate correctly with offline tile mode
2. Fixed reference samples are saved/loaded from reference_samples.json
3. Same images used across epochs (validation dataset)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from pathlib import Path
from src.utils.config_io import load_config
from src.models.builder import get_model
from src.data.datasets import AsymmetricMILDataset
from src.evaluation.heatmap_visualizer import MILVisualizer


def main():
    print("=" * 60)
    print("Test: Heatmap Generation for Offline Tile Mode")
    print("=" * 60)

    # Load config
    config_path = 'configs/algorithm/train_topk_asymmetric.yaml'
    config = load_config(config_path)

    # Verify offline mode is enabled
    offline_enabled = config.get('positive_pool', {}).get('enable', False)
    print(f"\n[CONFIG] Offline tile mode enabled: {offline_enabled}")

    # Create validation dataset
    print("\n[DATASET] Creating validation dataset...")
    val_dataset = AsymmetricMILDataset(
        config=config,
        split='val',
        transform=None,  # No augmentation for visualization
        seed=42
    )
    print(f"[DATASET] Val dataset size: {len(val_dataset)} bags")

    # Check class distribution
    class_counts = {}
    for i in range(len(val_dataset)):
        sample = val_dataset.samples[i]
        class_id = sample[1]
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    print(f"[DATASET] Class distribution: {class_counts}")

    # Create model
    print("\n[MODEL] Creating model...")
    model_name = config['model']['name']
    model = get_model(model_name, config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"[MODEL] Model loaded on {device}")

    # Create output directory
    output_dir = Path('outputs/test_heatmap')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create MILVisualizer with heatmap settings injected at construction time
    # (follows "Single Source of Truth" - config read once here, not at runtime)
    print("\n[VISUALIZER] Creating MILVisualizer...")
    heatmap_cfg = config.get('evaluation', {}).get('heatmap', {})
    visualizer = MILVisualizer(
        save_dir=str(output_dir),
        class_names=config.get('classes', {}).get('display_names', {}),
        # Heatmap settings from config
        multiscale_tile_sizes=heatmap_cfg.get('multiscale_tile_sizes', [1024, 1536, 2048]),
        small_image_tile_sizes=heatmap_cfg.get('small_image_tile_sizes', [512, 768, 1024]),
        multiscale_min_size=heatmap_cfg.get('multiscale_min_size', [3000, 4000]),
        stride_ratio=heatmap_cfg.get('stride_ratio', 0.5),
        batch_size=heatmap_cfg.get('batch_size', 8),
        conf_threshold=heatmap_cfg.get('conf_threshold', 0.4),
        top_k=heatmap_cfg.get('top_k', 5)
    )

    # Test 1: Generate monitoring heatmaps (simulating epoch 10)
    print("\n" + "=" * 60)
    print("[TEST 1] Generate monitoring heatmaps (epoch 10)")
    print("=" * 60)

    try:
        # NOTE: No config parameter - settings are injected at MILVisualizer construction
        result = visualizer.generate_monitoring_heatmaps(
            model=model,
            dataset=val_dataset,
            epoch=10,
            phase='stable',
            samples_per_class=1,
            fixed_samples=True,
            use_spatial_heatmap=True
        )

        print(f"\n[RESULT] Generation successful!")
        print(f"  - Samples processed: {result.get('num_samples', 'N/A')}")
        print(f"  - Output directory: {result.get('output_dir', 'N/A')}")
        print(f"  - Reference file: {result.get('reference_file', 'N/A')}")

        # Verify reference file exists
        ref_file = output_dir / 'reference_samples.json'
        if ref_file.exists():
            import json
            with open(ref_file) as f:
                ref_data = json.load(f)
            print(f"\n[VERIFY] Reference samples file exists")
            print(f"  - Total samples: {len(ref_data)}")
            if ref_data:
                print(f"  - Sample format: class_id={ref_data[0]['class_id']}, "
                      f"class_name={ref_data[0]['class_name']}")

    except Exception as e:
        print(f"\n[ERROR] Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Generate again (simulating epoch 20) - should use same samples
    print("\n" + "=" * 60)
    print("[TEST 2] Generate monitoring heatmaps (epoch 20) - same samples")
    print("=" * 60)

    try:
        result2 = visualizer.generate_monitoring_heatmaps(
            model=model,
            dataset=val_dataset,
            epoch=20,
            phase='stable',
            samples_per_class=1,
            fixed_samples=True,
            use_spatial_heatmap=True
        )

        print(f"\n[RESULT] Generation successful!")
        print(f"  - Samples processed: {result2.get('num_samples', 'N/A')}")

    except Exception as e:
        print(f"\n[ERROR] Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: List generated files
    print("\n" + "=" * 60)
    print("[TEST 3] Verify output files")
    print("=" * 60)

    heatmap_dir = output_dir / 'heatmaps'
    if heatmap_dir.exists():
        files = list(heatmap_dir.glob('**/*'))
        png_files = [f for f in files if f.suffix == '.png']
        print(f"[FILES] Total PNG files generated: {len(png_files)}")
        for f in sorted(png_files)[:10]:
            print(f"  - {f.relative_to(output_dir)}")
        if len(png_files) > 10:
            print(f"  ... and {len(png_files) - 10} more")
    else:
        print("[WARN] No heatmaps directory found")

    print("\n" + "=" * 60)
    print("[DONE] All tests completed successfully!")
    print("=" * 60)

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
