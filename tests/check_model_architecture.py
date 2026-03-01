"""
Model Architecture Verification Script.

Tests all architecture configurations:
1. Original (all enable=false)
2. ViT only
3. FPN only
4. HeatmapHead only
5. Full hybrid (ViT + FPN + HeatmapHead)

Also reports VRAM usage for each configuration.

Usage:
    python scripts/tools/check_model_architecture.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import gc
from copy import deepcopy

from src.utils.config_io import load_config
from src.models.builder import get_model


def get_vram_mb():
    """Get current VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def get_max_vram_mb():
    """Get peak VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def reset_vram():
    """Reset VRAM tracking."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def test_configuration(config, name, device):
    """Test a specific model configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    reset_vram()
    vram_before = get_vram_mb()

    try:
        # Build model
        model = get_model(config['model']['name'], config)
        model = model.to(device)

        vram_after_model = get_vram_mb()
        print(f"[VRAM] Model loaded: {vram_after_model:.1f} MB")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[PARAMS] Total: {total_params/1e6:.2f}M | Trainable: {trainable_params/1e6:.2f}M")

        # Test forward pass (batch=1, K=4 tiles)
        model.eval()
        with torch.no_grad():
            # Simulate bag input
            x = torch.randn(1, 4, 3, 384, 384).to(device)

            # Forward pass
            output = model(x)
            print(f"[SHAPE] Input: {x.shape} -> Output: {output.shape}")

            # Test predict_instances
            instance_output = model.predict_instances(x)
            print(f"[SHAPE] predict_instances: {instance_output.shape}")

            # Test extract_features (for Feature Critic)
            features = model.extract_features(x)
            print(f"[SHAPE] extract_features: {features.shape}")

            # Test spatial heatmap (if enabled)
            if hasattr(model, 'heatmap_head') and model.heatmap_head is not None:
                tiles = torch.randn(4, 3, 384, 384).to(device)
                heatmap = model.get_spatial_heatmap(tiles)
                print(f"[SHAPE] get_spatial_heatmap: {heatmap.shape}")

        vram_after_forward = get_max_vram_mb()
        print(f"[VRAM] Peak (eval): {vram_after_forward:.1f} MB")

        # Test training forward+backward
        model.train()
        x = torch.randn(1, 4, 3, 384, 384).to(device)
        output = model(x)
        loss = output.sum()
        loss.backward()

        vram_after_backward = get_max_vram_mb()
        print(f"[VRAM] Peak (train): {vram_after_backward:.1f} MB")

        print(f"[OK] {name} passed all tests")

        # Cleanup
        del model, x, output
        reset_vram()

        return True, vram_after_backward

    except Exception as e:
        print(f"[ERROR] {name} failed: {e}")
        import traceback
        traceback.print_exc()
        reset_vram()
        return False, 0


def main():
    print("="*60)
    print("Model Architecture Verification")
    print("="*60)

    # Load base config
    config_path = 'configs/algorithm/train_topk_asymmetric.yaml'
    base_config = load_config(config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Define test configurations
    configs = [
        ("Original (baseline)", {}),
        ("ViT only", {"vit_block": {"enable": True}}),
        ("FPN only", {"fpn_neck": {"enable": True}}),
        ("HeatmapHead only", {"heatmap_head": {"enable": True}}),
        ("ViT + FPN", {"vit_block": {"enable": True}, "fpn_neck": {"enable": True}}),
        ("Full Hybrid", {
            "vit_block": {"enable": True},
            "fpn_neck": {"enable": True},
            "heatmap_head": {"enable": True}
        }),
    ]

    results = []

    for name, overrides in configs:
        config = deepcopy(base_config)

        # Apply overrides
        if 'model' not in config:
            config['model'] = {}
        for key, value in overrides.items():
            if key not in config['model']:
                config['model'][key] = {}
            config['model'][key].update(value)

        success, vram = test_configuration(config, name, device)
        results.append((name, success, vram))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Configuration':<25} {'Status':<10} {'Peak VRAM':<15}")
    print("-"*50)
    for name, success, vram in results:
        status = "PASS" if success else "FAIL"
        vram_str = f"{vram:.1f} MB" if vram > 0 else "N/A"
        print(f"{name:<25} {status:<10} {vram_str:<15}")

    all_passed = all(r[1] for r in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed!"))

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
