"""
分析 Feature Critic 的问题：正样本 vs 负样本的相似度分布

运行：
    python scripts/tools/analyze_feature_critic.py \
        --config configs/algorithm/train_topk_asymmetric.yaml
"""

import sys
from pathlib import Path
# tests/ 目录在项目根目录下，需要上一级
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.utils.config_io import load_config
from src.data.positive_pool import PositiveTilePool
from src.utils.device import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=5000,
                        help="Number of positive samples to analyze")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    # Load prototypes
    fc_cfg = config.get('feature_critic', {})
    prototype_path = fc_cfg.get('construction', {}).get(
        'save_path', 'outputs/prototypes/background_prototypes.pth')

    print(f"[LOAD] Loading prototypes from {prototype_path}...")
    proto_data = torch.load(prototype_path, map_location='cpu', weights_only=False)
    prototypes = proto_data['prototypes']  # (K, D)
    threshold_stats = proto_data.get('threshold_stats', {})

    print(f"[INFO] Prototypes shape: {prototypes.shape}")
    print(f"[INFO] Recommended threshold: {proto_data.get('recommended_threshold', 'N/A')}")

    # 打印负样本的相似度分布（从prototype文件中）
    print("\n" + "="*60)
    print("负样本与原型的相似度分布 (from build_prototypes.py)")
    print("="*60)
    if threshold_stats:
        print(f"  Min:  {threshold_stats.get('min', 'N/A'):.4f}")
        print(f"  P5:   {threshold_stats.get('p5', 'N/A'):.4f}")
        print(f"  P25:  {threshold_stats.get('p25', 'N/A'):.4f}")
        print(f"  P50:  {threshold_stats.get('p50', 'N/A'):.4f}")
        print(f"  Mean: {threshold_stats.get('mean', 'N/A'):.4f} ± {threshold_stats.get('std', 'N/A'):.4f}")
        print(f"  P75:  {threshold_stats.get('p75', 'N/A'):.4f}")
        print(f"  P95:  {threshold_stats.get('p95', 'N/A'):.4f}")
        print(f"  Max:  {threshold_stats.get('max', 'N/A'):.4f}")
    else:
        print("  [WARNING] No threshold_stats in prototype file")

    # Load positive pool
    print(f"\n[LOAD] Loading positive pool...")
    pos_pool = PositiveTilePool(config, split='train')

    if not pos_pool.meta.get('store_features', False):
        print("[ERROR] Positive pool has no precomputed features!")
        print("        Run: python scripts/tools/build_positive_pool.py --force-rebuild")
        return

    # Sample positive features
    print(f"[SAMPLE] Sampling {args.num_samples} positive tiles...")
    total_tiles = len(pos_pool)
    sample_size = min(args.num_samples, total_tiles)

    # 随机选择 tile_ids
    all_tile_ids = list(pos_pool.tile_infos.keys())
    # 过滤当前 split 的 tiles
    split_tile_ids = [tid for tid in all_tile_ids if pos_pool.tile_infos[tid].split == pos_pool.split]
    sample_tile_ids = np.random.choice(split_tile_ids, sample_size, replace=False).tolist()

    # 批量加载
    print(f"[LOAD] Loading {sample_size} tiles with features...")
    tiles, features_array = pos_pool.load_tiles_batch(sample_tile_ids, load_features=True)

    if features_array is None:
        print("[ERROR] No features found in positive pool!")
        print("        Run: python scripts/tools/build_positive_pool.py --force-rebuild")
        return

    pos_features = features_array  # (N, D)

    print(f"[INFO] Loaded {len(pos_features)} positive features, shape: {pos_features.shape}")

    # Compute similarity
    print("\n[COMPUTE] Computing cosine similarity with prototypes...")
    pos_features_t = torch.from_numpy(pos_features).float()
    prototypes_t = prototypes.float()

    # Both should already be L2 normalized, but let's make sure
    pos_features_norm = F.normalize(pos_features_t, p=2, dim=1)
    prototypes_norm = F.normalize(prototypes_t, p=2, dim=1)

    # Similarity matrix: (N, K)
    sim_matrix = torch.mm(pos_features_norm, prototypes_norm.T)

    # Max similarity to any prototype
    max_sim = sim_matrix.max(dim=1)[0].numpy()

    # 打印正样本的相似度分布
    print("\n" + "="*60)
    print("正样本与原型的相似度分布")
    print("="*60)
    print(f"  Min:  {np.min(max_sim):.4f}")
    print(f"  P5:   {np.percentile(max_sim, 5):.4f}")
    print(f"  P25:  {np.percentile(max_sim, 25):.4f}")
    print(f"  P50:  {np.percentile(max_sim, 50):.4f}")
    print(f"  Mean: {np.mean(max_sim):.4f} ± {np.std(max_sim):.4f}")
    print(f"  P75:  {np.percentile(max_sim, 75):.4f}")
    print(f"  P95:  {np.percentile(max_sim, 95):.4f}")
    print(f"  Max:  {np.max(max_sim):.4f}")

    # 对比分析
    print("\n" + "="*60)
    print("诊断结论")
    print("="*60)

    neg_mean = threshold_stats.get('mean', 0)
    pos_mean = np.mean(max_sim)

    print(f"\n  负样本平均相似度: {neg_mean:.4f}")
    print(f"  正样本平均相似度: {pos_mean:.4f}")
    print(f"  差异: {pos_mean - neg_mean:+.4f}")

    if pos_mean > neg_mean:
        print("\n  [WARNING] 正样本比负样本更像背景原型！")
        print("  这说明在预训练特征空间中，病害区域和背景无法区分。")
        print("  Feature Critic 的设计假设不成立。")
        print("\n  建议:")
        print("    1. 禁用 Feature Critic (feature_critic.enable: false)")
        print("    2. 或改用正样本原型（排斥不像病害的 tiles）")
    else:
        diff = neg_mean - pos_mean
        print(f"\n  [OK] 正样本与原型的相似度比负样本低 {diff:.4f}")
        print("  Feature Critic 的假设成立。")

    # 计算合理的阈值
    print("\n" + "="*60)
    print("阈值建议")
    print("="*60)

    # 找一个阈值使得 <5% 的正样本被抑制
    for target_suppress in [5, 10, 20]:
        threshold = np.percentile(max_sim, 100 - target_suppress)
        print(f"  阈值 {threshold:.4f}: 抑制 {target_suppress}% 正样本")

    # === 关键分析：低相似度 tiles 的分布 ===
    print("\n" + "="*60)
    print("低相似度 tiles 分析（可能是真正的病害 tiles）")
    print("="*60)

    # 找出相似度最低的 tiles（最不像背景的）
    sorted_indices = np.argsort(max_sim)
    low_sim_count = int(len(max_sim) * 0.05)  # 最低 5%

    low_sim_values = max_sim[sorted_indices[:low_sim_count]]
    print(f"  最低 5% tiles ({low_sim_count} 个):")
    print(f"    相似度范围: [{low_sim_values.min():.4f}, {low_sim_values.max():.4f}]")
    print(f"    平均相似度: {low_sim_values.mean():.4f}")

    # 与负样本的 P5 比较
    neg_p5 = threshold_stats.get('p5', 0)
    print(f"\n  负样本 P5 (最低 5%): {neg_p5:.4f}")
    print(f"  正样本最低 5% 平均: {low_sim_values.mean():.4f}")

    if low_sim_values.mean() < neg_p5:
        print("\n  [GOOD] 正样本中最低相似度的 tiles 比负样本的更'不像背景'")
        print("         这些可能是真正的病害 tiles！")
        print("         Feature Critic 的问题是阈值设置，而非设计本身。")

        # 建议阈值：让这些低相似度 tiles 不被抑制
        suggested_threshold = low_sim_values.max() + 0.02
        print(f"\n  建议阈值: {suggested_threshold:.4f}")
        print(f"  （保护最低 5% 的正样本 tiles 不被抑制）")
    else:
        print("\n  [WARNING] 即使正样本中最不像背景的 tiles，也比负样本更像背景")
        print("            Feature Critic 在这个特征空间可能无效。")

    # === JPEG 压缩影响分析 ===
    print("\n" + "="*60)
    print("JPEG 压缩影响分析")
    print("="*60)

    # 实际测试 JPEG vs 无损的特征差异
    test_jpeg_compression_effect(pos_pool, prototypes, device)


def test_jpeg_compression_effect(pos_pool, prototypes, device):
    """测试 JPEG 压缩对特征的影响"""
    import cv2

    print("\n  [测试] 比较 JPEG vs 无损特征...")

    # 加载 backbone
    import timm
    backbone = timm.create_model('efficientnetv2_rw_s', pretrained=True, num_classes=0, global_pool='')
    backbone = backbone.to(device)
    backbone.eval()

    # 随机选几个 tile 测试
    split_tile_ids = [tid for tid in pos_pool.tile_infos.keys()
                      if pos_pool.tile_infos[tid].split == pos_pool.split]
    test_tile_ids = np.random.choice(split_tile_ids, min(100, len(split_tile_ids)), replace=False).tolist()

    jpeg_features = []
    png_features = []

    for tile_id in tqdm(test_tile_ids, desc="Testing compression"):
        # 加载原始 tile
        tiles, _ = pos_pool.load_tiles_batch([tile_id], load_features=False)
        if tiles is None or len(tiles) == 0:
            continue
        tile_np = tiles[0]  # (H, W, C) uint8

        # 测试：重新用不同质量 JPEG 压缩后的差异

        # 高质量 JPEG (quality=95)
        _, jpeg_high_buf = cv2.imencode('.jpg', cv2.cvtColor(tile_np, cv2.COLOR_RGB2BGR),
                                         [cv2.IMWRITE_JPEG_QUALITY, 95])
        tile_high = cv2.imdecode(jpeg_high_buf, cv2.IMREAD_COLOR)
        tile_high = cv2.cvtColor(tile_high, cv2.COLOR_BGR2RGB)

        # 低质量 JPEG (quality=50)
        _, jpeg_low_buf = cv2.imencode('.jpg', cv2.cvtColor(tile_np, cv2.COLOR_RGB2BGR),
                                        [cv2.IMWRITE_JPEG_QUALITY, 50])
        tile_low = cv2.imdecode(jpeg_low_buf, cv2.IMREAD_COLOR)
        tile_low = cv2.cvtColor(tile_low, cv2.COLOR_BGR2RGB)

        # 提取特征
        with torch.no_grad():
            def extract_feat(tile):
                t = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
                t = t.unsqueeze(0).to(device)
                fm = backbone.forward_features(t)
                f = F.adaptive_max_pool2d(fm, 1).squeeze(-1).squeeze(-1)
                f = F.normalize(f, p=2, dim=1)
                return f.cpu().numpy()

            feat_high = extract_feat(tile_high)
            feat_low = extract_feat(tile_low)

        jpeg_features.append(feat_high)
        png_features.append(feat_low)

    if len(jpeg_features) == 0:
        print("  [ERROR] No tiles loaded for compression test")
        return

    jpeg_features = np.concatenate(jpeg_features, axis=0)
    png_features = np.concatenate(png_features, axis=0)

    # 计算 JPEG high vs low 的余弦相似度
    cos_sim = np.sum(jpeg_features * png_features, axis=1)

    print(f"\n  JPEG Q95 vs Q50 余弦相似度:")
    print(f"    Min:  {cos_sim.min():.6f}")
    print(f"    Mean: {cos_sim.mean():.6f}")
    print(f"    Max:  {cos_sim.max():.6f}")

    if cos_sim.mean() > 0.99:
        print("\n  [结论] JPEG 压缩质量对特征影响很小 (<1%)")
        print("         压缩不是 Feature Critic 问题的原因")
    else:
        print(f"\n  [结论] JPEG 压缩导致 {(1-cos_sim.mean())*100:.2f}% 的特征差异")
        print("         这可能是问题的一部分")


if __name__ == '__main__':
    main()
