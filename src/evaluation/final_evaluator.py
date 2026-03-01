"""
Post-Training Evaluation Module.
训练后完整评估模块,负责计算指标、生成热力图、输出报告。

Recent Updates:
    - [2026-02-27] Refactor: Full-dataset evaluation (no sampling); dual confusion matrix (filtered + unfiltered)
    - [2026-02-09] Refactor: Use UnifiedInferenceEngine for heatmap generation
    - [2026-01-06] Refactor: Decoupled visualization logic to heatmap_visualizer.py
    - [2025-12-26] Logic: Confusion Matrix Hit-Acc Filtering (抗噪逻辑)

Key Responsibilities:
    - Metric Computation: Negative Recall, Top-K Lift, Hit-Acc, Confusion Matrix
    - Heatmap Generation: Delegated to UnifiedInferenceEngine (single source of truth)
    - Report Export: JSON metrics + visualizations (via MILVisualizer)

Usage:
    >>> from src.evaluation import FinalEvaluator
    >>> evaluator = FinalEvaluator(model, val_dataset, neg_pool, config, device, logger)
    >>> metrics = evaluator.evaluate_all()
    >>> evaluator.save_evaluation_report(metrics)

Configuration (config['evaluation']):
    - confusion_matrix.use_hit_acc_filter: 是否开启抗噪过滤 (default: True)
    - confusion_matrix.min_confidence: 过滤阈值 (default: 0.3)
    - heatmap.use_adaptive_tiles: 是否启用自适应多尺度 (default: True)
    - heatmap.batch_size: 热力图推理时的 Batch Size
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json

# Import unified visualizer
from src.evaluation.heatmap_visualizer import MILVisualizer, CLASS_NAMES, CLASS_NAMES_SHORT


class FinalEvaluator:
    """
    Post-Training Complete Evaluator.

    Computes all evaluation metrics and generates visualizations after training.
    Delegates all plotting to MILVisualizer for clean separation of concerns.

    Args:
        model: 训练完成的模型
        val_dataset: 验证数据集
        val_negative_pool: 验证负样本池
        config: 配置字典
        device: 设备
        logger: LocalLogger实例
    """

    def __init__(
        self,
        model,
        val_dataset,
        val_negative_pool,
        config: Dict,
        device: torch.device,
        logger
    ):
        self.model = model
        self.val_dataset = val_dataset
        self.val_negative_pool = val_negative_pool
        self.config = config
        self.device = device
        self.logger = logger
        self.num_classes = val_dataset.num_classes

        # Confusion Matrix configuration
        cm_cfg = config.get('evaluation', {}).get('confusion_matrix', {})
        self.cm_use_hit_acc_filter = cm_cfg.get('use_hit_acc_filter', True)
        self.cm_min_confidence = cm_cfg.get('min_confidence', 0.3)

        # Heatmap configuration
        heatmap_cfg = config.get('evaluation', {}).get('heatmap', {})
        self.heatmap_use_adaptive = heatmap_cfg.get('use_adaptive_tiles', True)
        self.heatmap_multiscale_tiles = heatmap_cfg.get('multiscale_tile_sizes', [1024, 1536, 2048])
        # Small image: multi-scale (matches training dataset.tile_config.small_image_scales)
        self.heatmap_small_image_tiles = heatmap_cfg.get('small_image_tile_sizes', [512, 768, 1024])
        self.heatmap_multiscale_min_size = heatmap_cfg.get('multiscale_min_size', [3000, 4000])
        self.heatmap_stride_ratio = heatmap_cfg.get('stride_ratio', 0.5)
        self.heatmap_batch_size = heatmap_cfg.get('batch_size', 8)
        self.heatmap_conf_threshold = heatmap_cfg.get('conf_threshold', 0.4)
        self.heatmap_top_k = heatmap_cfg.get('top_k', 5)
        self.heatmap_num_samples = heatmap_cfg.get('num_samples_per_class', 3)

        # Model input size (for tile resizing during inference)
        self.model_input_size = config.get('model', {}).get('img_size', 384)

        # Create unified inference engine (single source of truth for heatmap generation)
        from src.inference.engine import UnifiedInferenceEngine
        self._inference_engine = UnifiedInferenceEngine(
            model=model,
            device=device,
            model_input_size=self.model_input_size,
            tile_sizes=self.heatmap_multiscale_tiles,
            stride_ratio=self.heatmap_stride_ratio,
            batch_size=self.heatmap_batch_size,
            conf_threshold=self.heatmap_conf_threshold,
            adaptive_scale=self.heatmap_use_adaptive,
            large_image_tiles=self.heatmap_multiscale_tiles,
            small_image_tiles=self.heatmap_small_image_tiles,
            large_image_min_size=tuple(self.heatmap_multiscale_min_size),
        )

        # Initialize visualizer
        vis_save_dir = self.logger.get_evaluation_dir() / "visualizations"
        self.visualizer = MILVisualizer(save_dir=str(vis_save_dir))

        # Validation transform (no augmentation, only resize + normalize)
        from src.data.transforms import get_val_transforms_from_config
        self.val_transform = get_val_transforms_from_config(config)

        print(f"[FINAL-EVAL] Initialized for {self.num_classes} disease classes + Class 0")
        print(f"[FINAL-EVAL] Confusion Matrix: Hit-Acc filter={'ON' if self.cm_use_hit_acc_filter else 'OFF'} (min_conf={self.cm_min_confidence})")
        print(f"[FINAL-EVAL] Heatmap config: adaptive={'ON' if self.heatmap_use_adaptive else 'OFF'}, "
              f"large_img={self.heatmap_multiscale_tiles}, small_img={self.heatmap_small_image_tiles}, model_input={self.model_input_size}")

    def evaluate_all(self) -> Dict:
        """
        运行所有评估指标.

        Returns:
            完整评估指标字典
        """
        print("\n" + "=" * 60)
        print("[PHASE-4] Running Final Evaluation")
        print("=" * 60)

        metrics = {}

        # 1. 负样本完整评估
        print("\n[1/5] Evaluating Negative Pool (Background Suppression)...")
        neg_metrics = self.evaluate_negative_pool()
        metrics.update(neg_metrics)

        # 2. Top-K质量评估（全量）
        print("\n[2/5] Evaluating Top-K Selection Quality (full dataset)...")
        topk_metrics = self.evaluate_topk_quality()
        metrics.update(topk_metrics)

        # 3. 混淆矩阵（全量，双版本）
        print("\n[3/5] Computing Confusion Matrix (full dataset, filtered + unfiltered)...")
        cm_metrics = self.compute_confusion_matrix(
            use_hit_acc_filter=self.cm_use_hit_acc_filter,
            min_confidence=self.cm_min_confidence
        )
        metrics.update(cm_metrics)

        # 4. Per-class指标（filtered版本为主，unfiltered为参考）
        print("\n[4/5] Computing Per-class Precision/Recall...")
        class_metrics = self.compute_per_class_metrics(cm_metrics['confusion_matrix'])
        metrics.update(class_metrics)
        # Unfiltered per-class metrics (reference)
        class_metrics_unfiltered = self.compute_per_class_metrics(cm_metrics['confusion_matrix_unfiltered'])
        metrics['per_class_precision_unfiltered'] = class_metrics_unfiltered['per_class_precision']
        metrics['per_class_recall_unfiltered'] = class_metrics_unfiltered['per_class_recall']
        metrics['per_class_f1_unfiltered'] = class_metrics_unfiltered['per_class_f1']

        # 5. 整体准确率
        print("\n[5/5] Computing Overall Metrics...")
        overall_metrics = self.compute_overall_metrics(cm_metrics['confusion_matrix'])
        metrics.update(overall_metrics)
        overall_unfiltered = self.compute_overall_metrics(cm_metrics['confusion_matrix_unfiltered'])
        metrics['overall_accuracy_unfiltered'] = overall_unfiltered['overall_accuracy']

        print("\n[PHASE-4] Final Evaluation Completed!")
        return metrics

    def evaluate_negative_pool(self) -> Dict:
        """
        评估全部验证负样本池（背景抑制能力）.

        Returns:
            {
                'negative_recall': float,
                'neg_disease_hallucination': float,
                'negative_total': int,
                'negative_correct': int
            }
        """
        self.model.eval()

        num_neg_total = len(self.val_negative_pool)
        print(f"  Evaluating {num_neg_total} negative tiles...")

        batch_size = 128
        neg_correct = 0
        neg_hallucinations = []

        # sample() returns (tiles, indices) tuple
        all_neg_tiles, _ = self.val_negative_pool.sample(num_neg_total)

        with torch.no_grad():
            for i in tqdm(range(0, num_neg_total, batch_size), desc="  Negative Pool Evaluation"):
                batch_tiles_np = all_neg_tiles[i:i + batch_size]

                # Flatten tiles to handle variable dimensions
                flat_tiles = []
                for tile_idx, t in enumerate(batch_tiles_np):
                    arr = np.asarray(t)

                    if arr.ndim == 4:  # (K, H, W, C)
                        for ti in arr:
                            flat_tiles.append(np.asarray(ti))
                    elif arr.ndim == 3:  # (H, W, C)
                        flat_tiles.append(arr)
                    elif arr.ndim == 1:  # Corrupted tile
                        print(f"[WARNING] Corrupted tile at batch {i}, index {tile_idx}: shape {arr.shape}")
                        total_pixels = arr.shape[0]
                        if total_pixels % 3 == 0:
                            num_pixels = total_pixels // 3
                            H = int(np.sqrt(num_pixels))
                            if H * H == num_pixels:
                                try:
                                    arr_reshaped = arr.reshape(H, H, 3)
                                    print(f"[WARNING] Successfully reshaped to {arr_reshaped.shape}")
                                    flat_tiles.append(arr_reshaped)
                                except:
                                    print(f"[WARNING] Failed to reshape, skipping this tile")
                                    continue
                            else:
                                print(f"[WARNING] Cannot determine image dimensions, skipping")
                                continue
                        else:
                            print(f"[WARNING] Array size not divisible by 3, skipping")
                            continue
                    else:
                        print(f"[ERROR] Unexpected tile shape: {arr.shape}, skipping...")
                        continue

                if len(flat_tiles) == 0:
                    print(f"[WARNING] Batch {i//batch_size} has no valid tiles, skipping...")
                    continue

                # Convert to tensor with proper normalization (CRITICAL FIX 2026-02-05)
                batch_tiles_list = []
                for ti in flat_tiles:
                    transformed = self.val_transform(image=np.ascontiguousarray(ti))
                    if isinstance(transformed['image'], torch.Tensor):
                        batch_tiles_list.append(transformed['image'])
                    else:
                        batch_tiles_list.append(
                            torch.from_numpy(transformed['image']).permute(2, 0, 1).float() / 255.0
                        )
                batch_tiles = torch.stack(batch_tiles_list).to(self.device)

                # Inference
                outputs = self.model.predict_instances(batch_tiles)
                probs = F.softmax(outputs, dim=1)

                # Statistics
                preds = outputs.argmax(dim=1)
                neg_correct += (preds == 0).sum().item()

                # Negative confidence = max(P(Class 1-N))
                disease_probs = probs[:, 1:]
                max_disease_conf = disease_probs.max(dim=1)[0]
                neg_hallucinations.extend(max_disease_conf.cpu().tolist())

        neg_recall = neg_correct / num_neg_total if num_neg_total > 0 else 0.0
        neg_disease_hallucination = np.mean(neg_hallucinations) if neg_hallucinations else 0.0

        print(f"  Negative Recall: {neg_recall:.4f} ({neg_correct}/{num_neg_total})")
        print(f"  Neg Disease Hallucination: {neg_disease_hallucination:.4f} (avg max disease prob on neg tiles)")

        return {
            'negative_recall': neg_recall,
            'neg_disease_hallucination': neg_disease_hallucination,
            'negative_total': num_neg_total,
            'negative_correct': neg_correct
        }

    def evaluate_topk_quality(self, k: int = None) -> Dict:
        """
        评估Top-K选择质量（全量 val_dataset）.

        Args:
            k: Top-K值（None则从config读取）

        Returns:
            {
                'topk_lift': float,
                'topk_avg_confidence': float,
                'avg_top1_conf': float,
                'hit_acc': float
            }
        """
        if k is None:
            k = self.config.get('asymmetric_mil', {}).get('stable_k', 2)

        self.model.eval()

        topk_lifts = []
        topk_confidences = []
        avg_top1_confs = []
        hit_correct = 0
        hit_total = 0

        # Full val dataset - no sampling
        sample_indices = range(len(self.val_dataset))

        min_confidence_threshold = self.cm_min_confidence

        print(f"  Evaluating {len(self.val_dataset)} val bags (full dataset)...")
        with torch.no_grad():
            for idx in tqdm(sample_indices, desc="  Top-K Quality Evaluation"):
                bag = self.val_dataset[idx]
                tiles = bag['tiles']
                class_id = bag['class_id']

                # Convert to tensor with proper normalization (CRITICAL FIX 2026-02-05)
                tiles_list = []
                for t in tiles:
                    transformed = self.val_transform(image=t)
                    if isinstance(transformed['image'], torch.Tensor):
                        tiles_list.append(transformed['image'])
                    else:
                        tiles_list.append(
                            torch.from_numpy(transformed['image']).permute(2, 0, 1).float() / 255.0
                        )
                tiles_tensor = torch.stack(tiles_list).to(self.device)

                # Inference
                outputs = self.model.predict_instances(tiles_tensor)
                scores = outputs[:, class_id]
                probs = F.softmax(outputs, dim=1)

                # Top-K selection
                topk_k = min(k, len(scores))
                topk_indices = scores.topk(topk_k).indices
                topk_scores = scores[topk_indices]
                topk_probs = probs[topk_indices, class_id]

                # Random selection
                random_indices = torch.randperm(len(scores))[:topk_k]
                random_scores = scores[random_indices]
                random_probs = probs[random_indices, class_id]

                # Compute metrics
                topk_avg_prob = topk_probs.mean().item()
                random_avg_prob = random_probs.mean().item()
                lift = topk_avg_prob - random_avg_prob

                topk_avg_confidence = topk_probs.mean().item()

                topk_lifts.append(lift)
                topk_confidences.append(topk_avg_confidence)

                # Calculate Top-1 confidence
                top1_idx = scores.argmax()
                top1_prob = probs[top1_idx, class_id]
                avg_top1_confs.append(top1_prob.item())

                # Hit-Acc (Top-1 classification accuracy) - FINAL EVALUATION DEFINITION
                # ====================================================================
                # FILTERED: Exclude background predictions and low-confidence tiles
                # Definition: pred != 0 (non-background) AND conf >= threshold
                #
                # RATIONALE (differs from WarmupEvaluator):
                #   - WarmupEvaluator: Uses raw hit (pred==label) to track learning progress
                #   - FinalEvaluator: Uses stricter filter to exclude noise for deployment
                #     Low-confidence background predictions are filtered to prevent false positives
                # ====================================================================
                top1_pred_class = outputs[top1_idx].argmax()
                top1_max_prob = probs[top1_idx].max()

                if top1_pred_class != 0 and top1_max_prob >= min_confidence_threshold:
                    if top1_pred_class == class_id:
                        hit_correct += 1
                    hit_total += 1

        avg_lift = np.mean(topk_lifts) if topk_lifts else 0.0
        avg_confidence = np.mean(topk_confidences) if topk_confidences else 0.0
        avg_top1_conf = np.mean(avg_top1_confs) if avg_top1_confs else 0.0
        hit_acc = hit_correct / hit_total if hit_total > 0 else 0.0

        print(f"  Top-K Lift: {avg_lift:.4f}")
        print(f"  Avg-Top1-Conf: {avg_top1_conf:.4f}")
        print(f"  Hit-Acc: {hit_acc:.4f} ({hit_correct}/{hit_total})")
        print(f"  [Reference] TopK-Avg-Confidence: {avg_confidence:.4f}")

        return {
            'topk_lift': avg_lift,
            'topk_avg_confidence': avg_confidence,
            'avg_top1_conf': avg_top1_conf,
            'hit_acc': hit_acc
        }

    def compute_confusion_matrix(
        self,
        use_hit_acc_filter: bool = True,
        min_confidence: float = 0.3
    ) -> Dict:
        """
        计算混淆矩阵（全量 val_dataset + 全量 val_negative_pool，双版本输出）.

        同时构建过滤前和过滤后两版混淆矩阵：
          - cm_unfiltered: 所有 tile 直接记录（无 Hit-Acc 过滤）
          - cm_filtered:   pred != 0 AND max_conf >= min_confidence 才记录（Hit-Acc 过滤）

        Args:
            use_hit_acc_filter: 是否启用过滤（影响 cm_filtered 计算；cm_unfiltered 始终保存）
            min_confidence: Hit-Acc 过滤阈值

        Returns:
            {
                'confusion_matrix': np.ndarray,            # filtered version (primary)
                'confusion_matrix_unfiltered': np.ndarray, # unfiltered version (reference)
                'positive_tiles_total': int,
                'positive_tiles_filtered': int
            }
        """
        self.model.eval()

        num_total_classes = self.num_classes + 1
        cm_filtered = np.zeros((num_total_classes, num_total_classes), dtype=int)
        cm_unfiltered = np.zeros((num_total_classes, num_total_classes), dtype=int)

        # Full val dataset - no sampling
        num_pos_bags = len(self.val_dataset)
        print(f"  Evaluating {num_pos_bags} val bags (full dataset)...")
        pos_tiles_total = 0
        pos_tiles_filtered = 0

        with torch.no_grad():
            # Positive bags - full dataset
            for idx in tqdm(range(num_pos_bags), desc="  Confusion Matrix (Positive)"):
                bag = self.val_dataset[idx]
                tiles = bag['tiles']
                true_class = bag['class_id']

                # Convert to tensor with proper normalization (CRITICAL FIX 2026-02-05)
                tiles_list = []
                for t in tiles:
                    transformed = self.val_transform(image=t)
                    if isinstance(transformed['image'], torch.Tensor):
                        tiles_list.append(transformed['image'])
                    else:
                        tiles_list.append(
                            torch.from_numpy(transformed['image']).permute(2, 0, 1).float() / 255.0
                        )
                tiles_tensor = torch.stack(tiles_list).to(self.device)

                outputs = self.model.predict_instances(tiles_tensor)
                preds = outputs.argmax(dim=1)
                probs = F.softmax(outputs, dim=1)

                pos_tiles_total += len(preds)

                for i, pred in enumerate(preds.cpu().numpy()):
                    cm_unfiltered[true_class, pred] += 1
                    if use_hit_acc_filter:
                        max_prob = probs[i].max().item()
                        if pred != 0 and max_prob >= min_confidence:
                            cm_filtered[true_class, pred] += 1
                            pos_tiles_filtered += 1
                    else:
                        cm_filtered[true_class, pred] += 1
                        pos_tiles_filtered += 1

            # Negative pool - full pool
            num_neg_total = len(self.val_negative_pool)
            print(f"  Evaluating {num_neg_total} negative tiles (full pool)...")
            neg_tiles_np, _ = self.val_negative_pool.sample(num_neg_total)

            batch_size = 128
            for i in tqdm(range(0, num_neg_total, batch_size), desc="  Confusion Matrix (Negative)"):
                batch_tiles_np = neg_tiles_np[i:i+batch_size]

                # Convert to tensor with proper normalization (CRITICAL FIX 2026-02-05)
                batch_tiles_list = []
                for t in batch_tiles_np:
                    transformed = self.val_transform(image=t)
                    if isinstance(transformed['image'], torch.Tensor):
                        batch_tiles_list.append(transformed['image'])
                    else:
                        batch_tiles_list.append(
                            torch.from_numpy(transformed['image']).permute(2, 0, 1).float() / 255.0
                        )
                batch_tiles = torch.stack(batch_tiles_list).to(self.device)

                outputs = self.model.predict_instances(batch_tiles)
                preds = outputs.argmax(dim=1)

                for pred in preds.cpu().numpy():
                    cm_unfiltered[0, pred] += 1
                    cm_filtered[0, pred] += 1  # Negative tiles always included in both

        filter_ratio = pos_tiles_filtered / pos_tiles_total if pos_tiles_total > 0 else 0.0
        print(f"  Confusion Matrix computed: {cm_filtered.shape}")
        if use_hit_acc_filter:
            print(f"  Hit-Acc Filter: {pos_tiles_filtered}/{pos_tiles_total} pos tiles retained ({filter_ratio:.1%})")

        return {
            'confusion_matrix': cm_filtered,
            'confusion_matrix_unfiltered': cm_unfiltered,
            'positive_tiles_total': pos_tiles_total,
            'positive_tiles_filtered': pos_tiles_filtered
        }

    def compute_per_class_metrics(self, cm: np.ndarray) -> Dict:
        """
        计算每类精确率/召回率.

        Args:
            cm: 混淆矩阵

        Returns:
            {
                'per_class_precision': Dict[int, float],
                'per_class_recall': Dict[int, float],
                'per_class_f1': Dict[int, float]
            }
        """
        num_classes = cm.shape[0]

        precision_dict = {}
        recall_dict = {}
        f1_dict = {}

        for c in range(num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            fn = cm[c, :].sum() - tp
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            precision_dict[c] = precision
            recall_dict[c] = recall
            f1_dict[c] = f1

        return {
            'per_class_precision': precision_dict,
            'per_class_recall': recall_dict,
            'per_class_f1': f1_dict
        }

    def compute_overall_metrics(self, cm: np.ndarray) -> Dict:
        """
        计算整体指标.

        Args:
            cm: 混淆矩阵

        Returns:
            {'overall_accuracy': float}
        """
        total_correct = np.trace(cm)
        total_samples = cm.sum()
        overall_acc = total_correct / total_samples if total_samples > 0 else 0.0

        print(f"  Overall Accuracy: {overall_acc:.4f} (Reference only)")

        return {'overall_accuracy': overall_acc}

    def generate_heatmap(self, bag: Dict, tile_sizes: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        生成单个bag的多尺度全局热力图.

        Delegates to UnifiedInferenceEngine for sliding window inference.
        Converts multi-class output to single-channel max-disease heatmap
        for backward compatibility with visualization code.

        Args:
            bag: Bag字典 {'tiles': List[np.ndarray], 'class_id': int, 'image_path': str}
            tile_sizes: Tile尺寸列表（None时使用engine默认配置）

        Returns:
            (heatmap_global, original_image, tiles_info):
                - heatmap_global: (H, W) 全局热力图 (max disease confidence)
                - original_image: (H, W, 3) RGB图像
                - tiles_info: List[Dict] tile详细信息
        """
        self.model.eval()

        # Read original image
        image_path = bag.get('image_path', None)
        if image_path is None or not os.path.exists(image_path):
            print(f"[WARNING] No valid image_path, using first tile as fallback")
            tile = bag['tiles'][0]
            img_rgb = np.asarray(tile)
            h_orig, w_orig = img_rgb.shape[:2]
            heatmap_global = np.zeros((h_orig, w_orig), dtype=np.float32)
            return heatmap_global, img_rgb, []

        # Load image with unified utility
        from src.utils.io_utils import load_and_preprocess_image
        img_rgb = load_and_preprocess_image(image_path, target_size=None, color_mode='RGB')

        # Run unified inference engine
        result = self._inference_engine.run(img_rgb, tile_sizes=tile_sizes)

        # Convert multi-class (C, H, W) → single-channel (H, W) max disease
        # Exclude class 0 (healthy/background), take max across disease classes
        if result.heatmap.shape[0] > 1:
            heatmap_global = result.heatmap[1:].max(axis=0)  # (H, W)
        else:
            heatmap_global = result.heatmap[0]

        return heatmap_global, img_rgb, result.tiles_info

    def save_final_heatmaps(
        self,
        num_samples_per_class: Optional[int] = None,
        seed: int = 42,
        use_fixed_samples: bool = True,
        reference_samples_path: Optional[str] = None
    ):
        """
        训练结束后为每个正类别生成heatmap.

        CRITICAL: Can use same fixed reference samples as training monitoring heatmaps
        to enable direct comparison of model attention changes during training.

        Args:
            num_samples_per_class: 每个类别抽取的样本数
            seed: 随机种子 (only used if use_fixed_samples=False)
            use_fixed_samples: If True, load samples from reference_samples.json (default: True)
            reference_samples_path: Path to reference_samples.json (default: auto-detect from logger)
        """
        import json

        if num_samples_per_class is None:
            num_samples_per_class = self.heatmap_num_samples

        heatmap_dir = self.logger.get_evaluation_dir() / "final_heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"[FINAL-HEATMAP] Generating final heatmaps (num_classes={self.num_classes})...")
        print("=" * 60)

        # Try to load fixed reference samples (same as training monitoring)
        reference_samples = None
        if use_fixed_samples:
            # Try to find reference_samples.json from training heatmaps directory
            if reference_samples_path is None:
                # Check common locations
                possible_paths = [
                    self.logger.run_dir / 'heatmaps' / 'reference_samples.json',
                    self.logger.run_dir / 'reference_samples.json',
                ]
                for p in possible_paths:
                    if p.exists():
                        reference_samples_path = str(p)
                        break

            if reference_samples_path and Path(reference_samples_path).exists():
                try:
                    with open(reference_samples_path, 'r') as f:
                        reference_samples = json.load(f)
                    print(f"[FINAL-HEATMAP] Loaded {len(reference_samples)} fixed reference samples")
                    print(f"[FINAL-HEATMAP] Using same samples as training monitoring for comparison")
                except (json.JSONDecodeError, IOError) as e:
                    print(f"[FINAL-HEATMAP] Warning: Failed to load reference_samples.json: {e}")
                    reference_samples = None

        # If no fixed samples, select new ones
        if reference_samples is None:
            print("[FINAL-HEATMAP] Selecting new samples (not using fixed reference)")
            reference_samples = []
            rng = np.random.RandomState(seed)

            # Group samples by class
            class_to_indices = {}
            for idx in range(len(self.val_dataset)):
                bag = self.val_dataset[idx]
                class_id = bag['class_id']
                if class_id not in class_to_indices:
                    class_to_indices[class_id] = []
                class_to_indices[class_id].append(idx)

            # Sample from each class
            for class_id in sorted(class_to_indices.keys()):
                if class_id == 0:  # Skip background
                    continue
                indices = class_to_indices[class_id]
                num_samples = min(num_samples_per_class, len(indices))
                selected = rng.choice(indices, num_samples, replace=False)

                for idx in selected:
                    bag = self.val_dataset[idx]
                    class_name = CLASS_NAMES_SHORT.get(class_id, f'Class{class_id}')
                    reference_samples.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'image_path': bag.get('image_path', 'unknown'),
                        'bag_id': idx
                    })

        # Create a dedicated visualizer that saves to heatmap_dir
        # (self.visualizer saves to evaluation/visualizations, but final heatmaps
        #  should go to evaluation/final_heatmaps for clean separation)
        final_viz = MILVisualizer(
            save_dir=str(heatmap_dir),
            class_names=self.visualizer.class_names,
            short_names=self.visualizer.short_names
        )

        # Generate heatmaps for reference samples
        for sample in reference_samples:
            class_id = sample['class_id']
            class_name = sample.get('class_name', CLASS_NAMES_SHORT.get(class_id, f'Class{class_id}'))
            image_path = sample['image_path']

            print(f"\n[Class {class_id}] {CLASS_NAMES.get(class_id, class_name)}...")

            # Find the bag in dataset
            bag_idx = sample.get('bag_id')
            if bag_idx is not None and bag_idx < len(self.val_dataset):
                bag = self.val_dataset[bag_idx]
            else:
                # Search by image_path
                bag = None
                for idx in range(len(self.val_dataset)):
                    b = self.val_dataset[idx]
                    if b.get('image_path') == image_path:
                        bag = b
                        break
                if bag is None:
                    print(f"  [WARNING] Sample not found: {image_path}")
                    continue

            print(f"    Generating multi-scale heatmap...")
            heatmap_global, original_image, tiles_info = self.generate_heatmap(bag)

            # Save visualization using dedicated final heatmap visualizer
            save_name = f"final_class{class_id:02d}_{class_name}.png"
            final_viz.plot_spatial_heatmap(
                original_image=original_image,
                heatmap_global=heatmap_global,
                tiles_info=tiles_info,
                class_id=class_id,
                save_name=save_name,
                conf_threshold=self.heatmap_conf_threshold,
                top_k=self.heatmap_top_k
            )

            print(f"  Saved: {heatmap_dir / save_name}")

        # Save reference samples for this final evaluation
        final_ref_path = heatmap_dir / 'final_reference_samples.json'
        with open(final_ref_path, 'w') as f:
            json.dump(reference_samples, f, indent=2)
        print(f"\n[FINAL-HEATMAP] Reference samples saved to: {final_ref_path}")
        print(f"[FINAL-HEATMAP] All heatmaps saved to: {heatmap_dir}")

    def save_evaluation_report(self, metrics: Dict):
        """
        保存完整评估报告（JSON + 可视化）.

        Delegates all visualization to MILVisualizer for clean separation.

        Args:
            metrics: evaluate_all()返回的指标字典
        """
        eval_dir = self.logger.get_evaluation_dir()

        # 1. Save JSON metrics
        json_path = eval_dir / "final_metrics.json"
        metrics_serializable = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                metrics_serializable[k] = v.tolist()
            elif isinstance(v, dict):
                metrics_serializable[k] = {int(kk) if isinstance(kk, (int, np.integer)) else kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                                           for kk, vv in v.items()}
            else:
                metrics_serializable[k] = float(v) if isinstance(v, (float, np.floating)) else v

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)
        print(f"\n[FINAL-EVAL] Metrics saved to: {json_path}")

        # 1.5 Save filter statistics markdown report
        self._save_filter_statistics_report(metrics, eval_dir)

        # 2. Plot confusion matrix - both filtered and unfiltered versions
        if 'confusion_matrix' in metrics:
            filter_info = "(Hit-Acc Filtered, Full Dataset)" if self.cm_use_hit_acc_filter else "(Full Dataset)"
            self.visualizer.plot_confusion_matrix(
                cm=metrics['confusion_matrix'],
                save_name="confusion_matrix_filtered.png",
                filter_info=filter_info
            )
            print(f"  Filtered confusion matrix saved via MILVisualizer")

        if 'confusion_matrix_unfiltered' in metrics:
            self.visualizer.plot_confusion_matrix(
                cm=metrics['confusion_matrix_unfiltered'],
                save_name="confusion_matrix_unfiltered.png",
                filter_info="(Unfiltered, Full Dataset)"
            )
            print(f"  Unfiltered confusion matrix saved via MILVisualizer")

        # 3. Plot per-class metrics - filtered (primary) and unfiltered (reference)
        if all(k in metrics for k in ['per_class_precision', 'per_class_recall', 'per_class_f1']):
            self.visualizer.plot_per_class_metrics(
                metrics_dict={
                    'per_class_precision': metrics['per_class_precision'],
                    'per_class_recall': metrics['per_class_recall'],
                    'per_class_f1': metrics['per_class_f1']
                },
                save_name="per_class_metrics_filtered.png"
            )
            print(f"  Filtered per-class metrics plot saved via MILVisualizer")

        if all(k in metrics for k in ['per_class_precision_unfiltered', 'per_class_recall_unfiltered', 'per_class_f1_unfiltered']):
            self.visualizer.plot_per_class_metrics(
                metrics_dict={
                    'per_class_precision': metrics['per_class_precision_unfiltered'],
                    'per_class_recall': metrics['per_class_recall_unfiltered'],
                    'per_class_f1': metrics['per_class_f1_unfiltered']
                },
                save_name="per_class_metrics_unfiltered.png"
            )
            print(f"  Unfiltered per-class metrics plot saved via MILVisualizer")

        # 4. Generate final heatmaps
        print("\n[FINAL-EVAL] Generating final heatmaps...")
        self.save_final_heatmaps(num_samples_per_class=3)

        print(f"\n[FINAL-EVAL] Evaluation report completed: {eval_dir}")

    def _save_filter_statistics_report(self, metrics: Dict, eval_dir: Path):
        """
        Save confusion matrix filter statistics as markdown report.

        Args:
            metrics: Evaluation metrics dict (contains positive_tiles_total, positive_tiles_filtered)
            eval_dir: Evaluation output directory
        """
        md_path = eval_dir / "confusion_matrix_filter_stats.md"

        pos_total = metrics.get('positive_tiles_total', 0)
        pos_filtered = metrics.get('positive_tiles_filtered', 0)
        pos_excluded = pos_total - pos_filtered
        filter_ratio = pos_filtered / pos_total if pos_total > 0 else 0.0
        exclude_ratio = pos_excluded / pos_total if pos_total > 0 else 0.0

        report_lines = [
            "# Confusion Matrix Filter Statistics",
            "",
            "## Hit-Acc Filter Configuration",
            f"- **Filter Enabled**: {'Yes' if self.cm_use_hit_acc_filter else 'No'}",
            f"- **Minimum Confidence Threshold**: {self.cm_min_confidence:.2f}",
            "",
            "## Tile Statistics",
            f"- **Total Positive Tiles Evaluated**: {pos_total:,}",
            f"- **Tiles Retained (passed filter)**: {pos_filtered:,} ({filter_ratio:.1%})",
            f"- **Tiles Excluded (below threshold or background prediction)**: {pos_excluded:,} ({exclude_ratio:.1%})",
            "",
            "## Filter Logic",
            "The Hit-Acc filter excludes tiles from the confusion matrix when:",
            "1. The predicted class is Background (Class 0), OR",
            f"2. The maximum class probability is below {self.cm_min_confidence:.2f}",
            "",
            "**Rationale**: In weakly-supervised MIL, not all tiles in a positive bag are truly diseased.",
            "Low-confidence predictions likely represent healthy/background tiles within the bag.",
            "Filtering these prevents annotation noise from artificially inflating errors.",
            "",
            "## Interpretation Guide",
            f"- A high exclude ratio (>{30:.0%}) may indicate:",
            "  - Many healthy tiles in positive bags (expected in MIL)",
            "  - Model uncertainty on certain tile types",
            "  - Potential need for threshold tuning",
            f"- Current exclude ratio: **{exclude_ratio:.1%}**",
        ]

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"  Filter statistics report saved to: {md_path}")


__all__ = ['FinalEvaluator']
