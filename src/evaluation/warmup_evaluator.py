"""
Warm-up Evaluation Module for Asymmetric MIL Training.

Recent Updates:
    - [2025-12-10] BUGFIX: Fixed Per-Class Quality metrics calculation (CRITICAL)
        - class_coverage denominator: use total_num_classes (not sampled classes)
        - min_class_conf: unsampled classes treated as 0.0 (worst case)
        - qualified_classes threshold: use config (not hardcoded 0.5)
        - Added num_sampled_classes / total_num_classes tracking
    - [2025-12-10] BUGFIX: Fixed 4 critical bugs (tensor conversion, tuple unpacking, config access)
        - NegativeTilePool.sample() tuple unpacking (TypeError fix)
        - Hit-Acc CUDA tensor to Python bool conversion (TypeError fix)
        - pred_class tensor to Python int conversion (consistency)
        - Config safe access with .get() chains (KeyError prevention)
    - [2025-12-10] REDESIGN: Simplified P0 criteria (6 -> 3)
        - Removed class_coverage (too complex, ratio calculation confusing)
        - Downgraded min_class_conf to P1 reference (fine-grained classification for Stable phase)
        - Downgraded avg_top1_conf to P1 reference (not critical for warmup termination)
        - Warmup goal: Background vs Foreground separation only (coarse-grained)
        - Expected 9 positive classes (Class 1-9, no Class 0 background)
    - [2025-12-10] CRITICAL: Per-class analysis in P0 (prevents某类别死亡but平均值通过)
    - [2025-12-10] CRITICAL: Hit-Acc now includes ALL tiles (no filtering) for true hit rate
    - [2025-12-10] Removed: C0 and low-confidence filtering (was hiding model weaknesses)
    - [2025-12-05] CRITICAL: Simplified to 4 P0 criteria (removed redundant pos_acc metrics)
    - [2025-12-05] Refactor: All sampling parameters now configurable via yaml (warmup_eval_*)
    - [2025-12-05] Fixed: Added fixed seed for bag sampling (reproducible metrics across epochs)
    - [2025-12-03] Fixed: Added tile_size initialization from val_dataset or config
    - [2025-01-29] Fixed: Evaluation sampling stability (fixed seed for reproducible metrics)
    - [2025-01-29] Fixed: Default max_eval_samples corrected (110 → 200)
    - [2025-01-28] CRITICAL: Redesigned warm-up termination criteria (removed misleading Tile Acc)
    - [2025-01-28] FIXED: Negative Confidence now measures max disease confidence (not P(Class 0))
    - [2025-01-28] Feature: Increased eval samples (100 → 200) for more reliable evaluation
    - [2025-01-28] Feature: Top-K quality metrics (Lift and Avg Confidence) - P0 priority
    - [2025-01-27] Initial: Four-criteria evaluation system

P0 Criteria (ALL 3 must pass - SIMPLIFIED 2025-12-10):
    Background Suppression:
        1. Negative Recall > 0.7 (model can identify backgrounds)
        2. Negative Confidence < 0.3 (model doesn't hallucinate diseases on backgrounds)

    Top-K Quality:
        3. Top-K Lift > 0.2 (Top-K selection is not random)

P1 Reference Metrics (informational only, not used for warmup termination):
    - Min-Class-Conf: Hardest class confidence (fine-grained classification for Stable phase)
    - Avg-Top1-Conf: Global avg top1 confidence across all bags
    - Hit-Acc: Top-1 classification accuracy (includes ALL tiles, no filtering)
    - TopK-Avg-Confidence: Average confidence of Top-K tiles

RATIONALE:
    - Warmup goal: Background vs Foreground separation (coarse-grained)
    - Fine-grained classification (per-class quality) is Stable phase's job
    - Min-Class-Conf downgraded to P1 reference (too strict for warmup)

REMOVED:
    - Class-Coverage (too complex, ratio calculation confusing)
    - Min-Class-Conf as P0 metric (downgraded to P1 reference)
    - Top1-Confidence as P0 metric (downgraded to P1 reference)
    - Tile-level Acc (misleading: high acc may indicate poor background suppression)

Usage:
    >>> evaluator = WarmupEvaluator(model, val_dataset, negative_pool, config, max_eval_samples=200)
    >>> metrics = evaluator.evaluate_all_criteria(epoch, train_loss)
    >>> if evaluator.should_end_warmup(epoch, metrics):
    ...     print("Ending warm-up!")
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm


class WarmupEvaluator:
    """
    Warm-up phase evaluator.

    Evaluates if model has learned basic target vs background recognition ability.

    Args:
        model: Model being trained
        val_dataset: Validation dataset (AsymmetricMILDataset)
        negative_pool: Negative tile pool
        config: Configuration dict
        device: Device
        max_eval_samples: Max evaluation samples (prevents excessive eval time)
    """

    def __init__(
        self,
        model,
        val_dataset,
        negative_pool,
        config: Dict,
        device: torch.device = torch.device('cuda:0'),
        max_eval_samples: int = None  # Deprecated: use config instead
    ):
        self.model = model
        self.val_dataset = val_dataset
        self.negative_pool = negative_pool
        self.config = config
        self.device = device

        # Get tile_size from val_dataset
        # self.tile_size = getattr(val_dataset, 'tile_size',
        #                          config.get('dataset', {}).get('tile_config', {}).get('tile_size', 1024))
        # NEW: 获取最终输入尺寸 (用于某些手动resize的场合，虽然现在 _load_tiles 已经处理了)
        self.tile_size = getattr(val_dataset, 'final_tile_size', 384)

        # CRITICAL FIX [2026-02-05]: Create validation transform for proper preprocessing
        # Without this, evaluation uses /255 only while training uses ImageNet normalization
        # This mismatch causes incorrect evaluation metrics (topk_lift, hit_acc, etc.)
        from src.data.transforms import get_validation_transforms
        self.val_transform = get_validation_transforms(img_size=self.tile_size)
        # ========== Sampling Configuration (from yaml) ==========
        training_cfg = config.get('training', {})

        # Negative recognition evaluation
        self.neg_tiles = training_cfg.get('warmup_eval_neg_tiles', 1600)

        # Top-K quality evaluation (for topk_lift and topk_avg_confidence)
        self.topk_max_bags = training_cfg.get('warmup_eval_topk_max_bags', 100)

        # Legacy support (deprecated)
        if max_eval_samples is not None:
            print(f"[WARMUP-EVAL] Warning: max_eval_samples parameter is deprecated, using config instead")
            self.pos_max_bags = max_eval_samples
            self.topk_max_bags = max_eval_samples // 2

        # Threshold configuration (all from yaml, no hardcoded values)
        warmup_cfg = config.get('asymmetric_mil', {}).get('warmup_criteria', {})

        # P0 Core - Background Suppression
        self.neg_recall_threshold = warmup_cfg.get('neg_recall_threshold', 0.70)
        self.neg_disease_hallucination_threshold = warmup_cfg.get('neg_disease_hallucination_threshold', 0.30)

        # P0 Core - Top-K Quality
        self.topk_lift_threshold = warmup_cfg.get('topk_lift_threshold', 0.2)

        # Deprecated thresholds (kept for backward compatibility)
        # self.top1_confidence_threshold = warmup_cfg.get('top1_confidence_threshold', 0.5)  # Deprecated: downgraded to P1 reference
        # self.topk_avg_confidence_threshold = warmup_cfg.get('topk_avg_confidence_threshold', 0.7)  # Deprecated

        # Deprecated - Redundant with topk_avg_confidence
        # self.pos_acc_threshold = warmup_cfg.get('pos_acc_threshold', 0.5)
        # self.pos_confidence_threshold = warmup_cfg.get('pos_confidence_threshold', 0.5)
        # self.pos_pred_as_bg_threshold = warmup_cfg.get('pos_pred_as_bg_threshold', 0.2)

        # Legacy thresholds (not used in current P0 criteria)
        # self.tile_acc_threshold = warmup_cfg.get('tile_acc_threshold', 0.50)  # Not used anymore (misleading)
        # self.topk_consistency_threshold = warmup_cfg.get('topk_consistency_threshold', 0.60)  # P1 reference
        # self.loss_delta_threshold = warmup_cfg.get('loss_delta_threshold', 0.15)  # P1 reference

        # History tracking
        self.history = []

    # REMOVED: evaluate_tile_classification (2025-01-29)
    # Reason: Tile-Acc is misleading (high acc may indicate poor background suppression)
    # Use evaluate_negative_recognition instead for neg_recall

    # REMOVED: evaluate_topk_consistency (2025-01-29)
    # Reason: Extremely slow (2-5x forward passes per bag × 100 bags = 200-500 forwards)
    # Only P1 reference metric - Loss-Delta is sufficient for P1 criteria

    def evaluate_topk_quality(self, k: int = 2) -> Tuple[float, float, float, float, Dict]:
        """
        Evaluate Top-K selection quality with per-class analysis (NEW: 2025-12-10).

        This is the MOST CRITICAL warm-up metric according to evaluation_schedule.md.
        Validates if Top-K tiles have significantly higher scores than random tiles.

        CRITICAL ALIGNMENT WITH LOSS DESIGN:
            - Top-1: CE strong supervision → should have high confidence (> 0.5)
            - Top-2~K: Negative rejection/KL → suppressed, low confidence is normal
            - Therefore: Top-1 confidence is the key metric, NOT Top-K avg confidence

        NEW (2025-12-10): Per-class breakdown to detect "困难类别"
            - Prevents某些类别完全不达标but平均值通过P0的问题
            - Allows 20% classes不达标（允许困难类别存在）

        Args:
            k: Top-k value

        Returns:
            (topk_avg_confidence, topk_lift, hit_acc, avg_top1_conf, per_class_stats)
            - topk_avg_confidence: [P1 Reference] Average confidence of Top-K tiles
            - topk_lift: [P0 Core] Top-K avg score - Random tiles avg score (should > 0.2)
            - hit_acc: [P1 Reference] Hit Accuracy - Top-1 classification accuracy (includes ALL tiles)
            - avg_top1_conf: [P1 Reference] Top-1 tile confidence (GLOBAL AVG, renamed from top1_confidence)
            - per_class_stats: Dict[class_id, {avg_conf, hit_acc, count}] per-class breakdown
        """
        self.model.eval()
        topk_confidences = []
        lifts = []
        avg_top1_confs = []  # RENAMED: Track Top-1 confidence separately (from top1_confidences)

        # NEW: Hit-Acc tracking (ALL tiles included, no filtering)
        hit_correct = 0
        hit_total = 0

        # NEW (2025-12-10): Per-class statistics
        from collections import defaultdict
        per_class_data = defaultdict(lambda: {'confidences': [], 'hits': [], 'lifts': []})  # NEW: Track lifts per class

        # Sample validation bags (with fixed seed for consistency)
        # NEW (2025-12-11): Class-balanced sampling (e.g., 55 bags = 11 classes × 5 per class)
        eval_rng = np.random.RandomState(self.config.get('training_strategy', {}).get('seed', 42) + 2)  # +2 for different subset

        # Group indices by class
        class_to_indices = defaultdict(list)
        for idx in range(len(self.val_dataset)):
            bag = self.val_dataset[idx]
            class_id = bag['class_id']
            class_to_indices[class_id].append(idx)

        # Sample equal number per class
        num_classes = len(class_to_indices)
        samples_per_class = max(1, self.topk_max_bags // num_classes)  # Default: 55 // 11 = 5

        sample_indices = []
        for class_id in sorted(class_to_indices.keys()):
            indices = class_to_indices[class_id]
            if len(indices) > 0:
                # Sample with replacement if class has fewer samples than needed
                if samples_per_class <= len(indices):
                    sampled = eval_rng.choice(indices, samples_per_class, replace=False)
                else:
                    sampled = eval_rng.choice(indices, samples_per_class, replace=True)
                sample_indices.extend(sampled.tolist())

        # Shuffle final sample indices for better diversity in tqdm progress
        eval_rng.shuffle(sample_indices)

        with torch.no_grad():
            for idx in tqdm(sample_indices, desc="Eval TopK-Quality", leave=False):
                bag = self.val_dataset[idx]
                tiles = bag['tiles']
                class_id = bag['class_id']

                # CRITICAL FIX (2025-12-12): Ensure all tiles have same size for multi-scale
                # Dataset should already resize, but double-check here for safety
                tiles_resized = []
                for t in tiles:
                    t_arr = np.asarray(t)
                    # If tile size doesn't match final_tile_size, resize it
                    if t_arr.shape[0] != self.tile_size or t_arr.shape[1] != self.tile_size:
                        from src.utils.resize_utils import resize_keep_aspect_ratio_crop
                        t_arr = resize_keep_aspect_ratio_crop(t_arr, self.tile_size)
                    tiles_resized.append(t_arr)

                # Convert to tensor with proper normalization (CRITICAL FIX 2026-02-05)
                # Must use same preprocessing as training (ImageNet normalize)
                tiles_transformed = []
                for t in tiles_resized:
                    transformed = self.val_transform(image=t)
                    if isinstance(transformed['image'], torch.Tensor):
                        tiles_transformed.append(transformed['image'])
                    else:
                        tiles_transformed.append(
                            torch.from_numpy(transformed['image']).permute(2, 0, 1).float() / 255.0
                        )
                tiles_tensor = torch.stack(tiles_transformed).to(self.device)

                # Forward
                outputs = self.model.predict_instances(tiles_tensor)
                scores = outputs[:, class_id]  # Class-specific scores (logits)
                probs = F.softmax(outputs, dim=1)  # Full probability distribution

                # Top-K selection (based on scores/logits)
                topk_k = min(k, len(scores))
                topk_indices = scores.topk(topk_k).indices
                # topk_scores = scores[topk_indices]
                topk_probs = probs[:, class_id][topk_indices]  # Class-specific probabilities

                # Random selection (for Lift calculation)
                random_indices = torch.randperm(len(scores))[:topk_k]
                # random_scores = scores[random_indices]
                random_probs = probs[:, class_id][random_indices]

                # Metrics (FIXED: Use probabilities for lift, not logits)
                topk_avg_confidence = topk_probs.mean().item()
                topk_avg_prob = topk_probs.mean().item()
                random_avg_prob = random_probs.mean().item()
                lift = topk_avg_prob - random_avg_prob  # Lift based on probabilities (0-1 range)

                topk_confidences.append(topk_avg_confidence)
                lifts.append(lift)

                # NEW: Calculate Top-1 confidence (CRITICAL: aligned with loss design)
                # Find Top-1 tile (highest class-specific score)
                top1_idx = scores.argmax()
                top1_prob = probs[top1_idx, class_id]  # Top-1 tile's confidence on true class
                avg_top1_confs.append(top1_prob.item())  # RENAMED: from top1_confidences

                # Hit-Acc (Top-1 classification accuracy) - WARMUP DEFINITION
                # ====================================================================
                # NO FILTERING - Include ALL tiles to track learning progress
                # Definition: Top-1 pred == Bag label (simple match, no confidence filter)
                #
                # RATIONALE (differs from FinalEvaluator):
                #   - WarmupEvaluator: Tracks raw learning progress without noise filtering
                #   - FinalEvaluator: Uses stricter filter (pred!=0 AND conf>=threshold)
                #     to exclude low-confidence background predictions for deployment
                # ====================================================================
                top1_pred_class = outputs[top1_idx].argmax()  # Predicted class

                # Check: Predicted class == Bag true label?
                hit = (top1_pred_class == class_id).item()  # Convert tensor to Python bool
                if hit:
                    hit_correct += 1
                hit_total += 1

                # NEW (2025-12-10): Per-class tracking
                per_class_data[class_id]['confidences'].append(top1_prob.item())
                per_class_data[class_id]['hits'].append(hit)
                per_class_data[class_id]['lifts'].append(lift)  # NEW: Track lift per class

        # Global metrics
        avg_confidence = np.mean(topk_confidences) if len(topk_confidences) > 0 else 0.0
        avg_lift = np.mean(lifts) if len(lifts) > 0 else 0.0
        hit_acc = hit_correct / hit_total if hit_total > 0 else 0.0
        avg_top1_conf = np.mean(avg_top1_confs) if len(avg_top1_confs) > 0 else 0.0  # RENAMED: from top1_confidence

        # NEW (2025-12-10): Per-class statistics
        per_class_stats = {}
        for cls_id, data in per_class_data.items():
            per_class_stats[cls_id] = {
                'avg_conf': np.mean(data['confidences']),
                'hit_acc': np.mean(data['hits']),
                'avg_lift': np.mean(data['lifts']),  # NEW: Average lift per class
                'count': len(data['confidences'])
            }

        return avg_confidence, avg_lift, hit_acc, avg_top1_conf, per_class_stats  # RENAMED: top1_confidence → avg_top1_conf

    def evaluate_negative_recognition(self, num_samples: int = None) -> Tuple[float, float]:
        """
        Criterion 3: Negative sample recognition ability.

        CRITICAL METRIC DEFINITION:
            - Negative Recall: Fraction of negative tiles correctly predicted as Class 0
            - Negative Confidence: Avg **MAX disease confidence** on negative tiles
              (i.e., max(P(Class 1), P(Class 2), ..., P(Class N)))
              This measures if model hallucinates diseases on healthy backgrounds.

        Args:
            num_samples: Number of negative tiles to evaluate (deprecated, use config instead)

        Returns:
            (neg_recall, avg_max_disease_confidence)
        """
        self.model.eval()

        # Use config value if num_samples not provided
        if num_samples is None:
            num_samples = self.neg_tiles

        neg_tiles, _ = self.negative_pool.sample(min(num_samples, len(self.negative_pool)))

        correct_neg = 0
        max_disease_confidences = []

        with torch.no_grad():
            for tile_np in tqdm(neg_tiles, desc="Eval Neg-Recognition", leave=False):
                # CRITICAL FIX (2025-12-12): Ensure tile size matches for multi-scale
                tile_np = np.asarray(tile_np)
                if tile_np.shape[0] != self.tile_size or tile_np.shape[1] != self.tile_size:
                    from src.utils.resize_utils import resize_keep_aspect_ratio_crop
                    tile_np = resize_keep_aspect_ratio_crop(tile_np, self.tile_size)

                # Apply proper normalization (CRITICAL FIX 2026-02-05)
                transformed = self.val_transform(image=tile_np)
                if isinstance(transformed['image'], torch.Tensor):
                    tile_tensor = transformed['image']
                else:
                    tile_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float() / 255.0
                tile_tensor = tile_tensor.unsqueeze(0).to(self.device)

                output = self.model.predict_instances(tile_tensor)
                probs = F.softmax(output, dim=1)[0]

                pred_class = probs.argmax().item()  # Convert tensor to Python int

                # FIXED: Negative Confidence = max disease confidence (NOT P(Class 0))
                # probs[0] is P(Class 0 = healthy)
                # probs[1:] is P(Class 1-N = diseases)
                max_disease_conf = probs[1:].max().item() if len(probs) > 1 else 0.0

                if pred_class == 0:
                    correct_neg += 1
                max_disease_confidences.append(max_disease_conf)

        neg_recall = correct_neg / len(neg_tiles) if len(neg_tiles) > 0 else 0.0
        avg_max_disease_conf = np.mean(max_disease_confidences) if len(max_disease_confidences) > 0 else 0.0

        return neg_recall, avg_max_disease_conf

    # DEPRECATED: Removed in favor of topk_avg_confidence metric
    # This function is no longer used - positive classification metrics are redundant
    # with topk_avg_confidence in evaluate_topk_quality()
    def _deprecated_evaluate_positive_classification(self) -> Tuple[float, float, float]:
        """
        Evaluate positive tile classification quality using Top-K tile selection.

        CRITICAL STRATEGY:
            - Uses Scout forward to select Top-K tiles (K from config, default=2)
            - K=1: Most strict (only model's best tile)
            - K=2: Balanced (top 2 tiles, allows slight tolerance)
            - Strict quality check ensures warmup termination is reliable

        Rationale:
            - Top-K tiles = tiles model is most confident about
            - If model can't classify its BEST tiles correctly → not ready for stable phase
            - Random sampling would include low-quality tiles and produce misleading metrics

        Detects "confusion collapse" where model learns foreground/background
        but fails to learn C1-C9 disease classification.

        Returns:
            Tuple of (pos_acc, pos_confidence, pos_pred_as_bg):
                - pos_acc: Positive tile accuracy (on Top-K tiles, strict standard)
                - pos_confidence: Average max probability when correct
                - pos_pred_as_bg: Fraction of Top-K tiles predicted as Class 0 (background)
        """
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        # Sample positive bags (with fixed seed for consistency)
        num_samples = min(len(self.val_dataset), self.pos_max_bags)
        eval_rng = np.random.RandomState(self.config.get('training_strategy', {}).get('seed', 42) + 3)  # +3 for different subset
        sample_indices = eval_rng.choice(len(self.val_dataset), num_samples, replace=False)

        for idx in tqdm(sample_indices, desc="Eval Pos-Classification", leave=False):
            image_path, class_id = self.val_dataset.samples[idx]

            # Load ALL tiles (need all for Top-K selection)
            tiles_np = self.val_dataset._load_tiles(image_path)

            # Convert to tensors (resize to uniform size)
            tiles_list = []
            for tile_np in tiles_np:
                if tile_np.shape[0] != self.tile_size or tile_np.shape[1] != self.tile_size:
                    from src.utils.resize_utils import resize_keep_aspect_ratio_crop
                    tile_np = resize_keep_aspect_ratio_crop(tile_np, self.tile_size)

                tile_tensor = torch.from_numpy(tile_np).permute(2, 0, 1).float() / 255.0
                tiles_list.append(tile_tensor)

            tiles_tensor = torch.stack(tiles_list).to(self.device)

            # Scout forward: Get class-specific scores for all tiles
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
                outputs = self.model.predict_instances(tiles_tensor)
                scores = outputs[:, class_id]  # Class-specific scores for this bag

                # Select Top-K tiles (strict quality check)
                # K=1: Most strict (only best tile)
                # K=2: Balanced (top 2 tiles)
                k_actual = min(self.pos_topk, len(scores))
                topk_indices = scores.topk(k_actual).indices

                # Get predictions and probabilities for Top-K tiles
                topk_outputs = outputs[topk_indices]
                probs = torch.softmax(topk_outputs, dim=1)

            preds = topk_outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend([class_id] * k_actual)
            all_probs.extend(probs.cpu().numpy())

        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Positive accuracy (correct C1-C11 classification)
        pos_acc = (all_preds == all_labels).sum() / len(all_labels) if len(all_labels) > 0 else 0.0

        # Positive confidence (when correct)
        correct_mask = all_preds == all_labels
        if correct_mask.sum() > 0:
            pos_confidence = all_probs[correct_mask].max(axis=1).mean()
        else:
            pos_confidence = 0.0

        # Positive predicted as background (false negative rate)
        pos_pred_as_bg = (all_preds == 0).sum() / len(all_labels) if len(all_labels) > 0 else 0.0

        return pos_acc, pos_confidence, pos_pred_as_bg

    # REMOVED: evaluate_all_criteria (2025-01-29)
    # Use evaluate_warmup_criteria instead (topk_consistency removed)
    def evaluate_warmup_criteria(self, epoch: int, train_loss: float) -> Dict:
        """
        Evaluate all warm-up criteria.

        Args:
            epoch: Current epoch
            train_loss: Current training loss

        Returns:
            Metrics dict
        """
        print(f"\n[WARMUP-EVAL] Evaluating epoch {epoch}...")

        # Criterion 1: Negative recognition (P0 Core)
        neg_recall, neg_disease_hallucination = self.evaluate_negative_recognition()
        print(f"  Neg-Recall: {neg_recall:.4f}, Neg-Disease-Hallucination: {neg_disease_hallucination:.4f}")

        # Criterion 2: Top-K quality
        # topk_lift: Top-K selection is not random (P0 Core)
        # avg_top1_conf: Global avg top1 confidence (P1 Reference)
        # hit_acc: Top-1 classification accuracy (P1 Reference)
        # topk_avg_confidence: Average confidence of Top-K tiles (P1 Reference)
        # per_class_stats: Per-class breakdown for min_class_conf
        k = self.config.get('asymmetric_mil', {}).get('warmup_k', 4)
        topk_avg_confidence, topk_lift, hit_acc, avg_top1_conf, per_class_stats = self.evaluate_topk_quality(k=k)
        print(f"  TopK-Lift: {topk_lift:.4f}")
        print(f"  [P1 Reference] Avg-Top1-Conf: {avg_top1_conf:.4f}, Hit-Acc: {hit_acc:.4f}, TopK-Avg-Conf: {topk_avg_confidence:.4f}")

        # NEW (2025-12-10): Per-class analysis (Simplified: only min_class_conf)
        # Note: Evaluation samples ~100 bags from validation set, should cover all classes
        num_sampled_classes = len(per_class_stats)  # Classes actually sampled

        # Min-Class-Conf: Worst class's avg top1 confidence (hardest class must pass threshold)
        min_class_conf = min((stats['avg_conf'] for stats in per_class_stats.values()), default=0.0)

        # Diagnostic info
        print(f"  [Per-Class] Min-Conf: {min_class_conf:.3f}, Sampled: {num_sampled_classes} classes")
        if num_sampled_classes < 9:  # Expected: 9 positive classes (Class 1-9, no Class 0)
            print(f"  [Per-Class WARNING] Expected 9 classes, only {num_sampled_classes} sampled (increase eval samples?)")

        # Detailed per-class breakdown (for debugging)
        if num_sampled_classes > 0:
            sorted_classes = sorted(per_class_stats.items(), key=lambda x: x[1]['avg_conf'])
            print(f"  [Per-Class DEBUG] Worst 3 classes:")
            for cls_id, stats in sorted_classes[:3]:
                print(f"    Class {cls_id}: avg_conf={stats['avg_conf']:.3f}, lift={stats['avg_lift']:.3f}, hit_acc={stats['hit_acc']:.3f}, count={stats['count']}")  # NEW: Added lift

        metrics = {
            'epoch': epoch,
            # P0 Core: Background suppression
            'neg_recall': neg_recall,
            'neg_disease_hallucination': neg_disease_hallucination,
            # P0 Core: Top-K quality (sufficient for warmup termination)
            'topk_lift': topk_lift,
            # P0 Core: Per-class quality (NEW 2025-12-10, Simplified)
            'min_class_conf': min_class_conf,  # Worst class's avg top1 confidence
            # P1 Reference metrics (not used for warmup termination)
            'avg_top1_conf': avg_top1_conf,  # Avg top1 confidence across all bags
            'hit_acc': hit_acc,  # Hit Accuracy (Top-1 classification accuracy)
            'topk_avg_confidence': topk_avg_confidence,  # Reference only
            # Diagnostic info
            'num_sampled_classes': num_sampled_classes,  # Number of classes sampled
            'per_class_stats': per_class_stats,  # Detailed per-class breakdown
            # Training state
            'train_loss': train_loss,
        }

        self.history.append(metrics)
        return metrics

    def should_end_warmup(self, epoch: int, metrics: Dict) -> bool:
        """
        Determine if warm-up should end.

        REDESIGNED LOGIC (2025-12-10 v3 - Minimalist):
            Downgraded min_class_conf to P1 reference (fine-grained classification for Stable).
            Downgraded avg_top1_conf to P1 reference (not critical for warmup termination).
            Warmup goal: Background vs Foreground separation only (coarse-grained).

        P0 CORE CRITERIA (ALL 3 must pass):
            Background Suppression:
                1. Negative Recall > 0.7 (model recognizes backgrounds)
                2. Negative Confidence < 0.3 (model doesn't hallucinate diseases on backgrounds)

            Top-K Quality:
                3. Top-K Lift > 0.2 (Top-K selection is not random)

        P1 REFERENCE METRICS (not used for warmup termination):
            - min_class_conf: Hardest class confidence (fine-grained, for Stable phase)
            - avg_top1_conf: Global avg top1 confidence (informational only)
            - hit_acc: Top-1 classification accuracy
            - topk_avg_confidence: Avg confidence of Top-K tiles

        RATIONALE:
            - Warmup focuses on coarse-grained: Background vs Foreground
            - Fine-grained classification (per-class quality) is Stable phase's job
            - Only 3 P0 criteria: Minimal but sufficient for warmup termination

        Args:
            epoch: Current epoch
            metrics: Evaluation metrics

        Returns:
            True if should end warm-up
        """
        # Hard requirement: at least N warm-up epochs
        warmup_epochs = self.config.get('asymmetric_mil', {}).get('warmup_epochs', 5)
        if epoch < warmup_epochs:
            print(f"[WARMUP-EVAL] Continue warm-up (min epochs: {warmup_epochs})")
            return False

        # ========== P0 Core Criteria (ALL 3 must pass) ==========
        # Background Suppression
        p0_neg_recall = metrics['neg_recall'] > self.neg_recall_threshold
        p0_neg_hallucination = metrics['neg_disease_hallucination'] < self.neg_disease_hallucination_threshold  # NOTE: < not >

        # Top-K Quality
        p0_topk_lift = metrics['topk_lift'] > self.topk_lift_threshold

        p0_conditions = [
            p0_neg_recall, p0_neg_hallucination,  # Background suppression
            p0_topk_lift,                      # Top-K quality
        ]
        p0_passed = all(p0_conditions)

        print(f"\n[WARMUP-EVAL] Warm-up Termination Criteria Check:")
        print(f"\n  === P0 CORE (ALL 3 must pass - Minimalist) ===")
        print(f"\n  Background Suppression:")
        print(f"    1. Neg-Recall > {self.neg_recall_threshold}: {'[PASS]' if p0_neg_recall else '[FAIL]'} ({metrics['neg_recall']:.4f})")
        print(f"       -> Model can identify backgrounds (high recall = good)")
        print(f"    2. Neg-Disease-Hallucination < {self.neg_disease_hallucination_threshold}: {'[PASS]' if p0_neg_hallucination else '[FAIL]'} ({metrics['neg_disease_hallucination']:.4f})")
        print(f"       -> Model doesn't hallucinate diseases on backgrounds (low = good, = max disease prob on neg tiles)")

        print(f"\n  Top-K Quality:")
        print(f"    3. TopK-Lift > {self.topk_lift_threshold}: {'[PASS]' if p0_topk_lift else '[FAIL]'} ({metrics['topk_lift']:.4f})")
        print(f"       -> Top-K selection is not random (high lift = good)")

        print(f"\n  === P1 REFERENCE (informational, not used for warmup termination) ===")
        print(f"    Min-Class-Conf: {metrics['min_class_conf']:.3f} (hardest class, fine-grained for Stable)")
        print(f"    Avg-Top1-Conf: {metrics['avg_top1_conf']:.4f} (global avg top1 confidence)")
        print(f"    Hit-Acc: {metrics['hit_acc']:.4f} (Top-1 classification accuracy)")
        print(f"    TopK-Avg-Confidence: {metrics['topk_avg_confidence']:.4f}")
        print(f"    Sampled: {metrics['num_sampled_classes']} classes")

        print(f"\n  P0 Status: {'ALL PASSED' if p0_passed else 'FAILED'}")

        # Decision: P0 must ALL pass (no P1 criteria)
        should_end = p0_passed

        if should_end:
            print(f"\n[WARMUP-EVAL] All 3 P0 criteria met! Ending warm-up at epoch {epoch}")
            print(f"  Background Suppression:")
            print(f"    - Neg-Recall: {metrics['neg_recall']:.2%}")
            print(f"    - Neg-Disease-Hallucination: {metrics['neg_disease_hallucination']:.3f}")
            print(f"  Top-K Quality:")
            print(f"    - TopK-Lift: {metrics['topk_lift']:.3f}")
            print(f"  P1 Reference (for info):")
            print(f"    - Min-Class-Conf: {metrics['min_class_conf']:.3f} (hardest class)")
            return True
        else:
            print(f"\n[WARMUP-EVAL] Continue warm-up (P0 core criteria not met)")
            if not p0_neg_recall:
                print(f"    - Neg-Recall too low ({metrics['neg_recall']:.2%} < {self.neg_recall_threshold:.2%})")
            if not p0_neg_hallucination:
                print(f"    - Neg-Disease-Hallucination too high ({metrics['neg_disease_hallucination']:.3f} >= {self.neg_disease_hallucination_threshold:.3f})")
            if not p0_topk_lift:
                print(f"    - TopK-Lift too low ({metrics['topk_lift']:.3f} < {self.topk_lift_threshold})")
            return False


__all__ = ['WarmupEvaluator']
