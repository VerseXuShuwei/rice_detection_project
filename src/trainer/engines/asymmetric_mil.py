"""
Asymmetric MIL Training and Validation Engines.

Recent Updates:
    - [2026-01-05] Refactor: Extracted from train_topk_asymmetric.py (no logic changes)
    - [2026-01-05] CRITICAL: Scout-Snipe logic preserved exactly

Key Features:
    - train_one_epoch: Scout (eval+no_grad) → Snipe (train+grad) with AugmentedTileCache
    - validate: Quick validation with random tile sampling
    - Hard negative mining lifecycle hooks preserved
    - Mosaic augmentation support in stable phase

Scout-Snipe Architecture (CRITICAL - Do NOT modify):
    Scout Pass: model.eval() + torch.no_grad()
        → Top-K selection + tile_cache.store()
    Snipe Pass: model.train() + torch.enable_grad()
        → Forward on cached tiles (tile_cache.get_topk())
        → Gradient descent

Usage:
    >>> from src.trainer.engines.asymmetric_mil import train_one_epoch, validate
    >>> epoch_state = {
    ...     'K': 4,
    ...     'is_warmup': True,
    ...     'hard_ratio': 0.25,
    ...     'hard_mining_enable': True
    ... }
    >>> avg_loss, avg_acc, epoch_tier_stats = train_one_epoch(
    ...     model=model,
    ...     dataloader=train_loader,
    ...     negative_pool=train_neg_pool,
    ...     criterion=loss_fn,
    ...     optimizer=optimizer,
    ...     scaler=scaler,
    ...     device=device,
    ...     config=config,
    ...     epoch=1,
    ...     epoch_state=epoch_state
    ... )

"""

import gc
from typing import Dict, Tuple, Optional, Any, Union, TypedDict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import numpy as np

from src.data import get_augmentation_from_config, AugmentedTileCache
from src.data.negative_pool import NegativeTilePool
from src.data.spatial_nms import spatial_nms_from_tile_infos, scale_diverse_topk


def train_one_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    train_sampler: 'ClassAwareBagSampler',
    negative_pool: 'NegativeTilePool',
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    config: Dict,
    epoch: int,
    epoch_state: Dict,
    feature_critic: Optional[nn.Module] = None,
    logger: Optional[Any] = None
) -> tuple[Union[float, Any], Union[float, Any], dict[str, Any]]:
    """
    Train for one epoch using Top-K Asymmetric strategy.

    Three-phase training flow (Scout-Snipe Architecture):
        1. Load: Sample positive bags from DataLoader
        2. Scout: model.eval() + no_grad → Top-K tile selection
        3. Snipe: model.train() + grad → train on Top-K + negative samples

    Args:
        model: Model to train
        train_dataloader: Persistent DataLoader (created once in Trainer)
        train_sampler: Class-aware bag sampler (for metadata access)
        negative_pool: Negative tile pool for hard mining
        criterion: Loss function (TopKAnchoredMILLoss or AsymmetricMILLoss)
        optimizer: Optimizer
        scaler: GradScaler for AMP
        device: Device
        config: Configuration dict
        epoch: Current epoch
        epoch_state: Dict containing epoch-level training state
            - 'K': Top-K value (4 for warmup, 2 for stable)
            - 'is_warmup': Whether in warm-up phase
            - 'hard_ratio': Hard negative mining ratio (0.0-1.0)
            - 'hard_mining_enable': Whether hard negative mining is enabled

    Returns:
        Tuple of (average_loss, average_accuracy, epoch_tier_stats, last_batch_diag)

    CRITICAL - Scout-Snipe Sanctity:
        - Scout MUST use model.eval() + torch.no_grad()
        - Snipe MUST use model.train() + torch.enable_grad()
        - MUST use AugmentedTileCache to ensure Scout/Snipe see same tiles
    """
    # Extract epoch-level state (Centralized state management)
    K = epoch_state['K']
    is_warmup = epoch_state['is_warmup']
    hard_ratio = epoch_state['hard_ratio']
    hard_mining_enable = epoch_state['hard_mining_enable']

    # Read config
    train_cfg = config.get('training', {})
    mil_cfg = config.get('asymmetric_mil', {})
    nms_cfg = config.get('spatial_nms', {})
    scale_div_cfg = config.get('scale_diversity', {})
    anti_collapse_cfg = config.get('anti_collapse', {})
    use_amp = train_cfg.get('use_amp', True)
    gradient_clip = train_cfg.get('gradient_clip', 1.0)
    scout_batch_size = mil_cfg.get('scout_batch_size', 32)
    cache_augmented = mil_cfg.get('cache_augmented_tiles', True)

    # Spatial NMS config (with stable-phase override)
    enable_nms = nms_cfg.get('enable', False)
    nms_iou_threshold = nms_cfg.get('iou_threshold', 0.5)
    # [2026-02-02] Allow disabling NMS in stable phase to preserve Top-K diversity
    nms_stable_phase_enable = nms_cfg.get('stable_phase_enable', True)
    if not is_warmup and not nms_stable_phase_enable:
        enable_nms = False

    # Scale-Diversity Top-K config (NEW 2026-02-08)
    enable_scale_diversity = scale_div_cfg.get('enable', False)
    scale_diversity_penalty = scale_div_cfg.get('penalty', 0.5)

    # Anti-collapse config
    freeze_bn_in_snipe = anti_collapse_cfg.get('freeze_bn_in_snipe', True)
    monitor_collapse = anti_collapse_cfg.get('monitor_collapse_indicators', True)
    debug_score_drift = anti_collapse_cfg.get('debug_score_drift', False)

    # Metrics
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(train_sampler)

    # Hard mining: global sample counter across all batches in epoch
    # Used to create unique keys for current_epoch_samples mapping
    global_sample_offset = 0

    # Initialize augmented tile cache
    tile_cache = AugmentedTileCache() if cache_augmented else None

    # Get augmentation transform
    img_size = config.get('dataset', {}).get('final_tile_size', 384)
    train_transform = get_augmentation_from_config({
        'mode': 'base',
        'img_size': img_size,
        'augmentation': config.get('augmentation', {})
    })

    # ========== Use Persistent DataLoader (Created Once in Trainer) ==========
    # NOTE: DataLoader is now passed as parameter (created once in Trainer.__init__)
    # Workers stay alive across all epochs (persistent_workers=True) → no startup overhead

    # Create progress bar
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} ({'WARMUP' if is_warmup else 'STABLE'})", total=num_batches)
    epoch_critic_stats = {'bg_ratio': [], 'avg_sim': []} if feature_critic is not None else None
    for batch_idx, batch_bags in enumerate(pbar):
        # ========== Phase 1: LOAD ==========
        # Data already loaded by DataLoader workers (multi-process)

        # ========== Phase 2: SCOUT (Inference for Top-K selection) ==========
        model.eval()  # CRITICAL: Use eval mode to prevent BN statistics shift

        topk_indices_list = []
        class_ids = []

        # Feature Critic stats accumulator (if enabled)
        # epoch_critic_stats = {'bg_ratio': [], 'avg_sim': []} if hasattr(trainer, 'feature_critic') else None
        # epoch_critic_stats = {'bg_ratio': [], 'avg_sim': []} if feature_critic is not None else None

        with torch.no_grad():
            for bag_idx, bag in enumerate(batch_bags):
                tiles_np = bag['tiles']  # List[np.ndarray]
                class_id = bag['class_id']
                class_ids.append(class_id)

                # Check for precomputed features (from offline positive pool)
                precomputed_features = bag.get('precomputed_features', None)

                # Apply augmentation
                tiles_augmented = []
                for tile_np in tiles_np:
                    transformed = train_transform(image=tile_np)

                    # Handle both Tensor and numpy array outputs
                    if isinstance(transformed['image'], torch.Tensor):
                        # Already a tensor from ToTensorV2 (C, H, W, float, 0-1)
                        tile_tensor = transformed['image']
                    else:
                        # Numpy array (H, W, C, uint8, 0-255)
                        tile_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float() / 255.0

                    tiles_augmented.append(tile_tensor)

                tiles_tensor = torch.stack(tiles_augmented).to(device)  # (N, C, H, W)

                # Cache augmented tiles (if enabled)
                if tile_cache is not None:
                    tile_cache.store(bag_idx, tiles_tensor)

                # Scout inference (in mini-batches to save memory)
                all_scores = []
                all_outputs = []  # Store full outputs for Feature Critic
                for i in range(0, len(tiles_tensor), scout_batch_size):
                    batch_tiles = tiles_tensor[i:i + scout_batch_size]
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        outputs = model.predict_instances(batch_tiles)  # (batch, num_classes+1)
                    all_outputs.append(outputs)
                    scores = outputs[:, class_id]  # Class-specific scores
                    all_scores.append(scores)

                all_scores = torch.cat(all_scores, dim=0)  # (N,)
                all_outputs = torch.cat(all_outputs, dim=0)  # (N, num_classes+1)

                # === Feature Critic Intervention (Phase-Configurable) ===
                # DESIGN RATIONALE (2026-01-16 Updated):
                #   External prior most useful when model is weak (Warmup phase).
                #   apply_phase configured in feature_critic.runtime.apply_phase
                #
                # FEATURE DRIFT SOLUTION (2026-01-29):
                #   Use precomputed features from offline pool (same backbone as prototypes)
                #   to avoid feature drift between prototype space and training model space.
                should_apply_critic = False
                if feature_critic is not None and feature_critic.loaded:
                    fc_phase = feature_critic.apply_phase  # "warmup" | "stable" | "both"
                    should_apply_critic = (
                        fc_phase == 'both' or
                        (fc_phase == 'warmup' and is_warmup) or
                        (fc_phase == 'stable' and not is_warmup)
                    )
                if should_apply_critic:
                    # Determine feature source: precomputed (preferred) or online extraction (fallback)
                    if precomputed_features is not None:
                        # Use precomputed features (zero drift, zero VRAM overhead)
                        # Shape: (N, D) -> (1, N, D) for Feature Critic
                        features = torch.from_numpy(precomputed_features).float().to(device)
                        features = features.unsqueeze(0)  # (1, N, D)
                    else:
                        # Fallback: Online feature extraction (has drift, uses VRAM)
                        # This path is used when:
                        #   1. Online mode (positive_pool.enable=false)
                        #   2. Pool built without store_features=true
                        features = model.extract_features(tiles_tensor.unsqueeze(0))  # (1, N, D)

                    # Apply Feature Critic
                    # Reshape outputs to (1, N, num_classes) for critic
                    outputs_reshaped = all_outputs.unsqueeze(0)  # (1, N, num_classes)
                    filtered_outputs, critic_stats = feature_critic(outputs_reshaped, features)

                    # Extract filtered outputs back to (N, num_classes)
                    all_outputs = filtered_outputs.squeeze(0)

                    # Update class-specific scores after filtering
                    all_scores = all_outputs[:, class_id]

                    # Accumulate stats
                    if epoch_critic_stats is not None:
                        epoch_critic_stats['bg_ratio'].append(critic_stats['background_like_ratio'])
                        epoch_critic_stats['avg_sim'].append(critic_stats['avg_similarity'])

                # === Top-K Selection Strategy ===
                # Priority: Scale-Diversity > Spatial NMS > Plain Top-K
                tile_infos = bag.get('tile_infos', None)

                if enable_scale_diversity and tile_infos is not None and len(tile_infos) > 0:
                    # Scale-Diversity Top-K (NEW 2026-02-08)
                    # Penalizes same-scale tiles to encourage multi-scale representation in Top-K
                    scores_np = all_scores.cpu().numpy()
                    diverse_indices = scale_diverse_topk(
                        tile_infos, scores_np, k=K, penalty=scale_diversity_penalty
                    )
                    topk_indices = torch.tensor(diverse_indices, device=all_scores.device)

                elif enable_nms and tile_infos is not None and len(tile_infos) > 0:
                    # Spatial NMS (legacy, kept for ablation)
                    scores_np = all_scores.cpu().numpy()
                    nms_indices = spatial_nms_from_tile_infos(
                        tile_infos, scores_np, iou_threshold=nms_iou_threshold
                    )
                    nms_indices_tensor = torch.tensor(nms_indices, device=all_scores.device)
                    all_scores_filtered = all_scores[nms_indices_tensor]
                    k_actual = min(K, len(all_scores_filtered))
                    topk_values, topk_in_filtered = all_scores_filtered.topk(k_actual)
                    topk_indices = nms_indices_tensor[topk_in_filtered]

                else:
                    # Plain Top-K (online mode or all diversity disabled)
                    k_actual = min(K, len(all_scores))
                    topk_values, topk_indices = all_scores.topk(k_actual)

                topk_indices_list.append(topk_indices.cpu())

        # ========== Sample Negative Tiles ==========
        # Asymmetric sampling strategy: Controlled ratio (positive bags : negative tiles)
        # Rationale:
        #   - Positive: N_bags × K tiles (K tiles from each of N_bags different disease classes)
        #   - Negative: Configurable (not per-bag)
        #   - Avoids artificially inflating Class 0 (background) weight
        #   - Balances background suppression and GPU utilization
        if is_warmup:
            # Warmup: configurable negative count (default 4)
            num_neg_total = mil_cfg.get('warmup_neg_tiles', 4)
        else:
            # Stable: configurable negative count (default 10)
            num_neg_total = mil_cfg.get('stable_neg_tiles', 10)

        # Sample negative tiles (v2.0: multi-scale proportional sampling)
        # Mosaic Augmentation removed - replaced by real multi-scale tiles
        neg_tiles_np, neg_tile_indices = negative_pool.sample(num_neg_total, hard_ratio=hard_ratio)

        # Convert negative tiles to tensor
        neg_tiles_list = []
        for tile_np in neg_tiles_np:
            # Apply same augmentation
            transformed = train_transform(image=tile_np)

            # Handle both Tensor and numpy array outputs
            if isinstance(transformed['image'], torch.Tensor):
                tile_tensor = transformed['image']
            else:
                tile_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float() / 255.0

            neg_tiles_list.append(tile_tensor)

        neg_tiles_tensor = torch.stack(neg_tiles_list).to(device)  # (num_neg_total, C, H, W)

        # ========== Phase 3: SNIPE (Training on selected tiles) ==========
        model.train()  # CRITICAL: Switch back to train mode

        # [2026-02-02] Anti-Collapse: Freeze BN layers in Snipe Pass
        # Rationale: Scout uses eval() (global stats), Snipe uses train() (batch stats).
        # The Snipe batch is a "Franken-batch" (high-activation patches from different images),
        # whose statistics differ drastically from natural images.
        # Freezing BN ensures Scout-selected scores remain consistent in Snipe.
        if freeze_bn_in_snipe:
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.eval()

        # Collect all Top-K tiles
        all_topk_tiles = []
        all_topk_labels = []
        all_is_top1 = []  # Mark Top-1 tiles for TopKAnchoredMILLoss

        for bag_idx, bag in enumerate(batch_bags):
            topk_indices = topk_indices_list[bag_idx].to(device)

            if tile_cache is not None:
                # Use cached augmented tiles (consistency guaranteed!)
                topk_tiles = tile_cache.get_topk(bag_idx, topk_indices, device=device)
            else:
                # Re-augment (inconsistent with Scout, not recommended)
                tiles_np = bag['tiles']
                tiles_augmented = []
                for idx in topk_indices.cpu().tolist():
                    transformed = train_transform(image=tiles_np[idx])

                    # Handle both Tensor and numpy array outputs
                    if isinstance(transformed['image'], torch.Tensor):
                        tile_tensor = transformed['image']
                    else:
                        tile_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float() / 255.0

                    tiles_augmented.append(tile_tensor)
                topk_tiles = torch.stack(tiles_augmented).to(device)

            all_topk_tiles.append(topk_tiles)
            all_topk_labels.extend([bag['class_id']] * len(topk_indices))

            # Mark Top-1 tile (first tile in topk_indices)
            # Top-1 = highest Scout confidence → classification anchor for TopKAnchoredMILLoss
            is_top1_flags = [True] + [False] * (len(topk_indices) - 1)
            all_is_top1.extend(is_top1_flags)

        # Concatenate positive and negative tiles
        batch_tiles = torch.cat(all_topk_tiles + [neg_tiles_tensor], dim=0)  # (N_pos + N_neg, C, H, W)
        batch_labels = torch.tensor(all_topk_labels + [0] * num_neg_total, device=device)  # Class 0 for negatives

        # Concatenate is_top1 flags (negative tiles are marked as False)
        is_top1 = torch.tensor(all_is_top1 + [False] * num_neg_total, device=device)

        # Register batch mapping for hard negative mining
        # CRITICAL: Use global_sample_offset to create unique keys across all batches
        # Without offset, each batch's positions [0,1,2...] would overwrite previous batch's mapping
        if hard_mining_enable:
            num_pos = len(all_topk_labels)
            # Global positions = offset + local positions within batch
            neg_global_positions = list(range(
                global_sample_offset + num_pos,
                global_sample_offset + num_pos + num_neg_total
            ))
            negative_pool.register_batch_mapping(neg_tile_indices, neg_global_positions)

        # Forward pass with AMP
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model.predict_instances(batch_tiles)  # (N, num_classes+1)

            # Pass is_top1 and is_warmup flags to loss function
            # TopKAnchoredMILLoss requires both parameters
            # Standard losses (CE, Focal, etc.) will ignore these parameters
            if hasattr(criterion, 'forward') and 'is_top1' in criterion.forward.__code__.co_varnames:
                loss = criterion(outputs, batch_labels, is_top1=is_top1, is_warmup=is_warmup)
            elif hasattr(criterion, 'forward') and 'is_warmup' in criterion.forward.__code__.co_varnames:
                loss = criterion(outputs, batch_labels, is_warmup=is_warmup)
            else:
                loss = criterion(outputs, batch_labels)

        # Defense Layer: Loss NaN Check (detect numerical instability)
        if not torch.isfinite(loss):
            print(f"\n[ERROR] Loss became NaN/Inf at Epoch {epoch}, Batch {batch_idx}")
            print(f"  Outputs range: [{outputs.min():.2f}, {outputs.max():.2f}]")
            print(f"  Labels (first 10): {batch_labels[:10].tolist()}")
            print(f"  Skipping this batch to prevent gradient corruption...")

            # Skip this batch - do not update parameters
            optimizer.zero_grad()
            del batch_bags, all_topk_tiles, batch_tiles, batch_labels, outputs
            del topk_indices_list, neg_tiles_tensor
            if tile_cache is not None:
                tile_cache.clear()
            continue  # Jump to next batch

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping
        if gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Compute metrics
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == batch_labels).sum().item()
        total += len(batch_labels)

        # Record negative sample predictions for hard negative mining
        # CRITICAL: Convert batch-local indices to global indices using offset
        if hard_mining_enable:
            # Extract negative sample predictions
            neg_mask = (batch_labels == 0)
            neg_local_indices = torch.where(neg_mask)[0].cpu().numpy()
            neg_predictions = preds[neg_mask].cpu().numpy()
            # Convert to global indices matching register_batch_mapping keys
            neg_global_indices = [global_sample_offset + idx for idx in neg_local_indices]
            negative_pool.record_prediction(neg_global_indices, neg_predictions)

        # [2026-02-06] Collect batch diagnostic snapshot (before del)
        with torch.no_grad():
            probs = F.softmax(outputs, dim=1)
            pos_mask_diag = (batch_labels != 0)
            neg_mask_diag = (batch_labels == 0)
            num_pos_diag = pos_mask_diag.sum().item()
            num_neg_diag = neg_mask_diag.sum().item()

            last_batch_diag = {
                'num_pos': num_pos_diag,
                'num_neg': num_neg_diag,
            }

            if num_neg_diag > 0:
                neg_correct_diag = (preds[neg_mask_diag] == 0).sum().item()
                neg_class0_conf = probs[neg_mask_diag, 0].mean().item()
                # max disease prob (same metric as validation's neg_conf)
                neg_disease_conf = probs[neg_mask_diag, 1:].max(dim=-1)[0].mean().item()
                last_batch_diag['neg_correct'] = neg_correct_diag
                last_batch_diag['neg_class0_conf'] = neg_class0_conf
                last_batch_diag['neg_disease_conf'] = neg_disease_conf

            if num_pos_diag > 0:
                pos_correct_diag = (preds[pos_mask_diag] == batch_labels[pos_mask_diag]).sum().item()
                pos_pred_class0 = (preds[pos_mask_diag] == 0).sum().item()
                pos_target_conf = probs[pos_mask_diag].gather(
                    1, batch_labels[pos_mask_diag].unsqueeze(1)
                ).mean().item()
                pos_class0_conf = probs[pos_mask_diag, 0].mean().item()
                last_batch_diag['pos_correct'] = pos_correct_diag
                last_batch_diag['pos_pred_class0'] = pos_pred_class0
                last_batch_diag['pos_target_conf'] = pos_target_conf
                last_batch_diag['pos_class0_conf'] = pos_class0_conf

        # Update progress bar
        postfix_dict = {
            'loss': f'{total_loss / (batch_idx + 1):.4f}',
            'K': K,
            'S': f'{num_pos_diag}+{num_neg_diag}',
        }

        # Per-batch P/N accuracy
        if num_pos_diag > 0:
            postfix_dict['P%'] = f'{100.*last_batch_diag.get("pos_correct",0)/num_pos_diag:.0f}'
        if num_neg_diag > 0:
            postfix_dict['N%'] = f'{100.*last_batch_diag.get("neg_correct",0)/num_neg_diag:.0f}'

        # Stable phase: Add tier statistics
        if not is_warmup and hasattr(criterion, 'last_tier1_count'):
            t1 = criterion.last_tier1_count
            t2 = criterion.last_tier2_count
            t3 = criterion.last_tier3_count
            if t1 + t2 + t3 > 0:
                postfix_dict['T1/T2/T3'] = f'{t1}/{t2}/{t3}'

        # Pos/neg loss ratio
        if hasattr(criterion, 'last_pos_loss'):
            pos_l = criterion.last_pos_loss
            neg_l = getattr(criterion, 'last_neg_loss', 0)
            if neg_l > 1e-8:
                postfix_dict['P/N'] = f'{pos_l / neg_l:.2f}'

        pbar.set_postfix(postfix_dict)

        # Update global sample offset for next batch (CRITICAL for hard mining index mapping)
        # batch_size = num_positive_tiles + num_negative_tiles
        batch_size = len(all_topk_labels) + num_neg_total
        global_sample_offset += batch_size

        # Memory Cleanup
        # Explicitly delete large temporary variables to free RAM
        del batch_bags, all_topk_tiles, batch_tiles, batch_labels, outputs
        del topk_indices_list, neg_tiles_tensor
        if tile_cache is not None:
            tile_cache.clear()  # Clear cache at end of batch

        # Trigger garbage collection every 10 batches to prevent RAM accumulation
        if (batch_idx + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    avg_loss = total_loss / num_batches
    avg_acc = correct / total

    # Log Feature Critic statistics (if enabled and applied)
    critic_metrics = {}
    if epoch_critic_stats is not None and len(epoch_critic_stats['bg_ratio']) > 0:
        critic_metrics['train/critic_bg_suppressed_ratio'] = np.mean(epoch_critic_stats['bg_ratio'])
        critic_metrics['train/critic_avg_similarity'] = np.mean(epoch_critic_stats['avg_sim'])

        if logger is not None:
            logger.log(critic_metrics, step=epoch)

    # ========== [DIAGNOSTIC] Epoch 级别诊断 (last batch snapshot) ==========
    print(f"\n[DIAG] Epoch {epoch} Summary (last batch snapshot):")
    print(f"  Phase: {'WARMUP' if is_warmup else 'STABLE'}, K={K}")
    d = last_batch_diag
    n_pos, n_neg = d['num_pos'], d['num_neg']
    print(f"  Samples: pos={n_pos}, neg={n_neg}")
    if n_neg > 0:
        nc = d['neg_correct']
        print(f"  Neg ACC: {100.*nc/n_neg:.1f}% (correct={nc}/{n_neg})")
        print(f"  Neg→Class0 Conf: {d['neg_class0_conf']:.2f} (bg prob, higher=better)")
        print(f"  Neg→Disease Halluc: {d['neg_disease_conf']:.2f} (max disease prob on neg, lower=better, P0 threshold < 0.30)")
    if n_pos > 0:
        pc = d['pos_correct']
        p0 = d['pos_pred_class0']
        print(f"  Pos ACC: {100.*pc/n_pos:.1f}% (correct={pc}/{n_pos})")
        print(f"  Pos→Class0 Pred: {p0}/{n_pos} ({100.*p0/n_pos:.1f}% misclassified as background)")
        print(f"  Pos→Target Conf: {d['pos_target_conf']:.2f}")
        print(f"  Pos→Class0 Conf: {d['pos_class0_conf']:.2f}")
    if hasattr(criterion, 'last_pos_loss') and hasattr(criterion, 'last_neg_loss'):
        pos_l = criterion.last_pos_loss
        neg_l = criterion.last_neg_loss
        print(f"  Loss: pos={pos_l:.4f}, neg={neg_l:.4f}")
    if epoch_critic_stats is not None and len(epoch_critic_stats['bg_ratio']) > 0:
        avg_bg = np.mean(epoch_critic_stats['bg_ratio'])
        print(f"  Feature Critic: avg_bg_suppressed={avg_bg:.1%}")
    # ========== [END DIAGNOSTIC] ==========

    # Collect epoch-level tier statistics (Stable phase only)
    epoch_tier_stats = {}
    if not is_warmup and hasattr(criterion, 'last_tier1_count'):
        epoch_tier_stats['tier1_count'] = criterion.last_tier1_count
        epoch_tier_stats['tier2_count'] = criterion.last_tier2_count
        epoch_tier_stats['tier3_count'] = criterion.last_tier3_count
    if hasattr(criterion, 'current_dynamic_weight'):
        epoch_tier_stats['dynamic_weight'] = criterion.current_dynamic_weight

    # [2026-02-02] Always log pos/neg loss ratio (both phases)
    if hasattr(criterion, 'last_pos_loss') and hasattr(criterion, 'last_neg_loss'):
        epoch_tier_stats['last_pos_loss'] = criterion.last_pos_loss
        epoch_tier_stats['last_neg_loss'] = criterion.last_neg_loss

    # [2026-02-02] Anti-Collapse: Monitor key indicators
    if monitor_collapse and not is_warmup:
        tier1 = epoch_tier_stats.get('tier1_count', 0)
        tier2 = epoch_tier_stats.get('tier2_count', 0)
        tier3 = epoch_tier_stats.get('tier3_count', 0)
        total_top1 = tier1 + tier2 + tier3

        if total_top1 > 0:
            tier1_ratio = tier1 / total_top1
            epoch_tier_stats['tier1_ratio'] = tier1_ratio

            # Collapse Warning: Tier 1 ratio < 30% indicates model losing confidence
            if tier1_ratio < 0.3:
                print(f"\n[COLLAPSE WARNING] Epoch {epoch}: Tier 1 ratio = {tier1_ratio:.1%} < 30%")
                print(f"  Model is losing qualified Top-1 predictions!")
                print(f"  Tier distribution: T1={tier1}, T2={tier2}, T3={tier3}")

        # Pos/Neg loss ratio (if available)
        if hasattr(criterion, 'last_pos_loss') and hasattr(criterion, 'last_neg_loss'):
            pos_loss = criterion.last_pos_loss
            neg_loss = criterion.last_neg_loss
            if neg_loss > 0:
                pos_neg_ratio = pos_loss / neg_loss
                epoch_tier_stats['pos_neg_loss_ratio'] = pos_neg_ratio

                # Collapse Warning: Pos loss << Neg loss indicates over-suppression
                if pos_neg_ratio < 0.5:
                    print(f"\n[COLLAPSE WARNING] Epoch {epoch}: Pos/Neg loss ratio = {pos_neg_ratio:.2f} < 0.5")
                    print(f"  Model may be over-suppressing positive predictions!")
                    print(f"  Pos Loss: {pos_loss:.4f}, Neg Loss: {neg_loss:.4f}")

    return avg_loss, avg_acc, epoch_tier_stats, last_batch_diag


@torch.no_grad()
def validate(
    model: nn.Module,
    val_dataset,  # AsymmetricMILDataset
    val_negative_pool: NegativeTilePool,
    device: torch.device,
    config: Dict,
    max_samples: int = None,
    criterion=None,  # NEW: Training loss function for aligned loss calculation
    is_warmup: bool = True  # NEW: Current training phase
) -> Dict[str, float]:
    """
    Validate model using tile-level classification with random sampling.

    PURPOSE: Quick trend monitoring (does NOT affect training decisions).
    For strict quality check that determines warmup termination, see WarmupEvaluator.

    Loss Calculation Strategy (2026-01-21 Update):
        - 'loss': CE Loss baseline (always computed, for reference)
        - 'loss_aligned': Training-aligned loss (if criterion provided)
          Uses same loss function as training, enabling direct comparison

    Sampling Strategy:
        - Random sampling (fixed seed per image_path for reproducibility)
        - Provides general trend overview
        - Fast execution (no Scout forward needed)

    Sampling Configuration (from config['training']):
        - val_max_bags: Number of positive bags to sample (default: 200)
        - val_tiles_per_bag: Tiles randomly sampled per bag (default: 8)
        - val_neg_tiles_ratio: Negative:Positive tile ratio (default: 1.0)
        - val_bags_per_batch: Batch size for DataLoader (default: 8)
        - val_tile_batch_size: Mini-batch size for inference (default: 32)

    Args:
        model: Model to validate
        val_dataset: Validation dataset
        val_negative_pool: Validation negative tile pool (for Class 0 samples)
        device: Device
        config: Configuration dict
        max_samples: Maximum number of bags (deprecated, use config instead)
        criterion: Training loss function (optional). If provided, computes aligned loss.
        is_warmup: Current training phase flag for criterion

    Returns:
        Dictionary with validation metrics:
            - 'loss': Average CE loss (baseline reference)
            - 'loss_aligned': Average aligned loss (matches training criterion, if provided)
            - 'accuracy': Tile-level accuracy (including negative tiles)
            - 'neg_disease_hallucination': Avg max disease probability on negative tiles (range: [0, 1])
              Lower is better: < 0.3 means model doesn't hallucinate diseases on backgrounds
            - 'negative_recall': Fraction of negative tiles correctly predicted as Class 0
              Higher is better: > 0.8 means model recognizes backgrounds
    """
    model.eval()

    use_amp = config.get('training', {}).get('use_amp', True)
    ce_criterion = nn.CrossEntropyLoss()  # CE baseline (always computed)

    # Read Sampling Configuration from yaml
    dataset_cfg = config.get('dataset', {})
    training_cfg = config.get('training', {})

    # Positive bag sampling
    val_max_bags = training_cfg.get('val_max_bags', 200)
    if max_samples is not None:
        print(f"[VALIDATE] Warning: max_samples parameter is deprecated, using config instead")
        val_max_bags = max_samples

    # Tile sampling
    tiles_per_bag = training_cfg.get('val_tiles_per_bag', 8)

    # Negative tile sampling
    neg_tiles_ratio = training_cfg.get('val_neg_tiles_ratio', 1.0)

    # DataLoader configuration
    val_bags_per_batch = training_cfg.get('val_bags_per_batch', 8)
    tile_batch_size = training_cfg.get('val_tile_batch_size', 32)
    num_workers = min(dataset_cfg.get('num_workers', 6), 4)  # Use fewer workers for validation
    tile_size = dataset_cfg.get('final_tile_size', 384)  # Target tile size for model input

    total_ce_loss = 0.0  # CE baseline loss (always computed)
    total_aligned_loss = 0.0  # Aligned loss (if criterion provided)
    correct = 0
    total = 0

    # Track negative tile metrics (Phase 1 evaluation)
    neg_hallucinations = []
    neg_correct = 0
    neg_total = 0

    # Track positive tile metrics (Phase 2 evaluation)
    all_pos_preds = []
    all_pos_labels = []
    all_pos_probs = []
    pos_foreground_count = 0  # Count tiles predicted as non-background (Class 1-9)

    # Create Validation DataLoader
    # Sample subset of validation data (with fixed seed for consistency)
    num_bags = min(len(val_dataset), val_max_bags)

    # CRITICAL: Use fixed seed to ensure same validation subset across epochs
    # This makes validation metrics comparable across epochs
    val_rng = np.random.RandomState(config.get('training_strategy', {}).get('seed', 42))
    sampled_indices = val_rng.choice(len(val_dataset), num_bags, replace=False)

    from torch.utils.data import Subset
    from src.data import mil_collate_fn  # Import global collate_fn (pickle-safe)

    val_subset = Subset(val_dataset, sampled_indices)

    val_loader = DataLoader(
        val_subset,
        batch_size=val_bags_per_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=mil_collate_fn
    )

    # Get validation transforms
    from src.data import get_validation_transforms
    val_transform = get_validation_transforms(img_size=tile_size)

    # Sample negative tiles (once, before loop)
    n_pos_tiles_total = num_bags * tiles_per_bag
    n_neg_tiles_total = int(n_pos_tiles_total * neg_tiles_ratio)
    neg_tiles_np, _ = val_negative_pool.sample(n_neg_tiles_total, hard_ratio=0.0)  # Pure random for validation

    # Process validation batches
    pbar = tqdm(val_loader, desc="Validating", total=len(val_loader))

    for batch_bags in pbar:
        # Process positive bags
        batch_tiles_list = []
        batch_labels_list = []

        for bag in batch_bags:
            tiles_np = bag['tiles']  # List[np.ndarray]
            class_id = bag['class_id']
            image_path = bag['image_path']

            # CRITICAL: Use hash(image_path) as seed for reproducible sampling
            # This ensures same tiles are sampled for each image across epochs
            bag_seed = hash(image_path) % (2**32)
            bag_rng = np.random.RandomState(bag_seed)
            sampled_indices = bag_rng.choice(len(tiles_np), min(tiles_per_bag, len(tiles_np)), replace=False)

            for idx in sampled_indices:
                tile_np = tiles_np[idx]
                transformed = val_transform(image=tile_np)

                # Handle both Tensor and numpy array outputs
                if isinstance(transformed['image'], torch.Tensor):
                    tile_tensor = transformed['image']
                else:
                    tile_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float() / 255.0

                batch_tiles_list.append(tile_tensor)
                batch_labels_list.append(class_id)

                # Track positive samples
                all_pos_labels.append(class_id)

        # Concatenate positive tiles
        if len(batch_tiles_list) > 0:
            pos_tiles = torch.stack(batch_tiles_list).to(device)
            pos_labels = torch.tensor(batch_labels_list, device=device)
        else:
            continue

        # Forward pass on positive tiles (in mini-batches)
        for i in range(0, len(pos_tiles), tile_batch_size):
            batch_tiles = pos_tiles[i:i + tile_batch_size]
            batch_labels = pos_labels[i:i + tile_batch_size]

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model.predict_instances(batch_tiles)

            # CE Loss (baseline, always computed)
            ce_loss = ce_criterion(outputs, batch_labels)
            total_ce_loss += ce_loss.item() * len(batch_labels)

            # Aligned Loss (if criterion provided)
            # Note: TopKAnchoredMILLoss requires is_top1 mask which is not available in validation
            # (no Scout Pass). Use CE loss as aligned loss for validation consistency.
            if criterion is not None:
                try:
                    # Try calling with is_warmup (works for simple losses)
                    aligned_loss = criterion(outputs, batch_labels, is_warmup=is_warmup)
                    total_aligned_loss += aligned_loss.item() * len(batch_labels)
                except (TypeError, ValueError):
                    # Fallback to CE loss for TopKAnchoredMILLoss or other complex losses
                    total_aligned_loss += ce_loss.item() * len(batch_labels)

            preds = outputs.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += len(batch_labels)

            # Track positive predictions and probabilities
            all_pos_preds.extend(preds.cpu().numpy())
            probs = F.softmax(outputs, dim=-1)
            all_pos_probs.append(probs.cpu().numpy())

            # Count foreground predictions (pred != 0)
            pos_foreground_count += (preds != 0).sum().item()

        pbar.set_postfix({
            'pos_loss': f'{total_ce_loss / total:.4f}' if total > 0 else '0.0000',
            'acc': f'{100. * correct / total:.2f}%' if total > 0 else '0.00%'
        })

        # Cleanup positive tiles after processing
        del pos_tiles, pos_labels, batch_tiles_list, batch_labels_list

    # Memory cleanup after positive tile processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Process negative tiles in mini-batches (memory-efficient)
    # Instead of stacking all tiles at once, process in chunks
    neg_tiles_to_process = neg_tiles_np[:n_neg_tiles_total]

    for batch_start in range(0, len(neg_tiles_to_process), tile_batch_size):
        batch_end = min(batch_start + tile_batch_size, len(neg_tiles_to_process))
        batch_neg_tiles_np = neg_tiles_to_process[batch_start:batch_end]

        # Transform and stack only current batch
        batch_tiles_list = []
        for tile_np in batch_neg_tiles_np:
            transformed = val_transform(image=tile_np)
            if isinstance(transformed['image'], torch.Tensor):
                tile_tensor = transformed['image']
            else:
                tile_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float() / 255.0
            batch_tiles_list.append(tile_tensor)

        if len(batch_tiles_list) == 0:
            continue

        batch_tiles = torch.stack(batch_tiles_list).to(device)
        batch_labels = torch.zeros(len(batch_tiles), dtype=torch.long, device=device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model.predict_instances(batch_tiles)

        # CE Loss (baseline)
        ce_loss = ce_criterion(outputs, batch_labels)
        total_ce_loss += ce_loss.item() * len(batch_labels)

        # Aligned Loss (if criterion provided)
        # Note: TopKAnchoredMILLoss requires is_top1 mask which is not available in validation
        # Use CE loss as aligned loss for validation consistency.
        if criterion is not None:
            try:
                aligned_loss = criterion(outputs, batch_labels, is_warmup=is_warmup)
                total_aligned_loss += aligned_loss.item() * len(batch_labels)
            except (TypeError, ValueError):
                # Fallback to CE loss for TopKAnchoredMILLoss or other complex losses
                total_aligned_loss += ce_loss.item() * len(batch_labels)

        preds = outputs.argmax(dim=1)
        correct += (preds == batch_labels).sum().item()
        total += len(batch_labels)

        # Track negative tile metrics
        probs = F.softmax(outputs, dim=-1)
        max_disease_prob = probs[:, 1:].max(dim=-1)[0]  # Max probability of disease classes
        neg_hallucinations.extend(max_disease_prob.cpu().numpy())

        neg_correct += (preds == 0).sum().item()
        neg_total += len(batch_labels)

        # Cleanup batch tensors
        del batch_tiles, batch_labels, outputs, probs
        if (batch_start // tile_batch_size) % 5 == 0:
            torch.cuda.empty_cache()

    # Update tqdm with final loss (pos+neg combined)
    if total > 0:
        pbar.set_postfix({
            'loss': f'{total_ce_loss / total:.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    # Compute final metrics
    avg_ce_loss = total_ce_loss / total if total > 0 else 0.0
    avg_aligned_loss = total_aligned_loss / total if total > 0 and criterion is not None else None
    avg_acc = correct / total if total > 0 else 0.0

    # Positive foreground ratio (tiles predicted as non-background / total positive tiles)
    pos_total = len(all_pos_preds)
    foreground_ratio = pos_foreground_count / pos_total if pos_total > 0 else 0.0

    # Negative metrics
    avg_neg_hallucination = np.mean(neg_hallucinations) if len(neg_hallucinations) > 0 else 0.0
    neg_recall = neg_correct / neg_total if neg_total > 0 else 0.0

    # Cleanup before return (prevent memory accumulation)
    del neg_tiles_to_process, neg_tiles_np
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    result = {
        'loss': avg_ce_loss,  # CE baseline (for reference)
        'accuracy': avg_acc,
        'foreground_ratio': foreground_ratio,  # Disease detection sensitivity
        'neg_disease_hallucination': avg_neg_hallucination,
        'negative_recall': neg_recall
    }

    # Add aligned loss if criterion was provided
    if avg_aligned_loss is not None:
        result['loss_aligned'] = avg_aligned_loss

    return result


__all__ = ['train_one_epoch', 'validate']
