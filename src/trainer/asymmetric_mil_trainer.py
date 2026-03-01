"""
Asymmetric MIL Trainer - Orchestrates Scout-Snipe training pipeline.

Recent Updates:
    - [2026-01-15] Major Refactor: God Class elimination
        - Extracted TrainerBuilder (component construction)
        - Extracted TrainingState (state management)
        - Reduced from 1038 lines to ~450 lines
        - Trainer now focuses ONLY on training loop orchestration

Key Features:
    - Scout-Snipe Training Loop (model.eval → model.train)
    - Warmup-to-Stable Phase Transition (P0 Criteria)
    - Hard Negative Mining Lifecycle
    - Concept Drift Monitoring (Stable Phase)
    - Checkpoint Management (Best Model + Regular)

Architecture (After Refactor):
    1. TrainerBuilder: Constructs all components
    2. TrainingState: Manages phase state and transitions
    3. AsymmetricMILTrainer (this class): Orchestrates training loop

Usage:
    >>> from src.trainer import AsymmetricMILTrainer
    >>> trainer = AsymmetricMILTrainer(config)
    >>> trainer.fit()
"""

import gc
import torch
from typing import Dict, Any, Optional

# Import builders and state
from src.trainer.trainer_builder import TrainerBuilder
from src.trainer.training_state import TrainingState

# Import engines
from src.trainer.engines import train_one_epoch, validate

# Import callbacks
from src.trainer.callbacks import (
    CheckpointCallback,
    DriftMonitorCallback,
    HeatmapCallback
)


class AsymmetricMILTrainer:
    """
    Asymmetric MIL Trainer - Simplified orchestrator after God Class refactor.

    Responsibilities (ONLY):
        - Execute training loop (fit() method)
        - Coordinate components built by TrainerBuilder
        - Delegate state management to TrainingState
        - Trigger callbacks at appropriate points
        - Handle checkpoint resume

    Does NOT handle:
        - Component construction (delegated to TrainerBuilder)
        - State transitions (delegated to TrainingState)
        - LR scheduling details (encapsulated in Scheduler)
        - Evaluation details (delegated to Evaluators)

    Args:
        config: Complete configuration dict from YAML files
        resume_checkpoint: Path to checkpoint for resume training (optional)

    Example:
        >>> config = load_config('configs/algorithm/train_topk_asymmetric.yaml')
        >>> trainer = AsymmetricMILTrainer(config)
        >>> trainer.fit()
    """

    def __init__(self, config: Dict[str, Any], resume_checkpoint: Optional[str] = None):
        """
        Initialize trainer by building components and setting up state.

        Args:
            config: Complete configuration dict from YAML files
            resume_checkpoint: Path to checkpoint for resume training (optional)
        """
        self.config = config
        self.resume_checkpoint = resume_checkpoint

        # Build all components using TrainerBuilder
        print("\n[TRAINER] Initializing AsymmetricMILTrainer (Refactored v2.0)...")
        builder = TrainerBuilder(config)
        components = builder.build_all()

        # Unpack components
        self.model = components['model']
        self.optimizer = components['optimizer']
        self.scheduler = components['scheduler']
        self.criterion = components['loss_function']  # V1 naming convention
        self.feature_critic = components['feature_critic']

        self.train_dataset = components['train_dataset']
        self.val_dataset = components['val_dataset']
        self.train_sampler = components['train_sampler']
        self.val_sampler = components['val_sampler']
        self.train_dataloader = components['train_dataloader']
        self.val_dataloader = components['val_dataloader']
        self.train_negative_pool = components['train_negative_pool']
        self.val_negative_pool = components['val_negative_pool']

        self.warmup_evaluator = components['warmup_evaluator']
        self.final_evaluator = components['final_evaluator']
        self.visualizer = components['visualizer']

        self.scaler = components['scaler']
        self.logger = components['logger']
        self.checkpoint_manager = components['checkpoint_manager']

        self.device = builder.device
        self.num_classes = components['num_classes']

        # Config references for callbacks
        self.train_cfg = config.get('training', {})
        self.hard_mining_cfg = config.get('asymmetric_mil', {}).get('hard_negative_mining', {})
        self.checkpoint_root = self.logger.get_checkpoints_dir()

        # Initialize training state
        self.state = TrainingState(config)

        # Setup callbacks
        self._setup_callbacks()

        # Resume from checkpoint if provided
        if self.resume_checkpoint is not None:
            self._resume_from_checkpoint()

        print("[TRAINER] Initialization complete\n")

    @property
    def is_warmup(self) -> bool:
        """Convenience property for callback compatibility."""
        return self.state.is_warmup

    def _setup_callbacks(self):
        """Setup training callbacks for modular event handling."""
        self.callbacks = []

        # Checkpoint callback (saves best + regular checkpoints)
        # Note: CheckpointCallback accesses trainer state directly via on_epoch_end(trainer, ...)
        self.callbacks.append(CheckpointCallback())

        # Drift monitor callback (stable phase only)
        drift_cfg = self.config.get('asymmetric_mil', {}).get('concept_drift', {})
        if drift_cfg.get('enable', True):
            self.callbacks.append(DriftMonitorCallback())

        # Heatmap generation callback
        heatmap_cfg = self.config.get('evaluation', {}).get('heatmap', {})
        if heatmap_cfg.get('enable', True):
            heatmap_freq = heatmap_cfg.get('interval', 5)
            fixed_samples = heatmap_cfg.get('fixed_samples', True)
            num_samples_per_class = heatmap_cfg.get('num_samples_per_class', 3)
            use_spatial_heatmap = heatmap_cfg.get('use_spatial_heatmap', True)  # Full spatial heatmap
            self.callbacks.append(HeatmapCallback(
                frequency=heatmap_freq,
                phases=['stable'],
                fixed_samples=fixed_samples,
                samples_per_class=num_samples_per_class,
                use_spatial_heatmap=use_spatial_heatmap
            ))

    def fit(self):
        """
        Execute complete training pipeline (Warmup + Stable phases).

        Training Flow:
            1. Warmup Phase (epoch 1 to warmup_epochs)
                - K=4, hard_ratio=0.5
                - Evaluate P0 criteria at end of warmup
            2. Transition Check (at epoch warmup_epochs + 1)
                - If P0 met → Stable Phase
                - If P0 not met → Extend warmup
            3. Stable Phase
                - K=2, hard_ratio=0.8
                - Concept drift monitoring
                - Multi-scale tiling
        """
        num_epochs = self.config.get('training', {}).get('num_epochs', 30)
        mil_warmup_epochs = self.config.get('asymmetric_mil', {}).get('warmup_epochs', 8)

        print("\n" + "="*80)
        print("TRAINING START")
        print("="*80)
        print(f"Total epochs: {num_epochs}")
        print(f"Warmup epochs: {mil_warmup_epochs}")
        if self.state.hybrid_warmup_enable:
            print(f"Hybrid warmup: Backbone frozen for {self.state.freeze_backbone_epochs} epochs")
        print(f"Device: {self.device}")
        print("="*80 + "\n")

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch}/{num_epochs} - {'WARMUP' if self.state.is_warmup else 'STABLE'}")
            print(f"{'='*80}")

            # Get epoch-level state
            epoch_state = self.state.get_epoch_state(epoch)

            # Hybrid warmup: backbone freeze/unfreeze lifecycle
            if epoch_state.get('freeze_backbone', False) and not self.state.backbone_frozen:
                self._freeze_backbone()
            elif not epoch_state.get('freeze_backbone', False) and self.state.backbone_frozen:
                self._unfreeze_backbone(epoch)

            # Hard mining lifecycle: start_epoch hook
            if self.hard_mining_cfg.get('enable', False):
                self.train_negative_pool.start_epoch()

            # Train one epoch
            train_loss, train_acc, epoch_tier_stats, last_batch_diag = train_one_epoch(
                model=self.model,
                train_dataloader=self.train_dataloader,
                train_sampler=self.train_sampler,
                negative_pool=self.train_negative_pool,
                criterion=self.criterion,
                optimizer=self.optimizer,
                scaler=self.scaler,
                device=self.device,
                config=self.config,
                epoch=epoch,
                epoch_state=epoch_state,
                feature_critic=self.feature_critic,
                logger=self.logger
            )

            # Memory cleanup after training (before validation)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Validate (with aligned loss if criterion supports it)
            val_metrics = validate(
                model=self.model,
                val_dataset=self.val_dataset,
                val_negative_pool=self.val_negative_pool,
                device=self.device,
                config=self.config,
                criterion=self.criterion,  # Pass training criterion for aligned loss
                is_warmup=self.state.is_warmup  # Pass current phase
            )

            val_loss = val_metrics['loss']  # CE baseline
            val_loss_aligned = val_metrics.get('loss_aligned')  # Aligned with training criterion
            val_acc = val_metrics['accuracy']
            foreground_ratio = val_metrics['foreground_ratio']
            neg_disease_hallucination = val_metrics['neg_disease_hallucination']
            neg_recall = val_metrics['negative_recall']

            # Cleanup validation artifacts (prevent memory leak)
            del val_metrics
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Top-K quality evaluation and warmup transition check
            topk_metrics = self._evaluate_topk_and_check_transition(
                epoch, train_loss, mil_warmup_epochs
            )

            # Update learning rate
            current_lr = self._step_scheduler(epoch)

            # Build metrics dict
            metrics = self._build_metrics_dict(
                train_loss, train_acc, val_loss, val_acc,
                neg_disease_hallucination, neg_recall, foreground_ratio, current_lr, topk_metrics,
                val_loss_aligned=val_loss_aligned,
                epoch_tier_stats=epoch_tier_stats,
                last_batch_diag=last_batch_diag
            )

            # Print epoch summary
            self._print_epoch_summary(
                epoch, train_loss, train_acc, val_loss, val_acc,
                neg_recall, neg_disease_hallucination, foreground_ratio, topk_metrics, current_lr
            )

            # Store topk_metrics for callback access
            self.topk_metrics = topk_metrics

            # Increment stable epoch count (for STABLE-{count} display and loss alpha decay)
            if not self.state.is_warmup:
                self.state.increment_stable_epoch()
                # Update loss function's stable epoch counter for alpha decay
                if hasattr(self.criterion, 'epoch_in_stable'):
                    self.criterion.epoch_in_stable = self.state.stable_epoch_count

            # Hard mining lifecycle: end_epoch hook
            if self.hard_mining_cfg.get('enable', False):
                self.train_negative_pool.end_epoch(current_epoch=epoch - 1)

            # Trigger callbacks (pass self as trainer for callback to access state)
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, metrics)

            # Log to local logger
            self.logger.log(metrics, step=epoch)

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80 + "\n")

        # Final evaluation (V1-compatible: evaluate_all + save_evaluation_report)
        print("[FINAL] Running post-training evaluation...")
        final_results = self.final_evaluator.evaluate_all()
        self.final_evaluator.save_evaluation_report(final_results)
        print(f"[FINAL] Results saved to: {self.logger.run_dir / 'evaluation'}")

        # Close logger and generate training curves
        self.logger.finish()
        print("[LOGGER] Training session closed")

        return final_results

    def _evaluate_topk_and_check_transition(
        self,
        epoch: int,
        train_loss: float,
        mil_warmup_epochs: int
    ) -> Dict[str, float]:
        """
        Evaluate Top-K quality and check for Warmup→Stable transition.

        V1-compatible logic:
        - Warmup phase: Full P0 criteria evaluation via evaluate_warmup_criteria()
        - Stable phase: Simplified Top-K monitoring via evaluate_topk_quality(k)

        Args:
            epoch: Current epoch
            train_loss: Training loss
            mil_warmup_epochs: Configured warmup duration

        Returns:
            Top-K metrics dict (avg_top1_conf, topk_lift, hit_acc, topk_avg_confidence,
                               neg_avg_conf, top1_hit_acc)
        """
        topk_metrics = {}
        mil_cfg = self.config.get('asymmetric_mil', {})

        # Warmup evaluation and transition check
        if self.state.is_warmup and epoch >= mil_warmup_epochs:
            print(f"\n{'='*60}")
            print(f"[WARMUP-EVAL] Evaluating warm-up termination criteria...")
            print(f"{'='*60}")

            # Run full P0 criteria evaluation
            warmup_metrics = self.warmup_evaluator.evaluate_warmup_criteria(epoch, train_loss)

            # Extract Top-K metrics for display and logging
            # Key mapping: warmup_evaluator returns 'neg_recall' (0-1), 'neg_disease_hallucination'
            topk_metrics = {
                'avg_top1_conf': warmup_metrics.get('avg_top1_conf', 0.0),
                'topk_lift': warmup_metrics.get('topk_lift', 0.0),
                'hit_acc': warmup_metrics.get('hit_acc', 0.0),
                'topk_avg_confidence': warmup_metrics.get('topk_avg_confidence', 0.0),
                'neg_avg_conf': warmup_metrics.get('neg_disease_hallucination', 0.0),
                'top1_hit_acc': warmup_metrics.get('hit_acc', 0.0) * 100  # Convert to %
            }

            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Check warmup termination using evaluator's should_end_warmup
            if self.warmup_evaluator.should_end_warmup(epoch, warmup_metrics):
                self._switch_to_stable(epoch, warmup_metrics)

        # Stable phase: Continuous Top-K quality monitoring
        elif not self.state.is_warmup:
            print(f"\n{'='*60}")
            print(f"[TOP-K EVAL] Evaluating Top-K quality (Stable Phase, Epoch {epoch})...")
            print(f"{'='*60}")

            k_for_eval = mil_cfg.get('stable_k', 2)
            topk_avg_confidence, topk_lift, hit_acc, avg_top1_conf, _ = \
                self.warmup_evaluator.evaluate_topk_quality(k=k_for_eval)

            print(f"  Avg-Top1-Conf: {avg_top1_conf:.4f}")
            print(f"  TopK-Lift: {topk_lift:.4f}")
            print(f"  Hit-Acc: {hit_acc:.4f}")

            # Also evaluate negative recognition for display
            neg_recall, neg_conf = self.warmup_evaluator.evaluate_negative_recognition()

            topk_metrics = {
                'avg_top1_conf': avg_top1_conf,
                'topk_lift': topk_lift,
                'hit_acc': hit_acc,
                'topk_avg_confidence': topk_avg_confidence,
                'neg_avg_conf': neg_conf,
                'top1_hit_acc': hit_acc * 100  # Convert to %
            }

            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Update best model metrics (if we have metrics)
        if topk_metrics:
            hit_acc = topk_metrics.get('hit_acc', 0.0)
            topk_lift = topk_metrics.get('topk_lift', 0.0)
            is_new_best = self.state.update_best_metrics(hit_acc, topk_lift)

            if is_new_best:
                print(f"[BEST MODEL] New best! Score: {self.state.best_score:.4f} "
                      f"(hit_acc={hit_acc:.2%}, topk_lift={topk_lift:.2%})")

        return topk_metrics

    def _switch_to_stable(self, epoch: int, warmup_metrics: Dict[str, float]):
        """
        Execute Warmup → Stable phase transition.

        Args:
            epoch: Current epoch
            warmup_metrics: Metrics from warmup evaluation
        """
        mil_cfg = self.config.get('asymmetric_mil', {})
        warmup_bags_per_batch = mil_cfg.get('warmup_bags_per_batch', 7)
        stable_bags_per_batch = mil_cfg.get('stable_bags_per_batch', 11)
        stable_k = mil_cfg.get('stable_k', 2)

        print(f"\n{'='*60}")
        print(f"[PHASE SWITCH] Warmup Complete → Transitioning to Stable Phase")
        print(f"{'='*60}")
        print(f"  K: {self.state.current_k} → {stable_k}")
        print(f"  Bags per batch: {warmup_bags_per_batch} → {stable_bags_per_batch}")

        # Use hit_acc as initial_pos_acc for drift monitoring
        initial_pos_acc = warmup_metrics.get('hit_acc', 0.0)
        self.state.transition_to_stable(initial_pos_acc)

        # Update sampler batch sizes
        if hasattr(self.train_sampler, 'bags_per_batch'):
            self.train_sampler.bags_per_batch = stable_bags_per_batch
            print(f"  Train sampler updated: {stable_bags_per_batch} bags/batch")

        print(f"{'='*60}\n")

    def _step_scheduler(self, epoch: int) -> float:
        """
        Update learning rate via scheduler.

        Args:
            epoch: Current epoch (1-indexed)

        Returns:
            Current learning rate
        """
        if self.scheduler is not None:
            if hasattr(self.scheduler, 'step'):
                if 'epoch' in self.scheduler.step.__code__.co_varnames:
                    return self.scheduler.step(epoch)
                else:
                    self.scheduler.step()

        return self.optimizer.param_groups[0]['lr']

    def _build_metrics_dict(
        self,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        neg_disease_hallucination: float,
        neg_recall: float,
        foreground_ratio: float,
        current_lr: float,
        topk_metrics: Dict[str, float],
        val_loss_aligned: float = None,
        epoch_tier_stats: Dict = None,
        last_batch_diag: Dict = None
    ) -> Dict[str, float]:
        """Build complete metrics dict for logging.

        Merges all metric sources into a single dict per epoch:
        - Core training/validation metrics
        - TopK evaluation metrics
        - Loss decomposition (from criterion attributes)
        - Tier stratification (from epoch_tier_stats)
        - Batch-level diagnostics (from last_batch_diag)
        """
        metrics = {
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'neg_disease_hallucination': neg_disease_hallucination,
            'negative_recall': neg_recall,
            'foreground_ratio': foreground_ratio,
            'learning_rate': current_lr,
            **topk_metrics
        }

        # Aligned validation loss (matches training criterion)
        if val_loss_aligned is not None:
            metrics['val_loss_aligned'] = val_loss_aligned

        # Loss decomposition (from criterion attributes)
        if hasattr(self.criterion, 'last_pos_loss'):
            metrics['pos_loss'] = self.criterion.last_pos_loss
            metrics['neg_loss'] = self.criterion.last_neg_loss

        if hasattr(self.criterion, 'last_inter_loss'):
            metrics['inter_loss'] = self.criterion.last_inter_loss

        if hasattr(self.criterion, 'current_dynamic_weight'):
            metrics['dynamic_weight'] = self.criterion.current_dynamic_weight

        # Tier stratification + anti-collapse indicators (from epoch_tier_stats)
        if epoch_tier_stats:
            for key in ('tier1_count', 'tier2_count', 'tier3_count',
                        'tier1_ratio', 'pos_neg_loss_ratio'):
                if key in epoch_tier_stats:
                    metrics[key] = epoch_tier_stats[key]

        # Fallback: compute pos_neg_loss_ratio from loss decomposition if not in tier stats
        # (ensures warmup phase also has P/N ratio data for plotting)
        if 'pos_neg_loss_ratio' not in metrics:
            pos_l = metrics.get('pos_loss')
            neg_l = metrics.get('neg_loss')
            if pos_l is not None and neg_l is not None and neg_l > 0:
                metrics['pos_neg_loss_ratio'] = pos_l / neg_l

        if hasattr(self.criterion, 'last_gate_filtered_ratio'):
            metrics['gate_filtered_ratio'] = self.criterion.last_gate_filtered_ratio

        # Batch-level diagnostics (last batch snapshot from train_one_epoch)
        if last_batch_diag:
            d = last_batch_diag
            n_pos = d.get('num_pos', 0)
            n_neg = d.get('num_neg', 0)

            if n_pos > 0:
                metrics['train_pos_accuracy'] = d['pos_correct'] / n_pos
                metrics['train_pos_class0_ratio'] = d['pos_pred_class0'] / n_pos
                metrics['train_pos_target_conf'] = d['pos_target_conf']
                metrics['train_pos_class0_conf'] = d['pos_class0_conf']

            if n_neg > 0:
                metrics['train_neg_accuracy'] = d['neg_correct'] / n_neg
                metrics['train_neg_class0_conf'] = d['neg_class0_conf']
                metrics['train_neg_disease_conf'] = d.get('neg_disease_conf', 0.0)

        return metrics

    def _print_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        neg_recall: float,
        neg_disease_hallucination: float,
        foreground_ratio: float,
        topk_metrics: Dict[str, float],
        current_lr: float
    ):
        """
        Print validation diagnostics after validate().

        Training diagnostics (last batch snapshot) are printed by train_one_epoch.
        This method prints validation metrics + stable phase metrics + LR.
        """
        num_epochs = self.config.get('training', {}).get('num_epochs', 30)
        phase = "WARMUP" if self.state.is_warmup else f"STABLE-{self.state.stable_epoch_count}"

        print(f"[Epoch {epoch}/{num_epochs} {phase}] Validation:")
        print(f"  Val Loss: {val_loss:.4f}, Val ACC: {val_acc:.1%}")
        print(f"  FG Ratio: {foreground_ratio:.2f} (positive tiles predicted as disease)")
        print(f"  Neg Recall: {neg_recall:.2f}, Neg Disease Halluc: {neg_disease_hallucination:.3f}")

        if topk_metrics is not None and len(topk_metrics) > 0:
            avg_top1 = topk_metrics.get('avg_top1_conf', 0)
            topk_lift = topk_metrics.get('topk_lift', 0)
            hit_acc = topk_metrics.get('hit_acc', 0)
            print(f"  TopK: AvgTop1={avg_top1:.3f}, Lift={topk_lift:.3f}, HitAcc={hit_acc:.2f}")

        if not self.state.is_warmup:
            parts = []
            if hasattr(self.criterion, 'current_dynamic_weight'):
                parts.append(f"DynW={self.criterion.current_dynamic_weight:.2f}")
            if hasattr(self.criterion, 'last_inter_loss'):
                parts.append(f"InterL={self.criterion.last_inter_loss:.3f}")
            if parts:
                print(f"  Stable: {', '.join(parts)}")

        # Print per-group LR using scheduler base_lrs to identify groups
        # Each module produces 2 param_groups (decay + no_decay), so collect by role
        if self.scheduler is not None and hasattr(self.scheduler, 'base_lrs'):
            backbone_lr_base = self.config.get('optimizer', {}).get('backbone_lr', 1e-5)
            hybrid_lr_base = self.config.get('optimizer', {}).get('hybrid_lr', 5e-5)
            classifier_lr_base = self.config.get('optimizer', {}).get('classifier_lr', 1e-4)
            bb_lr, hy_lr, hd_lr = None, None, None
            for i, base in enumerate(self.scheduler.base_lrs):
                actual = self.optimizer.param_groups[i]['lr']
                if abs(base - backbone_lr_base) < 1e-10:
                    if bb_lr is None:
                        bb_lr = actual
                elif abs(base - hybrid_lr_base) < 1e-10:
                    if hy_lr is None:
                        hy_lr = actual
                elif abs(base - classifier_lr_base) < 1e-10:
                    if hd_lr is None:
                        hd_lr = actual
            frozen_tag = " (frozen)" if self.state.backbone_frozen else ""
            if bb_lr is not None and hy_lr is not None and hd_lr is not None:
                print(f"  LR: backbone={bb_lr:.2e}{frozen_tag}, hybrid={hy_lr:.2e}, head={hd_lr:.2e}")
            elif bb_lr is not None and hd_lr is not None:
                print(f"  LR: backbone={bb_lr:.2e}{frozen_tag}, classifier={hd_lr:.2e}")
            else:
                print(f"  LR: {current_lr:.2e}")
        else:
            print(f"  LR: {current_lr:.2e}")

    def _freeze_backbone(self):
        """
        Freeze backbone parameters for hybrid warmup.

        During hybrid warmup, backbone outputs stable features (frozen target)
        so that new components (FPN/ViT/HeatmapHead) can learn on a stable foundation.
        Only backbone parameters are frozen; FPN/ViT/Head remain trainable.
        """
        frozen_count = 0
        for param in self.model.backbone.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += 1

        self.state.backbone_frozen = True
        freeze_epochs = self.state.freeze_backbone_epochs
        print(f"\n{'='*60}")
        print(f"[HYBRID WARMUP] Backbone FROZEN for {freeze_epochs} epochs")
        print(f"  Frozen parameters: {frozen_count}")
        print(f"  Trainable: FPN, ViT, HeatmapHead (hybrid_lr)")
        print(f"{'='*60}\n")

    def _unfreeze_backbone(self, epoch: int):
        """
        Unfreeze backbone parameters after hybrid warmup period.

        Respects permanent freeze_stages from model config (stages 0, 1, 2).
        Only unfreezes stages 3-6 + stem/head that were temporarily frozen.
        Registers a mini warmup with the scheduler to gradually ramp backbone LR.

        Args:
            epoch: Current epoch (for logging)
        """
        freeze_stages = self.config.get('model', {}).get('freeze_stages', [0, 1, 2])
        unfrozen_count = 0
        kept_frozen_count = 0

        for name, param in self.model.backbone.named_parameters():
            if 'blocks.' in name:
                # Extract stage index: blocks.{stage_idx}.{rest}
                stage_idx = int(name.split('.')[1])
                if stage_idx not in freeze_stages:
                    param.requires_grad = True
                    unfrozen_count += 1
                else:
                    kept_frozen_count += 1
            else:
                # Stem (conv_stem, bn1) and head (conv_head, bn2)
                param.requires_grad = True
                unfrozen_count += 1

        self.state.backbone_frozen = False

        # Register backbone mini warmup with scheduler
        # Find backbone param_group indices (lowest LR groups = backbone)
        hybrid_warmup_cfg = self.config.get('training', {}).get('hybrid_warmup', {})
        unfreeze_warmup_epochs = hybrid_warmup_cfg.get('backbone_unfreeze_warmup_epochs', 3)
        unfreeze_start_fraction = hybrid_warmup_cfg.get('backbone_unfreeze_start_fraction', 0.1)

        if self.scheduler is not None and hasattr(self.scheduler, 'notify_backbone_unfreeze'):
            backbone_lr = self.config.get('optimizer', {}).get('backbone_lr', 1e-5)
            backbone_indices = []
            for i, pg in enumerate(self.optimizer.param_groups):
                # Backbone groups have the lowest base_lr
                if hasattr(self.scheduler, 'base_lrs') and \
                        abs(self.scheduler.base_lrs[i] - backbone_lr) < 1e-10:
                    backbone_indices.append(i)

            if backbone_indices:
                self.scheduler.notify_backbone_unfreeze(
                    epoch=epoch,
                    backbone_group_indices=backbone_indices,
                    warmup_epochs=unfreeze_warmup_epochs,
                    start_fraction=unfreeze_start_fraction
                )
                # Immediately apply initial fraction to current LR
                # (scheduler.step() won't run until end of this epoch)
                for idx in backbone_indices:
                    self.optimizer.param_groups[idx]['lr'] *= unfreeze_start_fraction

        print(f"\n{'='*60}")
        print(f"[HYBRID WARMUP] Backbone UNFROZEN at epoch {epoch}")
        print(f"  Unfrozen parameters: {unfrozen_count}")
        print(f"  Permanently frozen (stages {freeze_stages}): {kept_frozen_count}")
        print(f"  Backbone mini warmup: {unfreeze_start_fraction:.0%} → 100% over {unfreeze_warmup_epochs} epochs")
        print(f"{'='*60}\n")

    def _resume_from_checkpoint(self):
        """Load checkpoint and resume training state."""
        import os
        if not os.path.exists(self.resume_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {self.resume_checkpoint}")

        print(f"\n[RESUME] Loading checkpoint: {self.resume_checkpoint}")

        checkpoint = torch.load(self.resume_checkpoint, map_location=self.device)

        # Load model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("[RESUME] Model weights loaded")

        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("[RESUME] Optimizer state loaded")

        # Load scheduler
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("[RESUME] Scheduler state loaded")

        # Load AMP scaler
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("[RESUME] AMP scaler state loaded")

        # Load training state
        if 'training_state' in checkpoint:
            self.state.load_state_dict(checkpoint['training_state'])
            print(f"[RESUME] Training state loaded: {self.state}")

        # Load hard mining state
        if 'hard_mining_state' in checkpoint:
            hard_mining_state = checkpoint['hard_mining_state']
            if self.train_negative_pool is not None:
                self.train_negative_pool.hard_negatives = hard_mining_state.get('train_hard_negatives', {})
                self.train_negative_pool.blacklisted_tiles = hard_mining_state.get('train_blacklisted_tiles', {})
            if self.val_negative_pool is not None:
                self.val_negative_pool.hard_negatives = hard_mining_state.get('val_hard_negatives', {})
                self.val_negative_pool.blacklisted_tiles = hard_mining_state.get('val_blacklisted_tiles', {})
            print("[RESUME] Hard mining state loaded")

        resume_epoch = checkpoint.get('epoch', 0)
        print(f"[RESUME] Resuming from epoch {resume_epoch + 1}")
        print(f"[RESUME] Best model score: {self.state.best_score:.4f}\n")

    def _save_best_checkpoint_if_improved(
        self,
        epoch: int,
        topk_metrics: Dict[str, float],
        metrics: Dict[str, float]
    ):
        """Save best model checkpoint if weighted score improved."""
        current_hit_acc = topk_metrics.get('hit_acc', 0.0)
        current_topk_lift = topk_metrics.get('topk_lift', 0.0)

        # Calculate weighted score (Hit-Acc: 60%, TopK-Lift: 40%)
        current_score = 0.6 * current_hit_acc + 0.4 * current_topk_lift

        if self.state.update_best_metrics(current_hit_acc, current_topk_lift):
            print(f"\n[CHECKPOINT] New best model! Epoch {epoch}, Score: {current_score:.4f} "
                  f"(Hit-Acc: {current_hit_acc:.4f}, TopK-Lift: {current_topk_lift:.4f})")

            # Save best checkpoint
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch,
                metrics=metrics,
                config=self.config,
                training_state=self.state.state_dict(),
                filename='best_model.pth'
            )

    def _save_regular_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save regular checkpoint every N epochs (stable phase only)."""
        if self.state.is_warmup:
            return

        # Save checkpoint with zero-padded naming
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            metrics=metrics,
            config=self.config,
            training_state=self.state.state_dict(),
            filename=f'epoch_{epoch:03d}.pth'
        )

    def _monitor_concept_drift(self, val_acc: float):
        """Monitor concept drift in stable phase using TrainingState."""
        threshold = self.config.get('asymmetric_mil', {}).get('concept_drift', {}).get('threshold', 0.05)
        if self.state.check_concept_drift(val_acc, threshold):
            print(f"[WARNING] Concept drift detected! Current: {val_acc:.4f}, "
                  f"Initial: {self.state.stable_start_pos_acc:.4f}")
