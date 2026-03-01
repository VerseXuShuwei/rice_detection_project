"""
Asymmetric MIL Trainer - Core Training Engine for Scout-Snipe MIL.

Recent Updates:
    - [2026-01-05] Refactor: Extracted from train_topk_asymmetric.py (no logic changes)

Key Features:
    - Scout-Snipe Training Loop (model.eval → model.train)
    - Warmup-to-Stable Phase Transition (P0 Criteria)
    - Hard Negative Mining Lifecycle (start_epoch → end_epoch)
    - Differential Learning Rates (Backbone vs Classifier)
    - Trapezoidal LR Schedule (Warmup → Hold → Decay)
    - Concept Drift Monitoring (Stable Phase)
    - Checkpoint Management (Best Model + Regular)

Usage:
    >>> from src.trainer import AsymmetricMILTrainer
    >>> trainer = AsymmetricMILTrainer(config)
    >>> trainer.fit()
"""

import gc
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, Optional

# Import engines
from src.trainer.engines import train_one_epoch, validate

# Import builders
from src.models.builder import get_model
from src.losses.builder import create_loss_function
from src.utils.builder import create_optimizer, create_scheduler

# Import utilities
from src.utils.local_logger import init_logger
from src.utils.device import get_device
from src.core.checkpoint_manager import CheckpointManager

# Import evaluation
from src.evaluation.warmup_evaluator import WarmupEvaluator
from src.evaluation.heatmap_visualizer import MILVisualizer
from src.evaluation.final_evaluator import FinalEvaluator


class AsymmetricMILTrainer:
    """
    Asymmetric MIL Trainer for Scout-Snipe training pipeline.

    Architecture:
        1. Builder Layer: Construct components from config
        2. Engine Layer: train_one_epoch + validate functions
        3. Trainer Layer (this class): State management + orchestration

    State Machine:
        - Warmup Phase: K=4, hard_ratio=0.5
        - Stable Phase: K=2, hard_ratio=0.8

    Args:
        config: Complete configuration dict from YAML files

    Example:
        >>> config = load_config('configs/algorithm/train_topk_asymmetric.yaml')
        >>> trainer = AsymmetricMILTrainer(config)
        >>> trainer.fit()
    """

    def __init__(self, config: Dict[str, Any], resume_checkpoint: Optional[str] = None):
        """
        Initialize trainer with all components.

        Args:
            config: Complete configuration dict from YAML files
            resume_checkpoint: Path to checkpoint for resume training (optional)
        """
        self.config = config
        self.device = get_device()
        self.resume_checkpoint = resume_checkpoint

        # Extract key configs
        self.train_cfg = config.get('training', {})
        self.mil_cfg = config.get('asymmetric_mil', {})
        self.hard_mining_cfg = self.mil_cfg.get('hard_negative_mining', {})

        # Initialize state variables
        self.current_epoch = 0
        self.is_warmup = True
        self.current_k = self.mil_cfg.get('warmup_k', 4)

        # Stable phase tracking
        self.stable_start_pos_acc = None
        self.stable_epoch_count = 0

        # Best model tracking
        self.best_hit_acc = 0.0
        self.best_topk_lift = 0.0
        self.best_score = 0.0  # Weighted score: 0.6*hit_acc + 0.4*topk_lift

        # Build components
        print("\n[TRAINER] Initializing AsymmetricMILTrainer...")
        self._build_data()  # Build data first to infer num_classes
        self._build_model()  # Model needs num_classes from dataset
        self._build_optimizer_scheduler()
        self._build_loss()
        self._setup_amp()
        self._setup_logger()
        self._build_feature_critic()  # Feature Critic (after logger)
        self._setup_checkpoint_manager()
        self._build_evaluators()
        self._setup_callbacks()

        # Resume from checkpoint if provided
        if self.resume_checkpoint is not None:
            self._resume_from_checkpoint()

        print("[TRAINER] Initialization complete")

    def _build_model(self):
        """Build and verify model architecture."""
        model_name = self.config.get('model', {}).get('name', 'mil_efficientnetv2-s')
        print(f"\n[MODEL] Creating model: {model_name}")
        self.model = get_model(model_name, config=self.config)
        self.model = self.model.to(self.device)

        # Verify output dimension
        # num_classes excludes Class 0 (healthy/negative)
        num_classes = self.config.get('num_classes', 11)  # Default: 11 rice diseases
        num_output_classes = num_classes + 1  # +1 for Class 0 (healthy)

        with torch.no_grad():
            tile_size = self.config.get('dataset', {}).get('final_tile_size', 384)
            dummy_input = torch.randn(1, 3, tile_size, tile_size).to(self.device)
            dummy_output = self.model.predict_instances(dummy_input)
            actual_output_dim = dummy_output.shape[1]

            if actual_output_dim != num_output_classes:
                raise RuntimeError(
                    f"Model output dimension mismatch!\n"
                    f"  Expected: {num_output_classes} (num_classes={num_classes} + 1 for Class 0)\n"
                    f"  Actual: {actual_output_dim}"
                )

            print(f"[MODEL] Output dimension verified: {actual_output_dim}")

        # Test numerical stability
        with torch.no_grad():
            dummy_batch = torch.randn(16, 3, tile_size, tile_size).to(self.device)
            dummy_outputs = self.model.predict_instances(dummy_batch)

            output_min = dummy_outputs.min().item()
            output_max = dummy_outputs.max().item()

            print(f"[MODEL] Logits range: [{output_min:.2f}, {output_max:.2f}]")

            if abs(output_max) > 50 or abs(output_min) > 50:
                print(f"  [WARNING] Large logits detected! May cause NaN")
            else:
                print(f"  [OK] Logits within safe range")

    def _build_data(self):
        """Build datasets, samplers, and negative pools."""
        from src.data import (
            AsymmetricMILDataset,
            ClassAwareBagSampler,
            NegativeTilePool,
            AugmentedTileCache
        )
        from pathlib import Path

        print("\n[DATA] Initializing datasets and pools...")
        seed = self.config.get('training_strategy', {}).get('seed', 42)

        # Build Negative Pools
        self.train_negative_pool = NegativeTilePool(self.config, split='train', seed=seed)
        self.val_negative_pool = NegativeTilePool(self.config, split='val', seed=seed)

        # Verify pool exists
        if not self.train_negative_pool.exists():
            raise RuntimeError(
                f"Negative tile pool not found!\n"
                f"Expected path: {self.train_negative_pool.lmdb_path}\n"
                f"Please run: python scripts/tools/build_negative_pool.py --config <config_path>"
            )

        print(f"[NEGATIVE-POOL] Train pool: {len(self.train_negative_pool)} tiles")
        print(f"[NEGATIVE-POOL] Val pool: {len(self.val_negative_pool)} tiles")

        # Build Datasets (Adaptive Multi-scale)
        print("[DATASET] Creating datasets...")
        self.train_dataset = AsymmetricMILDataset(
            self.config, split='train', seed=seed
        )
        self.val_dataset = AsymmetricMILDataset(
            self.config, split='val', seed=seed
        )

        # Verify num_classes consistency
        num_classes = self.train_dataset.num_classes
        config_num_classes = self.config.get('num_classes', num_classes)
        if config_num_classes != num_classes:
            print(f"[WARN] Config num_classes ({config_num_classes}) != Dataset num_classes ({num_classes})")
            print(f"       Using dataset value: {num_classes}")
            self.config['num_classes'] = num_classes

        print(f"[DATASET] Num classes: {num_classes} (Class 1-{num_classes}, Class 0 for negative)")
        print(f"[DATASET] Train samples: {len(self.train_dataset)}")
        print(f"[DATASET] Val samples: {len(self.val_dataset)}")

        # Build Sampler
        mil_cfg = self.config.get('asymmetric_mil', {})
        warmup_bags_per_batch = mil_cfg.get('warmup_bags_per_batch', 4)

        print(f"[SAMPLER] Creating class-aware sampler (warmup: {warmup_bags_per_batch} bags/batch)...")
        self.train_sampler = ClassAwareBagSampler(
            dataset=self.train_dataset,
            bags_per_batch=warmup_bags_per_batch,
            shuffle=True
        )

        # Build Tile Cache (for Scout-Snipe consistency)
        train_cfg = self.config.get('training', {})
        cache_augmented = train_cfg.get('cache_augmented', True)
        self.tile_cache = AugmentedTileCache() if cache_augmented else None
        if self.tile_cache:
            print("[TILE-CACHE] Augmented tile cache enabled")

        # Build Persistent DataLoader (avoid recreating workers every epoch)
        from torch.utils.data import DataLoader
        from src.data import mil_collate_fn

        dataset_cfg = self.config.get('dataset', {})
        num_workers = dataset_cfg.get('num_workers', 6)
        persistent_workers = dataset_cfg.get('persistent_workers', True)
        prefetch_factor = dataset_cfg.get('prefetch_factor', 2)
        pin_memory = dataset_cfg.get('pin_memory', True)

        print(f"[DATALOADER] Creating persistent DataLoader (workers={num_workers}, persistent={persistent_workers})...")
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            num_workers=num_workers,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=pin_memory,
            collate_fn=mil_collate_fn
        )
        print(f"[DATALOADER] Persistent workers will stay alive across all epochs")

    def _build_optimizer_scheduler(self):
        """Build optimizer and scheduler with differential learning rates."""
        print("\n[OPTIMIZER-SCHEDULER] Creating optimizer and scheduler...")
        self.optimizer = create_optimizer(self.model, self.config)
        self.scheduler = create_scheduler(self.optimizer, self.config)

    def _build_loss(self):
        """Build loss function from config."""
        loss_cfg = self.config.get('loss', {})
        self.criterion = create_loss_function(loss_cfg)
        print(f"[LOSS] Using: {loss_cfg.get('type', 'cross_entropy')}")

    def _build_feature_critic(self):
        """
        Build Feature Critic for Scout Pass background suppression.

        Feature Critic is applied ONLY in Scout Pass during Stable phase.
        It uses pre-computed background prototypes to suppress background tiles
        from being selected as Top-K.
        """
        from src.critics import FeatureCritic

        fc_cfg = self.config.get('feature_critic', {})
        fc_enabled = fc_cfg.get('enable', False)

        print(f"\n[CRITIC] Feature Critic: {'Enabled' if fc_enabled else 'Disabled'}")

        if fc_enabled:
            # Initialize critic (config-driven)
            self.feature_critic = FeatureCritic(self.config)
            self.feature_critic.to(self.device)  # Prototypes buffer moves to device

            # Log initialization status
            if self.feature_critic.loaded:
                runtime_cfg = fc_cfg.get('runtime', {})
                print(f"[CRITIC] Loaded {len(self.feature_critic.prototypes)} prototypes")
                print(f"[CRITIC] Threshold: {self.feature_critic.threshold:.2f}")
                print(f"[CRITIC] Penalty Scale: {self.feature_critic.penalty_scale:.2f}")
                print(f"[CRITIC] Apply Phase: {self.feature_critic.apply_phase}")

                # Log to tracker
                self.logger.log({
                    'system/feature_critic_enabled': True,
                    'system/feature_critic_n_prototypes': len(self.feature_critic.prototypes),
                    'system/feature_critic_threshold': self.feature_critic.threshold,
                    'system/feature_critic_penalty_scale': self.feature_critic.penalty_scale,
                    'system/feature_critic_apply_phase': self.feature_critic.apply_phase
                })
            else:
                construction_cfg = fc_cfg.get('construction', {})
                prototype_path = construction_cfg.get('save_path', 'outputs/prototypes/background_prototypes.pth')
                print(f"[CRITIC] WARNING: Prototypes not found at {prototype_path}")
                print(f"[CRITIC] Run: python scripts/tools/build_prototypes.py --config {self.config.get('_config_path', 'configs/...')}")
                print(f"[CRITIC] Feature Critic will be disabled during training.")

                self.logger.log({'system/feature_critic_enabled': False})
        else:
            # Disabled: Create dummy critic
            self.feature_critic = None
            self.logger.log({'system/feature_critic_enabled': False})

    def _setup_amp(self):
        """Setup Automatic Mixed Precision (AMP) for VRAM optimization."""
        use_amp = self.train_cfg.get('use_amp', True)
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        print(f"[AMP] Mixed Precision: {'Enabled' if use_amp else 'Disabled'}")

    def _setup_logger(self):
        """Initialize local training logger."""
        self.logger = init_logger(
            experiment_name="asymmetric_mil_training",
            config=self.config
        )
        print(f"[LOGGER] Logs directory: {self.logger.run_dir}")

    def _setup_checkpoint_manager(self):
        """Setup checkpoint manager for model saving."""
        checkpoint_root = self.logger.get_checkpoints_dir()
        keep_last_n = self.train_cfg.get('keep_last_n', 3)
        self.checkpoint_manager = CheckpointManager(checkpoint_root, keep_last_n=keep_last_n)
        self.checkpoint_root = Path(checkpoint_root)
        print(f"[CHECKPOINT] Directory: {checkpoint_root}, Keep last: {keep_last_n}")

    def _build_evaluators(self):
        """Build evaluators for warmup criteria and quality monitoring."""
        print("\n[EVALUATORS] Initializing evaluators...")

        # Warmup Evaluator (Phase 4 - already implemented)
        self.warmup_evaluator = WarmupEvaluator(
            model=self.model,
            val_dataset=self.val_dataset,
            negative_pool=self.val_negative_pool,
            config=self.config,
            device=self.device
        )
        print("  [OK] WarmupEvaluator initialized")

        # Heatmap Visualizer (Phase 7 - completed)
        # Load class name mappings from config (Single Source of Truth)
        classes_cfg = self.config.get('classes', {})
        display_names = classes_cfg.get('display_names', None)
        short_names = classes_cfg.get('short_names', None)

        self.visualizer = MILVisualizer(
            save_dir=str(self.logger.get_heatmaps_dir()),
            class_names=display_names,
            short_names=short_names
        )
        print("  [OK] HeatmapVisualizer initialized")

        # Final Evaluator (Phase 7 - completed)
        # NOTE: Requires val_dataset to be available (Phase 4 TODO)
        if self.val_dataset is not None:
            self.final_evaluator = FinalEvaluator(
                model=self.model,
                val_dataset=self.val_dataset,
                val_negative_pool=self.val_negative_pool,
                config=self.config,
                device=self.device,
                logger=self.logger
            )
            print("  [OK] FinalEvaluator initialized")
        else:
            self.final_evaluator = None
            print("  [SKIP] FinalEvaluator - val_dataset not available (Phase 4 TODO)")

    def _setup_callbacks(self):
        """Setup callback system for modular training hooks."""
        from src.trainer.callbacks import CallbackList, CheckpointCallback, DriftMonitorCallback, HeatmapCallback

        print("\n[CALLBACKS] Initializing callbacks...")

        heatmap_freq = self.config.get('evaluation', {}).get('heatmap_eval_frequency', 10)
        self.callbacks = CallbackList([
            CheckpointCallback(),
            DriftMonitorCallback(),
            HeatmapCallback(frequency=heatmap_freq, phases=['stable'])
        ])

        print(f"  [OK] CheckpointCallback")
        print(f"  [OK] DriftMonitorCallback")
        print(f"  [OK] HeatmapCallback (frequency={heatmap_freq})")

    def _get_active_scheduler(self):
        """
        Extract active scheduler from dict scheduler (for checkpoint loading).

        Returns:
            Active scheduler object (for dict schedulers) or self.scheduler (for standard schedulers)
        """
        if isinstance(self.scheduler, dict):
            active_phase = self.scheduler['current_phase']
            if active_phase == 'warmup':
                return self.scheduler['warmup_scheduler']
            elif active_phase == 'hold':
                return self.scheduler['hold_scheduler']
            else:
                return self.scheduler['cosine_scheduler']
        return self.scheduler

    def _resume_from_checkpoint(self):
        """
        Resume training from checkpoint.

        Loads:
            - Model/Optimizer/Scheduler/Scaler states (via CheckpointManager)
            - Training state (epoch, is_warmup, current_k, best_score)
            - Hard mining state (blacklist + hard_negatives)
        """
        print(f"\n[RESUME] Loading checkpoint from: {self.resume_checkpoint}")

        # Use CheckpointManager to load core states
        # NOTE: Handle dict schedulers by extracting active scheduler
        active_scheduler = self._get_active_scheduler()

        start_epoch, metrics = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=self.resume_checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=active_scheduler,
            scaler=self.scaler
        )

        # Restore training epoch (CheckpointManager returns next epoch)
        self.current_epoch = start_epoch - 1

        # Infer warmup state from K value in metrics
        current_k = metrics.get('K', self.mil_cfg.get('warmup_k', 4))
        warmup_k = self.mil_cfg.get('warmup_k', 4)
        stable_k = self.mil_cfg.get('stable_k', 2)

        if current_k == stable_k:
            self.is_warmup = False
            self.current_k = stable_k
            print(f"[RESUME] Phase: Stable (K={stable_k})")
        else:
            self.is_warmup = True
            self.current_k = warmup_k
            print(f"[RESUME] Phase: Warmup (K={warmup_k})")

        # Restore best model tracking
        if 'best_hit_acc' in checkpoint:
            self.best_hit_acc = checkpoint['best_hit_acc']
            self.best_topk_lift = checkpoint['best_topk_lift']
            self.best_score = checkpoint['best_score']
            print(f"[RESUME] Best Score: {self.best_score:.4f} (Hit-Acc: {self.best_hit_acc:.4f}, TopK-Lift: {self.best_topk_lift:.4f})")

        print(f"[RESUME] Training will resume from epoch {self.current_epoch + 1}")

        # Restore hard mining state (CRITICAL: prevents catastrophic forgetting)
        if self.hard_mining_cfg.get('enable', False):
            state_dir = checkpoint_path.parent
            # Try to find corresponding hard mining state file
            epoch_num = checkpoint_path.stem.split('_')[-1]
            state_path = state_dir / f'hard_mining_state_epoch_{epoch_num}.pkl'

            if state_path.exists():
                self.train_negative_pool.load_hard_mining_state(str(state_path))
                print(f"[RESUME] Hard mining state loaded from: {state_path.name}")
            else:
                print(f"[RESUME] [WARN]  Hard mining state not found: {state_path.name}")
                print(f"[RESUME] Training will continue without hard negatives history")

    def _get_epoch_state(self) -> Dict[str, Any]:
        """
        Build epoch-level state dictionary (centralized state management).

        Returns:
            epoch_state: {
                'K': int,
                'is_warmup': bool,
                'hard_ratio': float,
                'hard_mining_enable': bool
            }
        """
        # Calculate hard_ratio based on phase
        if self.hard_mining_cfg.get('enable', False):
            if self.is_warmup:
                current_hard_ratio = self.hard_mining_cfg.get('warmup_hard_ratio', 0.5)
            else:
                current_hard_ratio = self.hard_mining_cfg.get('stable_hard_ratio', 0.8)
        else:
            current_hard_ratio = 0.0

        return {
            'K': self.current_k,
            'is_warmup': self.is_warmup,
            'hard_ratio': current_hard_ratio,
            'hard_mining_enable': self.hard_mining_cfg.get('enable', False)
        }

    def fit(self):
        """
        Main training loop with phase transition and checkpoint management.

        Training Flow:
            For each epoch:
                1. Start epoch (hard mining lifecycle)
                2. Build epoch state (K, hard_ratio)
                3. Train one epoch (Scout-Snipe)
                4. Validate
                5. Check warmup criteria (Warmup → Stable transition)
                6. Update learning rate scheduler
                7. Log metrics
                8. End epoch (hard mining lifecycle)
                9. Save checkpoints
        """
        num_epochs = self.train_cfg.get('num_epochs', 50)
        mil_warmup_epochs = self.mil_cfg.get('warmup_epochs', 5)
        warmup_k = self.mil_cfg.get('warmup_k', 4)
        stable_k = self.mil_cfg.get('stable_k', 2)

        print(f"\n[TRAINING] Starting training...")
        print(f"[TRAINING] MIL Warm-up phase: Epochs 1-{mil_warmup_epochs}, K={warmup_k}")
        print(f"[TRAINING] MIL Stable phase: After warm-up, K={stable_k}")
        print(f"[TRAINING] Automatic warm-up termination enabled (P0 criteria)")

        # Resume from checkpoint if applicable
        start_epoch = self.current_epoch + 1 if self.resume_checkpoint is not None else 1

        # Callback: on_train_start
        self.callbacks.on_train_start(self)

        for epoch in range(start_epoch, num_epochs + 1):
            self.current_epoch = epoch

            # Callback: on_epoch_start
            self.callbacks.on_epoch_start(self, epoch)

            # Phase start: Hard mining lifecycle hook
            if self.hard_mining_cfg.get('enable', False):
                self.train_negative_pool.start_epoch()

            # Build epoch state
            epoch_state = self._get_epoch_state()

            # Train one epoch (DataLoader created internally)
            train_loss, train_acc = train_one_epoch(
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

            # Validate
            val_metrics = validate(
                model=self.model,
                val_dataset=self.val_dataset,
                val_negative_pool=self.val_negative_pool,
                device=self.device,
                config=self.config
            )

            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']
            foreground_ratio = val_metrics['foreground_ratio']
            neg_confidence = val_metrics['negative_confidence']
            neg_recall = val_metrics['negative_recall']

            # Cleanup validation artifacts to prevent memory leak
            del val_metrics
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Top-K quality evaluation and warmup transition
            topk_metrics = self._evaluate_topk_and_check_transition(
                epoch, train_loss, mil_warmup_epochs
            )

            # Update learning rate scheduler
            current_lr = self._step_scheduler(epoch)

            # Log metrics
            metrics = self._build_metrics_dict(
                train_loss, train_acc, val_loss, val_acc,
                neg_confidence, neg_recall, foreground_ratio, current_lr, topk_metrics
            )

            # Print epoch summary
            self._print_epoch_summary(
                epoch, num_epochs, train_loss, train_acc,
                val_loss, val_acc, neg_recall, neg_confidence,
                foreground_ratio, topk_metrics, current_lr
            )

            # Log to file
            self.logger.log(metrics, step=epoch)

            # Phase end: Hard mining lifecycle hook
            if self.hard_mining_cfg.get('enable', False):
                self.train_negative_pool.end_epoch(current_epoch=epoch - 1)

            # Callback: on_epoch_end (handles checkpointing, drift monitoring, heatmap generation)
            self.callbacks.on_epoch_end(self, epoch, metrics)

        # Post-training complete evaluation (Phase 7 - activated)
        if self.config.get('evaluation', {}).get('enable', True):
            print("\n" + "=" * 80)
            print("[PHASE-7] Running Post-Training Complete Evaluation")
            print("=" * 80)
            final_metrics = self.final_evaluator.evaluate_all()
            self.final_evaluator.save_evaluation_report(final_metrics)

            # Print summary
            print("\n" + "=" * 80)
            print("[PHASE-7] Training Summary:")
            print("=" * 80)
            print(f"Metrics:")
            print(f"  Neg-Recall: {final_metrics['negative_recall']:.2f}  Neg-Conf: {final_metrics['negative_confidence']:.2f}")
            print(f"  Top-K Lift: {final_metrics['topk_lift']:.2f}")
            print("=" * 80)

            # Save to logger
            self.logger.save_metrics_summary(final_metrics)

        # Callback: on_train_end
        self.callbacks.on_train_end(self)

        print("\n[TRAINING] Training complete!")
        print(f"[TRAINING] Best Score: {self.best_score:.4f} (Hit-Acc: {self.best_hit_acc:.4f}, TopK-Lift: {self.best_topk_lift:.4f})")

        # Close logger
        self.logger.finish()
        print("[LOGGER] Training session closed")

    def _monitor_concept_drift(self, val_acc: float):
        """Monitor concept drift in stable phase."""
        if not self.is_warmup:
            # Initialize baseline on first stable epoch
            if self.stable_start_pos_acc is None:
                self.stable_start_pos_acc = val_acc
                print(f"[Stable] Baseline Val-Acc: {self.stable_start_pos_acc:.1%}")

            # Monitor drift
            drift_amount = val_acc - self.stable_start_pos_acc
            self.stable_epoch_count += 1

            # Alert on significant drift
            if drift_amount < -0.05:
                print(f"\n[ALERT] Concept drift! Val-Acc: {self.stable_start_pos_acc:.1%}→{val_acc:.1%} (Δ{drift_amount:+.1%})")
                print(f"  Causes: Soft Bootstrap drift / LR too high")
            elif drift_amount < -0.02:
                print(f"[CAUTION] Minor drift: Val-Acc Δ{drift_amount:+.1%}")

            # Update epoch counter in criterion for progressive alpha
            if hasattr(self.criterion, 'epoch_in_stable'):
                self.criterion.epoch_in_stable = self.stable_epoch_count
        else:
            # Reset tracking in warmup phase
            self.stable_start_pos_acc = None
            self.stable_epoch_count = 0

    def _evaluate_topk_and_check_transition(
        self,
        epoch: int,
        train_loss: float,
        mil_warmup_epochs: int
    ) -> Optional[Dict[str, float]]:
        """
        Evaluate Top-K quality and check warmup termination criteria.

        Returns:
            topk_metrics: Dict with avg_top1_conf, topk_lift, hit_acc, topk_avg_confidence
        """
        topk_metrics = None

        # Warmup evaluation and transition
        if self.is_warmup and epoch >= mil_warmup_epochs:
            print(f"\n{'='*60}")
            print(f"[WARMUP-EVAL] Evaluating warm-up termination criteria...")
            print(f"{'='*60}")

            # Run full P0 criteria evaluation
            warmup_metrics = self.warmup_evaluator.evaluate_warmup_criteria(epoch, train_loss)

            # Extract Top-K metrics
            topk_metrics = {
                'avg_top1_conf': warmup_metrics['avg_top1_conf'],
                'topk_lift': warmup_metrics['topk_lift'],
                'hit_acc': warmup_metrics['hit_acc'],
                'topk_avg_confidence': warmup_metrics['topk_avg_confidence']
            }

            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Check warmup termination
            if self.warmup_evaluator.should_end_warmup(epoch, warmup_metrics):
                self._switch_to_stable(epoch)

                # TODO (Phase 7): Generate heatmaps to verify warmup quality
                # print(f"\n{'='*60}")
                # print(f"[WARMUP-SWITCH] Generating heatmaps to verify warmup quality...")
                # print(f"{'='*60}")
                # self.heatmap_visualizer.save_representative_heatmaps(epoch=epoch, num_samples=2)
                # print(f"[WARMUP-SWITCH] Heatmaps saved successfully")

        # Stable phase: Continuous Top-K quality monitoring
        elif not self.is_warmup:
            print(f"\n{'='*60}")
            print(f"[TOP-K EVAL] Evaluating Top-K quality (Stable Phase, Epoch {epoch})...")
            print(f"{'='*60}")

            k_for_eval = self.mil_cfg.get('stable_k', 2)
            topk_avg_confidence, topk_lift, hit_acc, avg_top1_conf, _ = \
                self.warmup_evaluator.evaluate_topk_quality(k=k_for_eval)

            print(f"  Avg-Top1-Conf: {avg_top1_conf:.4f}")
            print(f"  TopK-Lift: {topk_lift:.4f}")
            print(f"  Hit-Acc: {hit_acc:.4f}")

            topk_metrics = {
                'avg_top1_conf': avg_top1_conf,
                'topk_lift': topk_lift,
                'hit_acc': hit_acc,
                'topk_avg_confidence': topk_avg_confidence
            }

            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        return topk_metrics

    def _switch_to_stable(self, epoch: int):
        """Execute Warmup → Stable phase transition."""
        warmup_bags_per_batch = self.mil_cfg.get('warmup_bags_per_batch', 3)
        stable_bags_per_batch = self.mil_cfg.get('stable_bags_per_batch', 2)
        stable_k = self.mil_cfg.get('stable_k', 2)

        print(f"\n[WARMUP-SWITCH] Ending warm-up at epoch {epoch}")
        print(f"[WARMUP-SWITCH] Switching to stable phase:")
        print(f"  Bags per batch: {warmup_bags_per_batch} → {stable_bags_per_batch}")
        print(f"  K value: {self.current_k} → {stable_k}")
        print(f"[MULTI-SCALE] Negative pool now samples proportionally from all scales")
        print(f"[MULTI-SCALE] Scale ratios: 768=40%, 1024=35%, 1536=20%, 2048=5%")

        # Update trainer state
        self.train_sampler.set_num_bags(stable_bags_per_batch)
        self.current_k = stable_k
        self.is_warmup = False

        # Learning rate continues scheduler (no manual adjustment)
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"[WARMUP-SWITCH] Current LR: {current_lr:.2e} (continues scheduler)")

        # Generate heatmaps to verify warmup quality
        print(f"\n{'='*60}")
        print(f"[WARMUP-SWITCH] Generating heatmaps to verify warmup quality...")
        print(f"{'='*60}")
        self.visualizer.generate_monitoring_heatmaps(
            self.model, self.train_dataset, epoch, phase='warmup_end'
        )

    def _step_scheduler(self, epoch: int) -> float:
        """
        Update learning rate via scheduler.

        Scheduler internally handles phase logic (Warmup/Hold/Decay).
        Trainer no longer needs to know scheduler implementation details.

        Args:
            epoch: Current epoch (1-indexed)

        Returns:
            current_lr: Updated learning rate
        """
        if self.scheduler is not None:
            # Custom schedulers (TrapezoidalLR, ManualSequentialLR) handle step(epoch)
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
        neg_confidence: float,
        neg_recall: float,
        foreground_ratio: float,
        current_lr: float,
        topk_metrics: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Build complete metrics dictionary for logging."""
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'neg_confidence': neg_confidence,
            'neg_recall': neg_recall,
            'foreground_ratio': foreground_ratio,  # NEW
            'learning_rate': current_lr,
            'K': self.current_k,
            'epoch': self.current_epoch
        }

        # Add loss decomposition
        if hasattr(self.criterion, 'last_pos_loss'):
            metrics['pos_loss'] = self.criterion.last_pos_loss
            metrics['neg_loss'] = self.criterion.last_neg_loss

        # Add dynamic weight and inter-loss
        if hasattr(self.criterion, 'current_dynamic_weight'):
            metrics['dynamic_weight'] = self.criterion.current_dynamic_weight
        if hasattr(self.criterion, 'last_inter_loss'):
            metrics['inter_loss'] = self.criterion.last_inter_loss

        # Add Top-K metrics
        if topk_metrics is not None:
            metrics['avg_top1_conf'] = topk_metrics['avg_top1_conf']
            metrics['topk_lift'] = topk_metrics['topk_lift']
            metrics['hit_acc'] = topk_metrics['hit_acc']
            metrics['topk_avg_confidence'] = topk_metrics['topk_avg_confidence']

        # Add gate filter ratio and tier statistics (Stable phase)
        if not self.is_warmup:
            if hasattr(self.criterion, 'last_gate_filtered_ratio'):
                metrics['gate_filtered_ratio'] = self.criterion.last_gate_filtered_ratio
            if hasattr(self.criterion, 'last_tier1_count'):
                metrics['tier1_count'] = self.criterion.last_tier1_count
                metrics['tier2_count'] = self.criterion.last_tier2_count
                metrics['tier3_count'] = self.criterion.last_tier3_count

        return metrics

    def _print_epoch_summary(
        self,
        epoch: int,
        num_epochs: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        neg_recall: float,
        neg_confidence: float,
        foreground_ratio: float,
        topk_metrics: Optional[Dict[str, float]],
        current_lr: float
    ):
        """Print compact epoch summary."""
        phase = "WARMUP" if self.is_warmup else f"STABLE-{self.stable_epoch_count}"

        summary = (f"[Epoch {epoch}/{num_epochs} {phase}] "
                   f"Train: Loss {train_loss:.3f} Acc {train_acc:.2f}  |  "
                   f"Val: Loss {val_loss:.3f} Acc {val_acc:.2f}  |  "
                   f"FG {foreground_ratio:.2f}  |  "
                   f"Neg: Recall {neg_recall:.2f} Conf {neg_confidence:.2f}")

        # Add Top-K metrics
        if topk_metrics is not None:
            summary += f"  |  AvgTop1 {topk_metrics['avg_top1_conf']:.3f}"

        # Add stable phase metrics
        if not self.is_warmup:
            if hasattr(self.criterion, 'current_dynamic_weight'):
                summary += f"  |  DynW {self.criterion.current_dynamic_weight:.2f}"
            if hasattr(self.criterion, 'last_inter_loss'):
                summary += f"  |  InterL {self.criterion.last_inter_loss:.3f}"

            # Three-tier statistics
            if hasattr(self.criterion, 'last_tier1_count'):
                tier1 = self.criterion.last_tier1_count
                tier2 = self.criterion.last_tier2_count
                tier3 = self.criterion.last_tier3_count
                total_top1 = tier1 + tier2 + tier3
                if total_top1 > 0:
                    summary += f"  |  Tiers[T1:{tier1} T2:{tier2} T3:{tier3}]"

        summary += f"  |  LR={current_lr:.2e}"
        print(summary)

    def _save_best_checkpoint_if_improved(
        self,
        epoch: int,
        topk_metrics: Dict[str, float],
        metrics: Dict[str, float]
    ):
        """Save best model checkpoint if weighted score improved (simple naming)."""
        current_hit_acc = topk_metrics.get('hit_acc', 0.0)
        current_topk_lift = topk_metrics.get('topk_lift', 0.0)

        # Calculate weighted score (Hit-Acc: 60%, TopK-Lift: 40%)
        current_score = 0.6 * current_hit_acc + 0.4 * current_topk_lift

        if current_score > self.best_score:
            self.best_hit_acc = current_hit_acc
            self.best_topk_lift = current_topk_lift
            self.best_score = current_score

            print(f"\n[CHECKPOINT] New best model! Epoch {epoch}, Score: {self.best_score:.4f} "
                  f"(Hit-Acc: {self.best_hit_acc:.4f}, TopK-Lift: {self.best_topk_lift:.4f})")

            # Get active scheduler for saving
            scheduler_to_save = self._get_scheduler_for_checkpoint(epoch)

            # Save best checkpoint with simple name (easy to locate)
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=scheduler_to_save,
                scaler=self.scaler,
                epoch=epoch,
                metrics=metrics,
                config=self.config,
                filename='best_model.pth'  # Simple naming, latest best always here
            )

            # Save hard mining state for best model
            if self.hard_mining_cfg.get('enable', False):
                state_path = self.checkpoint_root / 'hard_mining_state_best.pkl'
                self.train_negative_pool.save_hard_mining_state(str(state_path))

    def _save_regular_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save regular checkpoint every N epochs (stable phase only)."""
        # Only save regular checkpoints in stable phase
        if self.is_warmup:
            return

        save_every = self.train_cfg.get('save_every', 10)  # Default to 10 epochs

        if epoch % save_every == 0:
            # Get active scheduler for saving
            scheduler_to_save = self._get_scheduler_for_checkpoint(epoch)

            # Save checkpoint with zero-padded naming
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=scheduler_to_save,
                scaler=self.scaler,
                epoch=epoch,
                metrics=metrics,
                config=self.config,
                filename=f'epoch_{epoch:03d}.pth'  # e.g., epoch_010.pth
            )

            # Save hard mining state for regular checkpoint
            if self.hard_mining_cfg.get('enable', False):
                state_path = self.checkpoint_root / f'hard_mining_state_epoch_{epoch:03d}.pkl'
                self.train_negative_pool.save_hard_mining_state(str(state_path))

    def _get_scheduler_for_checkpoint(self, epoch: int):
        """
        Get the active scheduler for checkpoint saving (epoch-aware).

        Args:
            epoch: Current epoch (1-indexed)

        Returns:
            Active scheduler object based on current epoch and scheduler type
        """
        if isinstance(self.scheduler, dict) and self.scheduler.get('type') == 'trapezoidal':
            warmup_epochs = self.scheduler['warmup_epochs']
            hold_epochs = self.scheduler['hold_epochs']

            if epoch <= warmup_epochs:
                return self.scheduler['warmup_scheduler']
            elif epoch <= warmup_epochs + hold_epochs:
                return self.scheduler['hold_scheduler']
            else:
                return self.scheduler['cosine_scheduler']

        elif isinstance(self.scheduler, dict) and self.scheduler.get('type') == 'manual_sequential':
            lr_warmup_epochs = self.scheduler['warmup_epochs']

            if epoch <= lr_warmup_epochs:
                return self.scheduler['warmup_scheduler']
            else:
                return self.scheduler['cosine_scheduler']

        else:
            return self.scheduler


# ============================================================================
# Logic Verification (from train_topk_asymmetric.py)
# ============================================================================
# ✅ Preserved: Scout-Snipe training flow (delegated to train_one_epoch engine)
# ✅ Preserved: Warmup → Stable phase transition (L1323-1383)
# ✅ Preserved: P0 criteria evaluation using WarmupEvaluator (L1329)
# ✅ Preserved: Hard mining lifecycle hooks (start_epoch + end_epoch)
# ✅ Preserved: Differential learning rates (delegated to create_optimizer)
# ✅ Preserved: Trapezoidal/Manual scheduler stepping (L1421-1474)
# ✅ Preserved: Concept drift monitoring (L1287-1316)
# ✅ Preserved: Best model checkpointing (weighted score: 0.6*hit_acc + 0.4*topk_lift)
# ✅ Preserved: Mosaic augmentation control (warmup: 0.0, stable: 0.3)
# ✅ Preserved: Three-tier statistics logging (L1510-1518)
# ✅ Preserved: Memory cleanup after evaluation (torch.cuda.empty_cache + gc.collect)
#

# ============================================================================
# Logic Verification (from train_topk_asymmetric.py)
# ============================================================================
# ✅ Preserved: Scout-Snipe training flow (delegated to train_one_epoch engine)
# ✅ Preserved: Warmup → Stable phase transition (L1323-1383)
# ✅ Preserved: P0 criteria evaluation using WarmupEvaluator (L1329)
# ✅ Preserved: Hard mining lifecycle hooks (start_epoch + end_epoch)
# ✅ Preserved: Differential learning rates (delegated to create_optimizer)
# ✅ Preserved: Trapezoidal/Manual scheduler stepping (L1421-1474)
# ✅ Preserved: Concept drift monitoring (L1287-1316)
# ✅ Preserved: Best model checkpointing (weighted score: 0.6*hit_acc + 0.4*topk_lift)
# ✅ Preserved: Mosaic augmentation control (warmup: 0.0, stable: 0.3)
# ✅ Preserved: Three-tier statistics logging (L1510-1518)
# ✅ Preserved: Memory cleanup after evaluation (torch.cuda.empty_cache + gc.collect)
#
# ✅ New (2026-01-11):
# - Fixed reference sample selection for consistent heatmap monitoring
# - Heatmap generation at warmup end + every 10 epochs in stable phase
