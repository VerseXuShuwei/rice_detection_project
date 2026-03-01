"""
Trainer Component Builder - Separates construction logic from training orchestration.

Recent Updates:
    - [2026-01-15] Refactor: Extracted from AsymmetricMILTrainer to eliminate God Class

Key Features:
    - Builder Pattern for Trainer components
    - Decouples "what to build" from "how to use"
    - Single Responsibility: Only handles component construction
    - Enables independent testing of construction logic

Usage:
    >>> from src.trainer.trainer_builder import TrainerBuilder
    >>> builder = TrainerBuilder(config)
    >>> components = builder.build_all()
    >>> model = components['model']
    >>> optimizer = components['optimizer']

Configuration:
    Reads from config dict (same structure as AsymmetricMILTrainer)
"""

import torch
import torch.nn as nn
from torch.amp import GradScaler
from typing import Dict, Any, Tuple, Optional
from torch.utils.data import DataLoader

# Import builders
from src.models.builder import get_model
from src.losses.builder import create_loss_function
from src.utils.builder import create_optimizer, create_scheduler

# Import data components
from src.data import (
    AsymmetricMILDataset,
    NegativeTilePool,
    ClassAwareBagSampler,
    mil_collate_fn
)

# Import evaluation
from src.evaluation import WarmupEvaluator, FinalEvaluator, MILVisualizer

# Import utilities
from src.utils.device import get_device
from src.utils.local_logger import init_logger
from src.core.checkpoint_manager import CheckpointManager

# Import Feature Critic
from src.critics import FeatureCritic


class TrainerBuilder:
    """
    Builder class for constructing all Trainer components.

    Separates component construction logic from training orchestration.
    This class handles the "Build" phase of the Trainer lifecycle.

    Responsibilities:
        - Build data loaders (train + validation)
        - Build model and move to device
        - Build optimizer and scheduler
        - Build loss function
        - Build evaluators (warmup + final)
        - Setup utilities (AMP, logger, checkpoint manager)

    Does NOT handle:
        - Training loop execution
        - State management (warmup/stable transition)
        - Checkpoint loading/saving during training
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize builder with configuration.

        Args:
            config: Complete configuration dict from YAML files
        """
        self.config = config
        self.device = get_device()

        # Extract key configs
        self.train_cfg = config.get('training', {})
        self.mil_cfg = config.get('asymmetric_mil', {})
        self.dataset_cfg = config.get('dataset', {})

        # [2026-02-05] CONFIG LOADING LOG - MIL Strategy
        print("=" * 60)
        print("[TRAINER-BUILDER] Asymmetric MIL Configuration")
        print("=" * 60)
        print(f"[MIL] Warmup Phase:")
        print(f"  warmup_epochs = {self.mil_cfg.get('warmup_epochs', 6)}")
        print(f"  warmup_bags_per_batch = {self.mil_cfg.get('warmup_bags_per_batch', 6)}")
        print(f"  warmup_k = {self.mil_cfg.get('warmup_k', 4)}")
        print(f"  warmup_neg_tiles = {self.mil_cfg.get('warmup_neg_tiles', 8)}")
        print(f"[MIL] Stable Phase:")
        print(f"  stable_bags_per_batch = {self.mil_cfg.get('stable_bags_per_batch', 9)}")
        print(f"  stable_k = {self.mil_cfg.get('stable_k', 3)}")
        print(f"  stable_neg_tiles = {self.mil_cfg.get('stable_neg_tiles', 9)}")

        # Hard negative mining config
        hn_cfg = self.mil_cfg.get('hard_negative_mining', {})
        print(f"[MIL] Hard Negative Mining:")
        print(f"  enable = {hn_cfg.get('enable', True)}")
        print(f"  warmup_hard_ratio = {hn_cfg.get('warmup_hard_ratio', 0.25)}")
        print(f"  stable_hard_ratio = {hn_cfg.get('stable_hard_ratio', 0.25)}")

        # Warmup criteria config
        wc_cfg = self.mil_cfg.get('warmup_criteria', {})
        print(f"[MIL] Warmup Termination Criteria (P0):")
        print(f"  neg_recall_threshold = {wc_cfg.get('neg_recall_threshold', 0.70)}")
        print(f"  neg_disease_hallucination_threshold = {wc_cfg.get('neg_disease_hallucination_threshold', 0.30)}")
        print(f"  topk_lift_threshold = {wc_cfg.get('topk_lift_threshold', 0.15)}")

        # Training config
        print(f"[TRAINING] Basic Config:")
        print(f"  num_epochs = {self.train_cfg.get('num_epochs', 30)}")
        print(f"  gradient_clip = {self.train_cfg.get('gradient_clip', 1.0)}")
        print(f"  use_amp = {self.train_cfg.get('use_amp', True)}")

        # Hybrid warmup config
        hw_cfg = self.train_cfg.get('hybrid_warmup', {})
        print(f"[TRAINING] Hybrid Warmup:")
        print(f"  enable = {hw_cfg.get('enable', False)}")
        print(f"  freeze_backbone_epochs = {hw_cfg.get('freeze_backbone_epochs', 10)}")
        print(f"  backbone_unfreeze_warmup_epochs = {hw_cfg.get('backbone_unfreeze_warmup_epochs', 3)}")
        print(f"  backbone_unfreeze_start_fraction = {hw_cfg.get('backbone_unfreeze_start_fraction', 0.1)}")

        # Optimizer LR config
        opt_cfg = config.get('optimizer', {})
        print(f"[OPTIMIZER] Learning Rates:")
        print(f"  backbone_lr = {opt_cfg.get('backbone_lr', 1e-5):.1e}")
        print(f"  hybrid_lr = {opt_cfg.get('hybrid_lr', 5e-5):.1e}")
        print(f"  classifier_lr = {opt_cfg.get('classifier_lr', 3e-4):.1e}")
        print(f"  weight_decay = {opt_cfg.get('weight_decay', 1e-2):.1e}")

        # Scheduler config
        sched_cfg = config.get('scheduler', {})
        print(f"[SCHEDULER] LR Schedule:")
        print(f"  name = {sched_cfg.get('name', 'trapezoidal')}")
        print(f"  warmup_epochs = {sched_cfg.get('warmup_epochs', 5)}")
        print(f"  hold_epochs = {sched_cfg.get('hold_epochs', 5)}")
        print(f"  min_lr = {sched_cfg.get('min_lr', 1e-6):.1e}")

        # Anti-collapse config
        ac_cfg = config.get('anti_collapse', {})
        print(f"[ANTI-COLLAPSE] Config:")
        print(f"  freeze_bn_in_snipe = {ac_cfg.get('freeze_bn_in_snipe', True)}")
        print(f"  monitor_collapse_indicators = {ac_cfg.get('monitor_collapse_indicators', True)}")

        # Scale Diversity / Spatial NMS config
        sd_cfg = config.get('scale_diversity', {})
        nms_cfg = config.get('spatial_nms', {})
        print(f"[SCOUT] Tile Selection:")
        print(f"  scale_diversity.enable = {sd_cfg.get('enable', False)}")
        print(f"  scale_diversity.penalty = {sd_cfg.get('penalty', 0.5)}")
        print(f"  spatial_nms.enable = {nms_cfg.get('enable', False)}")
        print(f"  spatial_nms.iou_threshold = {nms_cfg.get('iou_threshold', 0.5)}")

        print("=" * 60)

    def build_data(self) -> Dict[str, Any]:
        """
        Build train and validation data loaders.

        Returns:
            dict with keys:
                - train_dataset: AsymmetricMILDataset
                - val_dataset: AsymmetricMILDataset
                - train_dataloader: DataLoader
                - val_dataloader: DataLoader
                - train_negative_pool: NegativeTilePool
                - val_negative_pool: NegativeTilePool
                - num_classes: int (inferred from dataset)
        """
        print("\n[BUILDER] Building data components...")

        # Build negative pools (NegativeTilePool reads lmdb_path from config internally)
        seed = self.config.get('training', {}).get('seed', 42)
        train_negative_pool = NegativeTilePool(config=self.config, split='train', seed=seed)
        val_negative_pool = NegativeTilePool(config=self.config, split='val', seed=seed)

        # Verify pool exists
        if not train_negative_pool.exists():
            raise RuntimeError(
                f"Negative tile pool not found!\n"
                f"Expected path: {train_negative_pool.lmdb_path}\n"
                f"Please run: python scripts/tools/build_negative_pool.py --config <config_path>"
            )

        print(f"[BUILDER] Train pool: {len(train_negative_pool)} tiles")
        print(f"[BUILDER] Val pool: {len(val_negative_pool)} tiles")

        # Build datasets (AsymmetricMILDataset reads all params from config internally)
        train_dataset = AsymmetricMILDataset(
            config=self.config,
            split='train',
            seed=seed
        )

        val_dataset = AsymmetricMILDataset(
            config=self.config,
            split='val',
            seed=seed
        )

        # Verify and set num_classes
        num_classes = train_dataset.num_classes

        # CRITICAL: Always set num_classes in config for model to read
        # BaseModel.__init__ reads config.get('num_classes', 10) - must be set!
        if 'num_classes' in self.config:
            config_num_classes = self.config['num_classes']
            if config_num_classes != num_classes:
                print(f"[BUILDER] WARN: Config num_classes ({config_num_classes}) != Dataset num_classes ({num_classes})")
                print(f"[BUILDER]       Using dataset value: {num_classes}")

        self.config['num_classes'] = num_classes  # Always set, not conditional!
        print(f"[BUILDER] Num classes: {num_classes} (Class 1-{num_classes}, Class 0 for negative)")

        # Build samplers and dataloaders
        # Read from asymmetric_mil config (not training config)
        mil_cfg = self.config.get('asymmetric_mil', {})
        warmup_bags_per_batch = mil_cfg.get('warmup_bags_per_batch', 7)
        print(f"[BUILDER] Creating class-aware sampler (warmup: {warmup_bags_per_batch} bags/batch)...")

        train_sampler = ClassAwareBagSampler(
            dataset=train_dataset,
            bags_per_batch=warmup_bags_per_batch,
            shuffle=True
        )

        val_sampler = ClassAwareBagSampler(
            dataset=val_dataset,
            bags_per_batch=warmup_bags_per_batch,
            shuffle=False
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=mil_collate_fn,
            num_workers=self.train_cfg.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=mil_collate_fn,
            num_workers=self.train_cfg.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True
        )

        print(f"[BUILDER] Train bags: {len(train_dataset)} | Val bags: {len(val_dataset)}")

        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'train_sampler': train_sampler,
            'val_sampler': val_sampler,
            'train_dataloader': train_dataloader,
            'val_dataloader': val_dataloader,
            'train_negative_pool': train_negative_pool,
            'val_negative_pool': val_negative_pool,
            'num_classes': num_classes
        }

    def build_model(self, num_classes: int) -> nn.Module:
        """
        Build and verify model architecture.

        Args:
            num_classes: Number of disease classes (excluding Class 0)

        Returns:
            model: nn.Module moved to device
        """
        model_name = self.config.get('model', {}).get('name', 'mil_efficientnetv2-s')
        print(f"\n[BUILDER] Creating model: {model_name}")

        model = get_model(model_name, config=self.config)
        model = model.to(self.device)

        # Verify output dimension
        num_output_classes = num_classes + 1  # +1 for Class 0 (healthy)

        with torch.no_grad():
            tile_size = self.dataset_cfg.get('final_tile_size', 384)
            dummy_input = torch.randn(1, 3, tile_size, tile_size).to(self.device)
            dummy_output = model.predict_instances(dummy_input)
            actual_output_dim = dummy_output.shape[1]

            if actual_output_dim != num_output_classes:
                raise RuntimeError(
                    f"Model output dimension mismatch!\n"
                    f"  Expected: {num_output_classes} (num_classes={num_classes} + 1 for Class 0)\n"
                    f"  Actual: {actual_output_dim}"
                )

            print(f"[BUILDER] Output dimension verified: {actual_output_dim}")

        # Test numerical stability
        with torch.no_grad():
            dummy_batch = torch.randn(16, 3, tile_size, tile_size).to(self.device)
            dummy_outputs = model.predict_instances(dummy_batch)
            if torch.isnan(dummy_outputs).any() or torch.isinf(dummy_outputs).any():
                raise RuntimeError("Model produces NaN/Inf outputs! Check initialization.")

        print(f"[BUILDER] Model numerical stability verified")
        return model

    def build_optimizer_scheduler(self, model: nn.Module) -> Tuple[Any, Any]:
        """
        Build optimizer and learning rate scheduler.

        Args:
            model: Model to optimize

        Returns:
            (optimizer, scheduler)
        """
        print(f"\n[BUILDER] Creating optimizer and scheduler...")
        optimizer = create_optimizer(model, self.config)
        scheduler = create_scheduler(optimizer, self.config)
        return optimizer, scheduler

    def build_loss_function(self) -> nn.Module:
        """Build loss function from config['loss']."""
        print(f"\n[BUILDER] Creating loss function...")
        loss_config = self.config.get('loss', {})
        print(f"[BUILDER] Loss type: {loss_config.get('type', 'ce')}")
        return create_loss_function(loss_config)

    def build_feature_critic(self) -> Optional[FeatureCritic]:
        """
        Build Feature Critic if enabled in config.

        Returns:
            FeatureCritic instance or None if disabled
        """
        fc_cfg = self.config.get('feature_critic', {})

        # [2026-02-05] CONFIG LOADING LOG - Feature Critic
        print("=" * 60)
        print("[FEATURE-CRITIC] Configuration")
        print("=" * 60)
        print(f"[FC] enable = {fc_cfg.get('enable', False)}")

        if not fc_cfg.get('enable', False):
            print("[FC] Feature Critic DISABLED")
            print("=" * 60)
            return None

        # Print runtime config
        runtime_cfg = fc_cfg.get('runtime', {})
        print(f"[FC] Runtime Config:")
        print(f"  threshold = {runtime_cfg.get('threshold', 0.86)}")
        print(f"  auto_threshold = {runtime_cfg.get('auto_threshold', False)}")
        print(f"  penalty_scale = {runtime_cfg.get('penalty_scale', 0.5)}")
        print(f"  apply_phase = {runtime_cfg.get('apply_phase', 'warmup')}")

        # Print construction config
        construction_cfg = fc_cfg.get('construction', {})
        print(f"[FC] Construction Config:")
        print(f"  save_path = {construction_cfg.get('save_path', 'outputs/prototypes/background_prototypes.pth')}")
        print("=" * 60)

        print(f"[BUILDER] Creating Feature Critic...")
        critic = FeatureCritic(self.config)
        critic = critic.to(self.device)
        return critic

    def build_evaluators(self, model, val_dataset, val_negative_pool) -> Dict[str, Any]:
        """
        Build warmup evaluator, final evaluator, and visualizer.

        Args:
            model: Trained model
            val_dataset: Validation dataset
            val_negative_pool: Validation negative pool

        Returns:
            dict with keys: warmup_evaluator, final_evaluator, visualizer
        """
        print(f"\n[BUILDER] Creating evaluators...")

        # Load class name mappings from config
        classes_cfg = self.config.get('classes', {})
        display_names = classes_cfg.get('display_names', None)
        short_names = classes_cfg.get('short_names', None)

        # Get save_dir from logger (created in build_logging)
        heatmaps_dir = str(self.logger.get_heatmaps_dir()) if hasattr(self, 'logger') else 'outputs/heatmaps'

        # Load heatmap settings from config (injected at construction time)
        # This follows "Single Source of Truth" - config read once here, not at runtime
        heatmap_cfg = self.config.get('evaluation', {}).get('heatmap', {})

        visualizer = MILVisualizer(
            save_dir=heatmaps_dir,
            class_names=display_names,
            short_names=short_names,
            # Heatmap generation settings (from config)
            multiscale_tile_sizes=heatmap_cfg.get('multiscale_tile_sizes', [1024, 1536, 2048]),
            small_image_tile_sizes=heatmap_cfg.get('small_image_tile_sizes', [512, 768, 1024]),
            multiscale_min_size=heatmap_cfg.get('multiscale_min_size', [3000, 4000]),
            stride_ratio=heatmap_cfg.get('stride_ratio', 0.5),
            batch_size=heatmap_cfg.get('batch_size', 8),
            conf_threshold=heatmap_cfg.get('conf_threshold', 0.4),
            top_k=heatmap_cfg.get('top_k', 5)
        )

        warmup_evaluator = WarmupEvaluator(
            model=model,
            val_dataset=val_dataset,
            negative_pool=val_negative_pool,
            config=self.config,
            device=self.device
        )

        final_evaluator = FinalEvaluator(
            model=model,
            val_dataset=val_dataset,
            val_negative_pool=val_negative_pool,
            config=self.config,
            device=self.device,
            logger=self.logger
        )

        return {
            'warmup_evaluator': warmup_evaluator,
            'final_evaluator': final_evaluator,
            'visualizer': visualizer
        }

    def setup_amp(self) -> GradScaler:
        """Setup Automatic Mixed Precision scaler."""
        use_amp = self.train_cfg.get('use_amp', True)
        print(f"\n[BUILDER] AMP enabled: {use_amp}")
        return GradScaler('cuda', enabled=use_amp)

    def setup_logger(self) -> Any:
        """Setup local logger.

        Reads paths from training.paths config (YAML: default_schedule.yaml).
        """
        print(f"\n[BUILDER] Initializing local logger...")
        paths_cfg = self.train_cfg.get('paths', {})
        logs_root = paths_cfg.get('logs_root', 'outputs/logs')
        checkpoints_root = paths_cfg.get('checkpoints_root', 'outputs/checkpoints')
        logger = init_logger(
            experiment_name="asymmetric_mil_training",
            config=self.config,
            logs_root=logs_root,
            checkpoints_root=checkpoints_root
        )
        print(f"[BUILDER] Logs directory: {logger.run_dir}")
        return logger

    def setup_checkpoint_manager(self) -> CheckpointManager:
        """
        Setup checkpoint manager.

        Uses logger's checkpoints directory for consistency.
        All outputs are under outputs/{logs,checkpoints}/{experiment_name}_{timestamp}/
        """
        # Use logger's checkpoint directory for unified path management
        checkpoint_root = str(self.logger.get_checkpoints_dir())
        keep_last_n = self.train_cfg.get('keep_last_n', 3)
        print(f"\n[BUILDER] Checkpoint root: {checkpoint_root}")
        return CheckpointManager(checkpoint_root=checkpoint_root, keep_last_n=keep_last_n)

    def build_all(self) -> Dict[str, Any]:
        """
        Build all trainer components in correct order.

        Returns:
            dict with all components:
                - model, optimizer, scheduler, loss_function
                - train_dataset, val_dataset, train_dataloader, val_dataloader
                - train_negative_pool, val_negative_pool
                - warmup_evaluator, final_evaluator, visualizer
                - feature_critic (optional)
                - scaler, logger, checkpoint_manager
                - num_classes
        """
        print("\n" + "="*60)
        print("TRAINER COMPONENT BUILDER")
        print("="*60)

        # Build in dependency order
        data_components = self.build_data()
        model = self.build_model(data_components['num_classes'])
        optimizer, scheduler = self.build_optimizer_scheduler(model)
        loss_function = self.build_loss_function()
        feature_critic = self.build_feature_critic()

        # Logger must be created before evaluators (evaluators need heatmaps_dir)
        scaler = self.setup_amp()
        self.logger = self.setup_logger()
        checkpoint_manager = self.setup_checkpoint_manager()

        evaluators = self.build_evaluators(
            model=model,
            val_dataset=data_components['val_dataset'],
            val_negative_pool=data_components['val_negative_pool']
        )

        print("\n" + "="*60)
        print("COMPONENT BUILDING COMPLETE")
        print("="*60 + "\n")

        return {
            # Core training components
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'loss_function': loss_function,
            'feature_critic': feature_critic,

            # Data components
            'train_dataset': data_components['train_dataset'],
            'val_dataset': data_components['val_dataset'],
            'train_sampler': data_components['train_sampler'],
            'val_sampler': data_components['val_sampler'],
            'train_dataloader': data_components['train_dataloader'],
            'val_dataloader': data_components['val_dataloader'],
            'train_negative_pool': data_components['train_negative_pool'],
            'val_negative_pool': data_components['val_negative_pool'],

            # Evaluation components
            'warmup_evaluator': evaluators['warmup_evaluator'],
            'final_evaluator': evaluators['final_evaluator'],
            'visualizer': evaluators['visualizer'],

            # Utilities
            'scaler': scaler,
            'logger': self.logger,
            'checkpoint_manager': checkpoint_manager,

            # Metadata
            'num_classes': data_components['num_classes']
        }
