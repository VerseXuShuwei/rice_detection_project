"""
Training State Manager - Encapsulates training phase state and transitions.

Recent Updates:
    - [2026-01-15] Refactor: Extracted from AsymmetricMILTrainer for state isolation

Key Features:
    - Manages Warmup ↔ Stable phase transition
    - Tracks epoch-level training state
    - Encapsulates P0 criteria checking
    - Provides epoch_state dict for engines

Usage:
    >>> from src.trainer.training_state import TrainingState
    >>> state = TrainingState(config)
    >>> epoch_state = state.get_epoch_state(epoch=1)
    >>> if state.should_transition_to_stable(p0_metrics):
    >>>     state.transition_to_stable()

Configuration:
    Reads from asymmetric_mil config section
"""

from typing import Dict, Any, Optional


class TrainingState:
    """
    Manages training state across Warmup and Stable phases.

    Responsibilities:
        - Track current phase (Warmup/Stable)
        - Track current K value (top-K selection)
        - Monitor P0 criteria for phase transition
        - Generate epoch_state dict for training engines
        - Track best model metrics

    Does NOT handle:
        - Component construction (handled by TrainerBuilder)
        - Training loop execution (handled by AsymmetricMILTrainer)
        - Checkpoint saving (handled by CheckpointManager)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize training state from config.

        Args:
            config: Complete configuration dict
        """
        self.config = config
        self.mil_cfg = config.get('asymmetric_mil', {})
        self.hard_mining_cfg = self.mil_cfg.get('hard_negative_mining', {})

        # Phase state
        self.is_warmup = True
        self.current_epoch = 0

        # K values
        self.warmup_k = self.mil_cfg.get('warmup_k', 4)
        self.stable_k = self.mil_cfg.get('stable_k', 2)
        self.current_k = self.warmup_k

        # Hard mining state
        self.warmup_hard_ratio = self.hard_mining_cfg.get('warmup_hard_ratio', 0.5)
        self.stable_hard_ratio = self.hard_mining_cfg.get('stable_hard_ratio', 0.8)
        self.current_hard_ratio = self.warmup_hard_ratio

        # Stable phase tracking
        self.stable_start_pos_acc = None
        self.stable_epoch_count = 0

        # Hybrid warmup state (backbone freeze/unfreeze lifecycle)
        hybrid_warmup_cfg = config.get('training', {}).get('hybrid_warmup', {})
        self.hybrid_warmup_enable = hybrid_warmup_cfg.get('enable', False)
        self.freeze_backbone_epochs = hybrid_warmup_cfg.get('freeze_backbone_epochs', 10)
        self.backbone_frozen = False  # tracks current freeze state

        # P0 criteria thresholds (from config, 0-1 scale)
        # NOTE: Actual P0 checking is done by WarmupEvaluator.should_end_warmup(),
        #       which reads these same config values independently.
        #       This method is kept for programmatic access / fallback.
        warmup_criteria_cfg = self.mil_cfg.get('warmup_criteria', {})
        self.p0_neg_recall_threshold = warmup_criteria_cfg.get('neg_recall_threshold', 0.70)
        self.p0_neg_hallucination_threshold = warmup_criteria_cfg.get('neg_disease_hallucination_threshold', 0.30)
        self.p0_topk_lift_threshold = warmup_criteria_cfg.get('topk_lift_threshold', 0.25)

        # Best model tracking
        self.best_hit_acc = 0.0
        self.best_topk_lift = 0.0
        self.best_score = 0.0  # Weighted: 0.6*hit_acc + 0.4*topk_lift

    def get_epoch_state(self, epoch: int) -> Dict[str, Any]:
        """
        Generate epoch state dict for training engines.

        This dict is passed to train_one_epoch() and validate() functions
        to configure their behavior based on current phase.

        Args:
            epoch: Current epoch number (1-indexed)

        Returns:
            epoch_state dict with keys:
                - is_warmup: bool
                - K: int (top-K value)
                - hard_mining_enable: bool
                - hard_ratio: float
                - mosaic_ratio: float (deprecated, kept for compatibility)
                - freeze_backbone: bool (True if backbone should be frozen this epoch)
        """
        self.current_epoch = epoch

        # Hard mining enable flag (V1-compatible: just check config.enable)
        # Note: start_epoch/end_epoch logic removed - not in V1 implementation
        hard_mining_enable = self.hard_mining_cfg.get('enable', False)

        return {
            'is_warmup': self.is_warmup,
            'K': self.current_k,
            'hard_mining_enable': hard_mining_enable,
            'hard_ratio': self.current_hard_ratio,
            'mosaic_ratio': 0.0,  # Deprecated after multi-scale negative pool v2.0
            'freeze_backbone': self.hybrid_warmup_enable and epoch <= self.freeze_backbone_epochs
        }

    def should_transition_to_stable(self, p0_metrics: Dict[str, float]) -> bool:
        """
        Check if P0 criteria are met for Warmup → Stable transition.

        NOTE: In practice, WarmupEvaluator.should_end_warmup() is the primary
              P0 checker. This method provides a simplified programmatic interface.

        Thresholds read from config (asymmetric_mil.warmup_criteria), all 0-1 scale:
            - neg_recall_threshold (default 0.70)
            - neg_disease_hallucination_threshold (default 0.30)
            - topk_lift_threshold (default 0.25)

        Args:
            p0_metrics: Dict with keys (0-1 scale):
                - neg_recall: float
                - neg_disease_hallucination: float
                - topk_lift: float

        Returns:
            True if all P0 criteria met
        """
        neg_recall = p0_metrics.get('neg_recall', 0.0)
        neg_hallucination = p0_metrics.get('neg_disease_hallucination', 1.0)
        topk_lift = p0_metrics.get('topk_lift', 0.0)

        neg_recall_ok = neg_recall > self.p0_neg_recall_threshold
        neg_hallucination_ok = neg_hallucination < self.p0_neg_hallucination_threshold
        topk_lift_ok = topk_lift > self.p0_topk_lift_threshold

        return neg_recall_ok and neg_hallucination_ok and topk_lift_ok

    def transition_to_stable(self, initial_pos_acc: float):
        """
        Execute Warmup → Stable phase transition.

        Args:
            initial_pos_acc: Positive accuracy at transition point (for drift monitoring)
        """
        print("\n" + "="*60)
        print("PHASE TRANSITION: WARMUP → STABLE")
        print("="*60)

        self.is_warmup = False
        self.current_k = self.stable_k
        self.current_hard_ratio = self.stable_hard_ratio
        self.stable_start_pos_acc = initial_pos_acc
        self.stable_epoch_count = 0

        print(f"[STATE] New K value: {self.current_k}")
        print(f"[STATE] New hard_ratio: {self.current_hard_ratio:.2f}")
        print(f"[STATE] Baseline pos_acc: {initial_pos_acc:.2%} (for drift monitoring)")
        print("="*60 + "\n")

    def update_best_metrics(
        self,
        hit_acc: float,
        topk_lift: float
    ) -> bool:
        """
        Update best model metrics if current is better.

        Scoring function: 0.6 * hit_acc + 0.4 * topk_lift

        Args:
            hit_acc: Top-1 hit accuracy (0-1)
            topk_lift: Top-K lift over Top-1 (0-1)

        Returns:
            True if new best model
        """
        current_score = 0.6 * hit_acc + 0.4 * topk_lift

        if current_score > self.best_score:
            self.best_score = current_score
            self.best_hit_acc = hit_acc
            self.best_topk_lift = topk_lift
            return True

        return False

    def check_concept_drift(self, current_pos_acc: float, threshold: float = 0.05) -> bool:
        """
        Check for concept drift in Stable phase.

        Drift detected if: current_pos_acc < baseline - threshold

        Args:
            current_pos_acc: Current positive accuracy (0-1)
            threshold: Drift threshold (default 5%)

        Returns:
            True if drift detected
        """
        if self.is_warmup or self.stable_start_pos_acc is None:
            return False

        drift_amount = self.stable_start_pos_acc - current_pos_acc
        return drift_amount > threshold

    def should_unfreeze_backbone(self, epoch: int) -> bool:
        """
        Check if backbone should be unfrozen at this epoch.

        Returns True exactly once: when epoch crosses freeze_backbone_epochs boundary.

        Args:
            epoch: Current epoch (1-indexed)

        Returns:
            True if backbone should be unfrozen this epoch
        """
        if not self.hybrid_warmup_enable:
            return False
        return self.backbone_frozen and epoch > self.freeze_backbone_epochs

    def increment_stable_epoch(self):
        """Increment stable epoch counter."""
        if not self.is_warmup:
            self.stable_epoch_count += 1

    def state_dict(self) -> Dict[str, Any]:
        """
        Save state for checkpointing.

        Returns:
            State dict for checkpoint
        """
        return {
            'is_warmup': self.is_warmup,
            'current_epoch': self.current_epoch,
            'current_k': self.current_k,
            'current_hard_ratio': self.current_hard_ratio,
            'stable_start_pos_acc': self.stable_start_pos_acc,
            'stable_epoch_count': self.stable_epoch_count,
            'best_hit_acc': self.best_hit_acc,
            'best_topk_lift': self.best_topk_lift,
            'best_score': self.best_score,
            'backbone_frozen': self.backbone_frozen
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load state from checkpoint.

        Args:
            state_dict: State dict from checkpoint
        """
        self.is_warmup = state_dict.get('is_warmup', True)
        self.current_epoch = state_dict.get('current_epoch', 0)
        self.current_k = state_dict.get('current_k', self.warmup_k)
        self.current_hard_ratio = state_dict.get('current_hard_ratio', self.warmup_hard_ratio)
        self.stable_start_pos_acc = state_dict.get('stable_start_pos_acc', None)
        self.stable_epoch_count = state_dict.get('stable_epoch_count', 0)
        self.best_hit_acc = state_dict.get('best_hit_acc', 0.0)
        self.best_topk_lift = state_dict.get('best_topk_lift', 0.0)
        self.best_score = state_dict.get('best_score', 0.0)
        self.backbone_frozen = state_dict.get('backbone_frozen', False)

    def __repr__(self):
        phase = "WARMUP" if self.is_warmup else "STABLE"
        return (f"TrainingState(phase={phase}, epoch={self.current_epoch}, "
                f"K={self.current_k}, hard_ratio={self.current_hard_ratio:.2f})")
