"""
Top-K Anchored MIL Loss - Core Innovation

Recent Updates:
    - [2026-02-26] CRITICAL: Per-bag KL with dual-gate admission (Gate A: Tier1 conf, Gate B: top2k correct direction)
    - [2026-01-05] Refactor: Migrated from losses.py (preserved all logic)
    - [2025-12-18] MAJOR: Tier 3 Enhanced - Noise Drop + Correction Ranking
    - [2025-12-18] MAJOR: Top-2~K三级联动策略 (Teacher质量分级)
    - [2025-12-10] CRITICAL: Three-Tier Stratified Supervision in Stable Phase
    - [2025-12-10] CRITICAL: Fixed NR Loss Logic (reject disease, not background)
    - [2025-12-10] MAJOR: Dynamic weight + Inter-bag ranking + Stricter gate

Key Innovations:
    1. **Dynamic Top-1 CE weight**: Adapts to Top-K reliability (0.1 → 1.0)
    2. **Inter-bag ranking loss**: Positive Top-1 > All negatives
    3. **Stricter stable gate**: Conf > 0.5 AND Correct prediction
    4. **Three-Tier Stratification**: Qualified/Marginal/Wrong prediction handling
    5. **Tier-3 Dual Strategy**: Noise Drop (熔断) + Correction Ranking (纠错)
    6. **Top-2~K三级联动**: Teacher quality-aware student supervision

Loss Formula:
    Warm-up: L = L_neg + w(t)·λ1·CE(Top-1) + λ2·NR_disease(Top-2~K) + λ_r·L_inter
    Stable:  L = L_neg + w(t)·λ1·CE(Top-1) + λ3·KL(Top-2~K, qualified_Top-1) + λ_r·L_inter

    where:
        w(t) = dynamic weight (0.1 in warmup, grows to 1.0 in stable)
        NR_disease = -log(1 - max_disease_prob)  # FIXED: reject disease, not background

杠杆原则 (Leverage Principle):
    使用 reduction='sum' 后，样本数量会放大 loss 值。
    必须通过权重平衡，确保 Top-1 (少数样本) 仍然占主导地位。

    Example (7 bags, K=4):
        - Top-1: 7 个样本
        - Top-2~K: 21 个样本（数量优势！）
        - 如果 λ1=5.0, λ2=0.3:
            Top-1 贡献 = 5.0 × 7 = 35.0  →  77.3% (主导)
            Top-2~K 贡献 = 0.3 × 21 = 6.3  →  13.9%
            → Top-1 实现"真理独裁"！

Rationale:
    - MIL Assumption: Each positive bag contains at least one disease tile
    - Top-1 tile (highest Scout confidence): Most likely disease → CE anchor
    - Top-2~K tiles: May contain backgrounds → NR_disease for noise tolerance
    - CRITICAL: NR rejects disease confidence (not background confidence)
      Reason: Top-2~K may truly be background, forcing "not background" creates contradiction

Usage:
    >>> criterion = TopKAnchoredMILLoss(
    ...     top1_ce_weight=5.0,
    ...     top2k_nr_weight=0.3,
    ...     enable_dynamic_weight=True,
    ...     enable_ranking=True
    ... )
    >>> # Warmup phase
    >>> loss = criterion(outputs, labels, is_top1=is_top1_mask, is_warmup=True, epoch=0)
    >>> # Stable phase
    >>> loss = criterion(outputs, labels, is_top1=is_top1_mask, is_warmup=False, epoch=15)
    >>> print(f"Tier 1/2/3: {criterion.last_tier1_count}/{criterion.last_tier2_count}/{criterion.last_tier3_count}")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.losses.components.focal import FocalLoss


class TopKAnchoredMILLoss(nn.Module):
    """
    Top-K Anchored MIL Loss with dynamic weighting and ranking constraint.

    See module docstring for detailed explanation.
    """

    def __init__(
        self,
        top1_ce_weight: float = 5.0,        # 杠杆原则，确保 Top-1 主导
        top2k_nr_weight: float = 0.3,       # Warm-up NR weight
        top2k_soft_weight: float = 0.1,     # Stable Soft Bootstrap weight
        # Tier 3 策略参数
        noise_drop_threshold: float = 0.6,  # 噪声熔断阈值
        correction_margin: float = 0.4,     # 纠错 Ranking Margin
        correction_weight: float = 2.0,     # 纠错 Ranking 权重
        # Stable 阶段 Top-2~K 的 NR 权重
        stable_nr_weight: float = 0.3,
        alpha: float = 0.95,
        epsilon: float = 1e-8,
        # Dynamic weighting and ranking constraint
        enable_dynamic_weight: bool = False,
        warmup_ce_weight: float = 0.1,
        stable_ce_weight: float = 1.0,
        growth_epochs: int = 10,
        enable_ranking: bool = False,
        inter_weight: float = 1.0,
        margin: float = 0.2,
        stable_gate_conf: float = 0.5,
        stable_gate_correct: bool = True,
        # Focal Loss configuration
        enable_focal_loss: bool = False,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        # [2026-02-02] Anti-Collapse: Tier 2 Top-2~K strategy configuration
        # Default "weak_ce": gentle CE on Top-2~K when Gate A fails (Tier2 teacher)
        # "nr": legacy Negative Rejection (deprecated, kept for ablation only)
        # "silence": no supervision on Top-2~K (safest but zero gradient)
        tier2_top2k_strategy: str = "weak_ce",
        # Anti-collapse weights (previously hardcoded)
        tier2_weak_weight: float = 1.0,          # Tier 2 Top-1 supervision weight (full: correct pred, push conf up)
        tier3_fallback_top1_weight: float = 0.3, # Safety net Top-1 weight (all-Tier3 collapse)
        tier3_fallback_top2k_weight: float = 0.1,# Safety net Top-2~K weight
        tier2_top2k_weak_ce_weight: float = 0.1, # Tier 2 Top-2~K weak_ce weight
        # [2026-02-24] Ablation C: stable phase loss mode
        stable_loss_mode: str = "tiered",        # "tiered" | "simple"
    ):
        super().__init__()
        self.top1_ce_weight = top1_ce_weight
        self.top2k_nr_weight = top2k_nr_weight
        self.top2k_soft_weight = top2k_soft_weight
        self.alpha = alpha
        self.epsilon = epsilon

        # Tier 3 策略参数
        self.noise_drop_threshold = noise_drop_threshold
        self.correction_margin = correction_margin
        self.correction_weight = correction_weight

        # Stable阶段Top-2~K的NR权重
        self.stable_nr_weight = stable_nr_weight

        # [2026-02-02] Anti-Collapse: Tier 2 Top-2~K strategy
        self.tier2_top2k_strategy = tier2_top2k_strategy
        self.tier2_weak_weight = tier2_weak_weight
        self.tier3_fallback_top1_weight = tier3_fallback_top1_weight
        self.tier3_fallback_top2k_weight = tier3_fallback_top2k_weight
        self.tier2_top2k_weak_ce_weight = tier2_top2k_weak_ce_weight
        # [2026-02-24] Ablation C
        self.stable_loss_mode = stable_loss_mode

        # Dynamic weight configuration
        self.enable_dynamic_weight = enable_dynamic_weight
        self.warmup_ce_weight = warmup_ce_weight
        self.stable_ce_weight = stable_ce_weight
        self.growth_epochs = growth_epochs

        # Ranking constraint configuration
        self.enable_ranking = enable_ranking
        self.inter_weight = inter_weight
        self.margin = margin

        # Stricter gate for stable phase
        self.stable_gate_conf = stable_gate_conf
        self.stable_gate_correct = stable_gate_correct

        # Focal Loss configuration
        self.enable_focal_loss = enable_focal_loss
        if self.enable_focal_loss:
            self.focal_loss = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                reduction='sum'
            )

        # Loss decomposition for logging
        self.last_pos_loss = 0.0
        self.last_neg_loss = 0.0
        self.last_inter_loss = 0.0

        # Stable phase tracking
        self.epoch_in_stable = 0
        self.current_dynamic_weight = 1.0

        # Gate filtering monitoring
        self.last_gate_filtered_ratio = 0.0

        # Three-tier stratification tracking
        self.last_tier1_count = 0
        self.last_tier2_count = 0
        self.last_tier3_count = 0

    def compute_inter_ranking_loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        is_top1: torch.Tensor
    ) -> torch.Tensor:
        """
        Inter-bag Ranking Loss: Positive Top-1 should score higher than all negatives.

        Formula:
            L_inter = ReLU(margin + max_neg_score - pos_top1_score).mean()

        Args:
            outputs: (N, num_classes) logits
            labels: (N,) class labels (0=negative, 1-C=positive)
            is_top1: (N,) boolean tensor marking Top-1 tiles

        Returns:
            Ranking loss (scalar)
        """
        probs = F.softmax(outputs, dim=1)

        neg_mask = labels == 0
        pos_mask = ~neg_mask
        top1_mask = pos_mask & is_top1

        # Early return if no negatives or no Top-1 positives
        if neg_mask.sum() == 0 or top1_mask.sum() == 0:
            return torch.tensor(0.0, device=outputs.device)

        # Max disease probability on negative tiles (Class 1-9, exclude Class 0)
        neg_probs = probs[neg_mask, 1:]
        max_neg_prob = neg_probs.max()

        # Top-1 positive tiles: probability on their true class
        pos_top1_labels = labels[top1_mask]
        pos_top1_probs = probs[top1_mask]

        # Gather probabilities for true classes
        pos_top1_class_probs = pos_top1_probs[
            torch.arange(len(pos_top1_labels), device=outputs.device),
            pos_top1_labels
        ]

        # Ranking constraint
        inter_loss = F.relu(self.margin + max_neg_prob - pos_top1_class_probs).mean()

        return inter_loss

    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        is_top1: Optional[torch.Tensor] = None,
        is_warmup: bool = True,
        epoch: int = 0
    ) -> torch.Tensor:
        """
        Compute loss based on training phase.

        Args:
            outputs: (N, num_classes) logits
            labels: (N,) class labels (0=negative, 1-C=positive)
            is_top1: (N,) boolean tensor, True for Top-1 tiles
            is_warmup: Whether in warm-up phase
            epoch: Current epoch (for dynamic weight calculation)

        Returns:
            Loss value (scalar)
        """
        if is_top1 is None:
            raise ValueError("is_top1 mask is required for TopKAnchoredMILLoss")

        # Logits clipping for numerical stability
        outputs = torch.clamp(outputs, min=-50.0, max=50.0)

        # Dynamic weight calculation
        if self.enable_dynamic_weight:
            if is_warmup:
                dynamic_weight = self.warmup_ce_weight
            else:
                # Stable phase: linear growth
                growth_progress = min(1.0, self.epoch_in_stable / self.growth_epochs)
                dynamic_weight = (
                    self.warmup_ce_weight +
                    (self.stable_ce_weight - self.warmup_ce_weight) * growth_progress
                )
            self.current_dynamic_weight = dynamic_weight
        else:
            dynamic_weight = 1.0
            self.current_dynamic_weight = 1.0

        # Compute base loss
        if is_warmup:
            base_loss = self.topk_anchored_loss(outputs, labels, is_top1, dynamic_weight)
        elif self.stable_loss_mode == "simple":
            # Ablation C: no tier stratification, unified CE on all Top-K
            base_loss = self._simple_stable_loss(outputs, labels, is_top1, dynamic_weight)
        else:
            base_loss = self.soft_bootstrapping_loss(outputs, labels, is_top1, dynamic_weight)

        # Add ranking loss if enabled
        if self.enable_ranking:
            inter_loss = self.compute_inter_ranking_loss(outputs, labels, is_top1)
            self.last_inter_loss = inter_loss.item()
            total_loss = base_loss + self.inter_weight * inter_loss
        else:
            self.last_inter_loss = 0.0
            total_loss = base_loss

        return total_loss

    def _simple_stable_loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        is_top1: torch.Tensor,
        dynamic_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Ablation C: Simple stable loss — no tier stratification.

        Unified CE on all positive tiles (Top-1 with higher weight, Top-2~K weak),
        plus standard CE on negatives. No Tier1/2/3 gating, no KL distillation.

        Formula:
            L = L_neg + dynamic_weight * top1_ce_weight * CE(Top-1) + top2k_soft_weight * CE(Top-2~K)

        Rationale:
            When Tier1 ratio is low (model still weak), three-tier stratification
            mostly produces Tier2/3 which gives weak/no signal. Simple CE lets the
            model learn from all selected tiles without manual gating.
        """
        neg_mask = labels == 0
        pos_mask = ~neg_mask
        top1_mask = pos_mask & is_top1
        top2k_mask = pos_mask & ~is_top1

        # Negative: standard CE
        if neg_mask.sum() > 0:
            neg_loss = F.cross_entropy(outputs[neg_mask], labels[neg_mask], reduction='sum')
        else:
            neg_loss = torch.tensor(0.0, device=outputs.device)

        # Top-1: full weight CE
        if top1_mask.sum() > 0:
            top1_ce = F.cross_entropy(outputs[top1_mask], labels[top1_mask], reduction='sum')
        else:
            top1_ce = torch.tensor(0.0, device=outputs.device)

        # Top-2~K: weak CE (same weight as warmup top2k)
        if top2k_mask.sum() > 0:
            top2k_ce = F.cross_entropy(outputs[top2k_mask], labels[top2k_mask], reduction='sum')
        else:
            top2k_ce = torch.tensor(0.0, device=outputs.device)

        # Reset tier counts (not applicable in simple mode)
        self.last_tier1_count = top1_mask.sum().item()
        self.last_tier2_count = 0
        self.last_tier3_count = 0

        pos_loss = (
            dynamic_weight * self.top1_ce_weight * top1_ce +
            self.top2k_soft_weight * top2k_ce
        )

        num_pos = pos_mask.sum()
        num_neg = neg_mask.sum()
        self.last_pos_loss = (pos_loss / num_pos).item() if num_pos > 0 else 0.0
        self.last_neg_loss = (neg_loss / num_neg).item() if num_neg > 0 else 0.0

        return (pos_loss + neg_loss) / (num_pos + num_neg)

    def topk_anchored_loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        is_top1: torch.Tensor,
        dynamic_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Top-K Anchored Loss for warmup phase.

        Strategy:
            - Negative tiles: Standard CE
            - Top-1 positive: CE with dynamic weight (classification anchor)
            - Top-2~K positive: Weak CE (gentle positive supervision)

        Design rationale (2026-02-09):
            Original NR_disease (-log(1 - max_disease_prob)) pushes Top-2~K
            AWAY from disease prediction. But with offline multi-scale positive
            pool + Scout class-specific selection, Top-2~K are almost certainly
            real disease tiles. NR fights against Top-1 CE, causing the model
            to get stuck predicting everything as background.

            Replaced with weak CE (10% weight): same direction as Top-1 CE,
            gentle enough to tolerate occasional noise.

        Args:
            outputs: (N, num_classes) logits
            labels: (N,) class labels
            is_top1: (N,) boolean tensor
            dynamic_weight: Dynamic weight multiplier

        Returns:
            Loss value
        """
        # Split samples
        neg_mask = labels == 0
        pos_mask = ~neg_mask
        top1_mask = pos_mask & is_top1
        top2k_mask = pos_mask & ~is_top1

        # 1. Negative tiles: Standard CE
        if neg_mask.sum() > 0:
            neg_loss = F.cross_entropy(outputs[neg_mask], labels[neg_mask], reduction='sum')
        else:
            neg_loss = torch.tensor(0.0, device=outputs.device)

        # 2. Top-1 positive tiles: CE
        if top1_mask.sum() > 0:
            top1_ce_loss = F.cross_entropy(outputs[top1_mask], labels[top1_mask], reduction='sum')
        else:
            top1_ce_loss = torch.tensor(0.0, device=outputs.device)

        # 3. Top-2~K positive tiles: Weak CE (replaces NR_disease)
        # Weight 0.1 = gentle supervision, tolerates noise but provides correct gradient direction
        if top2k_mask.sum() > 0:
            top2k_ce = F.cross_entropy(outputs[top2k_mask], labels[top2k_mask], reduction='sum')
            top2k_weak_loss = 0.1 * top2k_ce
        else:
            top2k_weak_loss = torch.tensor(0.0, device=outputs.device)

        # Combine with dynamic weight
        pos_loss = (
            dynamic_weight * self.top1_ce_weight * top1_ce_loss +
            self.top2k_nr_weight * top2k_weak_loss
        )

        # Record for logging
        num_pos = pos_mask.sum()
        num_neg = neg_mask.sum()
        self.last_pos_loss = (pos_loss / num_pos).item() if num_pos > 0 else 0.0
        self.last_neg_loss = (neg_loss / num_neg).item() if num_neg > 0 else 0.0

        # Total loss (normalized)
        total_loss = (pos_loss + neg_loss) / (num_pos + num_neg)

        return total_loss

    def soft_bootstrapping_loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        is_top1: torch.Tensor,
        dynamic_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Three-Tier Stratified Supervision for Stable Phase.

        CORE PRINCIPLE: "不歧视低置信度，歧视错误预测"

        THREE-TIER STRATEGY (Top-1):
            Tier 1 - Qualified (conf>threshold AND correct): 全权重CE (dynamic_w × top1_ce_weight)
            Tier 2 - Marginal (conf≤threshold AND correct):
                     Top-1: tier2_weak_weight × top1_ce_weight CE
                            [2026-02-26] tier2_weak_weight=1.0 — prediction direction trustworthy,
                            full CE to push confidence up; noise_drop in Tier3 provides the safety net
                     Top-2~K: tier2_top2k_weak_ce_weight × CE (direction unreliable, gentle guidance only)
            Tier 3 - Wrong (pred wrong):
                - Extreme conflict (gap > noise_drop_threshold) → Noise Drop (Loss=0)
                - Normal error → Correction Ranking

        TOP-2~K Per-bag KL with Dual-Gate Admission (2026-02-26):
            Gate A (bag): Top-1 is Tier1 AND conf > threshold
            Gate B (tile): Top-2~K tile argmax == bag_label (prediction direction correct)
            Admitted (A✓ B✓): KL distill from THIS bag's Top-1 soft dist (not batch mean)
            Rejected (A✓ B✗): fallback weak_ce (bad direction tile, no distillation)
            Gate A fail (Tier2 Top-1): weak_ce on all Top-2~K (tier2_top2k_weak_ce_weight)
            Teacher is Tier 3 → Silence (Loss=0)

        Args:
            outputs: (N, num_classes) logits
            labels: (N,) class labels
            is_top1: (N,) boolean tensor
            dynamic_weight: Dynamic weight multiplier

        Returns:
            Loss value
        """
        num_classes = outputs.size(1)
        probs = F.softmax(outputs, dim=1)

        # Split samples
        neg_mask = labels == 0
        pos_mask = ~neg_mask
        top1_mask = pos_mask & is_top1
        top2k_mask = pos_mask & ~is_top1

        # Initialize losses
        top1_ce_loss = torch.tensor(0.0, device=outputs.device)
        top2k_soft_loss = torch.tensor(0.0, device=outputs.device)
        tier3_correction_loss = torch.tensor(0.0, device=outputs.device)

        # Track tier statistics
        tier1_count = 0
        tier2_count = 0
        tier3_count = 0

        if top1_mask.sum() > 0:
            top1_probs = probs[top1_mask]
            top1_preds = top1_probs.argmax(dim=1)
            top1_labels = labels[top1_mask]
            top1_conf = top1_probs.max(dim=1)[0]

            # Tier classification
            correct_pred = (top1_preds == top1_labels)

            tier1_mask = correct_pred & (top1_conf > self.stable_gate_conf)
            tier2_mask = correct_pred & (top1_conf <= self.stable_gate_conf)
            tier3_mask = ~correct_pred

            tier1_count = tier1_mask.sum().item()
            tier2_count = tier2_mask.sum().item()
            tier3_count = tier3_mask.sum().item()

            # Tier 1: Qualified → 强CE/Focal
            if tier1_mask.sum() > 0:
                if self.enable_focal_loss:
                    tier1_ce_loss = self.focal_loss(
                        outputs[top1_mask][tier1_mask],
                        labels[top1_mask][tier1_mask]
                    )
                else:
                    tier1_ce_loss = F.cross_entropy(
                        outputs[top1_mask][tier1_mask],
                        labels[top1_mask][tier1_mask],
                        reduction='sum'
                    )
                top1_ce_loss = top1_ce_loss + dynamic_weight * self.top1_ce_weight * tier1_ce_loss

            # Tier 2: Marginal → 弱CE
            if tier2_mask.sum() > 0:
                tier2_weak_weight = self.tier2_weak_weight
                if self.enable_focal_loss:
                    tier2_ce_loss = self.focal_loss(
                        outputs[top1_mask][tier2_mask],
                        labels[top1_mask][tier2_mask]
                    )
                else:
                    tier2_ce_loss = F.cross_entropy(
                        outputs[top1_mask][tier2_mask],
                        labels[top1_mask][tier2_mask],
                        reduction='sum'
                    )
                top1_ce_loss = top1_ce_loss + tier2_weak_weight * self.top1_ce_weight * tier2_ce_loss

            # Tier 3: Wrong → Noise Drop + Correction Ranking
            if tier3_mask.sum() > 0:
                tier3_wrong_conf = top1_conf[tier3_mask]
                tier3_true_probs = top1_probs[tier3_mask].gather(
                    1, top1_labels[tier3_mask].unsqueeze(1)
                ).squeeze()

                conflict_gap = tier3_wrong_conf - tier3_true_probs

                # Noise Drop: extreme conflict
                noise_mask = conflict_gap > self.noise_drop_threshold
                correctable_mask = ~noise_mask
                correctable_count = correctable_mask.sum().item()

                if correctable_count > 0:
                    # [2026-02-02] Fix: Use sum/pos_count instead of mean
                    # Rationale: "错误越多惩罚越重", mean would normalize away this effect
                    pos_count = pos_mask.sum().item()
                    tier3_correction_loss = F.relu(
                        conflict_gap[correctable_mask] + self.correction_margin
                    ).sum() / max(pos_count, 1) * self.correction_weight

            # Safety net: When ALL Top-1 are Tier 3 (T1+T2=0), noise_drop zeros out pos_loss
            # → only neg_loss remains → model can't learn positive features → unrecoverable
            # Fallback: weak CE on Top-1 and Top-2~K to provide minimal positive gradient
            if tier1_count + tier2_count == 0 and tier3_count > 0:
                fallback_ce = F.cross_entropy(
                    outputs[top1_mask],
                    labels[top1_mask],
                    reduction='sum'
                )
                top1_ce_loss = self.tier3_fallback_top1_weight * fallback_ce
                # Also apply weak CE to Top-2~K (avoid complete silence)
                if top2k_mask.sum() > 0:
                    top2k_fallback_ce = F.cross_entropy(
                        outputs[top2k_mask],
                        labels[top2k_mask],
                        reduction='sum'
                    )
                    top2k_soft_loss = self.tier3_fallback_top2k_weight * top2k_fallback_ce
                print(f"[FALLBACK] All Top-1 Tier 3 ({tier3_count}), using weak CE (w={self.tier3_fallback_top1_weight})")

            # Top-2~K: Per-bag KL with dual-gate admission control (2026-02-26)
            # ---------------------------------------------------------------
            # OLD: batch-level mean of all Tier1 Top-1 → single blurred soft
            #      target distilled to ALL Top-2~K across ALL bags → diffuse
            # NEW: per-bag processing —
            #   Gate A (bag level):  bag's Top-1 is Tier1 AND conf > threshold
            #   Gate B (tile level): Top-2~K tile's argmax == bag_label
            #   Only tiles passing BOTH gates get KL distillation from
            #   their *own* bag's Top-1 soft distribution.
            #   Non-admitted tiles fallback to weak_ce or silence.
            # ---------------------------------------------------------------
            if top2k_mask.sum() > 0 and (tier1_count + tier2_count > 0):
                # Reconstruct bag boundaries from is_top1 flags
                # Layout: [bag0_T1, bag0_T2..Tk, bag1_T1, bag1_T2..Tk, ...]
                # We only operate on positive tiles (pos_mask), so gather pos-only views
                pos_indices = pos_mask.nonzero(as_tuple=True)[0]  # indices into full batch
                pos_is_top1 = is_top1[pos_indices]                 # Top-1 flags for pos tiles only
                pos_outputs = outputs[pos_indices]
                pos_labels = labels[pos_indices]
                pos_probs = probs[pos_indices]

                # Segment into bags: each True in pos_is_top1 starts a new bag
                bag_starts = pos_is_top1.nonzero(as_tuple=True)[0]  # positions of Top-1 in pos array
                num_bags = len(bag_starts)

                kl_top2k_indices = []    # global indices (into full batch) admitted for KL
                kl_soft_targets = []     # corresponding per-tile soft targets
                weak_ce_top2k_indices = []  # fallback weak_ce global indices

                for b in range(num_bags):
                    start = bag_starts[b].item()
                    end = bag_starts[b + 1].item() if b + 1 < num_bags else len(pos_indices)

                    bag_top1_pos_idx = start          # position in pos array
                    bag_top2k_pos_slice = slice(start + 1, end)  # Top-2~K positions in pos array

                    if end <= start + 1:
                        continue  # K=1, no Top-2~K tiles

                    # Reconstruct Top-1 tier for this bag's Top-1
                    t1_prob = pos_probs[bag_top1_pos_idx]       # (num_classes,)
                    t1_pred = t1_prob.argmax().item()
                    t1_conf = t1_prob.max().item()
                    t1_label = pos_labels[bag_top1_pos_idx].item()
                    is_tier1 = (t1_pred == t1_label) and (t1_conf > self.stable_gate_conf)

                    top2k_pos_probs = pos_probs[bag_top2k_pos_slice]     # (K-1, C)
                    top2k_global_idx = pos_indices[bag_top2k_pos_slice]  # global indices

                    # Gate B: Top-2~K tiles predicted in correct direction
                    top2k_preds = top2k_pos_probs.argmax(dim=1)
                    correct_direction = (top2k_preds == t1_label)

                    if is_tier1:
                        # Admitted tiles (Gate A ✓ + Gate B ✓): KL from this bag's Top-1
                        admitted = correct_direction
                        soft_target = t1_prob.detach().unsqueeze(0)  # (1, C) — per-bag, NOT batch mean

                        if admitted.any():
                            admitted_global = top2k_global_idx[admitted]
                            kl_top2k_indices.append(admitted_global)
                            kl_soft_targets.append(soft_target.expand(admitted.sum(), -1))

                        # Rejected tiles (Gate B ✗): fallback weak_ce
                        rejected = ~correct_direction
                        if rejected.any():
                            weak_ce_top2k_indices.append(top2k_global_idx[rejected])
                    else:
                        # Gate A failed (Tier2 or Tier3 Top-1): weak_ce on all Top-2~K
                        if self.tier2_top2k_strategy == "weak_ce":
                            weak_ce_top2k_indices.append(top2k_global_idx)
                        # else: "nr" or "silence" — keep top2k_soft_loss = 0

                # Compute KL loss (per-bag distillation from own Top-1)
                if kl_top2k_indices:
                    all_kl_idx = torch.cat(kl_top2k_indices)
                    all_soft_targets = torch.cat(kl_soft_targets)  # (N_admitted, C)
                    kl_log_probs = F.log_softmax(outputs[all_kl_idx], dim=1)
                    top2k_soft_loss = F.kl_div(kl_log_probs, all_soft_targets, reduction='sum')

                # Compute weak_ce fallback for non-admitted tiles
                if weak_ce_top2k_indices:
                    all_weak_idx = torch.cat(weak_ce_top2k_indices)
                    weak_ce_labels = labels[all_weak_idx]
                    weak_ce_loss = F.cross_entropy(outputs[all_weak_idx], weak_ce_labels, reduction='sum')
                    top2k_soft_loss = top2k_soft_loss + self.tier2_top2k_weak_ce_weight * weak_ce_loss

                # Legacy path: only reached if tier2_top2k_strategy == "nr" (not default)
                # Default strategy is "weak_ce", handled per-bag in the loop above.
                if not kl_top2k_indices and not weak_ce_top2k_indices:
                    if tier2_count > 0 and self.tier2_top2k_strategy == "nr":
                        top2k_probs_all = F.softmax(outputs[top2k_mask], dim=1)
                        max_disease_prob = top2k_probs_all[:, 1:].max(dim=1)[0]
                        top2k_nr_loss = -torch.log(1.0 - max_disease_prob + self.epsilon).sum()
                        top2k_soft_loss = self.stable_nr_weight * top2k_nr_loss

                # 策略3: Teacher is Tier 3 → Silence (top2k_soft_loss stays 0)

            # Track gate filtering ratio
            if top1_mask.sum() > 0:
                self.last_gate_filtered_ratio = 1.0 - (tier1_count / top1_mask.sum().item())
            else:
                self.last_gate_filtered_ratio = 0.0

        else:
            self.last_gate_filtered_ratio = 0.0

        # Store tier counts
        self.last_tier1_count = tier1_count
        self.last_tier2_count = tier2_count
        self.last_tier3_count = tier3_count

        # Negative tiles: Standard CE
        if neg_mask.sum() > 0:
            neg_loss = F.cross_entropy(outputs[neg_mask], labels[neg_mask], reduction='sum')
        else:
            neg_loss = torch.tensor(0.0, device=outputs.device)

        # Combine
        pos_loss = (
            top1_ce_loss +
            self.top2k_soft_weight * top2k_soft_loss +
            tier3_correction_loss
        )

        # Record for logging
        num_pos = pos_mask.sum()
        num_neg = neg_mask.sum()
        self.last_pos_loss = (pos_loss / num_pos).item() if num_pos > 0 else 0.0
        self.last_neg_loss = (neg_loss / num_neg).item() if num_neg > 0 else 0.0

        # Total loss (normalized)
        total_loss = (pos_loss + neg_loss) / (num_pos + num_neg)

        return total_loss


__all__ = ['TopKAnchoredMILLoss']
