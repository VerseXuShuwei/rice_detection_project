# Weakly-Supervised Rice Disease Detection via Asymmetric Multiple Instance Learning

> **From image-level labels to pixel-level localization** — detecting and localizing 8 rice diseases without any bounding box or pixel annotations.

---

## Overview

Traditional rice disease detection requires expensive pixel-level or bounding-box annotations from domain experts. This project removes that requirement:

- **Input**: Image-level labels only ("this image contains Brown Spot")
- **Output**: Spatial heatmaps + detection boxes showing *where* the disease is
- **8 disease classes** + 1 background = **9-class output**
- **Hardware**: Trained entirely on RTX 4060 Laptop (8GB VRAM)

## Results

### Best Config: A3-corrected (EfficientNetV2-S bare, corrected loss weights, λ₁=2.0)

| Metric | Value |
|--------|-------|
| Overall Accuracy (filtered) | **91.25%** |
| Disease Macro-F1 (C1–C8) | **0.8569** |
| Background Recall (Neg Recall) | **99.15%** |
| Avg Top-1 Confidence | **99.24%** |
| Hit Accuracy (bag-level) | **99.89%** |
| Negative Hallucination | **0.94%** |
| Tiles Retained | **46.7%** (22,625 / 48,454) |

> **Lightweight alternative**: EfficientNet-B0 (bare) reaches 89.07% accuracy with ~4× fewer backbone parameters and faster training — preferred for deployment and rapid iteration.

### Per-Class Performance (A3-corrected — V2-S bare, λ₁=2.0)

| Class | Precision | Recall | F1 |
|:------|:---------:|:------:|:--:|
| Background | 1.000 | 0.992 | 0.996 |
| Leaf Blast | 0.934 | 0.963 | 0.949 |
| Rice Leaf Beetle | 0.965 | 0.917 | 0.940 |
| Brown Spot | 0.889 | 0.940 | 0.914 |
| Node Neck Blast | 0.916 | 0.907 | 0.912 |
| Rice Leaf Folder | 0.782 | 0.889 | 0.831 |
| Sheath Blight | 0.876 | 0.749 | 0.809 |
| False Smut | 0.805 | 0.784 | 0.794 |
| Bact. Leaf Blight | 0.612 | 0.830 | 0.704 |

---

## Method: Scout-Snipe Asymmetric MIL

### Core Idea

In weakly-supervised MIL, a "bag" (image) contains many tiles, but only some are truly diseased. Naively training on all tiles introduces massive label noise. Our solution: **separate the search from the learning**.

```
SCOUT PASS (no gradient)
  All tiles → model.eval() → score each → Select Top-K disease tiles
                                              ↓
SNIPE PASS (gradient enabled)
  Top-K tiles + negative tiles → model.train() → Tiered loss → Backprop
```

The Scout explores without contaminating gradients. The Snipe trains only on high-quality selections.

### Critical Foundation: Negative Pool

The entire framework rests on a confirmed-clean negative sample pool (46,391 tiles from verified healthy rice images). This establishes a calibrated background baseline — without it, the model bootstraps from its own uncertain beliefs.

---

## Architecture

The primary architecture is a **standard CNN backbone with Conv1×1 spatial classifier + global max pooling** — no additional neck or attention modules. Ablation results show the bare backbone outperforms the hybrid variant, confirming that the training strategy, not architectural complexity, drives performance.

### Production Architecture (Bare Backbone)

```
Input (B, 3, 384, 384)
    ↓
EfficientNet Backbone (V2-S or B0)
    Stages 0-2: FROZEN (ImageNet features)
    Stages 3+:  TRAINABLE (disease-specific)
    ↓
Conv1×1 (spatial classifier → 9-channel heatmap)
    ↓
Global Max Pooling → 9-class logits
```

The Conv1×1 acts as a minimal spatial classifier: applied before pooling, it produces per-location class scores (i.e., a spatial heatmap), and GMP selects the strongest response as the tile-level prediction. This enables pixel-level disease localization without any dedicated heatmap module.

| Backbone | Params | Overall Acc | Disease Macro-F1 |
|:---|:---:|:---:|:---:|
| EfficientNetV2-S (A3-corrected) | 22.15M | **91.25%** | **0.8569** |
| EfficientNet-B0 | 5.29M | 89.07% | 0.8199 |

### Experimental: CNN-ViT Hybrid with FPN

An experimental hybrid variant adds FPN neck, ViT residual block, and HeatmapHead after the backbone. While architecturally interesting, ablation shows it **underperforms** the bare backbone on V2-S (87.46% vs 91.25%) and provides only marginal gains on B0 (89.42% vs 89.07%). The additional training complexity (3-way LR, backbone freezing schedule, hybrid warmup) introduces optimization overhead that offsets representational benefits at 384×384 tile resolution.

<details>
<summary>Hybrid architecture details (click to expand)</summary>

```
Input (B, 3, 384, 384)
    ↓
EfficientNetV2-RW-S Backbone
    Stages 0-2: FROZEN (ImageNet features)
    Stages 3-7: TRAINABLE (disease-specific)
    ↓
FPN Neck (Multi-scale Fusion)
    Stage 3 (160ch, 24×24) + Stage 4 (1792ch, 12×12)
    → Fused output: (B, 256, 24×24)
    ↓
ViT Residual Block (Global Attention)
    576 tokens, 8 heads, learnable 2D PE
    0.94M params (41× reduction vs full-dim placement)
    ↓
HeatmapHead
    Conv1×1 → 9 channels (spatial disease response)
    TopK-Mean Pooling → class logits
```

**Total**: 24.18M parameters (Backbone 22.15M, FPN 1.09M, ViT 0.94M)

**Design decision**: ViT placed *after* FPN dimension reduction (256ch), not before (1792ch), achieving 41× parameter reduction. Placing ViT at 1792d would require 38.6M params and OOM on 8GB GPU.

</details>

---

## Training: Three-Tier Quality-Aware Loss

### Equal Contribution Principle

Sampling is unified across both phases (8 bags × K=3): 8 neg, 8 top-1, 16 top-2~K tiles per batch. Loss weights are calibrated so all three groups contribute equally (~8.0 total per batch):

- `top1_ce_weight = 1.0` → 8 × 1.0 = 8.0
- `top2k_nr_weight = 0.5` (warmup weak CE) → 16 × 0.5 = 8.0
- `top2k_soft_weight = 1.0` (stable, transparent passthrough; internal `tier1_kl_weight=0.5` controls KL)

### Phase 1 — Warmup (Epochs 1–15)
- Top-1: CE (×1.0) with dynamic weight (warmup=1.0, no compression)
- Top-2~K: Weak CE (×0.5, equal contribution baseline)
- Inter-bag ranking loss for margin-based separation (margin=0.3)

### Phase 2 — Stable (Epochs 16–30)

| Tier | Condition | Top-1 | Top-2~K |
|------|-----------|-------|---------|
| **Tier 1** (Qualified) | Correct & conf > 0.45 | CE (×1.0, dynamic) | Per-bag KL (weight=0.5) |
| **Tier 2** (Marginal) | Correct & conf ≤ 0.45 | CE (×1.0, full — direction trustworthy) | Weak CE (×0.25) |
| **Tier 3** (Wrong) | Incorrect | Noise Drop (gap>0.5) / Correction Ranking | Silence |

### Per-Bag KL Distillation (Key Innovation)

Standard batch-mean KL causes **heatmap diffusion** — mixing cross-class distributions makes the model "see disease everywhere." Our dual-gate per-bag approach ensures each tile learns from its own bag's highest-quality prediction:

- **Gate A** (bag-level): Is this bag's Top-1 a Tier 1 prediction?
- **Gate B** (tile-level): Does this tile's argmax match the bag label?
- Both pass → KL using *this bag's own* Top-1 soft distribution
- Either fails → Fallback to weak CE

### Top-1 Anchor Leverage (λ₁ ≥ 2.0)

Ablation reveals that the tier gating mechanism collapses when top-1 anchor weight equals other tiles (λ₁=1.0), causing 64% C2→C4 confusion. Raising to λ₁=2.0 restores tier gates to designed behavior. This **leverage principle** and the equal-contribution principle are both necessary simultaneously.

---

## Anti-Collapse Mechanisms

| Mechanism | Problem Solved |
|-----------|---------------|
| BN frozen in Snipe pass | Scout/Snipe statistics divergence |
| Tier 2 full weight (1.0) | Prevents Silence Spiral on marginal predictions |
| Per-bag KL (not batch-mean) | Prevents cross-class heatmap diffusion |
| Tier 3 safety net | Weak CE fallback when all Top-1s fail |
| Scale Diversity | Multi-scale tile deduplication |
| Confidence threshold 0.45 | Meaningful quality gate above random baseline |

---

## Inference Pipeline

Multi-scale sliding-window inference produces spatial disease heatmaps from full-resolution images.

```
Original Image (3000×4000)
    ↓
Multi-Scale Tiling
    Tile sizes: [768, 1024, 1536] px
    Stride: 50% overlap
    ↓
Batch Inference (batch_size=8)
    model.get_spatial_heatmap() → (B, 9, 24, 24)
    ↓
Scale-Weighted Heatmap Accumulation
    BG-aware downweighting (2×)
    Feather window (15% taper)
    TopK spatial normalization (k=3)
    ↓
Full-Resolution Probability Map (9, H, W)
    ↓
Post-Processing
    Gaussian blur + percentile threshold
    Connected Component Analysis (CCA)
    Greedy NMS (IoU > 0.3)
    Per-region class refinement
    BG-region filtering (mean_heatmap[0] > 0.5)
    ↓
Detection Results
    coords, class_id, confidence, area
```

### GUI Diagnostic Tool (6 Tabs)

| Tab | Name | Content |
|-----|------|---------|
| 1 | Detection | CCA boxes + NMS + tile gallery |
| 2 | Entropy | Per-scale disease entropy map |
| 3 | Per-Class | Heatmap + p90/max stats by class |
| 4 | Components | Backbone/FPN/ViT PCA + attention |
| 5 | Scale Debug | Per-scale top-K tile boxes |
| 6 | Ent-Filter | CCA + entropy gate (confidence × entropy dual threshold) |

All inference parameters are YAML-configurable via `configs/inference/default.yaml`.

---

## Ablation Study

### Design Rationale

The ablation study is structured along three orthogonal dimensions:

1. **Architecture** (Backbone × Components): Which components contribute? 2×2 factorial + component isolation.
2. **Loss** (Tiered vs Simple CE): Is three-tier quality-aware loss necessary?
3. **Pooling** (GMP vs TopK-Mean): Does pooling mode matter for hybrid architecture?

Loss and pooling ablations are conducted on B0, justified by architecture ablation showing < 1pp difference between backbones in matched configurations.

### Architecture Ablation

| Experiment | Backbone | Components | Pool Mode | Overall Acc | Macro-F1 | Neg Recall | Status |
|:-----------|:---------|:-----------|:---------:|:-----------:|:--------:|:----------:|:------:|
| **Arch-1** | V2-S | None | backbone GMP† | 0.9003 | 0.8434 | 0.9909 | ✅ |
| Arch-2 | B0 | None | backbone GMP† | 0.8907 | 0.8199 | 0.9848 | ✅ |
| Arch-3 | V2-S | FPN+ViT+Head | topk_mean | 0.8746 | 0.8015 | 0.9863 | ✅ |
| Arch-4 | B0 | FPN+ViT+Head | topk_mean | 0.8942 | 0.8243 | 0.9789 | ✅ |
| **Arch-5** | V2-S | FPN only | topk_mean | 0.9020 | 0.8305 | 0.9793 | ✅ |
| **Arch-6** | B0 | FPN only | topk_mean | 0.9017 | 0.8332 | 0.9761 | ✅ |

> † "backbone GMP" = `heatmap_head.enable: false`. Model uses internal Conv1×1 + backbone-native GMP. NOT `pool_mode: gmp`.

#### Key Findings

1. **Training strategy dominates.** Bare V2-S (Arch-1, 90.03%) outperforms full hybrid (Arch-3, 87.46%) by +2.57pp.
2. **FPN helps, more so on B0.** V2-S +0.17pp, B0 +1.10pp — smaller backbone benefits more from multi-scale fusion.
3. **ViT hurts both backbones.** V2-S −2.74pp, B0 −0.75pp — global attention at 384×384 introduces optimization overhead without proportional gain.
4. **Cross-backbone consistency.** V2-S vs B0 differ by < 1pp in matched configs, validating B0-only ablation for loss/pooling.

### Loss Ablation

| Experiment | Config | Loss Mode | Overall Acc | Macro-F1 | Hit Acc | Status |
|:-----------|:-------|:----------|:-----------:|:--------:|:-------:|:------:|
| Loss-1 | B0+FPN | Tiered (λ₁=1.0) | 0.8322 | 0.7466 | 0.9254 | ✅ |
| Loss-2 | B0+FPN | Simple CE | 0.8834 | 0.8128 | 0.9958 | ✅ |
| Loss-3 | B0+FPN | Tiered (λ₁=2.0, dw 0.7→1.0) | 0.8858 | 0.8130 | 0.9936 | ✅ |
| **Loss-4** | V2-S bare | Tiered (λ₁=2.0, corrected) | **0.9125** | **0.8569** | **0.9989** | ✅ |

**Critical finding**: λ₁=1.0 causes catastrophic class collapse (C2→C4 confusion at 64%). λ₁=2.0 restores tier gates. Simple CE is a surprisingly strong baseline — tier stratification's value comes primarily from the top-1 leverage mechanism, not the gating itself.

### Pooling Ablation (B0 + FPN+ViT+Head)

| Experiment | Pool Mode | Overall Acc | Macro-F1 | Neg Halluc. |
|:-----------|:----------|:-----------:|:--------:|:-----------:|
| Pool-1 | GMP | 0.8555 | 0.7610 | 3.51% |
| Pool-2 | TopK-Mean (k=3) | 0.8942 | 0.8243 | 2.40% |

TopK-Mean outperforms GMP by +3.87pp. GMP's winner-take-all gradient destabilizes training in the hybrid architecture; TopK-Mean distributes gradients across top-3 spatial locations for smoother optimization.

> **Note**: This result is specific to the hybrid architecture's HeatmapHead. Bare backbone experiments use backbone-native GMP and achieve top performance (90%+), confirming GMP itself is not inherently inferior — the hybrid's additional learnable layers benefit from softer gradients.

---

## Negative Results

These failed experiments informed our final design:

1. **Feature Critic with ImageNet features** — Pretrained features see healthy and diseased tiles as nearly identical (cosine sim gap: −0.0009). ImageNet provides no meaningful prior for rice pathology.

2. **Spatial NMS** — Limited effectiveness with tile overlap. Replaced with Scale Diversity.

3. **Negative Rejection loss** — Gradient direction ("don't be disease" applied to top-2~K) conflicts with Top-1 CE ("be this disease"), causing progressive suppression of positive responses (Silence Spiral).

---

## Future Work: Adaptive Perception-Decision Loop

> Preliminary exploration, not part of the current ablation study.

Investigating replacing the fixed multi-scale sliding window with a learned decision loop: a lightweight policy network decides {zoom_in, shift, stop} based on confidence, heatmap entropy, and deep features. Three progressive designs under evaluation:

1. **MVP**: MLP + manual state buffer (scalar observations only)
2. **GRU**: RNN + 4×4 spatial pooling (time-series memory)
3. **CNN Feature Injection** (recommended): FPN features + heatmap → lightweight CNN policy with continuous coordinate regression

Target: 30% tile coverage for 90%+ detection consistency vs full-coverage baseline.

---

## Dataset

| Property | Value |
|----------|-------|
| Positive bags (images) | 2,819 |
| Negative bags (images) | 1,127 |
| Disease classes | 8 |
| Total output classes | 9 (8 disease + 1 background) |
| Positive tiles (multi-scale) | 158,277 (100K train / 58K val) |
| Negative tiles (multi-scale) | 46,391 (37K train / 9K val) |
| Tile scales | 768 / 1024 / 1536 px |
| Model input size | 384 × 384 |

---

## Project Structure

```
rice_detection/
├── configs/                  # YAML configuration (all hyperparameters)
│   ├── algorithm/            # MIL strategy, loss, feature critic
│   ├── dataset/              # Class definitions, data paths, augmentation
│   ├── model/                # Architecture, hybrid components
│   ├── inference/            # Inference engine parameters
│   └── trainer/              # Optimizer, scheduler, evaluation
├── scripts/
│   ├── train.py              # Training entry point
│   └── tools/                # Data preprocessing (build pools, prototypes)
├── src/
│   ├── core/                 # Abstract interfaces, checkpoint manager
│   ├── data/                 # Dataset, tile pools (LMDB), sampler, augmentation
│   ├── models/               # EfficientNetV2-S / B0 / ResNet50 + FPN + ViT + heads
│   ├── losses/               # TopK-Anchored MIL loss (tiered, three-tier quality gating)
│   ├── trainer/              # Asymmetric MIL trainer (Builder + State + Engine)
│   ├── inference/            # Unified inference engine + GUI diagnostic tool
│   ├── evaluation/           # Warmup evaluator (P0), final evaluator, heatmap visualizer
│   ├── critics/              # Feature Critic (experimental, ineffective on this dataset)
│   └── utils/                # Config I/O, logging, scheduling, visualization
└── deploy/                   # ONNX export, inference SDK (WIP)
```

---

## Technical Environment

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 4060 Laptop (8GB VRAM) |
| Framework | PyTorch + timm + Albumentations |
| Backbone | EfficientNetV2-RW-S / EfficientNet-B0 (ImageNet pretrained) |
| Training VRAM | ~2.4 GB (8 tiles, with AMP) |
| Inference VRAM | ~1.0 GB (32 tiles batch) |
| Data storage | LMDB (tile pools with precomputed 1792-dim features) |

---

## Quick Start

```bash
# 1. Build negative tile pool (required before first training)
python scripts/tools/build_negative_pool.py --config configs/algorithm/train_topk_asymmetric.yaml

# 2. Build positive tile pool (offline multi-scale tiling)
python scripts/tools/build_positive_pool.py --config configs/algorithm/train_topk_asymmetric.yaml

# 3. Train
python scripts/train.py --config configs/algorithm/train_topk_asymmetric.yaml

# 4. Resume from checkpoint
python scripts/train.py --config configs/algorithm/train_topk_asymmetric.yaml \
    --resume outputs/checkpoints/exp_name/epoch_015.pth

# 5. Run tests
python -m pytest tests/ -v
```

---

## Publication

**Deep Learning Methods for Rice Disease Detection: Evolution and Challenges**
X. Shuwei et al. — Published in *RoViSP 2025* (International Conference on Robotics, Vision, Signal Processing and Power Applications)

---

## License

This project is part of an MSc thesis at Universiti Sains Malaysia.
