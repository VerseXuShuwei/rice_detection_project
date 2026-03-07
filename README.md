# Weakly-Supervised Rice Disease Detection via Asymmetric Multiple Instance Learning

> **From image-level labels to pixel-level localization** — detecting and localizing 9 rice diseases without any bounding box or pixel annotations.

<!--
<p align="center">
  <img src="assets/demo_detection.png" width="800"/>
</p>
-->

---

## Overview

Traditional rice disease detection requires expensive pixel-level or bounding-box annotations from domain experts. This project decreased that requirement:

- **Input**: Image-level labels only ("this image contains Brown Spot")
- **Output**: Spatial heatmaps + detection boxes showing *where* the disease is
- **9 disease classes** + background (10-class classification)
- **Hardware**: Trained entirely on RTX 4060 Laptop (8GB VRAM)

## Architecture

![Model Architecture](docs/architecture.png)

## Results

![detection box](results/detection_box_example.png)


> top-1 detection box scores 0.892; other top-2~5 boxes below 0.6

![Heatmap Visualization](results/perclass_heatmap_example.png)

### Key Results (Best: A3 — EfficientNetV2-S, no hybrid components)

| Metric | Value |
|--------|-------|
| Overall Accuracy (filtered) | **90.03%** |
| Disease Macro-F1 (C1–C8) | **0.8434** |
| Background Recognition (Neg Recall) | **99.09%** |
| Avg Top-1 Confidence | **99.42%** |
| Hit Accuracy (bag-level) | **99.89%** |
| Negative Hallucination | **1.03%** |
| Tiles Retained | **47.2%** (22,888 / 48,454) |

> **Lightweight alternative**: EfficientNet-B0 (no hybrid) reaches 89.07% accuracy with ~4× fewer backbone parameters and faster training, making it the preferred choice for deployment and future iteration.

<p align="center">
<img src="results/confusion_matrix_filtered.png" alt="confusion matrics" width="60%"/>
</p>

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

The primary architecture is a **standard CNN backbone with global max pooling** — no additional neck or attention modules. Ablation results show the bare backbone outperforms the hybrid variant, confirming that the training strategy, not architectural complexity, drives performance.

### Production Architecture (Bare Backbone)

```
Input (B, 3, 384, 384)
    ↓
EfficientNet Backbone (V2-S or B0)
    Stages 0-2: FROZEN (ImageNet features)
    Stages 3+:  TRAINABLE (disease-specific)
    ↓
Conv1×1 (spatial classifier → 10-channel heatmap)
    ↓
Global Max Pooling → 10-class logits
```

The Conv1×1 acts as a minimal spatial classifier: applied before pooling, it produces per-location class scores (i.e., a spatial heatmap), and GMP selects the strongest response as the tile-level prediction. This enables pixel-level disease localization without any dedicated heatmap module.

| Backbone | Params | Overall Acc | Disease Macro-F1 |
|:---|:---:|:---:|:---:|
| EfficientNetV2-S | 22.15M | **90.03%** | **0.8434** |
| EfficientNet-B0 | 5.29M | 89.07% | 0.8199 |

### Experimental: CNN-ViT Hybrid with FPN

An experimental hybrid variant adds FPN neck, ViT residual block, and HeatmapHead after the backbone. While architecturally interesting, ablation shows it **underperforms** the bare backbone on V2-S (87.46% vs 90.03%) and provides only marginal gains on B0 (89.42% vs 89.07%). The additional training complexity (3-way LR, backbone freezing schedule, hybrid warmup) introduces optimization overhead that offsets representational benefits at 384×384 tile resolution.

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
    Conv1×1 → 10 channels (spatial disease response)
    TopK-Mean Pooling → class logits
```

**Total**: 24.18M parameters (Backbone 22.15M, FPN 1.09M, ViT 0.94M)

**Design decision**: ViT placed *after* FPN dimension reduction (256ch), not before (1792ch), achieving 41× parameter reduction.

</details>

---

## Training: Three-Tier Quality-Aware Loss

### Phase 1 — Warmup (Epochs 1–15)
- Top-1: Strong CE (×5.0) as anchor signal
- Top-2~K: Weak CE (×0.1)
- Inter-bag ranking loss for margin-based separation

### Phase 2 — Stable (Epochs 16–30)

| Tier | Condition | Top-1 | Top-2~K |
|------|-----------|-------|---------|
| **Tier 1** (Qualified) | Correct & conf > 0.45 | Strong CE (×5.0) | Per-bag KL distillation |
| **Tier 2** (Marginal) | Correct & conf ≤ 0.45 | Full CE (×1.0) | Weak CE |
| **Tier 3** (Wrong) | Incorrect | Noise Drop / Correction | Silence |

### Per-Bag KL Distillation (Key Innovation)

Standard batch-mean KL causes **heatmap diffusion** — mixing cross-class distributions makes the model "see disease everywhere." Our dual-gate per-bag approach ensures each tile learns from its own bag's highest-quality prediction:

- **Gate A** (bag-level): Is this bag's Top-1 a Tier 1 prediction?
- **Gate B** (tile-level): Does this tile's argmax match the bag label?
- Both pass → KL using *this bag's own* Top-1 soft distribution
- Either fails → Fallback to weak CE

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

## Inference Pipeline()

```
raw image (H×W)
  │
  ├─ _select_tile_sizes() → [768, 1024, 1536]（large image）or [384, 512, 768]（small image）
  │
  ├─ tile_size:
  │    ├─ stride = tile_size × 0.5
  │    ├─ Sliding window cut tile（including edge handling）
  │    ├─ each tile → resize到384×384 → inference
  │    │    ├─ get_spatial_heatmap() → (N, 10, 24, 24) → softmax → resize to tile region
  │    │    └─ predict_instances() → (N, 10) → tiles_info
  │    │
  │    └─ Weighted cumulative heatmap_accum:
  │         ├─ scale_weight = ts / max_ts（large scale tile have high weight）
  │         ├─ bg_aware_weight = max(0, 1 - 2×bg_prob)（degrade high bg tile disease channel）
  │         └─ feather_window（Hann window eliminates splicing marks）
  │
  ├─ TopK spatial normalization （each disease channel divide by its top-3 average） 
  │
  └─ extract_detections():
       ├─ each diseases percentile threshold → binary → Morphology → CCA → bbox
       ├─ cross-class IoU NMS (>0.3 merge)
       ├─ BG-region filter（bg_prob mean >0.7 delete）
       └─ ranking by area×confidence → top_k

```

Features:
- **Adaptive tile selection**: Large images use [768, 1024, 1536], small images use [384, 512, 768]
- **Scale-weighted fusion**: Large tiles get higher weight to prevent small-tile signal domination
- **Tile-edge feathering**: Raised cosine windows eliminate grid artifacts at tile boundaries
- **BG-aware reweighting**: Tiles with bg_prob ≥ 0.5 contribute zero to the disease heatmap

---

## Changes being made on Inference Pipeline:

> Now considerating replace its function usingAdaptive Perception - Prototype of the decision loop（Preliminary plan :Depth Anything V2 + GRU decision module）

```
Freeze components (The model trained by this project's strategy):
├─ Disease classification model (ONNX)
└─ Depth Anything V2 Small (pre-trained)
Learnable components:
└─ Decision Module (lightweight GRU/MLP)
     input: class confidence + heatmap entropy + depth features + action history
     output: {zoom_in, zoom_out, shift, stop}
```

Progress: Writing a rule-based version - validating concepts(20260306)

---

## Ablation Study

### Architecture Ablation (March 2026)

Systematic 2×2 ablation isolating **backbone** (EfficientNetV2-S vs B0) × **hybrid components** (FPN + ViT + HeatmapHead).

All configs share: Scout-Snipe asymmetric MIL, TopK-Anchored loss with three-tier quality gating, negative tile pool, scale diversity, 30 epochs, seed 42.

#### Overall Metrics

| Experiment | Backbone | Components | Overall Acc | Disease Macro-F1 | Neg Recall | TopK Lift |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **A3** | V2-S | None | **0.9003** | **0.8434** | **0.9909** | 0.5100 |
| A2 | B0 | FPN+ViT+Head | 0.8942 | 0.8243 | 0.9789 | **0.6036** |
| A4 | B0 | None | 0.8907 | 0.8199 | 0.9848 | 0.5503 |
| Baseline | V2-S | FPN+ViT+Head | 0.8746 | 0.8015 | 0.9863 | 0.5776 |

#### Per-Class F1 Comparison (Filtered, C1–C8)

| Class | Baseline (V2S+Comp) | A2 (B0+Comp) | A3 (V2S bare) | A4 (B0 bare) | Best |
|:---|:---:|:---:|:---:|:---:|:---:|
| 1 Bact-Leaf-Blight | 0.5606 | **0.7368** | 0.6690 | 0.6208 | A2 |
| 2 Brown-Spot | 0.8662 | 0.8850 | **0.9129** | 0.9039 | A3 |
| 3 False-Smut | 0.7131 | **0.7528** | 0.7412 | 0.7262 | A2 |
| 4 Leaf-Blast | 0.9111 | 0.9023 | 0.9244 | **0.9353** | A4 |
| 5 Neck-Blast | 0.8437 | 0.8700 | **0.8888** | 0.8709 | A3 |
| 6 Leaf-Beetle | 0.9356 | 0.9205 | **0.9626** | 0.9334 | A3 |
| 7 Leaf-Folder | 0.7942 | 0.7415 | **0.8622** | 0.7808 | A3 |
| 8 Sheath-Blight | 0.7872 | 0.7854 | 0.7863 | **0.7877** | A4 |
| **Macro (C1–C8)** | 0.8015 | 0.8243 | **0.8434** | 0.8199 | **A3** |

#### Additional Evaluation Metrics

| Experiment | Hit Acc | Top1 Conf | Neg Hallucination | Filtered Tiles |
|:---|:---:|:---:|:---:|:---:|
| A3 (V2S bare) | 0.9989 | 0.9942 | 0.0103 | 22,888 / 48,454 |
| A2 (B0+Comp) | 0.9968 | 0.9902 | 0.0240 | 17,963 / 48,454 |
| A4 (B0 bare) | 1.0000 | 0.9906 | 0.0175 | 20,968 / 48,454 |
| Baseline (V2S+Comp) | 0.9947 | 0.9888 | 0.0150 | 19,890 / 48,454 |

#### Key Findings

1. **Training strategy dominates architecture.** The bare V2-S backbone (A3) outperforms the full hybrid stack (Baseline) by **+2.6pp accuracy** and **+4.2pp Macro-F1**. The Scout-Snipe framework with tiered loss is the primary driver of performance — not architectural complexity.

2. **Hybrid components hurt V2-S, marginally help B0.** Adding FPN+ViT+HeatmapHead degrades V2-S performance across the board, while giving B0 a small boost (+0.35pp acc, +0.44pp F1). The larger backbone already extracts sufficient multi-scale features; the hybrid modules introduce optimization overhead without proportional representational gain at 384×384 resolution.

3. **B0 is a viable lightweight alternative.** At ~4× fewer backbone parameters, B0 (bare) reaches 89.07% accuracy — only 0.96pp behind V2-S (bare). With components, B0 narrows the gap further to 0.61pp. B0 also trains faster and uses less VRAM, making it attractive for deployment and rapid iteration.

4. **Per-class patterns reveal backbone-specific strengths.** A2 (B0+components) wins on diagnostically challenging classes (Bact-Leaf-Blight, False-Smut), suggesting the FPN attention mechanism helps B0 focus on subtle lesion patterns that its smaller feature space would otherwise miss. A3 (V2-S bare) dominates on texture-rich classes (Beetle, Folder, Neck-Blast).

### Training Strategy Ablation (Historical)

Earlier ablation on training strategy components (all using EfficientNetV2-S + FPN + ViT + HeatmapHead):

| Config | Neg Recall | Hit Acc | Overall Acc (filtered) | Neg Hallucination |
|--------|:---:|:---:|:---:|:---:|
| 0209 (Baseline) | 94.7% | 98.1% | 71.2% | 6.9% |
| 0224 | 94.1% | 98.3% | 67.9% | 7.5% |
| 0225 | 90.9% | 99.1% | 66.0% | 10.7% |
| **0226** | **96.8%** | **98.7%** | **81.3%** | **3.6%** |

The combination of conf_threshold=0.45 + Tier 2 full weight + warmup 15 epochs produced a 10+ point jump in overall accuracy, demonstrating synergistic effects of training strategy refinements.

---

## Negative Results

These failed experiments informed our final design:

1. **Feature Critic with ImageNet features** — Pretrained features see healthy and diseased tiles as nearly identical (cosine sim gap: −0.0009). ImageNet provides no meaningful prior for rice pathology.

2. **Spatial NMS** — Limited effectiveness With tile overlap. Replaced with Scale Diversity.

3. **Negative Rejection loss** — Gradient direction ("don't be disease" to top2~k) conflicts with Top-1 CE ("be this disease"), causing progressive suppression of positive responses.

---

## Per-Class Performance (Best Config: A3 — V2-S, No Components)

| Class | Precision | Recall | F1 |
|:------|:---------:|:------:|:--:|
| Background | 1.000 | 0.991 | 0.995 |
| Rice Leaf Beetle | 0.977 | 0.948 | 0.963 |
| Leaf Blast | 0.909 | 0.941 | 0.924 |
| Brown Spot | 0.902 | 0.924 | 0.913 |
| Node Neck Blast | 0.905 | 0.873 | 0.889 |
| Rice Leaf Folder | 0.803 | 0.931 | 0.862 |
| Sheath Blight | 0.869 | 0.718 | 0.786 |
| False Smut | 0.802 | 0.689 | 0.741 |
| Bact. Leaf Blight | 0.590 | 0.772 | 0.669 |

---

## Dataset

| Property | Value |
|----------|-------|
| Positive bags (images) | 2,819 |
| Negative bags (images) | 1,127 |
| Disease classes | 9 |
| Positive tiles (multi-scale) | 158,277 (100K train / 58K val) |
| Negative tiles (multi-scale) | 46,391 (37K train / 9K val) |
| Tile scales | 768 / 1024 / 1536 px |
| Model input size | 384 × 384 |

---

## Project Structure

```
rice_detection/
├── configs/                  # YAML configuration files
│   ├── algorithm/            # MIL strategy, loss, feature critic
│   ├── dataset/              # Data paths, augmentation, pools
│   ├── model/                # Architecture, hybrid components
│   └── trainer/              # Optimizer, scheduler, evaluation
├── scripts/
│   ├── train.py              # Training entry point
│   └── tools/                # Data preprocessing utilities
├── src/
│   ├── core/                 # Config, registry, base classes
│   ├── data/                 # Dataset, tile pools, augmentation
│   ├── models/               # EfficientNetV2 + FPN + ViT + heads
│   ├── losses/               # TopK-Anchored MIL loss (tiered)
│   ├── trainer/              # Asymmetric MIL trainer loop
│   ├── inference/            # Unified engine + GUI + detection
│   ├── evaluation/           # Metrics, validation
│   ├── critics/              # Feature Critic (experimental)
│   └── utils/                # Logging, visualization
└── deploy/                   # Deployment utilities (WIP)
```

---

## Technical Environment

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 4060 Laptop (8GB VRAM) |
| Framework | PyTorch + timm + Albumentations |
| Backbone | EfficientNetV2-RW-S / EfficientNet-B0 (ImageNet pretrained) |
| Training VRAM | ~2.4 GB (with AMP) |
| Inference VRAM | ~1.0 GB (32 tiles batch) |
| Data storage | LMDB (tile pools with precomputed features) |

---

## Publication

**Deep Learning Methods for Rice Disease Detection: Evolution and Challenges**
X. Shuwei et al. — Published in *RoViSP 2025* (International Conference on Robotics, Vision, Signal Processing and Power Applications)

---

## License

This project is part of an MSc thesis at Universiti Sains Malaysia.
