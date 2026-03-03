# Weakly-Supervised Rice Disease Detection via Asymmetric Multiple Instance Learning

> **From Image-Level Labels to Pixel-Level Localization**
>
> A framework for detecting and localizing 9 rice diseases using only image-level annotations, eliminating the need for expensive pixel-level labeling.

---

## 1. Problem Statement

Traditional rice disease detection requires pixel-level or bounding-box annotations — expensive, slow, and dependent on domain experts. This project addresses:

**Can we achieve spatial disease localization using only image-level labels?**

- **Input**: "This image contains Brown Spot" (bag-level label)
- **Output**: Spatial heatmap + detection boxes showing *where* the disease is located
- **9 disease classes** + background (10-class classification)
- **Hardware constraint**: RTX 4060 Laptop, 8GB VRAM

---

## 2. Framework Design: Asymmetric MIL with Scout-Snipe Strategy(需要简化?)

### Core Insight

In weakly-supervised MIL, a "bag" (image) contains many tiles, but only some are truly diseased. 
Naïvely training on all tiles introduces massive label noise. Our solution: **separate the search from the learning**.

### Critical Dependency: High-Quality Negative Pool

This entire framework rests on one foundational assumption — **the existence of a reliable, confirmed-clean negative sample pool** (images guaranteed to contain no disease). Without it, Scout-Snipe is bootstrapping from nothing: the model scores tiles against its own uncertain beliefs, selects "best" tiles based on those beliefs, then trains on its own selections. That's left foot stepping on right foot — no ground truth anchors the process.

The negative pool provides that anchor. By training the model to confidently recognize "definitely not disease" (from confirmed healthy rice images), we establish a background baseline. Every positive prediction is then measured against this calibrated null hypothesis. The quality, diversity, and confirmed cleanliness of the negative pool directly determines the ceiling of the entire system.

In our case: 46,391 negative tiles from verified healthy rice images, with 25% hard negative mining to continuously challenge the background boundary.

### Two-Pass Architecture

```
SCOUT PASS (no gradient, inference mode)
  All tiles → model.eval() → score each tile → Select Top-K most likely disease tiles
                                                    ↓
SNIPE PASS (gradient enabled, training mode)
  Top-K tiles + negative tiles → model.train() → Tiered loss computation → Backprop
```

**Why asymmetric?** 
The Scout pass explores without contaminating gradients. 
The Snipe pass trains only on high-quality selections. 
This prevents noise-tile gradients from polluting the learned representations.

---

## 3. Model Architecture: CNN-ViT Hybrid

```
Input (B, 3, 384, 384)
        ↓
EfficientNetV2-RW-S Backbone
  Stages 0-2: FROZEN (ImageNet features)
  Stages 3-7: TRAINABLE (disease-specific)
  Stage 3 → (B, 160, 24, 24)  ──┐
  Stage 4 → (B, 1792, 12, 12) ──┤
        ↓                         ↓
FPN Neck (Multi-scale Fusion)
  Lateral: 160ch → 256ch
  Top-down: 1792ch → 256ch → Upsample 2×
  Fusion: Add + Conv3×3 → (B, 256, 24, 24)
        ↓
ViT Residual Block (Global Attention)
  Flatten → (B, 576, 256) + 2D Positional Encoding
  Multi-Head Self-Attention (8 heads)
  MLP (4× expansion) + Scaled Residual
  Output: (B, 256, 24, 24)
        ↓
HeatmapHead
  Conv1×1: 256 → 10 channels (spatial disease response)
  TopK-Mean Pooling → (B, 10) class logits
```

**Design decision**: ViT is placed *after* FPN dimension reduction (256ch), not before (1792ch). 
This yields **0.94M ViT parameters** vs 38.6M if placed at full backbone dimensionality — a 41× parameter reduction with no loss in spatial modeling capability.

**Total**: 24.18M parameters (Backbone 22.15M, FPN 1.09M, ViT 0.94M)

---

## 4. Training Strategy: Three-Tier Loss with Per-Bag Knowledge Distillation

### Core Insight

In weakly-supervised MIL, a "bag" (image) contains many tiles, but only some are truly diseased. 
Naïvely training on all tiles introduces massive label noise. Our solution: **separate the search from the learning**.

**Critical foundation**: 
High-quality negative pool (46,391 tiles from verified healthy images) anchors the entire system. 
Without confirmed-clean negatives, Scout-Snipe bootstraps from noise. 
The negative pool establishes a calibrated background baseline against which all positive predictions are measured.

### 4.1 Warmup Phase (Epochs 1–15)

Simple, stable learning to establish baseline representations:

- **Top-1**: Strong CE (weight = 5.0) — anchor signal
- **Top-2~K**: Weak CE (0.1× weight) — gentle guidance
- **Inter-bag ranking loss**: Margin-based separation between bags

### 4.2 Stable Phase (Epochs 16–30)

Introduces quality-aware tiered supervision:

| Tier | Condition | Top-1 Strategy | Top-2~K Strategy |
|------|-----------|----------------|------------------|
| **Tier 1** (Qualified) | Correct prediction AND conf > 0.45 | Strong CE (×5.0) | Per-bag KL distillation (dual-gate) |
| **Tier 2** (Marginal) | Correct prediction AND conf ≤ 0.45 | Full CE (×1.0) | Weak CE |
| **Tier 3** (Wrong) | Incorrect prediction | Noise Drop or Correction Ranking | Silence |

### 4.3 Per-Bag KL Dual-Gate (Key Innovation)

Standard knowledge distillation uses batch-mean teacher distributions — mixing disease classes causes **heatmap diffusion** (experimentally validated, see Section 7).

Our solution uses **per-bag** teacher signals with dual quality gates:

- **Gate A** (bag-level): Is this bag's Top-1 a Tier 1 prediction? → Teacher distribution trustworthy
- **Gate B** (tile-level): Does this Top-2~K tile's argmax match the bag label? → Student direction correct
- Gate A✓ + Gate B✓ → KL divergence using *this bag's own Top-1* soft probabilities
- Any gate fails → Fallback to weak CE

This ensures each tile learns from its own bag's highest-quality prediction, not a cross-class average.

### 4.4 Three-Timeline Alignment

```
Epoch:  1         8    15        16         30
        |=========|====|=========|==========|
Backbone:[==FROZEN=====][======TRAINABLE=========]
LR:     [LR Warmup    ][Hold    ][--Cosine Decay--]
MIL:    [=====MIL Warmup (K=4)==][Stable K=3======]
```

- LR warmup (1–8): Backbone frozen → hybrid components ramp up on stable features
- LR hold (9–15): Backbone just unfrozen → peak LR for joint fine-tuning
- MIL stable from Epoch 16: Cosine decay begins → tiered loss activates

---

## 5. Anti-Collapse Mechanisms

Multiple safeguards prevent training degeneration:

| Mechanism | Problem Solved |
|-----------|---------------|
| BN frozen in Snipe pass | Scout/Snipe statistics divergence |
| Tier 2 full weight (1.0) | Prevents Silence Spiral on marginal predictions |
| Per-bag KL (not batch-mean) | Prevents cross-class heatmap diffusion |
| Tier 3 safety net | Weak CE fallback when all Top-1s fail (prevents pos_loss=0) |
| Scale Diversity | Multi-scale tile deduplication (replaces ineffective Spatial NMS) |
| Confidence threshold 0.45 | Meaningful quality gate (0.25 was near 10-class random baseline) |

---

## 6. Negative Results as Design Insights

### 6.1 Feature Critic: ImageNet Features Cannot Separate Rice Diseases

We attempted to use pretrained EfficientNetV2 features with K-Means background prototypes to filter background tiles before training.

**Result**: Complete failure.

| Metric | Negative (Background) | Positive (Disease) |
|--------|----------------------|-------------------|
| Mean cosine similarity | 0.9336 | 0.9345 |
| Std | 0.0524 | 0.0559 |
| Gap | −0.0009 | — |

The pretrained backbone sees rice disease tiles and healthy tiles as nearly identical in feature space. Any filtering threshold removes both populations equally. This confirms that **ImageNet pretraining provides no meaningful prior for rice leaf pathology**.

### 6.2 Spatial NMS: Ineffective Due to Tiling Geometry

With tile overlap, same-scale IoU is Limited effectiveness, Replaced with Scale Diversity (preferring different scales for Top-2~K selection).

### 6.3 Negative Rejection (NR): Causes Silence Spiral

NR loss gradient direction ("don't be disease" to top2-k tiles) directly conflicts with Top-1 CE direction ("be this disease"), progressively suppressing positive sample responses. Deprecated in favor of weak CE for Tier 2 Top-2~K.

---

## 7. Experimental Results

### 7.1 Ablation Study (Full Validation Set Evaluation)

All experiments evaluated on the complete validation set (53,457 positive tiles + 9,279 negative tiles).

| Config | 0209 (Baseline) | 0224 | 0225 | **0226 (Final)** |
|--------|:---:|:---:|:---:|:---:|
| Total Epochs | 40 | 45 | 30 | **30** |
| Warmup Epochs | 15 | 20 | 20 | **15** |
| Conf Threshold | 0.40 | 0.25 | 0.25 | **0.45** |
| Tier 2 Weight | default | 0.3 | 0.3 | **1.0** |
| Pool Mode | default | GMP | TopK-Mean | **TopK-Mean** |
| Noise Drop | 0.5 | 0.7 | 0.7 | **0.7** |

| Metric | 0209 | 0224 | 0225 | **0226** |
|--------|:---:|:---:|:---:|:---:|
| **Neg Recall** | 94.7% | 94.1% | 90.9% | **96.8%** |
| **Hit Accuracy** | 98.1% | 98.3% | 99.1% | **98.7%** |
| **Overall Acc (filtered)** | 71.2% | 67.9% | 66.0% | **81.3%** |
| **Overall Acc (unfiltered)** | 33.4% | 36.6% | 39.5% | **44.9%** |
| Avg Top-1 Conf | 88.3% | 90.8% | 89.6% | **97.8%** |
| TopK Lift | 56.8% | 57.2% | 53.3% | **53.5%** |
| Neg Hallucination | 6.9% | 7.5% | 10.7% | **3.6%** |
| Tiles Retained | 26.0% | 31.4% | 41.5% | **47.3%** |

**Key finding**: The combination of conf_threshold 0.45 + Tier 2 full weight + warmup 15 (0226) produces the best results across all primary metrics. The 10+ percentage point jump in overall accuracy (filtered) from 0225 to 0226 demonstrates that these changes work synergistically.

### 7.2 Per-Class Performance (0226, Filtered)

| Class | Precision | Recall | F1 |
|-------|:---------:|:------:|:--:|
| Background | 1.000 | 0.969 | 0.984 |
| Leaf Blast | 0.895 | 0.882 | 0.888 |
| Rice Leaf Beetle | 0.955 | 0.811 | 0.877 |
| Brown Spot | 0.820 | 0.857 | 0.838 |
| Node Neck Blast | 0.879 | 0.686 | 0.770 |
| Sheath Blight | 0.765 | 0.695 | 0.728 |
| Bact. Sheath Brown Rot | 0.664 | 0.799 | 0.726 |
| False Smut | 0.752 | 0.578 | 0.653 |
| Rice Leaf Folder | 0.430 | 0.843 | 0.570 |
| Bact. Leaf Blight | 0.392 | 0.603 | 0.475 |

**Performance tiers**:
- **Strong** (F1 > 0.8): Background, Leaf Blast, Rice Beetle, Brown Spot — classes with visually distinctive, spatially compact lesions
- **Moderate** (F1 0.65–0.8): Node Neck Blast, Sheath Blight, Bact. Sheath Rot, Sheath Blight — classes with more diffuse or overlapping morphology
- **Challenging** (F1 < 0.65): False Smut, Rice Folder, Bact. Leaf Blight — classes where tile-level features are insufficient or morphologically similar to other classes

### 7.3 Heatmap Diffusion: Batch-Mean vs Per-Bag KL

Visual comparison on False Smut detection (same image):

- **Epoch 20 (warmup end)**: Focused heatmap with clear hot spot on disease lesion
- **Epoch 30 (after batch-mean KL)**: Diffused heatmap with scattered activation across entire image

This directly validates the per-bag KL design: batch-mean KL mixes cross-class distributions, causing the model to "see disease everywhere" rather than localizing it.



---

## 8. Inference Pipeline(已修改,需要更新)

```
Original Image
      ↓
Multi-scale Tiling (512/768/1024/1536/2048px, adaptive overlap)
      ↓
Per-tile Inference → 10-channel spatial heatmap per tile
      ↓
Global Heatmap Assembly (tile contributions weighted, overlap-averaged)
      ↓
Threshold → Connected Component Analysis (CCA)
      ↓
NMS + Top-K → Final Detection Boxes with class labels and confidence
```

### 8.1 Qualitative Results

[example images showing:]
- Original image
- Predicted heatmap
- Final detection boxes

Key observations:
- Model successfully localizes compact lesions (Leaf Blast, Brown Spot)
- Struggles with diffuse diseases (False Smut, BLB) where lesions lack clear boundaries
- Background tiles correctly suppressed (neg_recall 96.8%)

---

## 9. Dataset

| Property                | Value                                                         |
|-------------------------|---------------------------------------------------------------|
| **Images (bags)**       | **2,819 positive bags + 1,127 negative bags**                 |
| Disease classes         | 9 (Bacterial Leaf Blight, ...)                                |
| Positive tiles          | 158,277 (100,281 train / 57,996 val)(with Multi-scale tiling) |
| Negative tiles          | 46,391 (37,112 train / 9,279 val)(with Multi-scale tiling)    |
| Multi-scale tiling      | 512/768/1024/1536/2048px with adaptive overlap                |
| Tile size (model input) | 384 × 384                                                     |
| Split strategy          | Image-level (same image's tiles never split across train/val) |
---

## 10. Limitations and Future Work

### Current Limitations

1. **Tile boundary artifacts**: Disease lesions split across tile boundaries produce low-confidence predictions. Tiles capturing partial lesions often oscillate between background and disease predictions (observed particularly with Brown Spot and BLB).

2. **Morphologically similar classes**: BLB, Leaf Blast, and Rice Leaf Folder share similar local appearance at tile level. The model correctly localizes but misclassifies between these classes, suggesting tile-level features alone are insufficient for discrimination.

3. **Background co-occurrence bias**: Some classes may be distinguished by background context (e.g., image background color/texture) rather than lesion morphology — a form of spurious correlation inherent to weakly-supervised learning without pixel-level supervision.

### Planned Improvements

1. **BG-aware confidence reweighting**: Weight each tile's contribution to the global heatmap by `(1 - bg_confidence)`, suppressing ambiguous boundary tiles without hard thresholding.

2. **Post-training Feature Critic**: Rebuild background prototypes using fine-tuned (not pretrained) backbone features, where disease and healthy tiles should be separable.

3. **Inter-tile context modeling**: Aggregate spatial relationships across neighboring tiles to capture disease extent and morphology beyond single-tile receptive fields.

**Technical next steps:**
4. **Backbone ablation**: Evaluate alternative architectures (ConvNeXt, SwinV2) for rice disease feature extraction

5. **Inference optimization**: Implement TensorRT/ONNX quantization for real-time field deployment

6. **Production deployment**: Package as FastAPI service with Docker containerization for agricultural monitoring systems


---

## 11. Technical Environment

| Component      | Specification                               |
|----------------|---------------------------------------------|
| GPU            | NVIDIA RTX 4060 Laptop (8GB VRAM)           |
| Framework      | PyTorch + timm + Albumentations             |
| Backbone       | EfficientNetV2-RW-S (ImageNet pretrained)   |
| Training VRAM  | ~2.4GB (with AMP)                           |
| Inference VRAM | ~1.0GB (32 tiles batch)                     |
| Data storage   | LMDB (tile pools with precomputed features) |
