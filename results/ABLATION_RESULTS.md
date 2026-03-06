# Ablation Study: Backbone × Hybrid Components

> **Date**: 2026-03-07  
> **Framework**: Scout-Snipe Asymmetric MIL  
> **Dataset**: Total Rice Image (8 disease classes + background)  
> **Training**: 30 epochs, seed=42, deterministic  
> **Evaluation**: Filtered confusion matrix (min_confidence=0.45, hit_acc_filter=true)

---

## 1. Experiment Design

This ablation isolates two factors: **backbone architecture** and **hybrid components** (FPN Neck + ViT Residual Block + HeatmapHead).

| Experiment | Backbone | FPN | ViT | HeatmapHead | Hybrid Warmup |
|:---|:---|:---:|:---:|:---:|:---:|
| **Baseline** | EfficientNetV2-S | ✓ | ✓ | ✓ | ✓ |
| **A2** | EfficientNet-B0 | ✓ | ✓ | ✓ | ✓ |
| **A3** | EfficientNetV2-S | ✗ | ✗ | ✗ | ✗ |
| **A4** | EfficientNet-B0 | ✗ | ✗ | ✗ | ✗ |

**Shared configuration across all experiments:**
- Optimizer: AdamW (backbone_lr=1e-5, hybrid_lr=5e-5, classifier_lr=3e-4)
- Scheduler: Trapezoidal (warmup 8 epochs, hold 7, cosine decay)
- MIL: warmup_epochs=15, warmup_k=4, stable_k=3
- Loss: TopK Anchored MIL (tiered, top1_ce_weight=5.0, stable_gate_conf=0.45)
- Anti-collapse: BN frozen in Snipe, Scale Diversity enabled
- Feature Critic: disabled

---

## 2. Overall Results

| Experiment | Overall Acc | Disease Macro-F1 | Neg Recall | TopK Lift |
|:---|:---:|:---:|:---:|:---:|
| Baseline (V2-S + Components) | 0.8746 | 0.8015 | 0.9863 | 0.5776 |
| A2 (B0 + Components) | 0.8942 | 0.8243 | 0.9789 | 0.6036 |
| A3 (V2-S, No Components) | **0.9003** | **0.8434** | **0.9909** | 0.5100 |
| A4 (B0, No Components) | 0.8907 | 0.8199 | 0.9848 | 0.5503 |

**Key observations:**
- A3 (V2-S bare backbone) achieves the **highest overall accuracy** (0.9003) and **best Disease Macro-F1** (0.8434)
- Adding FPN+ViT components **hurts** both backbones on accuracy: V2-S drops 2.6pp (0.9003→0.8746), B0 drops ~0.4pp (0.8907→0.8942 — but B0 actually gains slightly)
- B0 benefits more from components than V2-S does: A2 improves over A4, while Baseline degrades from A3
- TopK Lift is highest for A2 (0.6036), suggesting components help Scout's tile selection quality on B0

---

## 3. Per-Class F1 Scores (Filtered)

### 3.1 Baseline — EfficientNetV2-S + FPN + ViT + HeatmapHead

| Class | F1 |
|:---|:---:|
| 0 BG | 0.9922 |
| 1 Bact-Leaf-Blight | 0.5606 |
| 2 Brown-Spot | 0.8662 |
| 3 False-Smut | 0.7131 |
| 4 Leaf-Blast | 0.9111 |
| 5 Neck-Blast | 0.8437 |
| 6 Leaf-Beetle | 0.9356 |
| 7 Leaf-Folder | 0.7942 |
| 8 Sheath-Blight | 0.7872 |
| **Macro (C1–C8)** | **0.8015** |

### 3.2 A2 — EfficientNet-B0 + FPN + ViT + HeatmapHead

| Class | F1 |
|:---|:---:|
| 0 BG | 0.9884 |
| 1 Bact-Leaf-Blight | 0.7368 |
| 2 Brown-Spot | 0.8850 |
| 3 False-Smut | 0.7528 |
| 4 Leaf-Blast | 0.9023 |
| 5 Neck-Blast | 0.8700 |
| 6 Leaf-Beetle | 0.9205 |
| 7 Leaf-Folder | 0.7415 |
| 8 Sheath-Blight | 0.7854 |
| **Macro (C1–C8)** | **0.8243** |

### 3.3 A3 — EfficientNetV2-S, No Components

| Class | F1 |
|:---|:---:|
| 0 BG | 0.9955 |
| 1 Bact-Leaf-Blight | 0.6690 |
| 2 Brown-Spot | 0.9129 |
| 3 False-Smut | 0.7412 |
| 4 Leaf-Blast | 0.9244 |
| 5 Neck-Blast | 0.8888 |
| 6 Leaf-Beetle | 0.9626 |
| 7 Leaf-Folder | 0.8622 |
| 8 Sheath-Blight | 0.7863 |
| **Macro (C1–C8)** | **0.8434** |

### 3.4 A4 — EfficientNet-B0, No Components

| Class | F1 |
|:---|:---:|
| 0 BG | 0.9918 |
| 1 Bact-Leaf-Blight | 0.6208 |
| 2 Brown-Spot | 0.9039 |
| 3 False-Smut | 0.7262 |
| 4 Leaf-Blast | 0.9353 |
| 5 Neck-Blast | 0.8709 |
| 6 Leaf-Beetle | 0.9334 |
| 7 Leaf-Folder | 0.7808 |
| 8 Sheath-Blight | 0.7877 |
| **Macro (C1–C8)** | **0.8199** |

---

## 4. Cross-Experiment Per-Class Comparison

| Class | Baseline | A2 (B0+Comp) | A3 (V2S bare) | A4 (B0 bare) | Best |
|:---|:---:|:---:|:---:|:---:|:---|
| 0 BG | 0.9922 | 0.9884 | **0.9955** | 0.9918 | A3 |
| 1 Bact-Leaf-Blight | 0.5606 | **0.7368** | 0.6690 | 0.6208 | A2 |
| 2 Brown-Spot | 0.8662 | 0.8850 | **0.9129** | 0.9039 | A3 |
| 3 False-Smut | 0.7131 | **0.7528** | 0.7412 | 0.7262 | A2 |
| 4 Leaf-Blast | 0.9111 | 0.9023 | 0.9244 | **0.9353** | A4 |
| 5 Neck-Blast | 0.8437 | 0.8700 | **0.8888** | 0.8709 | A3 |
| 6 Leaf-Beetle | 0.9356 | 0.9205 | **0.9626** | 0.9334 | A3 |
| 7 Leaf-Folder | 0.7942 | 0.7415 | **0.8622** | 0.7808 | A3 |
| 8 Sheath-Blight | 0.7872 | 0.7854 | 0.7863 | **0.7877** | A4 |
| **Macro (C1–C8)** | 0.8015 | 0.8243 | **0.8434** | 0.8199 | **A3** |

---

## 5. Additional Evaluation Metrics

| Experiment | Hit Acc | Avg Top1 Conf | TopK Avg Conf | Neg Hallucination | Filtered Tiles |
|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline | 0.9947 | 0.9888 | 0.9764 | 0.0150 | 19,890 / 48,454 |
| A2 | 0.9968 | 0.9902 | 0.9788 | 0.0240 | 17,963 / 48,454 |
| A3 | 0.9989 | 0.9942 | 0.9865 | 0.0103 | 22,888 / 48,454 |
| A4 | 1.0000 | 0.9906 | 0.9804 | 0.0175 | 20,968 / 48,454 |

---

## 6. Log References

| Experiment | Log Directory | Run Date |
|:---|:---|:---|
| Baseline | `asymmetric_mil_training_20260305_112901` | 2026-03-05 |
| A2 | `asymmetric_mil_training_20260303_193040` | 2026-03-03 |
| A3 | `asymmetric_mil_training_20260306_014846` | 2026-03-06 |
| A4 | `asymmetric_mil_training_20260304_035215` | 2026-03-04 |
