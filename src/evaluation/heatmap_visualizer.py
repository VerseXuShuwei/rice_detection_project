"""
Unified Visualization Module for MIL Inference.
统一可视化模块，集成训练监控(Tile分布)与终局评估(空间热力图)的绘图逻辑。

Recent Updates:
    - [2026-02-26] Fix: Build UnifiedInferenceEngine once per monitoring batch (not per-image);
                        use config-injected conf_threshold instead of hardcoded 0.1;
                        _generate_spatial_heatmap() now accepts optional pre-built engine arg.
    - [2026-01-06] Refactor: Consolidated visualization logic from heatmap_visualizer.py and final_evaluator.py
    - [2026-01-06] Feature: Support two visualization modes:
        1. plot_bag_distribution: 瓦片级激活分布 (3-row plot) - 适合分析Bag内部打分
        2. plot_spatial_heatmap: 空间热力叠加 (Overlay + Boxes) - 适合分析病灶定位
    - [2026-01-06] Feature: Centralized CLASS_NAMES mapping

Usage:
    >>> from src.evaluation.heatmap_visualizer import MILVisualizer
    >>> viz = MILVisualizer(save_dir='outputs/logs/exp_name/heatmaps')
    >>> # Mode 1: 画Tile分布
    >>> viz.plot_bag_distribution(heatmap_array, class_id, save_name='bag_123.png')
    >>> # Mode 2: 画空间热力图
    >>> viz.plot_spatial_heatmap(heatmap_global, original_img, tiles_info, class_id, save_name='spatial_123.png')
"""

import matplotlib

matplotlib.use('Agg')  # Backend safety
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import class mapping utilities (Single Source of Truth: config)
from src.utils.class_mapping import get_display_names, get_short_names

# DEPRECATED: Legacy hardcoded fallback (only used if config is not provided)
# [2026-01-27] Updated: Removed narrow-brown-spot and rice-hispa-damage
# Prefer loading from config['classes']['display_names']
DEFAULT_CLASS_NAMES = {
    0: 'Background',
    1: 'Bacterial Leaf Blight',
    2: 'Bacterial Sheath Brown Rot',
    3: 'Brown Spot',
    4: 'False Smut',
    5: 'Leaf Blast',
    6: 'Node Neck Blast',
    7: 'Rice Leaf Beetle',
    8: 'Rice Leaf Folder',
    9: 'Sheath Blight'
}

# DEPRECATED: Legacy short names (only used if config is not provided)
DEFAULT_CLASS_NAMES_SHORT = {
    0: 'BG',
    1: 'Bact-Leaf-Blight',
    2: 'Bact-Sheath-Rot',
    3: 'Brown-Spot',
    4: 'False-Smut',
    5: 'Leaf-Blast',
    6: 'Node-Neck-Blast',
    7: 'Rice-Beetle',
    8: 'Rice-Folder',
    9: 'Sheath-Blight'
}

# Backward compatibility aliases (module-level for imports)
# DEPRECATED: Use get_display_names(config) / get_short_names(config) instead
CLASS_NAMES = DEFAULT_CLASS_NAMES
CLASS_NAMES_SHORT = DEFAULT_CLASS_NAMES_SHORT


class MILVisualizer:
    """
    MIL 统一可视化器.
    负责所有绘图逻辑，不包含模型推理。

    Args:
        save_dir: Directory to save visualization outputs
        class_names: Optional dict mapping class_id -> display name (from config)
        short_names: Optional dict mapping class_id -> short name (from config)
    """

    def __init__(
            self,
            save_dir: str,
            class_names: Optional[Dict[int, str]] = None,
            short_names: Optional[Dict[int, str]] = None,
            # Heatmap generation settings (injected from config at construction time)
            multiscale_tile_sizes: Optional[List[int]] = None,
            small_image_tile_sizes: Optional[List[int]] = None,
            multiscale_min_size: Optional[List[int]] = None,
            stride_ratio: float = 0.5,
            batch_size: int = 8,
            conf_threshold: float = 0.4,
            top_k: int = 5
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Class name mappings (use config values or defaults)
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self.short_names = short_names or DEFAULT_CLASS_NAMES_SHORT

        # Heatmap generation settings (stored as attributes, not re-read from config)
        self.multiscale_tile_sizes = multiscale_tile_sizes or [1024, 1536, 2048]
        self.small_image_tile_sizes = small_image_tile_sizes or [512, 768, 1024]
        self.multiscale_min_size = multiscale_min_size or [3000, 4000]
        self.stride_ratio = stride_ratio
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.top_k = top_k

        # Color palette for bounding boxes
        self.box_colors = [(255, 0, 0), (255, 140, 0), (255, 20, 147), (0, 255, 255), (138, 43, 226)]

    def plot_bag_distribution(
            self,
            heatmap: np.ndarray,
            class_id: int,
            save_name: str,
            title_suffix: str = ""
    ):
        """
        绘制 Bag 内部的 Tile 激活分布图 (3行风格).
        来自原 heatmap_visualizer.py

        Args:
            heatmap: (num_classes+1, num_tiles) array, Transposed from softmax output
            class_id: True class ID
            save_name: Filename to save
        """
        num_tiles = heatmap.shape[1]
        save_path = self.save_dir / save_name

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # --- Row 1: Background (Class 0) ---
        ax1 = axes[0]
        im1 = ax1.imshow(heatmap[0:1, :], aspect='auto', cmap='Reds', vmin=0, vmax=1)
        ax1.set_title('Class 0 (Background) Activation - Should be LOW for disease bags', fontweight='bold')
        ax1.set_yticks([])
        ax1.set_ylabel('BG')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # --- Row 2: True Class ---
        ax2 = axes[1]
        im2 = ax2.imshow(heatmap[class_id:class_id + 1, :], aspect='auto', cmap='Greens', vmin=0, vmax=1)
        ax2.set_title(f'True Class {class_id} Activation ({self.class_names.get(class_id)}) - Should be HIGH',
                      fontweight='bold')
        ax2.set_yticks([])
        ax2.set_ylabel(f'C{class_id}')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # --- Row 3: Max Disease Class Prediction ---
        ax3 = axes[2]
        disease_heatmap = heatmap[1:, :]  # Exclude BG
        max_disease_activation = disease_heatmap.max(axis=0, keepdims=True)
        max_disease_class_idx = disease_heatmap.argmax(axis=0) + 1

        im3 = ax3.imshow(max_disease_activation, aspect='auto', cmap='Blues', vmin=0, vmax=1)
        pmr = np.max(max_disease_activation) / (np.mean(max_disease_activation) + 1e-6)
        ax3.set_title(f'Max Disease Activation (Peak-to-Mean Ratio: {pmr:.2f})', fontweight='bold')
        ax3.set_xlabel(f'Tiles Sequence (Total: {num_tiles})')
        ax3.set_yticks([])
        ax3.set_ylabel('Max')

        # Annotate indices
        for tile_idx in range(num_tiles):
            cls_idx = max_disease_class_idx[tile_idx]
            val = max_disease_activation[0, tile_idx]
            text_color = 'white' if val > 0.5 else 'black'
            ax3.text(tile_idx, 0, f'{cls_idx}', ha='center', va='center',
                     fontsize=8, fontweight='bold', color=text_color)

        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        plt.suptitle(f'Bag Distribution Analysis - {title_suffix}', fontsize=14, y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_spatial_heatmap(
            self,
            original_image: np.ndarray,
            heatmap_global: np.ndarray,
            tiles_info: List[Dict],
            class_id: int,
            save_name: str,
            conf_threshold: float = 0.4,
            top_k: int = 5
    ):
        """
        绘制空间热力叠加图 + 检测框 (Sniper View).
        来自原 final_evaluator.py

        Args:
            original_image: (H, W, 3) RGB image
            heatmap_global: (H, W) global heatmap 2D array
            tiles_info: List of dicts with keys 'score', 'coords', 'class_id', 'tile_size'
            class_id: True class ID
        """
        save_path = self.save_dir / save_name
        class_name = self.class_names.get(class_id, f"C{class_id}")

        fig, axes = plt.subplots(1, 2, figsize=(22, 10))

        # --- Left: Sniper View (Bounding Boxes) ---
        valid_tiles = [t for t in tiles_info if t['score'] > conf_threshold]
        valid_tiles = sorted(valid_tiles, key=lambda x: x['score'], reverse=True)[:top_k]

        img_boxes = original_image.copy()

        if not valid_tiles:
            cv2.putText(img_boxes, "No suspicious areas", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        else:
            for rank, item in enumerate(valid_tiles):
                y1, y2, x1, x2 = item['coords']
                color = self.box_colors[rank % len(self.box_colors)]
                cls_name = self.class_names.get(item['class_id'], f"C{item['class_id']}")

                # Draw Box
                cv2.rectangle(img_boxes, (x1, y1), (x2, y2), color, 4)

                # Label
                label = f"#{rank + 1} {cls_name} ({item['score']:.2f})"
                cv2.rectangle(img_boxes, (x1, y1 - 40), (x1 + 600, y1), color, -1)
                cv2.putText(img_boxes, label, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        axes[0].imshow(img_boxes)
        axes[0].set_title(f"Detection View (Top-{top_k}, Thresh={conf_threshold})", fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # --- Right: Heatmap Overlay ---
        heatmap_norm = np.uint8(255 * heatmap_global)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original_image, 0.6, heatmap_color, 0.4, 0)

        tile_sizes = sorted(list(set(t['tile_size'] for t in tiles_info))) if tiles_info else []
        axes[1].imshow(overlay)
        axes[1].set_title(f"Global Heatmap Overlay (Tile Sizes: {tile_sizes})", fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # --- Bottom: Detail Table ---
        if valid_tiles:
            table_data = [['Rank', 'Class', 'Conf', 'TileSize', 'Coords']]
            for rank, item in enumerate(valid_tiles):
                y1, y2, x1, x2 = item['coords']
                table_data.append([
                    f"#{rank + 1}",
                    self.short_names.get(item['class_id'], str(item['class_id'])),
                    f"{item['score']:.1%}",
                    str(item['tile_size']),
                    f"({x1},{y1})-({x2},{y2})"
                ])

            table = plt.table(cellText=table_data, loc='bottom', cellLoc='center',
                              colWidths=[0.05, 0.25, 0.1, 0.1, 0.3], bbox=[0.1, -0.3, 0.8, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)

        plt.suptitle(f"Spatial Analysis - True Class: {class_name}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, cm: np.ndarray, save_name: str, filter_info: str = ""):
        """
        绘制混淆矩阵.
        来自原 final_evaluator.py
        """
        save_path = self.save_dir / save_name

        plt.figure(figsize=(14, 12))
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)
        class_labels = [self.short_names.get(i, f'C{i}') for i in range(cm.shape[0])]

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels)

        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title(f'Confusion Matrix (Normalized) {filter_info}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def plot_per_class_metrics(self, metrics_dict: Dict, save_name: str):
        """
        绘制每类指标条形图.
        来自原 final_evaluator.py
        """
        save_path = self.save_dir / save_name

        classes = sorted(metrics_dict['per_class_precision'].keys())
        prec = [metrics_dict['per_class_precision'][c] for c in classes]
        rec = [metrics_dict['per_class_recall'][c] for c in classes]
        f1 = [metrics_dict['per_class_f1'][c] for c in classes]

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(16, 7))
        ax.bar(x - width, prec, width, label='Precision')
        ax.bar(x, rec, width, label='Recall')
        ax.bar(x + width, f1, width, label='F1')

        ax.set_xticks(x)
        ax.set_xticklabels([self.short_names.get(c, f'C{c}') for c in classes], rotation=45, ha='right')
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def generate_monitoring_heatmaps(
        self,
        model,
        dataset,
        epoch: int,
        phase: str = 'stable',
        samples_per_class: int = 1,
        fixed_samples: bool = True,
        use_spatial_heatmap: bool = True
    ) -> Dict:
        """
        Generate monitoring heatmaps for reference samples.

        Trainer delegates visualization to this method (decoupling).
        Handles sample selection + inference + heatmap generation internally.

        CRITICAL: Samples are selected from VALIDATION set to ensure:
            1. Same images across epochs (fixed_samples=True)
            2. Images not used in training (avoid overfitting visualization)

        Fixed Sample Mode (fixed_samples=True):
            - First call: Randomly select samples and save to reference_samples.json
            - Subsequent calls: Load existing samples from reference_samples.json
            - Enables tracking model learning progress on same images across epochs

        NOTE: Heatmap settings (tile_sizes, stride_ratio, etc.) are configured
        at MILVisualizer construction time via __init__ parameters, NOT read
        from config at runtime. This follows the "Single Source of Truth" principle.

        Args:
            model: Trained model
            dataset: VALIDATION dataset (AsymmetricMILDataset) - NOT train dataset
            epoch: Current epoch
            phase: Training phase ('warmup_end' | 'stable')
            samples_per_class: Samples per class (default: 1)
            fixed_samples: If True, reuse samples from reference_samples.json (default: True)
            use_spatial_heatmap: If True, generate full spatial heatmaps (default: True)

        Returns:
            Dict with 'saved_paths' and 'reference_samples'
        """
        import random
        import json
        import torch
        import cv2
        import os
        from src.data import get_validation_transforms
        from src.utils.resize_utils import resize_keep_aspect_ratio_crop

        ref_path = self.save_dir / 'reference_samples.json'

        # Step 1: Load or select reference samples
        reference_samples = None

        # Try to load existing fixed samples
        if fixed_samples and ref_path.exists():
            try:
                with open(ref_path, 'r') as f:
                    reference_samples = json.load(f)
                print(f"\n[HEATMAP] Loaded {len(reference_samples)} fixed reference samples from {ref_path}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"[HEATMAP] Warning: Failed to load reference_samples.json: {e}")
                reference_samples = None

        # Select new samples if needed
        if reference_samples is None:
            print(f"\n[HEATMAP] Selecting reference samples for monitoring (from validation set)...")
            reference_samples = []
            class_to_bags = {}

            for idx, (image_path, class_id) in enumerate(dataset.samples):
                if class_id not in class_to_bags:
                    class_to_bags[class_id] = []
                class_to_bags[class_id].append((idx, image_path))

            # Use fixed seed for reproducibility
            rng = random.Random(42)

            # Sample per positive class (skip Class 0)
            for class_id in sorted(class_to_bags.keys()):
                if class_id == 0:
                    continue

                bag_list = class_to_bags[class_id]
                sampled_idx, sampled_path = rng.choice(bag_list)
                class_name = dataset.idx_to_class.get(class_id, f'class_{class_id}')

                reference_samples.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'image_path': sampled_path,
                    'bag_id': sampled_idx
                })

            # Save metadata for future reuse
            with open(ref_path, 'w') as f:
                json.dump(reference_samples, f, indent=2)
            print(f"[HEATMAP] Saved {len(reference_samples)} reference samples to {ref_path}")

        # Step 2: Generate heatmaps
        saved_paths = []
        model.eval()
        device = next(model.parameters()).device
        tile_size = dataset.final_tile_size

        # Use heatmap settings from self (configured at __init__ time)
        # This follows "Single Source of Truth" - config is read once at construction
        multiscale_tiles = self.multiscale_tile_sizes
        small_image_tiles = self.small_image_tile_sizes
        multiscale_min_size = self.multiscale_min_size
        stride_ratio = self.stride_ratio
        batch_size = self.batch_size
        conf_threshold = self.conf_threshold
        top_k = self.top_k

        # Build engine once outside the loop — all samples share the same engine instance
        # so we avoid repeated construction overhead and guarantee identical inference config
        monitoring_engine = None
        if use_spatial_heatmap:
            from src.inference.engine import UnifiedInferenceEngine
            monitoring_engine = UnifiedInferenceEngine(
                model=model,
                device=device,
                model_input_size=tile_size,
                tile_sizes=multiscale_tiles,
                stride_ratio=stride_ratio,
                batch_size=batch_size,
                conf_threshold=conf_threshold,
                adaptive_scale=True,
                large_image_tiles=multiscale_tiles,
                small_image_tiles=small_image_tiles,
                large_image_min_size=tuple(multiscale_min_size),
            )

        for sample in reference_samples:
            image_path = sample['image_path']
            class_id = sample['class_id']
            class_name = sample['class_name']

            # Check if image exists
            if not os.path.exists(image_path):
                print(f"[HEATMAP] Warning: Image not found: {image_path}")
                continue

            # Load image with unified utility
            from src.utils.io_utils import load_and_preprocess_image
            try:
                image = load_and_preprocess_image(image_path, target_size=None, color_mode='RGB')
            except (FileNotFoundError, ValueError) as e:
                print(f"[HEATMAP] Warning: Failed to load {image_path}: {e}")
                continue

            h_orig, w_orig = image.shape[:2]

            if use_spatial_heatmap:
                # Generate full spatial heatmap using shared engine (built once above)
                heatmap_global, tiles_info = self._generate_spatial_heatmap(
                    model=model,
                    image=image,
                    device=device,
                    engine=monitoring_engine
                )

                # Save spatial heatmap
                save_name = f"epoch{epoch:03d}_{phase}_{class_name}_spatial.png"
                self.plot_spatial_heatmap(
                    original_image=image,
                    heatmap_global=heatmap_global,
                    tiles_info=tiles_info,
                    class_id=class_id,
                    save_name=save_name,
                    conf_threshold=conf_threshold,
                    top_k=top_k
                )
                saved_paths.append(self.save_dir / save_name)
                print(f"[HEATMAP] Saved {save_name}")

            else:
                # Simple tile distribution (legacy mode)
                # For offline mode, we need to load tiles differently
                if hasattr(dataset, 'offline_mode') and dataset.offline_mode:
                    # Offline mode: load tiles from LMDB
                    tile_infos = dataset.tile_pool.get_bag_tiles(image_path)
                    if not tile_infos:
                        continue
                    tile_ids = [t.tile_id for t in tile_infos]
                    tiles_resized = dataset.tile_pool.load_tiles_batch(tile_ids)
                    tiles_resized = [tiles_resized[i] for i in range(tiles_resized.shape[0])]
                else:
                    # Online mode: generate tiles
                    raw_tiles = dataset._generate_tiles(image)
                    tiles_resized = [
                        resize_keep_aspect_ratio_crop(t, tile_size) for t in raw_tiles
                    ]

                # Transform and inference
                val_transform = get_validation_transforms(img_size=tile_size)
                tiles_transformed = []
                for tile in tiles_resized:
                    transformed = val_transform(image=tile)
                    if isinstance(transformed['image'], torch.Tensor):
                        tile_tensor = transformed['image']
                    else:
                        tile_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float() / 255.0
                    tiles_transformed.append(tile_tensor)

                if len(tiles_transformed) == 0:
                    continue

                tiles_batch = torch.stack(tiles_transformed).to(device)

                with torch.no_grad():
                    outputs = model.predict_instances(tiles_batch)
                    probs = torch.softmax(outputs, dim=-1).cpu().numpy()

                save_name = f"epoch{epoch:03d}_{phase}_{class_name}.png"
                self.plot_bag_distribution(
                    probs.T, class_id, save_name=save_name
                )
                saved_paths.append(self.save_dir / save_name)
                print(f"[HEATMAP] Saved {save_name}")

        return {
            'saved_paths': saved_paths,
            'reference_samples': reference_samples
        }

    def _generate_spatial_heatmap(
        self,
        model,
        image: np.ndarray,
        device: torch.device,
        tile_size: int = 384,
        multiscale_tiles: List[int] = None,
        small_image_tiles: List[int] = None,
        multiscale_min_size: List[int] = None,
        stride_ratio: float = 0.5,
        batch_size: int = 8,
        engine=None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate spatial heatmap using UnifiedInferenceEngine.

        Delegates all sliding window logic to the unified engine.
        Returns single-channel max-disease heatmap for backward compatibility.

        Args:
            model: Trained model
            image: (H, W, 3) RGB image
            device: torch device
            tile_size: Model input size (default: 384)
            multiscale_tiles: Tile sizes for large images
            small_image_tiles: Tile sizes for small images
            multiscale_min_size: Threshold for large/small image detection
            stride_ratio: Stride as fraction of tile size
            batch_size: Batch size for inference
            engine: Optional pre-built UnifiedInferenceEngine. If provided, reused
                    directly (avoids per-call construction overhead and ensures
                    consistent conf_threshold with training config).

        Returns:
            (heatmap_global, tiles_info)
        """
        from src.inference.engine import UnifiedInferenceEngine

        if engine is None:
            multiscale_tiles = multiscale_tiles or [1024, 1536, 2048]
            small_image_tiles = small_image_tiles or [512, 768, 1024]
            multiscale_min_size = multiscale_min_size or [3000, 4000]

            engine = UnifiedInferenceEngine(
                model=model,
                device=device,
                model_input_size=tile_size,
                tile_sizes=multiscale_tiles,
                stride_ratio=stride_ratio,
                batch_size=batch_size,
                conf_threshold=self.conf_threshold,  # use config-injected value
                adaptive_scale=True,
                large_image_tiles=multiscale_tiles,
                small_image_tiles=small_image_tiles,
                large_image_min_size=tuple(multiscale_min_size),
            )

        result = engine.run(image)

        # Convert multi-class (C, H, W) → single-channel (H, W) max disease
        if result.heatmap.shape[0] > 1:
            heatmap_global = result.heatmap[1:].max(axis=0)
        else:
            heatmap_global = result.heatmap[0]

        return heatmap_global, result.tiles_info