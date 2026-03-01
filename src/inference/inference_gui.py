"""
Rice Disease MIL Diagnostic GUI v4.0

Recent Updates:
    - [2026-02-19] Fix: Per-class tab uses coverage% + hotspot_mean instead of misleading max conf
    - [2026-02-18] Major: Complete rewrite with 4-tab layout (Detection/Heatmap/Per-Class/Components)
    - [2026-02-18] Fix: 1-class0 + percentile detection, Top-1 fallback, gap-based line styles
    - [2026-02-18] New: Entropy visualization, PCA feature maps, ViT attention, click interaction
    - [2026-02-18] New: Dual checkpoint comparison, per-class analysis grid

Key Features:
    - Tab 1 Detection: Before/After NMS side-by-side + count banner + tile gallery
    - Tab 2 Heatmap: Disease confidence + entropy overlay side-by-side + stats
    - Tab 3 Per-Class: Class dropdown + large heatmap + 3x3 grid with max conf
    - Tab 4 Components: PCA feature maps (Backbone/FPN/ViT/Head) + attention + dual ckpt
    - Click interaction: 10-class probability bar chart at clicked location

Usage:
    python -m src.inference.inference_gui
    python -m src.inference.inference_gui --config configs/algorithm/train_topk_asymmetric.yaml
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
import threading
import argparse
from typing import Dict, List, Tuple, Optional, Any

# Project imports
from src.models.builder import get_model
from src.utils.config_io import load_config
from src.utils.class_mapping import build_class_map_for_inference
from src.inference.engine import UnifiedInferenceEngine, extract_detections


# ==============================================================================
# Layer 1: Logic Core (Model Layer)
# ==============================================================================
class RiceDiagnosisEngine:
    """
    Core inference engine for rice disease diagnosis.
    Delegates all inference logic to UnifiedInferenceEngine.
    """

    _LEGACY_CLASS_MAP = {
        0: "Healthy", 1: "Bacterial Leaf Blight", 2: "Bacterial Sheath Brown Rot",
        3: "Brown Spot", 4: "False Smut", 5: "Leaf Blast",
        6: "Node Neck Blast", 7: "Rice Leaf Beetle", 8: "Rice Leaf Folder",
        9: "Sheath Blight"
    }
    CLASS_MAP = _LEGACY_CLASS_MAP.copy()

    def __init__(self, config: Optional[Dict] = None):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or {}
        self._inference_engine: Optional[UnifiedInferenceEngine] = None

        inf_cfg = self.config.get('inference', {})
        self.tile_size = inf_cfg.get('tile_size', 384)
        self.default_stride_ratio = inf_cfg.get('stride_ratio', 0.5)
        self.default_batch_size = inf_cfg.get('batch_size', 16)
        self.conf_threshold = inf_cfg.get('conf_threshold', 0.05)

        if self.config:
            RiceDiagnosisEngine.CLASS_MAP = build_class_map_for_inference(self.config)
            print(f"[Engine] Loaded {len(RiceDiagnosisEngine.CLASS_MAP)} classes from config")

        print(f"[Engine] Initialized with device: {self.device}")

    def load_model(self, checkpoint_path: str) -> Tuple[bool, str]:
        """Load model weights from checkpoint. Returns (success, message)."""
        try:
            print(f"[Engine] Loading model from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            if 'config' in checkpoint:
                ckpt_config = checkpoint['config']
                self.config = {**self.config, **ckpt_config}

            model_name = self.config.get('model', {}).get('name', 'mil_efficientnetv2-s')
            self.model = get_model(model_name, self.config)

            state_dict = checkpoint.get('model_state_dict', checkpoint)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)

            self.model.to(self.device)
            self.model.eval()

            self._inference_engine = UnifiedInferenceEngine.from_config(
                model=self.model, device=self.device, config=self.config,
            )

            num_params = sum(p.numel() for p in self.model.parameters()) / 1e6
            return True, f"Model loaded ({num_params:.1f}M params)"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, str(e)

    def run_inference(
        self,
        image_path: str,
        scan_config: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict], List[Dict]]:
        """
        Run full image inference with spatial heatmaps.

        Args:
            image_path: Path to input image
            scan_config: Override scan parameters
                - top_k: Number of detections to return
                - percentile: Detection percentile threshold (70-95)

        Returns:
            - original_image: RGB numpy array (H, W, 3)
            - heatmap: Multi-class heatmap (num_classes, H, W) in [0, 1]
            - top_tiles: List of final detections (after NMS + top_k)
            - raw_tiles: List of all raw detections before NMS
        """
        if self.model is None or self._inference_engine is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        scan_config = scan_config or {}
        top_k = scan_config.get('top_k', 5)
        percentile = scan_config.get('percentile', 85.0)

        # Load image
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]

        print(f"[Engine] Processing {w_orig}x{h_orig} image...")

        # Run unified inference engine
        result = self._inference_engine.run(img_rgb)
        # Cache tiles_info on self for Scale Debug tab (tile-level summary)
        self._last_tiles_info = result.tiles_info

        # Extract detections via 1-class0 + percentile + CCA
        # return_raw=True: also get raw detections before NMS for visualization
        # bg_region_max: filter regions where class-0 mean > threshold (bg-dominated)
        bg_region_max = self.config.get('heatmap_generation', {}).get('bg_region_max', 0.7)
        top_tiles, raw_tiles = extract_detections(
            result.heatmap, img_rgb, top_k=top_k,
            conf_threshold=self.conf_threshold,
            class_names=self.CLASS_MAP,
            percentile=percentile,
            return_raw=True,
            bg_region_max=bg_region_max,
        )

        # Top-1 fallback: if CCA returns fewer than requested, supplement from tiles_info
        if len(top_tiles) < top_k and result.tiles_info:
            existing_coords = {d['coords'] for d in top_tiles}
            # Prefer large tiles: sort by (tile_size * score) so large-tile
            # detections rank higher than numerous small-tile detections
            fallback_tiles = sorted(
                result.tiles_info,
                key=lambda t: t.get('tile_size', 1024) * t['score'],
                reverse=True
            )
            for tile_info in fallback_tiles:
                if len(top_tiles) >= top_k:
                    break
                coords = tile_info['coords']
                if coords in existing_coords:
                    continue
                y1, y2, x1, x2 = coords
                tile_rgb = img_rgb[y1:y2, x1:x2].copy()
                class_id = tile_info['class_id']
                top_tiles.append({
                    'score': tile_info['score'],
                    'class_id': class_id,
                    'class_name': self.CLASS_MAP.get(class_id, f"Class {class_id}"),
                    'class_score': tile_info['score'],
                    'coords': coords,
                    'area': (y2 - y1) * (x2 - x1),
                    'tile_rgb': tile_rgb,
                    'top2_class': None,
                    'confidence_gap': tile_info['score'],
                    '_fallback': True,
                })
                existing_coords.add(coords)

        print(f"[Engine] Found {len(top_tiles)} candidate tiles ({len(raw_tiles)} before NMS)")
        return img_rgb, result.heatmap, top_tiles, raw_tiles, result.scale_heatmaps


# ==============================================================================
# Layer 2: Visualization Helpers
# ==============================================================================

def compute_disease_map(heatmap: np.ndarray) -> np.ndarray:
    """Compute disease confidence map: 1 - class0_prob. Returns (H, W) in [0,1]."""
    return 1.0 - heatmap[0]


def compute_entropy_map(heatmap: np.ndarray) -> np.ndarray:
    """
    Compute normalized entropy map from probability heatmap.

    Returns (H, W) in [0, 1] where 1 = maximum confusion.
    H = -sum(p * log(p + eps)) / log(num_classes)
    """
    num_classes = heatmap.shape[0]
    eps = 1e-10
    entropy = -np.sum(heatmap * np.log(heatmap + eps), axis=0)
    max_entropy = np.log(num_classes)
    if max_entropy > 0:
        entropy /= max_entropy
    return np.clip(entropy, 0, 1)


def compute_boundary_sharpness(disease_map: np.ndarray, detections: List[Dict]) -> float:
    """Compute mean gradient magnitude at detection box edges."""
    if not detections:
        return 0.0

    # Sobel gradients of disease_map
    grad_x = cv2.Sobel(disease_map.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(disease_map.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    edge_values = []
    for det in detections:
        y1, y2, x1, x2 = det['coords']
        # Sample along box edges
        if y2 > y1 and x2 > x1:
            edge_values.extend(grad_mag[y1, x1:x2].tolist())
            edge_values.extend(grad_mag[y2 - 1, x1:x2].tolist())
            edge_values.extend(grad_mag[y1:y2, x1].tolist())
            edge_values.extend(grad_mag[y1:y2, x2 - 1].tolist())

    return float(np.mean(edge_values)) if edge_values else 0.0


def overlay_on_image(
    img_rgb: np.ndarray,
    value_map: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a (H,W) value map on an image with colormap."""
    if value_map.shape[:2] != img_rgb.shape[:2]:
        value_map = cv2.resize(value_map, (img_rgb.shape[1], img_rgb.shape[0]))
    hm_uint8 = np.uint8(255 * np.clip(value_map, 0, 1))
    colored = cv2.applyColorMap(hm_uint8, colormap)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1 - alpha, colored_rgb, alpha, 0)


def draw_detection_boxes(
    img_rgb: np.ndarray,
    detections: List[Dict],
    class_map: Dict[int, str],
    style: str = 'final',
) -> np.ndarray:
    """
    Draw detection boxes with gap-based line styles and scaled text.

    style='final': colored solid/dashed boxes + full label (gap-based)
    style='raw':   semi-transparent grey boxes + class+score only
                   (visualizes all pre-NMS detections without crowding)

    gap < 10% → dashed (uncertain), gap >= 10% → solid (confident) [final only]
    Font/thickness scale with image width.
    """
    vis = img_rgb.copy()
    h, w = vis.shape[:2]

    font_scale = max(0.5, w / 2500)
    thickness = max(1, int(w / 1200))
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Color palettes
    final_colors = [
        (255, 50, 50), (255, 140, 0), (255, 220, 0),
        (50, 200, 50), (50, 150, 255),
    ]
    # For raw: class-indexed muted colors so same class = same hue
    raw_palette = [
        (160, 160, 160),  # class 0 (unused)
        (255, 120, 120),  # class 1
        (255, 180, 80),   # class 2
        (220, 220, 60),   # class 3
        (100, 220, 100),  # class 4
        (80, 180, 255),   # class 5
        (200, 80, 255),   # class 6
        (80, 240, 200),   # class 7
        (255, 80, 180),   # class 8
        (180, 140, 80),   # class 9
    ]

    if style == 'raw':
        # Draw a dim overlay so boxes are visible but not dominant
        overlay = vis.copy()
        for i, det in enumerate(detections):
            y1, y2, x1, x2 = det['coords']
            cid = det.get('class_id', 0)
            color = raw_palette[cid % len(raw_palette)]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, max(1, thickness))

            # Short label: class name only (no gap info)
            class_name = det.get('class_name', f"C{cid}")[:12]
            score = det.get('class_score', det.get('score', 0))
            label = f"{class_name} {score:.0%}"
            (tw, th), _ = cv2.getTextSize(label, font, font_scale * 0.65, 1)
            # Place label above box; if box is near top edge, place it inside the box top
            label_above_y = y1 - th - 6  # baseline sits (th+6) above box top
            if label_above_y < th:
                # Not enough room above — place inside box, just below top edge
                label_y = y1 + th + 4
            else:
                label_y = y1 - 4  # baseline 4px above box top
            cv2.rectangle(overlay, (x1, label_y - th - 2), (x1 + tw + 6, label_y + 3), color, -1)
            cv2.putText(overlay, label, (x1 + 3, label_y - 1),
                        font, font_scale * 0.65, (20, 20, 20), 1)
        # Blend: raw boxes at 70% opacity so image is still visible
        vis = cv2.addWeighted(vis, 0.3, overlay, 0.7, 0)
        return vis

    # style == 'final'
    for i, det in enumerate(detections):
        y1, y2, x1, x2 = det['coords']
        color = final_colors[i % len(final_colors)]
        gap = det.get('confidence_gap', 0.5)

        # Gap-based line style
        if gap < 0.10:
            _draw_dashed_rect(vis, (x1, y1), (x2, y2), color, thickness + 1)
        else:
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness + 1)

        # Label with top-1 and top-2
        class_name = det.get('class_name', f"Class {det['class_id']}")[:18]
        score = det.get('class_score', det['score'])
        label = f"#{i+1} {class_name} {score:.0%}"

        top2 = det.get('top2_class')
        if top2:
            top2_name = top2['class_name'][:12]
            label += f" (vs {top2_name} {top2['score']:.0%}, gap={gap:.0%})"

        # Label background — placed above box; falls inside box top if near image edge
        (tw, th), _ = cv2.getTextSize(label, font, font_scale * 0.7, max(1, thickness))
        if y1 - th - 8 < th:
            # Not enough room above box — place label inside box, just below top edge
            label_y = y1 + th + 6
        else:
            label_y = y1 - 6   # baseline 6px above box top
        cv2.rectangle(vis, (x1, label_y - th - 4), (x1 + tw + 10, label_y + 4), color, -1)
        cv2.putText(vis, label, (x1 + 5, label_y - 2),
                    font, font_scale * 0.7, (255, 255, 255), max(1, thickness))

    return vis


def _draw_dashed_rect(img, pt1, pt2, color, thickness, dash_len=15, gap_len=10):
    """Draw a dashed rectangle on img."""
    x1, y1 = pt1
    x2, y2 = pt2
    for edge in [
        ((x1, y1), (x2, y1)),  # top
        ((x2, y1), (x2, y2)),  # right
        ((x2, y2), (x1, y2)),  # bottom
        ((x1, y2), (x1, y1)),  # left
    ]:
        _draw_dashed_line(img, edge[0], edge[1], color, thickness, dash_len, gap_len)


def _draw_dashed_line(img, pt1, pt2, color, thickness, dash_len=15, gap_len=10):
    """Draw a dashed line from pt1 to pt2."""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    length = max(1, int(np.sqrt(dx * dx + dy * dy)))
    step = dash_len + gap_len

    for i in range(0, length, step):
        t1 = i / length
        t2 = min((i + dash_len) / length, 1.0)
        start = (int(pt1[0] + dx * t1), int(pt1[1] + dy * t1))
        end = (int(pt1[0] + dx * t2), int(pt1[1] + dy * t2))
        cv2.line(img, start, end, color, thickness)


def _letterbox(img: np.ndarray, target: int) -> np.ndarray:
    """Resize image to (target, target) with aspect-ratio preserved, black padding."""
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    y_off = (target - new_h) // 2
    x_off = (target - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def create_tile_gallery(
    tiles: List[Dict],
    top_k: int = 5,
    tile_display_size: int = 150,
) -> np.ndarray:
    """Create gallery of top-K tiles with labels (aspect-ratio preserved via letterbox)."""
    display_tiles = tiles[:top_k]
    if not display_tiles:
        placeholder = np.zeros((tile_display_size, tile_display_size * 2, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No disease detected", (10, tile_display_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return placeholder

    n = len(display_tiles)
    grid_h = tile_display_size + 40
    grid_w = tile_display_size * n
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    colors = [(255, 50, 50), (255, 140, 0), (255, 220, 0), (50, 200, 50), (50, 150, 255)]

    for i, tile in enumerate(display_tiles):
        tile_rgb = tile.get('tile_rgb')
        if tile_rgb is None:
            continue
        # Letterbox: keep aspect ratio, pad with black
        cell = _letterbox(tile_rgb, tile_display_size)
        color = colors[i % len(colors)]
        cv2.rectangle(cell, (0, 0), (tile_display_size - 1, tile_display_size - 1), color, 3)

        x_off = i * tile_display_size
        grid[0:tile_display_size, x_off:x_off + tile_display_size] = cell

        label = f"#{i+1} {tile['class_name'][:12]}"
        fallback = " [tile]" if tile.get('_fallback') else ""
        # Also show original tile dimensions
        h_t, w_t = tile_rgb.shape[:2]
        cv2.putText(grid, label + fallback, (x_off + 4, tile_display_size + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        cv2.putText(grid, f"{tile['score']:.1%} ({w_t}x{h_t})",
                    (x_off + 4, tile_display_size + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

    return grid


# ==============================================================================
# Layer 3: GUI Application (4-Tab Layout)
# ==============================================================================
class RiceAnalysisApp:
    """
    Tkinter-based GUI with 4-tab diagnostic layout.

    Tabs:
        1. Detection: Original + boxes + gallery
        2. Heatmap: Disease confidence + entropy + stats
        3. Per-Class: Dropdown + large view + 3x3 grid
        4. Components: PCA feature maps + attention + dual ckpt
    """

    def __init__(self, root: tk.Tk, config: Optional[Dict] = None):
        self.root = root
        self.root.title("Rice Disease MIL Diagnostic v4.0")
        self.root.geometry("1500x950")

        self.config = config or {}
        self.engine = RiceDiagnosisEngine(config)
        self.engine_b: Optional[RiceDiagnosisEngine] = None  # Dual checkpoint

        self.current_results = None  # (img_rgb, heatmap, detections, raw_detections, scale_hms)
        self.current_results_b = None
        self.img_path = None

        # Image display state (for click interaction)
        self._display_ratio = 1.0
        self._display_offset = (0, 0)

        self._init_ui()

    def _init_ui(self):
        style = ttk.Style()
        style.theme_use('clam')

        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_paned, padding=5)
        right_frame = ttk.Frame(main_paned, padding=5)

        main_paned.add(left_frame, weight=1)
        main_paned.add(right_frame, weight=5)

        self._build_control_panel(left_frame)
        self._build_tab_panel(right_frame)

    # ================================================================
    # Control Panel (Left)
    # ================================================================
    def _build_control_panel(self, parent: ttk.Frame):
        ttk.Label(parent, text="Controls", font=("Arial", 11, "bold")).pack(pady=5, anchor="w")

        # --- Inputs ---
        grp = ttk.LabelFrame(parent, text="Model & Image")
        grp.pack(fill=tk.X, pady=3)

        self.btn_ckpt = ttk.Button(grp, text="Load Checkpoint A", command=self.load_checkpoint)
        self.btn_ckpt.pack(fill=tk.X, padx=4, pady=2)
        self.lbl_ckpt = ttk.Label(grp, text="No model", foreground="red", wraplength=180)
        self.lbl_ckpt.pack(fill=tk.X, padx=4)

        self.btn_ckpt_b = ttk.Button(grp, text="Load Checkpoint B", command=self.load_checkpoint_b)
        self.btn_ckpt_b.pack(fill=tk.X, padx=4, pady=2)
        self.lbl_ckpt_b = ttk.Label(grp, text="(optional)", foreground="gray", wraplength=180)
        self.lbl_ckpt_b.pack(fill=tk.X, padx=4)

        self.btn_img = ttk.Button(grp, text="Select Image", command=self.load_image)
        self.btn_img.pack(fill=tk.X, padx=4, pady=2)
        self.lbl_img = ttk.Label(grp, text="No image", wraplength=180)
        self.lbl_img.pack(fill=tk.X, padx=4)

        # --- Parameters ---
        grp2 = ttk.LabelFrame(parent, text="Parameters")
        grp2.pack(fill=tk.X, pady=3)

        ttk.Label(grp2, text="Top-K:").pack(anchor="w", padx=4)
        self.entry_topk = ttk.Entry(grp2, width=8)
        self.entry_topk.insert(0, "5")
        self.entry_topk.pack(fill=tk.X, padx=4, pady=1)

        ttk.Label(grp2, text="Stride Ratio:").pack(anchor="w", padx=4)
        self.entry_stride = ttk.Entry(grp2, width=8)
        self.entry_stride.insert(0, "0.5")
        self.entry_stride.pack(fill=tk.X, padx=4, pady=1)

        # Percentile slider
        ttk.Label(grp2, text="Detection Percentile:").pack(anchor="w", padx=4)
        self.percentile_var = tk.IntVar(value=85)
        pctl_frame = ttk.Frame(grp2)
        pctl_frame.pack(fill=tk.X, padx=4, pady=1)
        self.percentile_slider = ttk.Scale(
            pctl_frame, from_=70, to=95, variable=self.percentile_var,
            orient=tk.HORIZONTAL, command=lambda v: self.lbl_pctl.config(text=f"{int(float(v))}%")
        )
        self.percentile_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_pctl = ttk.Label(pctl_frame, text="85%", width=5)
        self.lbl_pctl.pack(side=tk.RIGHT)

        # --- Run ---
        self.btn_run = ttk.Button(parent, text="START DIAGNOSIS", command=self.start_inference)
        self.btn_run.pack(fill=tk.X, pady=10)

        # --- Status ---
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN,
                  wraplength=200).pack(side=tk.BOTTOM, fill=tk.X)

    # ================================================================
    # Tab Panel (Right)
    # ================================================================
    def _build_tab_panel(self, parent: ttk.Frame):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Detection
        self.tab_detect = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_detect, text=" Detection ")
        self._build_tab_detection(self.tab_detect)

        # Tab 2: Heatmap + Entropy
        self.tab_heatmap = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_heatmap, text=" Heatmap ")
        self._build_tab_heatmap(self.tab_heatmap)

        # Tab 3: Per-Class
        self.tab_perclass = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_perclass, text=" Per-Class ")
        self._build_tab_perclass(self.tab_perclass)

        # Tab 4: Components
        self.tab_components = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_components, text=" Components ")
        self._build_tab_components(self.tab_components)

        # Tab 5: Scale Diagnostics
        self.tab_scale_debug = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_scale_debug, text=" Scale Debug ")
        self._build_tab_scale_debug(self.tab_scale_debug)

    # --- Tab 1: Detection ---
    def _build_tab_detection(self, parent: ttk.Frame):
        # Count comparison banner
        self.detect_count_var = tk.StringVar(value="Run inference to see before/after NMS")
        ttk.Label(parent, textvariable=self.detect_count_var,
                  font=("Consolas", 9, "bold"), relief=tk.GROOVE, padding=3).pack(fill=tk.X)

        # Side-by-side Raw / Final
        img_row = ttk.Frame(parent)
        img_row.pack(fill=tk.BOTH, expand=True)

        # Left: Raw detections (before NMS)
        raw_frame = ttk.LabelFrame(img_row, text="Before Post-Processing  (all CCA boxes)")
        raw_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.detect_raw_canvas = tk.Canvas(raw_frame, bg='#1a1a2e')
        self.detect_raw_canvas.pack(fill=tk.BOTH, expand=True)
        self.detect_raw_canvas.bind("<Button-1>", self._on_canvas_click)

        # Right: Final detections (after NMS + top_k)
        final_frame = ttk.LabelFrame(img_row, text="After Post-Processing  (NMS + top-K)")
        final_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=2)
        self.detect_canvas = tk.Canvas(final_frame, bg='#1a1a2e')
        self.detect_canvas.pack(fill=tk.BOTH, expand=True)
        self.detect_canvas.bind("<Button-1>", self._on_canvas_click)

        # Detection list + gallery (bottom)
        bottom_row = ttk.Frame(parent)
        bottom_row.pack(fill=tk.X)

        list_frame = ttk.LabelFrame(bottom_row, text="Final Detections", width=250)
        list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=3)
        list_frame.pack_propagate(False)

        self.detect_listbox = tk.Listbox(list_frame, font=("Consolas", 9), bg='#1a1a2e', fg='white',
                                          selectbackground='#3a3a5e', height=5)
        self.detect_listbox.pack(fill=tk.BOTH, expand=True)

        gallery_wrap = ttk.Frame(bottom_row)
        gallery_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=3)
        self.gallery_label = ttk.Label(gallery_wrap, text="Tile Gallery")
        self.gallery_label.pack(fill=tk.X)

    # --- Tab 2: Heatmap + Entropy ---
    def _build_tab_heatmap(self, parent: ttk.Frame):
        img_frame = ttk.Frame(parent)
        img_frame.pack(fill=tk.BOTH, expand=True)

        # Left: Disease confidence
        left = ttk.LabelFrame(img_frame, text="Disease Confidence (1 - class0)")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.heatmap_canvas = tk.Canvas(left, bg='#1a1a2e')
        self.heatmap_canvas.pack(fill=tk.BOTH, expand=True)
        self.heatmap_canvas.bind("<Button-1>", self._on_canvas_click)

        # Right: Entropy
        right = ttk.LabelFrame(img_frame, text="Normalized Entropy (confusion)")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=2)
        self.entropy_canvas = tk.Canvas(right, bg='#1a1a2e')
        self.entropy_canvas.pack(fill=tk.BOTH, expand=True)

        # Stats bar
        self.stats_var = tk.StringVar(value="Run inference to see stats")
        ttk.Label(parent, textvariable=self.stats_var, font=("Consolas", 9),
                  relief=tk.GROOVE, padding=4).pack(fill=tk.X, pady=2)

    # --- Tab 3: Per-Class ---
    def _build_tab_perclass(self, parent: ttk.Frame):
        top = ttk.Frame(parent)
        top.pack(fill=tk.X, pady=3)

        ttk.Label(top, text="Class:").pack(side=tk.LEFT, padx=4)
        self.class_combo = ttk.Combobox(top, state='readonly', width=30)
        self.class_combo.pack(side=tk.LEFT, padx=4)
        self.class_combo.bind("<<ComboboxSelected>>", self._on_class_selected)
        self.class_stats_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.class_stats_var, font=("Consolas", 9)).pack(side=tk.LEFT, padx=10)

        content = ttk.Frame(parent)
        content.pack(fill=tk.BOTH, expand=True)

        # Large class heatmap (left)
        self.class_heatmap_canvas = tk.Canvas(content, bg='#1a1a2e')
        self.class_heatmap_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

        # 3x3 grid (right)
        grid_frame = ttk.LabelFrame(content, text="Disease Overview (click to select)")
        grid_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=2)
        self.grid_canvas = tk.Canvas(grid_frame, bg='#1a1a2e', width=420, height=420)
        self.grid_canvas.pack(fill=tk.BOTH, expand=True)
        self.grid_canvas.bind("<Button-1>", self._on_grid_click)

    # --- Tab 4: Components ---
    def _build_tab_components(self, parent: ttk.Frame):
        top = ttk.Frame(parent)
        top.pack(fill=tk.X, pady=3)

        ttk.Label(top, text="Tile:").pack(side=tk.LEFT, padx=4)
        self.tile_combo = ttk.Combobox(top, state='readonly', width=40)
        self.tile_combo.pack(side=tk.LEFT, padx=4)
        self.tile_combo.bind("<<ComboboxSelected>>", self._on_tile_selected)

        # Component strip: 5 panels
        self.component_canvas = tk.Canvas(parent, bg='#1a1a2e', height=250)
        self.component_canvas.pack(fill=tk.X, pady=3)

        # Dual checkpoint comparison (bottom)
        self.dual_frame = ttk.LabelFrame(parent, text="Model A vs Model B Comparison")
        self.dual_frame.pack(fill=tk.BOTH, expand=True, pady=3)
        self.dual_canvas = tk.Canvas(self.dual_frame, bg='#1a1a2e')
        self.dual_canvas.pack(fill=tk.BOTH, expand=True)

    # --- Tab 5: Scale Debug ---
    def _build_tab_scale_debug(self, parent: ttk.Frame):
        """
        Tab 5: Scale Diagnostics
        Diagnose why certain diseases (e.g. BLB large-lesion) are missed.

        Layout:
          Top bar: class selector (shares class_combo value) + info label
          Row 1 (disease map): 3 canvases side-by-side, one per tile scale
                               each shows 1-heatmap[0] (disease confidence)
          Row 2 (per-class):   3 canvases side-by-side for selected class channel
          Bottom: Tile-level BG vs Disease scatter (text summary of tiles_info)
        """
        # ---- Top bar: class selector + help text ----
        top = ttk.Frame(parent)
        top.pack(fill=tk.X, pady=3, padx=4)
        ttk.Label(top, text="Class:").pack(side=tk.LEFT, padx=2)
        self.scale_debug_class_combo = ttk.Combobox(top, state='readonly', width=28)
        self.scale_debug_class_combo.pack(side=tk.LEFT, padx=4)
        self.scale_debug_class_combo.bind(
            "<<ComboboxSelected>>", self._on_scale_debug_class_selected
        )
        self.scale_debug_info_var = tk.StringVar(
            value="Run inference to populate scale-level heatmaps"
        )
        ttk.Label(top, textvariable=self.scale_debug_info_var,
                  font=("Consolas", 8), foreground='#aaaaaa').pack(side=tk.LEFT, padx=8)

        # ---- Row 1: disease map per scale ----
        row1_frame = ttk.LabelFrame(parent, text="Disease Confidence Map  (1 − BG prob)  per Scale")
        row1_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        self._scale_disease_canvases: Dict[str, tk.Canvas] = {}
        self._scale_disease_labels: Dict[str, ttk.Label] = {}
        for col, label in enumerate(["Small / 768 px", "Medium / 1024 px", "Large / 1536 px"]):
            col_frame = ttk.Frame(row1_frame)
            col_frame.grid(row=0, column=col, sticky='nsew', padx=2, pady=2)
            row1_frame.columnconfigure(col, weight=1)
            row1_frame.rowconfigure(0, weight=1)
            lbl = ttk.Label(col_frame, text=label, font=("Consolas", 8, "bold"),
                            anchor='center')
            lbl.pack(fill=tk.X)
            self._scale_disease_labels[label] = lbl
            canvas = tk.Canvas(col_frame, bg='#0d0d1a')
            canvas.pack(fill=tk.BOTH, expand=True)
            self._scale_disease_canvases[label] = canvas

        # ---- Row 2: per-class heatmap per scale ----
        row2_frame = ttk.LabelFrame(parent, text="Per-Class Heatmap  per Scale  (selected class above)")
        row2_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        self._scale_class_canvases: Dict[str, tk.Canvas] = {}
        for col, label in enumerate(["Small / 768 px", "Medium / 1024 px", "Large / 1536 px"]):
            col_frame = ttk.Frame(row2_frame)
            col_frame.grid(row=0, column=col, sticky='nsew', padx=2, pady=2)
            row2_frame.columnconfigure(col, weight=1)
            row2_frame.rowconfigure(0, weight=1)
            ttk.Label(col_frame, text=label, font=("Consolas", 8, "bold"),
                      anchor='center').pack(fill=tk.X)
            canvas = tk.Canvas(col_frame, bg='#0d0d1a')
            canvas.pack(fill=tk.BOTH, expand=True)
            self._scale_class_canvases[label] = canvas

        # ---- Bottom: tile-level BG/Disease scatter (text) ----
        bottom_frame = ttk.LabelFrame(parent, text="Tile-Level Score Summary (tiles_info — per scale)")
        bottom_frame.pack(fill=tk.X, padx=4, pady=2)
        self.scale_tile_text = tk.Text(
            bottom_frame, height=6, bg='#0d0d1a', fg='#cccccc',
            font=("Consolas", 8), state=tk.DISABLED
        )
        self.scale_tile_text.pack(fill=tk.X)

    # ================================================================
    # Actions
    # ================================================================
    def load_checkpoint(self):
        path = filedialog.askopenfilename(
            filetypes=[("PyTorch Checkpoint", "*.pth"), ("All Files", "*.*")])
        if path:
            threading.Thread(target=self._load_model_thread, args=(path, False), daemon=True).start()

    def load_checkpoint_b(self):
        path = filedialog.askopenfilename(
            filetypes=[("PyTorch Checkpoint", "*.pth"), ("All Files", "*.*")])
        if path:
            threading.Thread(target=self._load_model_thread, args=(path, True), daemon=True).start()

    def _load_model_thread(self, path: str, is_b: bool):
        self.status_var.set(f"Loading {'Model B' if is_b else 'Model A'}...")
        self.btn_run.config(state=tk.DISABLED)

        if is_b:
            if self.engine_b is None:
                self.engine_b = RiceDiagnosisEngine(self.config)
            success, msg = self.engine_b.load_model(path)
        else:
            success, msg = self.engine.load_model(path)

        def update_ui():
            lbl = self.lbl_ckpt_b if is_b else self.lbl_ckpt
            if success:
                lbl.config(text=os.path.basename(path), foreground="green")
                self.status_var.set(f"{'B' if is_b else 'A'}: {msg}")
            else:
                messagebox.showerror("Load Error", msg)
                self.status_var.set("Load Failed.")
            self.btn_run.config(state=tk.NORMAL)

        self.root.after(0, update_ui)

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")])
        if path:
            self.img_path = path
            self.lbl_img.config(text=os.path.basename(path))

    def start_inference(self):
        if self.engine.model is None:
            messagebox.showwarning("Warning", "Load Model A first.")
            return
        if not self.img_path:
            messagebox.showwarning("Warning", "Select an image first.")
            return

        try:
            top_k = int(self.entry_topk.get())
            percentile = float(self.percentile_var.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter: {e}")
            return

        scan_cfg = {'top_k': top_k, 'percentile': percentile}
        threading.Thread(target=self._inference_thread, args=(scan_cfg,), daemon=True).start()

    def _inference_thread(self, scan_cfg: Dict):
        self.status_var.set("Running inference (Model A)...")
        self.btn_run.config(state=tk.DISABLED)

        try:
            img_rgb, heatmap, tiles, raw_tiles, scale_hms = self.engine.run_inference(self.img_path, scan_cfg)
            self.current_results = (img_rgb, heatmap, tiles, raw_tiles, scale_hms)

            # Run Model B if loaded
            if self.engine_b is not None and self.engine_b.model is not None:
                self.status_var.set("Running inference (Model B)...")
                img_b, hm_b, tiles_b, raw_b, _scale_hms_b = self.engine_b.run_inference(self.img_path, scan_cfg)
                self.current_results_b = (img_b, hm_b, tiles_b, raw_b)
            else:
                self.current_results_b = None

            self.root.after(0, lambda: self._update_all_tabs(scan_cfg))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_var.set("Done."))

    # ================================================================
    # Display Update
    # ================================================================
    def _update_all_tabs(self, scan_cfg: Dict):
        if self.current_results is None:
            return

        img_rgb, heatmap, detections, raw_detections, scale_hms = self.current_results
        top_k = scan_cfg.get('top_k', 5)

        self._update_tab_detection(img_rgb, heatmap, detections, raw_detections, top_k)
        self._update_tab_heatmap(img_rgb, heatmap, detections)
        self._update_tab_perclass(img_rgb, heatmap)
        self._update_tab_components(img_rgb, heatmap, detections)
        self._update_tab_scale_debug(img_rgb, scale_hms)

    # --- Tab 1 Update ---
    def _update_tab_detection(self, img_rgb, heatmap, detections, raw_detections, top_k):
        # Left: Raw (before NMS) — draw ALL raw detections in grey/orange palette
        raw_vis = draw_detection_boxes(
            img_rgb, raw_detections, self.engine.CLASS_MAP, style='raw'
        )
        self._show_on_canvas(raw_vis, self.detect_raw_canvas)

        # Right: Final (after NMS + top_k)
        final_vis = draw_detection_boxes(img_rgb, detections, self.engine.CLASS_MAP, style='final')
        self._show_on_canvas(final_vis, self.detect_canvas)

        # Count comparison banner
        n_raw = len(raw_detections)
        n_final = len(detections)
        suppressed = n_raw - n_final
        self.detect_count_var.set(
            f"CCA raw: {n_raw} boxes  →  NMS removed: {suppressed}  →  Final kept: {n_final} (top-{top_k})"
        )

        # Detection list (final only)
        self.detect_listbox.delete(0, tk.END)
        for i, d in enumerate(detections):
            gap_str = f"gap={d.get('confidence_gap', 0):.0%}"
            fb = " [tile-fb]" if d.get('_fallback') else ""
            entry = f"#{i+1} {d['class_name'][:15]} {d['score']:.1%} area={d['area']:,} {gap_str}{fb}"
            self.detect_listbox.insert(tk.END, entry)

        # Gallery (final tiles)
        gallery = create_tile_gallery(detections, top_k=top_k)
        self._show_on_label(gallery, self.gallery_label, max_h=190)

    # --- Tab 2 Update ---
    def _update_tab_heatmap(self, img_rgb, heatmap, detections):
        disease_map = compute_disease_map(heatmap)
        entropy_map = compute_entropy_map(heatmap)

        # Left: Disease overlay
        disease_vis = overlay_on_image(img_rgb, disease_map, alpha=0.4, colormap=cv2.COLORMAP_JET)
        self._show_on_canvas(disease_vis, self.heatmap_canvas)

        # Right: Entropy overlay
        entropy_vis = overlay_on_image(img_rgb, entropy_map, alpha=0.4, colormap=cv2.COLORMAP_INFERNO)
        self._show_on_canvas(entropy_vis, self.entropy_canvas)

        # Stats
        mean_dis = float(disease_map.mean())
        max_dis = float(disease_map.max())
        mean_ent = float(entropy_map.mean())
        high_ent_pct = float((entropy_map > 0.65).sum()) / max(1, entropy_map.size) * 100
        fg_ratio = float((disease_map > 0.15).sum()) / max(1, disease_map.size) * 100
        boundary = compute_boundary_sharpness(disease_map, detections)

        self.stats_var.set(
            f"Disease: mean={mean_dis:.3f} max={max_dis:.3f} | "
            f"Entropy: mean={mean_ent:.3f} high>{0.65}:{high_ent_pct:.1f}% | "
            f"FG:{fg_ratio:.1f}% | BoundarySharp:{boundary:.4f}"
        )

    # --- Tab 3 Update ---
    def _update_tab_perclass(self, img_rgb, heatmap):
        num_classes = heatmap.shape[0]
        class_map = self.engine.CLASS_MAP

        # Populate combobox (disease classes only, skip class 0)
        items = []
        for c in range(1, num_classes):
            name = class_map.get(c, f"Class {c}")
            coverage = float((heatmap[c] > 0.2).sum()) / max(1, heatmap[c].size) * 100
            items.append(f"{c}: {name} (cov={coverage:.1f}%)")
        self.class_combo['values'] = items
        if items:
            self.class_combo.current(0)
            self._show_class_heatmap(img_rgb, heatmap, 1)

        # 3x3 Grid
        self._draw_class_grid(img_rgb, heatmap, class_map)

    def _show_class_heatmap(self, img_rgb, heatmap, class_id):
        """Show single-class heatmap on large canvas with coverage + hotspot metrics."""
        class_heat = heatmap[class_id]
        vis = overlay_on_image(img_rgb, class_heat, alpha=0.5, colormap=cv2.COLORMAP_JET)
        self._show_on_canvas(vis, self.class_heatmap_canvas)

        max_val = float(class_heat.max())
        # Coverage: % of pixels above meaningful threshold
        coverage = float((class_heat > 0.2).sum()) / max(1, class_heat.size) * 100
        # Hotspot mean: average probability in top-10% hottest pixels
        flat = class_heat.ravel()
        top10_threshold = np.percentile(flat, 90)
        hotspot_mask = flat >= top10_threshold
        hotspot_mean = float(flat[hotspot_mask].mean()) if hotspot_mask.any() else 0.0

        name = self.engine.CLASS_MAP.get(class_id, f"Class {class_id}")
        self.class_stats_var.set(
            f"{name}: coverage(>0.2)={coverage:.1f}%  hotspot_mean={hotspot_mean:.3f}  max={max_val:.3f}"
        )

    def _draw_class_grid(self, img_rgb, heatmap, class_map):
        """Draw 3x3 grid of disease class heatmaps."""
        num_classes = heatmap.shape[0]
        cell_size = 140
        grid_w = 3 * cell_size
        grid_h = 3 * cell_size
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        self._grid_cell_map = {}  # (row, col) -> class_id

        for idx in range(min(9, num_classes - 1)):
            class_id = idx + 1
            row, col = idx // 3, idx % 3
            self._grid_cell_map[(row, col)] = class_id

            y_off = row * cell_size
            x_off = col * cell_size
            inner = cell_size - 4

            # Mini heatmap overlay
            class_heat = heatmap[class_id]
            h_small = cv2.resize(class_heat, (inner, inner - 18))
            colored = cv2.applyColorMap(np.uint8(255 * np.clip(h_small, 0, 1)), cv2.COLORMAP_JET)
            colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

            grid[y_off + 2:y_off + 2 + inner - 18, x_off + 2:x_off + 2 + inner] = colored_rgb

            # Coverage % in corner (more informative than max which is always ~1.0)
            coverage = float((class_heat > 0.2).sum()) / max(1, class_heat.size) * 100
            cv2.putText(grid, f"{coverage:.1f}%", (x_off + inner - 50, y_off + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Class name at bottom
            name = class_map.get(class_id, f"C{class_id}")[:14]
            cv2.putText(grid, name, (x_off + 4, y_off + cell_size - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

        self._show_on_canvas(grid, self.grid_canvas)
        self._grid_cell_size = cell_size

    def _on_class_selected(self, event=None):
        if self.current_results is None:
            return
        sel = self.class_combo.get()
        if not sel:
            return
        class_id = int(sel.split(":")[0])
        img_rgb, heatmap, _, _raw, _shms = self.current_results
        self._show_class_heatmap(img_rgb, heatmap, class_id)

    def _on_grid_click(self, event):
        """Click on 3x3 grid to select class."""
        if not hasattr(self, '_grid_cell_size'):
            return
        cs = self._grid_cell_size
        col = event.x // cs
        row = event.y // cs
        class_id = self._grid_cell_map.get((row, col))
        if class_id is not None:
            # Update combobox
            for i, val in enumerate(self.class_combo['values']):
                if val.startswith(f"{class_id}:"):
                    self.class_combo.current(i)
                    self._on_class_selected()
                    break

    # --- Tab 4 Update ---
    def _update_tab_components(self, img_rgb, heatmap, detections):
        """Update component diagnostics tab."""
        # Populate tile selector
        items = []
        for i, d in enumerate(detections):
            items.append(f"#{i+1} {d['class_name'][:15]} ({d['score']:.1%})")
        if not items:
            items.append("(no detections)")
        self.tile_combo['values'] = items
        self.tile_combo.current(0)

        # Show component features for top-1
        if detections:
            self._show_component_strip(img_rgb, detections[0])

        # Dual checkpoint comparison
        if self.current_results_b is not None:
            self._show_dual_comparison(detections)

    # --- Tab 5 Update ---
    def _update_tab_scale_debug(self, img_rgb: np.ndarray, scale_hms: Optional[Dict]):
        """
        Populate Tab 5 Scale Diagnostics with per-scale heatmaps.

        scale_hms: {tile_size: (num_classes, H, W)} — raw confidence per scale.
        Layout: 3 columns (768/1024/1536), 2 rows (disease map / per-class).
        """
        if scale_hms is None:
            self.scale_debug_info_var.set("No scale data (scale_diagnostics disabled)")
            return

        # Sort scales
        sorted_ts = sorted(scale_hms.keys())
        col_labels = ["Small / 768 px", "Medium / 1024 px", "Large / 1536 px"]
        # Map available tile_sizes to column labels by position
        ts_to_label: Dict[int, str] = {}
        for i, ts in enumerate(sorted_ts):
            if i < len(col_labels):
                ts_to_label[ts] = col_labels[i]

        # Update class combo (disease classes only)
        num_classes = next(iter(scale_hms.values())).shape[0]
        class_map = self.engine.CLASS_MAP
        items = []
        for c in range(1, num_classes):
            name = class_map.get(c, f"Class {c}")
            items.append(f"{c}: {name}")
        self.scale_debug_class_combo['values'] = items
        if items and not self.scale_debug_class_combo.get():
            self.scale_debug_class_combo.current(0)

        scale_strs = " / ".join(str(ts) for ts in sorted_ts)
        self.scale_debug_info_var.set(f"Scales: {scale_strs}px  |  Click class dropdown to compare")

        # Store for redraw when class changes
        self._scale_hms = scale_hms
        self._scale_img_rgb = img_rgb
        self._scale_ts_to_label = ts_to_label

        # Render disease maps + per-class maps
        self._render_scale_debug()

        # Tile-level summary
        self._render_scale_tile_summary()

    def _render_scale_debug(self):
        """Render Row1 (disease map) and Row2 (per-class) for all scales."""
        if not hasattr(self, '_scale_hms') or self._scale_hms is None:
            return

        img_rgb = self._scale_img_rgb
        scale_hms = self._scale_hms
        ts_to_label = self._scale_ts_to_label

        # Which class is selected?
        sel = self.scale_debug_class_combo.get()
        try:
            selected_class_id = int(sel.split(":")[0]) if sel else 1
        except (ValueError, IndexError):
            selected_class_id = 1

        for ts, sh in scale_hms.items():
            label = ts_to_label.get(ts)
            if label is None:
                continue

            # Row 1: disease map = 1 - class0
            disease_map = np.clip(1.0 - sh[0], 0.0, 1.0)  # (H, W)
            disease_vis = overlay_on_image(img_rgb, disease_map, alpha=0.55,
                                           colormap=cv2.COLORMAP_JET)
            # Annotate with tile_size
            cv2.putText(disease_vis, f"ts={ts}px", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            if label in self._scale_disease_canvases:
                self._show_on_canvas(disease_vis, self._scale_disease_canvases[label])

            # Row 2: selected class heatmap
            if selected_class_id < sh.shape[0]:
                class_map_data = sh[selected_class_id]  # (H, W)
                class_vis = overlay_on_image(img_rgb, class_map_data, alpha=0.6,
                                             colormap=cv2.COLORMAP_JET)
                class_name = self.engine.CLASS_MAP.get(selected_class_id, f"C{selected_class_id}")
                cv2.putText(class_vis, f"{class_name} ts={ts}px", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                if label in self._scale_class_canvases:
                    self._show_on_canvas(class_vis, self._scale_class_canvases[label])

    def _render_scale_tile_summary(self):
        """Show per-scale tile-level BG vs Disease score summary."""
        if self.current_results is None:
            return

        lines = []
        engine = self.engine  # RiceDiagnosisEngine — _last_tiles_info set in run_inference()
        if (hasattr(engine, '_last_tiles_info') and engine._last_tiles_info):
            tiles_info = engine._last_tiles_info
            by_scale: Dict[int, list] = {}
            for t in tiles_info:
                ts = t.get('tile_size', 0)
                by_scale.setdefault(ts, []).append(t)

            for ts in sorted(by_scale.keys()):
                group = by_scale[ts]
                bg_probs = [t['bg_prob'] for t in group]
                scores = [t['score'] for t in group]
                n = len(group)
                bg_mean = np.mean(bg_probs)
                dis_mean = np.mean(scores)
                bg_hi = sum(1 for b in bg_probs if b > 0.5) / max(n, 1) * 100
                dis_hi = sum(1 for s in scores if s > 0.3) / max(n, 1) * 100
                lines.append(
                    f"ts={ts:4d}px │ n={n:4d} tiles │ "
                    f"bg_prob: mean={bg_mean:.3f}  >0.5={bg_hi:.0f}% │ "
                    f"disease: mean={dis_mean:.3f}  >0.3={dis_hi:.0f}%"
                )
        else:
            lines.append("(tiles_info not available — run inference first)")

        text = "\n".join(lines)
        self.scale_tile_text.config(state=tk.NORMAL)
        self.scale_tile_text.delete("1.0", tk.END)
        self.scale_tile_text.insert(tk.END, text)
        self.scale_tile_text.config(state=tk.DISABLED)

    def _on_scale_debug_class_selected(self, event=None):
        """Redraw Row 2 when user picks a different class."""
        self._render_scale_debug()

    def _show_component_strip(self, img_rgb, detection):
        """Show 5-panel component strip for selected detection."""
        try:
            from src.inference.feature_debug import (
                extract_component_features, features_to_pca_rgb, attention_map_to_overlay
            )
        except ImportError:
            return

        y1, y2, x1, x2 = detection['coords']
        tile_crop = img_rgb[y1:y2, x1:x2].copy()

        # Preprocess tile
        tile_tensor = self.engine._inference_engine._preprocess_tile(tile_crop)
        tile_tensor = tile_tensor.unsqueeze(0)

        # Extract features
        features = extract_component_features(
            self.engine.model, tile_tensor, self.engine.device
        )

        panel_sz = 200  # Square panels: all feature maps are square (24x24 grid)
        panel_h = panel_sz
        panel_w = panel_sz
        panels = []

        # Letterboxed tile: used as background for Input + Attention panels.
        # Feature map panels (Backbone/FPN/ViT/Heatmap) come from square grids
        # so direct resize to (panel_sz, panel_sz) is correct for them.
        tile_lb = _letterbox(tile_crop, panel_sz)  # (panel_sz, panel_sz, 3), no distortion

        # Panel 1: Input tile — letterboxed
        panels.append(('Input', tile_lb.copy()))

        # Panel 2: Backbone PCA (feature grid is square → direct resize OK)
        if features['backbone'] is not None:
            bb_pca = features_to_pca_rgb(features['backbone'], target_size=(panel_sz, panel_sz))
            panels.append(('Backbone', bb_pca))

        # Panel 3: FPN PCA
        if features['fpn'] is not None:
            fpn_pca = features_to_pca_rgb(features['fpn'], target_size=(panel_sz, panel_sz))
            panels.append(('FPN', fpn_pca))

        # Panel 4: ViT PCA
        if features['vit'] is not None:
            vit_pca = features_to_pca_rgb(features['vit'], target_size=(panel_sz, panel_sz))
            panels.append(('ViT', vit_pca))

        # Panel 5: Heatmap (max disease channel, square grid → direct resize OK)
        if features['heatmap'] is not None:
            hm = features['heatmap']
            if hm.shape[0] > 1:
                disease_hm = hm[1:].max(axis=0)
            else:
                disease_hm = hm[0]
            disease_hm = np.clip(disease_hm, 0, None)
            d_max = disease_hm.max()
            if d_max > 0:
                disease_hm /= d_max
            hm_colored = cv2.applyColorMap(np.uint8(255 * disease_hm), cv2.COLORMAP_JET)
            hm_rgb = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
            hm_resized = cv2.resize(hm_rgb, (panel_sz, panel_sz))
            panels.append(('Heatmap', hm_resized))

        # Panel 6: Attention overlay — use letterboxed tile as background
        if features.get('attn_map') is not None:
            attn_overlay = attention_map_to_overlay(
                features['attn_map'],
                tile_lb,   # already (panel_sz, panel_sz, 3), no distortion
                alpha=0.5
            )
            panels.append(('Attention', attn_overlay))

        # Compose strip
        strip_w = len(panels) * (panel_w + 10) + 10
        strip_h = panel_h + 30
        strip = np.zeros((strip_h, strip_w, 3), dtype=np.uint8)

        for i, (label, img) in enumerate(panels):
            x_off = 5 + i * (panel_w + 10)
            strip[0:panel_h, x_off:x_off + panel_w] = img[:panel_h, :panel_w]
            cv2.putText(strip, label, (x_off + 5, panel_h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        self._show_on_canvas(strip, self.component_canvas)

    def _show_dual_comparison(self, detections_a):
        """Show Model A vs Model B comparison."""
        if self.current_results_b is None or not detections_a:
            return

        img_a, hm_a, _, _raw_a, _shms_a = self.current_results
        img_b, hm_b, dets_b, _raw_b = self.current_results_b

        # Use same top detection coords from Model A
        d = detections_a[0]
        y1, y2, x1, x2 = d['coords']

        disease_a = compute_disease_map(hm_a)
        disease_b = compute_disease_map(hm_b)

        # Crop disease maps to detection region
        region_a = disease_a[y1:y2, x1:x2]
        region_b = disease_b[y1:y2, x1:x2]
        diff = region_a - region_b

        panel_h = 180
        panel_w = 250

        tile_crop = img_a[y1:y2, x1:x2]
        # Letterbox tile_crop to panel size (preserve aspect ratio)
        tc_lb = _letterbox(tile_crop, min(panel_w, panel_h))
        tc_panel = cv2.resize(tc_lb, (panel_w, panel_h))
        # Resize region maps to same panel size (float, no letterbox needed — same aspect)
        reg_a_panel = cv2.resize(region_a.astype(np.float32), (panel_w, panel_h))
        reg_b_panel = cv2.resize(region_b.astype(np.float32), (panel_w, panel_h))

        vis_a = overlay_on_image(tc_panel, reg_a_panel, alpha=0.4)
        vis_b = overlay_on_image(tc_panel, reg_b_panel, alpha=0.4)

        # Diff visualization (already same size as reg panels)
        diff_resized = reg_a_panel - reg_b_panel
        diff_norm = np.clip((diff_resized + 1) / 2, 0, 1)  # [-1,1] → [0,1]
        diff_vis = cv2.applyColorMap(np.uint8(255 * diff_norm), cv2.COLORMAP_COOLWARM)
        diff_vis = cv2.cvtColor(diff_vis, cv2.COLOR_BGR2RGB)

        # Compose
        strip_w = 3 * (panel_w + 10) + 10
        strip_h = panel_h + 25
        strip = np.zeros((strip_h, strip_w, 3), dtype=np.uint8)

        for i, (label, img) in enumerate([
            (f"Model A ({d['class_name'][:12]})", vis_a),
            ("Model B", vis_b),
            ("Diff (A-B)", diff_vis),
        ]):
            x_off = 5 + i * (panel_w + 10)
            strip[0:panel_h, x_off:x_off + panel_w] = img
            cv2.putText(strip, label, (x_off + 5, panel_h + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        self._show_on_canvas(strip, self.dual_canvas)

    def _on_tile_selected(self, event=None):
        """Handle tile selection in Components tab."""
        if self.current_results is None:
            return
        sel_idx = self.tile_combo.current()
        img_rgb, heatmap, dets, _raw, _shms = self.current_results
        if 0 <= sel_idx < len(dets):
            self._show_component_strip(img_rgb, dets[sel_idx])

    # ================================================================
    # Click Interaction
    # ================================================================
    def _on_canvas_click(self, event):
        """Handle click on detection/heatmap canvas — show probability bar chart."""
        if self.current_results is None:
            return

        canvas = event.widget
        img_rgb, heatmap, _, _raw, _shms = self.current_results

        # Use stored display params from _render_canvas (accurate)
        ratio = getattr(canvas, '_display_ratio', None)
        offset = getattr(canvas, '_display_offset', None)
        img_size = getattr(canvas, '_image_size', None)
        if ratio is None or offset is None or img_size is None:
            return

        x_offset, y_offset = offset
        iw, ih = img_size

        img_x = int((event.x - x_offset) / ratio)
        img_y = int((event.y - y_offset) / ratio)

        if img_x < 0 or img_x >= iw or img_y < 0 or img_y >= ih:
            return

        # Get per-class probabilities at this location
        num_classes = heatmap.shape[0]
        probs = []
        for c in range(num_classes):
            # Heatmap might be lower resolution than image
            hh, hw = heatmap.shape[1], heatmap.shape[2]
            hy = min(int(img_y * hh / ih), hh - 1)
            hx = min(int(img_x * hw / iw), hw - 1)
            probs.append(float(heatmap[c, hy, hx]))

        self._show_prob_popup(event, probs, img_x, img_y)

    def _show_prob_popup(self, event, probs: List[float], img_x: int, img_y: int):
        """Show popup with probability bar chart."""
        popup = tk.Toplevel(self.root)
        popup.title(f"Probabilities at ({img_x}, {img_y})")
        popup.geometry("380x320")
        popup.transient(self.root)

        # Note about spatial resolution
        ttk.Label(popup, text=f"Location: ({img_x}, {img_y}) | Resolution: 24x24 grid",
                  font=("Consolas", 8)).pack(pady=2)

        # Bar chart using Canvas
        chart = tk.Canvas(popup, bg='white', height=260)
        chart.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        num_classes = len(probs)
        bar_h = max(12, 240 // num_classes)
        max_bar_w = 250

        class_map = self.engine.CLASS_MAP
        for i in range(num_classes):
            y = 5 + i * (bar_h + 2)
            w = int(probs[i] * max_bar_w)
            color = '#e74c3c' if i == 0 else '#3498db'
            if probs[i] == max(probs):
                color = '#2ecc71'

            name = class_map.get(i, f"C{i}")[:16]
            chart.create_rectangle(100, y, 100 + max(w, 1), y + bar_h, fill=color, outline='')
            chart.create_text(95, y + bar_h // 2, text=name, anchor='e', font=("Consolas", 7))
            chart.create_text(105 + max(w, 1), y + bar_h // 2, text=f"{probs[i]:.3f}",
                              anchor='w', font=("Consolas", 7))

        # Auto-close after 10 seconds
        popup.after(10000, popup.destroy)

    # ================================================================
    # Canvas/Label display helpers
    # ================================================================
    def _show_on_canvas(self, cv_img: np.ndarray, canvas: tk.Canvas):
        """
        Display image on canvas, fitting to available canvas size.

        Stores the source image on the canvas so it can be re-rendered
        on <Configure> events (resize / tab switch).
        """
        # Store source image for re-render on resize
        canvas._source_cv_img = cv_img

        # Bind Configure event for auto-resize (only once per canvas)
        if not hasattr(canvas, '_configure_bound'):
            canvas.bind("<Configure>", lambda e, c=canvas: self._on_canvas_configure(c))
            canvas._configure_bound = True

        self._render_canvas(canvas)

    def _render_canvas(self, canvas: tk.Canvas):
        """Actually render the stored image onto the canvas at current size."""
        if not hasattr(canvas, '_source_cv_img') or canvas._source_cv_img is None:
            return

        cv_img = canvas._source_cv_img
        canvas.update_idletasks()

        cw = canvas.winfo_width()
        ch = canvas.winfo_height()

        # Fallback: if canvas hasn't been laid out yet, schedule a delayed render
        if cw < 50 or ch < 50:
            canvas.after(100, lambda: self._render_canvas(canvas))
            return

        ih, iw = cv_img.shape[:2]
        # Allow upscaling to fill canvas (no 1.0 cap)
        ratio = min(cw / iw, ch / ih)
        new_w = max(1, int(iw * ratio))
        new_h = max(1, int(ih * ratio))

        resized = cv2.resize(cv_img, (new_w, new_h))
        pil_img = Image.fromarray(resized)
        tk_img = ImageTk.PhotoImage(pil_img)

        canvas.delete("all")
        x_off = (cw - new_w) // 2
        y_off = (ch - new_h) // 2
        canvas.create_image(x_off, y_off, anchor=tk.NW, image=tk_img)
        canvas._tk_img = tk_img  # Prevent garbage collection

        # Store display params for click coordinate mapping
        canvas._display_ratio = ratio
        canvas._display_offset = (x_off, y_off)
        canvas._display_size = (new_w, new_h)
        canvas._image_size = (iw, ih)

    def _on_canvas_configure(self, canvas: tk.Canvas):
        """Re-render canvas image when canvas is resized or tab switches."""
        # Debounce: cancel pending render, schedule new one
        if hasattr(canvas, '_render_after_id'):
            canvas.after_cancel(canvas._render_after_id)
        canvas._render_after_id = canvas.after(50, lambda: self._render_canvas(canvas))

    def _show_on_label(self, cv_img: np.ndarray, label: ttk.Label, max_h: int = 400):
        """Display image on a ttk.Label."""
        h, w = cv_img.shape[:2]
        # Allow upscaling for small images
        ratio = min(max_h / max(h, 1), 2.0)
        new_w, new_h = max(1, int(w * ratio)), max(1, int(h * ratio))

        resized = cv2.resize(cv_img, (new_w, new_h))
        pil_img = Image.fromarray(resized)
        tk_img = ImageTk.PhotoImage(pil_img)
        label.config(image=tk_img, text="")
        label.image = tk_img

    def on_close(self):
        self.root.destroy()


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='Rice Disease MIL Diagnostic GUI v4.0')
    parser.add_argument('--config', type=str, default='configs/algorithm/train_topk_asymmetric.yaml')
    args = parser.parse_args()

    config = {}
    if os.path.exists(args.config):
        try:
            config = load_config(args.config)
            print(f"[GUI] Loaded config from: {args.config}")
        except Exception as e:
            print(f"[GUI] Warning: Failed to load config: {e}")

    root = tk.Tk()
    app = RiceAnalysisApp(root, config)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
