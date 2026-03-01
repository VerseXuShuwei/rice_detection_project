"""
Unified Inference Engine for Rice Disease MIL.

Recent Updates:
    - [2026-02-19] Fix: Align tile scales with training distribution [768,1024,1536] (was [1024,1536,2048])
    - [2026-02-19] Fix: Reverse fusion weighting — large tiles get MORE weight (not less)
    - [2026-02-18] Fix: extract_detections uses 1-class0 + adaptive percentile (not Otsu)
    - [2026-02-18] Cleanup: Remove has_heatmap_head flag and scalar fallback (HeatmapHead is permanent)
    - [2026-02-09] Created: Unified three duplicate inference implementations into one.

Key Features:
    - Multi-scale sliding window inference (from final_evaluator + heatmap_visualizer)
    - Edge tile handling (from model.inference_sliding_window)
    - Correct batch flushing (fixed bug from final_evaluator)
    - HeatmapHead spatial output (always available, scalar fallback removed)
    - CCA-based detection: 1-class0 + adaptive percentile → Morph → CCA → BBox
    - Multi-class output: (num_classes, H, W) probability heatmap
    - Accepts raw numpy RGB image with internal preprocessing
    - Returns heatmap + tiles_info for downstream visualization

Architecture:
    This engine is the SINGLE SOURCE OF TRUTH for all inference/heatmap generation.
    All consumers (final_evaluator, heatmap_visualizer, inference_gui) call this engine
    instead of reimplementing their own sliding window logic.

    Model API used:
        - model.get_spatial_heatmap(x) -> (N, C, h, w)  [spatial heatmap, always used]
        - model.predict_instances(x)   -> (N, C)          [per-tile metadata only]

Usage:
    from src.inference.engine import UnifiedInferenceEngine

    engine = UnifiedInferenceEngine(model, device, config)
    result = engine.run(image_rgb)
    # result.heatmap: (num_classes, H, W) probability map
    # result.tiles_info: List[Dict] with per-tile detection info
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from src.data.transforms import get_validation_transforms


@dataclass
class InferenceResult:
    """Container for inference output."""
    heatmap: np.ndarray  # (num_classes, H, W) probability map in [0, 1]
    tiles_info: List[Dict] = field(default_factory=list)  # per-tile detection info
    per_class_max: Optional[Dict[int, float]] = None  # max confidence per class
    per_class_coverage: Optional[Dict[int, float]] = None  # % pixels > 0.2 per class
    # Per-scale heatmaps for diagnostic visualization.
    # Key: tile_size (e.g. 768, 1024, 1536).
    # Value: (num_classes, H, W) normalized heatmap for that scale only,
    #        BEFORE topk_norm so you see raw model confidence per scale.
    # None when scale_diagnostics=False (default: True).
    scale_heatmaps: Optional[Dict[int, np.ndarray]] = None


class UnifiedInferenceEngine:
    """
    Unified sliding window inference engine for full-image heatmap generation.

    Merges best features from three previous implementations:
    - Multi-scale from final_evaluator/heatmap_visualizer
    - Edge tile handling from model.inference_sliding_window
    - Correct batch flushing (fixes final_evaluator bug)

    Args:
        model: Trained MIL model (must have predict_instances and get_spatial_heatmap)
        device: torch device
        model_input_size: Model input tile size (default: 384)
        tile_sizes: List of crop sizes for multi-scale inference
        stride_ratio: Stride as fraction of tile_size (default: 0.5)
        batch_size: GPU batch size (default: 8)
        conf_threshold: Minimum confidence to include in tiles_info (default: 0.1)
        adaptive_scale: Whether to auto-select tile sizes based on image dimensions
        large_image_tiles: Tile sizes for large images (adaptive mode)
        small_image_tiles: Tile sizes for small images (adaptive mode)
        large_image_min_size: (min_h, min_w) threshold for large image detection
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        model_input_size: int = 384,
        tile_sizes: Optional[List[int]] = None,
        stride_ratio: float = 0.5,
        batch_size: int = 8,
        conf_threshold: float = 0.1,
        adaptive_scale: bool = True,
        large_image_tiles: Optional[List[int]] = None,
        small_image_tiles: Optional[List[int]] = None,
        large_image_min_size: Optional[Tuple[int, int]] = None,
        topk_norm_k: int = 3,
        bg_aware_alpha: float = 2.0,
        feather_ratio: float = 0.15,
    ):
        self.model = model
        self.device = device
        self.model_input_size = model_input_size
        self.tile_sizes = tile_sizes or [768, 1024, 1536]
        self.stride_ratio = stride_ratio
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.adaptive_scale = adaptive_scale
        self.large_image_tiles = large_image_tiles or [768, 1024, 1536]
        self.small_image_tiles = small_image_tiles or [384, 512, 768]
        self.large_image_min_size = large_image_min_size or (2000, 2500)
        # Per-class Top-K spatial normalization:
        # After stitch, divide each disease channel by its top-K spatial mean
        # so disease peaks are self-normalized relative to background coverage.
        # Set to 0 to disable.
        self.topk_norm_k = topk_norm_k
        # BG-aware heatmap accumulation weight:
        # tile_weight = max(0, 1 - alpha * bg_prob)
        # Tiles with high background confidence contribute less to heatmap fusion.
        # alpha=2.0: bg_prob>0.5 → weight=0 (hard cutoff at majority-background)
        # alpha=0.0: disabled (original behavior, uniform weighting)
        self.bg_aware_alpha = bg_aware_alpha
        # Tile-edge feathering: blend tile borders to reduce stitching artifacts.
        # Each tile's spatial weight = outer_product(hann_window, hann_window)
        # feather_ratio: fraction of tile size that transitions from 0 to 1.
        # 0.0 = disabled (sharp edges, original behavior)
        # 0.15 = 15% border zone tapers to 0 (recommended)
        self.feather_ratio = feather_ratio
        # Cache for feather windows keyed by (h, w)
        self._feather_cache: Dict[Tuple[int, int], np.ndarray] = {}
        # Per-scale diagnostic accumulator context (set by run() per tile_size iteration)
        self._diag_accum: Optional[np.ndarray] = None
        self._diag_count: Optional[np.ndarray] = None

        self.num_classes = getattr(model, 'num_output_classes', 10)

        # Preprocessing: same ImageNet normalization as training
        self.val_transform = get_validation_transforms(img_size=model_input_size)

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        device: torch.device,
        config: Dict,
    ) -> 'UnifiedInferenceEngine':
        """
        Factory method: create engine from config dict.

        Reads parameters from config['heatmap_generation'] and config['inference'].
        """
        heatmap_cfg = config.get('heatmap_generation', {})
        inf_cfg = config.get('inference', {})

        model_input_size = config.get('model', {}).get('img_size', 384)

        return cls(
            model=model,
            device=device,
            model_input_size=model_input_size,
            tile_sizes=heatmap_cfg.get('multiscale_tiles', [768, 1024, 1536]),
            stride_ratio=heatmap_cfg.get('stride_ratio', 0.5),
            batch_size=heatmap_cfg.get('batch_size', 8),
            conf_threshold=inf_cfg.get('conf_threshold', 0.1),
            adaptive_scale=heatmap_cfg.get('use_adaptive', True),
            large_image_tiles=heatmap_cfg.get('multiscale_tiles', [768, 1024, 1536]),
            small_image_tiles=heatmap_cfg.get('small_image_tiles', [384, 512, 768]),
            large_image_min_size=tuple(
                heatmap_cfg.get('multiscale_min_size', [2000, 2500])
            ),
            topk_norm_k=heatmap_cfg.get('topk_norm_k', 3),
            bg_aware_alpha=heatmap_cfg.get('bg_aware_alpha', 2.0),
            feather_ratio=heatmap_cfg.get('feather_ratio', 0.15),
        )

    def run(
        self,
        image_rgb: np.ndarray,
        tile_sizes: Optional[List[int]] = None,
        return_per_class: bool = True,
        scale_diagnostics: bool = True,
    ) -> InferenceResult:
        """
        Run full-image inference on a raw RGB numpy image.

        Args:
            image_rgb: (H, W, 3) uint8 RGB image
            tile_sizes: Override tile sizes (None = use config/adaptive)
            return_per_class: Whether to compute per-class max confidence
            scale_diagnostics: If True, collect per-scale heatmaps separately
                for diagnostic visualization. Stored in result.scale_heatmaps.
                Each scale's heatmap is normalized independently (no topk_norm)
                so you can see raw model confidence at each receptive field size.

        Returns:
            InferenceResult with multi-class heatmap and tiles_info
        """
        self.model.eval()

        h_orig, w_orig = image_rgb.shape[:2]

        # Select tile sizes
        if tile_sizes is None:
            tile_sizes = self._select_tile_sizes(h_orig, w_orig)

        # Initialize multi-class accumulators
        heatmap_accum = np.zeros(
            (self.num_classes, h_orig, w_orig), dtype=np.float32
        )
        # Per-channel count map: Class 0 uses scale_weight (unpenalized),
        # disease channels use disease_weight (bg-penalized). Each channel
        # must be normalized by its own accumulated weight so the averages
        # are mathematically correct.
        count_map = np.zeros((self.num_classes, h_orig, w_orig), dtype=np.float32)
        all_tiles_info = []

        # Largest tile = reference for scale weighting (weight=1.0)
        self._max_tile_size = max(tile_sizes)

        # Per-scale diagnostic accumulators (no feather, no bg-aware, no scale-weight)
        # Key: tile_size → (accum, count) for simple uniform average per scale
        scale_accum: Dict[int, np.ndarray] = {}
        scale_count: Dict[int, np.ndarray] = {}

        # Multi-scale sliding window
        for ts in tile_sizes:
            if h_orig < ts or w_orig < ts:
                continue

            # Initialize per-scale accumulators (uniform weight, no feather/bg-aware)
            if scale_diagnostics:
                scale_accum[ts] = np.zeros(
                    (self.num_classes, h_orig, w_orig), dtype=np.float32
                )
                scale_count[ts] = np.zeros(
                    (self.num_classes, h_orig, w_orig), dtype=np.float32
                )
                # Set context so _process_batch() can side-accumulate into this scale's buffers
                self._diag_accum = scale_accum[ts]
                self._diag_count = scale_count[ts]
            else:
                self._diag_accum = None
                self._diag_count = None

            stride = int(ts * self.stride_ratio)
            batch_tiles = []
            batch_coords = []

            # Main grid
            for y in range(0, h_orig - ts + 1, stride):
                for x in range(0, w_orig - ts + 1, stride):
                    tile_crop = image_rgb[y:y + ts, x:x + ts]
                    tile_tensor = self._preprocess_tile(tile_crop)
                    batch_tiles.append(tile_tensor)
                    batch_coords.append((y, x, y + ts, x + ts, ts))

                    if len(batch_tiles) >= self.batch_size:
                        self._process_batch(
                            batch_tiles, batch_coords,
                            heatmap_accum, count_map, all_tiles_info
                        )
                        batch_tiles = []
                        batch_coords = []

            # Edge tiles: right edge
            last_x_main = ((w_orig - ts) // stride) * stride
            if last_x_main + ts < w_orig:
                x = w_orig - ts
                for y in range(0, h_orig - ts + 1, stride):
                    tile_crop = image_rgb[y:y + ts, x:x + ts]
                    tile_tensor = self._preprocess_tile(tile_crop)
                    batch_tiles.append(tile_tensor)
                    batch_coords.append((y, x, y + ts, x + ts, ts))

                    if len(batch_tiles) >= self.batch_size:
                        self._process_batch(
                            batch_tiles, batch_coords,
                            heatmap_accum, count_map, all_tiles_info
                        )
                        batch_tiles = []
                        batch_coords = []

            # Edge tiles: bottom edge
            last_y_main = ((h_orig - ts) // stride) * stride
            if last_y_main + ts < h_orig:
                y = h_orig - ts
                for x in range(0, w_orig - ts + 1, stride):
                    tile_crop = image_rgb[y:y + ts, x:x + ts]
                    tile_tensor = self._preprocess_tile(tile_crop)
                    batch_tiles.append(tile_tensor)
                    batch_coords.append((y, x, y + ts, x + ts, ts))

                    if len(batch_tiles) >= self.batch_size:
                        self._process_batch(
                            batch_tiles, batch_coords,
                            heatmap_accum, count_map, all_tiles_info
                        )
                        batch_tiles = []
                        batch_coords = []

            # Edge tiles: bottom-right corner
            if last_y_main + ts < h_orig and last_x_main + ts < w_orig:
                y, x = h_orig - ts, w_orig - ts
                tile_crop = image_rgb[y:y + ts, x:x + ts]
                tile_tensor = self._preprocess_tile(tile_crop)
                batch_tiles.append(tile_tensor)
                batch_coords.append((y, x, y + ts, x + ts, ts))

            # Flush remaining batch (CRITICAL: fixes final_evaluator bug)
            if batch_tiles:
                self._process_batch(
                    batch_tiles, batch_coords,
                    heatmap_accum, count_map, all_tiles_info
                )

        # Clear diagnostic context after all scales done
        self._diag_accum = None
        self._diag_count = None

        # Build per-scale normalized heatmaps (before topk_norm, raw confidence)
        result_scale_heatmaps: Optional[Dict[int, np.ndarray]] = None
        if scale_diagnostics and scale_accum:
            result_scale_heatmaps = {}
            for ts, sa in scale_accum.items():
                sc = scale_count[ts]
                sh = np.divide(
                    sa, sc,
                    out=np.zeros_like(sa),
                    where=sc > 0
                )
                result_scale_heatmaps[ts] = sh
            print(f"[Engine] Scale diagnostics: {list(result_scale_heatmaps.keys())}")

        # Weighted average: heatmap_accum already has scale weights applied,
        # count_map accumulates the same weights → weighted average
        heatmap = np.divide(
            heatmap_accum, count_map,
            out=np.zeros_like(heatmap_accum),
            where=count_map > 0
        )

        # Per-class Top-K Spatial Normalization
        # Problem: After stitch, background tiles >> disease tiles numerically,
        # so softmax probability for disease classes is globally suppressed.
        # Fix: For each disease channel, divide by its spatial top-K mean,
        # making peaks self-relative. Class 0 (background) is intentionally skipped
        # so it still correctly anchors the probability landscape.
        if self.topk_norm_k > 0 and heatmap.shape[0] > 1:
            k = self.topk_norm_k
            for c in range(1, heatmap.shape[0]):  # disease classes only
                flat = heatmap[c].ravel()
                if len(flat) <= k:
                    continue
                # Partition to find top-K without full sort (O(n) average)
                topk_indices = np.argpartition(flat, -k)[-k:]
                topk_mean = float(flat[topk_indices].mean())
                if topk_mean > 1e-6:
                    heatmap[c] = np.clip(heatmap[c] / topk_mean, 0.0, 1.0)
            print(f"[Engine] Top-{k} spatial normalization applied to {heatmap.shape[0]-1} disease channels")

        # Per-class statistics
        per_class_max = None
        per_class_coverage = None
        if return_per_class:
            per_class_max = {}
            per_class_coverage = {}
            for c in range(self.num_classes):
                per_class_max[c] = float(heatmap[c].max())
                per_class_coverage[c] = float(
                    (heatmap[c] > 0.2).sum()
                ) / max(1, heatmap[c].size) * 100

        return InferenceResult(
            heatmap=heatmap,
            tiles_info=all_tiles_info,
            per_class_max=per_class_max,
            per_class_coverage=per_class_coverage,
            scale_heatmaps=result_scale_heatmaps,
        )

    def run_single_scale(
        self,
        image_tensor: torch.Tensor,
        stride: Optional[int] = None,
    ) -> InferenceResult:
        """
        Run single-scale inference on a pre-normalized image tensor.

        This is the lightweight path for model.inference_full_image() replacement.
        No multi-scale, no adaptive selection — just sliding window at model_input_size.

        Args:
            image_tensor: (C, H, W) normalized tensor (already preprocessed)
            stride: Sliding stride in pixels (default: model_input_size * stride_ratio)

        Returns:
            InferenceResult with multi-class heatmap
        """
        self.model.eval()

        C, H, W = image_tensor.shape
        ts = self.model_input_size
        if stride is None:
            stride = int(ts * self.stride_ratio)

        # Collect tiles + coords (with edge handling)
        tiles = []
        coords = []

        # Main grid
        for y in range(0, H - ts + 1, stride):
            for x in range(0, W - ts + 1, stride):
                tile = image_tensor[:, y:y + ts, x:x + ts]
                tiles.append(tile)
                coords.append((y, x, y + ts, x + ts))

        # Right edge
        if (W - ts) % stride != 0:
            x = W - ts
            for y in range(0, H - ts + 1, stride):
                tiles.append(image_tensor[:, y:y + ts, x:x + ts])
                coords.append((y, x, y + ts, x + ts))

        # Bottom edge
        if (H - ts) % stride != 0:
            y = H - ts
            for x in range(0, W - ts + 1, stride):
                tiles.append(image_tensor[:, y:y + ts, x:x + ts])
                coords.append((y, x, y + ts, x + ts))

        # Bottom-right corner
        if (H - ts) % stride != 0 and (W - ts) % stride != 0:
            y, x = H - ts, W - ts
            tiles.append(image_tensor[:, y:y + ts, x:x + ts])
            coords.append((y, x, y + ts, x + ts))

        if not tiles:
            raise ValueError(f"Image too small: {H}x{W} < tile_size {ts}")

        # Process in batches
        num_classes = self.num_classes
        heatmap_accum = np.zeros((num_classes, H, W), dtype=np.float32)
        count_map = np.zeros((1, H, W), dtype=np.float32)

        for i in range(0, len(tiles), self.batch_size):
            batch = torch.stack(tiles[i:i + self.batch_size]).to(self.device)
            batch_coords = coords[i:i + self.batch_size]

            with torch.no_grad():
                raw_maps = self.model.get_spatial_heatmap(batch)
                prob_maps = torch.softmax(raw_maps, dim=1).cpu().numpy()

            for j, (y1, x1, y2, x2) in enumerate(batch_coords):
                region_h, region_w = y2 - y1, x2 - x1

                tile_map = prob_maps[j]  # (C, h, w)
                h_map, w_map = tile_map.shape[1], tile_map.shape[2]
                if h_map != region_h or w_map != region_w:
                    resized = np.zeros(
                        (num_classes, region_h, region_w), dtype=np.float32
                    )
                    for c in range(num_classes):
                        resized[c] = cv2.resize(
                            tile_map[c], (region_w, region_h),
                            interpolation=cv2.INTER_LINEAR
                        )
                    tile_map = resized
                heatmap_accum[:, y1:y2, x1:x2] += tile_map

                count_map[0, y1:y2, x1:x2] += 1

        heatmap = np.divide(
            heatmap_accum, count_map,
            out=np.zeros_like(heatmap_accum),
            where=count_map > 0
        )

        return InferenceResult(heatmap=heatmap)

    def _select_tile_sizes(self, h: int, w: int) -> List[int]:
        """Select tile sizes based on image dimensions (adaptive mode)."""
        if not self.adaptive_scale:
            print(f"[Engine] Tile scales (fixed): {self.tile_sizes} for {w}x{h} image")
            return self.tile_sizes

        min_h, min_w = self.large_image_min_size
        if h >= min_h and w >= min_w:
            selected = self.large_image_tiles
            print(f"[Engine] Tile scales (large image): {selected} for {w}x{h} image")
        else:
            selected = self.small_image_tiles
            print(f"[Engine] Tile scales (small image): {selected} for {w}x{h} image")
        return selected

    def _preprocess_tile(self, tile_crop: np.ndarray) -> torch.Tensor:
        """
        Preprocess a raw tile crop for model input.

        Resizes to model_input_size and applies ImageNet normalization.

        Args:
            tile_crop: (H, W, 3) RGB uint8 numpy array

        Returns:
            (3, model_input_size, model_input_size) normalized tensor
        """
        h, w = tile_crop.shape[:2]
        if h != self.model_input_size or w != self.model_input_size:
            tile_resized = cv2.resize(
                tile_crop, (self.model_input_size, self.model_input_size)
            )
        else:
            tile_resized = tile_crop

        transformed = self.val_transform(image=np.ascontiguousarray(tile_resized))
        img = transformed['image']
        if isinstance(img, torch.Tensor):
            return img
        return torch.from_numpy(img).permute(2, 0, 1).float()

    def _get_feather_window(self, h: int, w: int) -> np.ndarray:
        """
        Build a 2D feathering window for a tile of size (h, w).

        Produces a smooth spatial weight mask: center = 1.0, edges taper to ~0
        using a raised-cosine (Hann) profile over `feather_ratio` of the tile.

        The mask is cached per (h, w) so repeated tile sizes do not re-compute.

        feather_ratio=0.0 → uniform weight 1.0 (no feathering, original behavior).
        feather_ratio=0.15 → 15% of tile width/height tapers from 1 → 0 at each edge.

        Args:
            h: tile height in pixels (after resize to image-space region)
            w: tile width in pixels

        Returns:
            (h, w) float32 array with values in [0, 1]
        """
        if self.feather_ratio <= 0.0:
            return np.ones((h, w), dtype=np.float32)

        key = (h, w)
        if key in self._feather_cache:
            return self._feather_cache[key]

        # Build 1-D raised cosine ramps
        def _hann_ramp(n: int, ramp_px: int) -> np.ndarray:
            """n-length window with cosine ramps at both ends of length ramp_px."""
            win = np.ones(n, dtype=np.float32)
            if ramp_px > 0:
                t = np.linspace(0.0, np.pi / 2.0, ramp_px, endpoint=False)
                ramp = np.sin(t) ** 2  # 0 → 1 over ramp_px samples
                win[:ramp_px] = ramp
                win[n - ramp_px:] = ramp[::-1]
            return win

        ramp_h = max(1, int(h * self.feather_ratio))
        ramp_w = max(1, int(w * self.feather_ratio))
        win_h = _hann_ramp(h, ramp_h)  # (h,)
        win_w = _hann_ramp(w, ramp_w)  # (w,)
        window = np.outer(win_h, win_w).astype(np.float32)  # (h, w)

        self._feather_cache[key] = window
        return window

    def _process_batch(
        self,
        batch_tiles: List[torch.Tensor],
        batch_coords: List[Tuple[int, int, int, int, int]],
        heatmap_accum: np.ndarray,
        count_map: np.ndarray,
        all_tiles_info: List[Dict],
    ):
        """
        Process a batch of tiles: run model and accumulate spatial heatmap.

        Uses get_spatial_heatmap() for per-pixel heatmap accumulation.
        Also calls predict_instances() for per-tile metadata (tiles_info).

        Fusion strategy: large-tile-priority weighting + tile-edge feathering.
        - Scale weight: larger tile = more weight (w = ts / max_ts).
        - Feathering: per-pixel spatial weight (raised cosine, center=1, edges→0)
          eliminates sharp count_map boundaries that cause visible grid artifacts.
        Small tiles are ~4x more numerous; scale weight prevents their signal
        from overwhelming fewer large-tile observations.

        Args:
            batch_tiles: List of preprocessed tile tensors
            batch_coords: List of (y1, x1, y2, x2, tile_size) tuples
            heatmap_accum: (num_classes, H, W) accumulator (weighted)
            count_map: (num_classes, H, W) per-channel weight accumulator.
                Class 0 uses scale_weight; disease channels use disease_weight
                (bg-penalized). Each channel divided by its own count for a
                correct per-channel weighted average.
            all_tiles_info: List to append tile detection info
        """
        input_tensor = torch.stack(batch_tiles).to(self.device)

        with torch.no_grad():
            # Scalar logits for per-tile metadata
            logits = self.model.predict_instances(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # (N, C)

            # Spatial heatmaps for pixel-level accumulation
            raw_maps = self.model.get_spatial_heatmap(input_tensor)
            spatial_maps = torch.softmax(raw_maps, dim=1).cpu().numpy()

        # Reference tile size for weighting (largest across ALL scales = weight 1.0)
        max_ts = getattr(self, '_max_tile_size', 1536)

        for i, prob in enumerate(probs):
            y1, x1, y2, x2, ts = batch_coords[i]
            h_orig = heatmap_accum.shape[1]
            w_orig = heatmap_accum.shape[2]
            y2 = min(y2, h_orig)
            x2 = min(x2, w_orig)
            h_slice = y2 - y1
            w_slice = x2 - x1

            if h_slice <= 0 or w_slice <= 0:
                continue

            # Record tile info (max disease class, excluding class 0)
            disease_probs = prob[1:]
            max_disease_idx = int(np.argmax(disease_probs))
            max_disease_conf = float(disease_probs[max_disease_idx])
            pred_class_id = max_disease_idx + 1

            if max_disease_conf > self.conf_threshold:
                all_tiles_info.append({
                    'score': max_disease_conf,
                    'class_id': pred_class_id,
                    'coords': (y1, y2, x1, x2),
                    'tile_size': ts,
                    'bg_prob': float(prob[0]),
                    'class_probs': prob.tolist(),
                })

            # Scale-priority weight: larger tile = more weight
            # w = ts / max_ts → e.g., 1536/1536 = 1.0, 768/1536 = 0.5
            # Rationale: small tiles are ~4x more numerous than large tiles
            # (due to smaller stride), so they dominate without downweighting.
            scale_weight = float(ts) / float(max_ts)

            # BG-aware weight: penalize DISEASE channels for high-bg tiles.
            # tile_bg_weight = max(0, 1 - alpha * bg_prob)
            # With alpha=2.0: bg_prob=0.0→1.0, bg_prob=0.3→0.4, bg_prob>=0.5→0.0
            #
            # CRITICAL: Class 0 (background) always uses scale_weight, NOT
            # total_weight. Reason: bg-aware suppression is meant to reduce
            # disease signal from background tiles, but those same tiles carry
            # the correct background probability for Class 0. If we also suppress
            # Class 0 accumulation, heatmap[0] gets underestimated in bg-heavy
            # regions, causing disease_map (=1-heatmap[0]) to spike falsely red.
            # Solution: split count_map into per-channel accumulators so
            # Class 0 and disease channels are normalized independently.
            if self.bg_aware_alpha > 0.0:
                bg_prob = float(prob[0])
                bg_weight = max(0.0, 1.0 - self.bg_aware_alpha * bg_prob)
            else:
                bg_weight = 1.0
            disease_weight = scale_weight * bg_weight  # for channels 1..C-1
            # Class 0 always accumulates with scale_weight (unpenalized)
            bg_channel_weight = scale_weight

            # Accumulate spatial heatmap (resize from model output to tile region)
            tile_spatial = spatial_maps[i]  # (C, h, w)
            num_classes = tile_spatial.shape[0]
            resized = np.zeros(
                (num_classes, h_slice, w_slice), dtype=np.float32
            )
            for c in range(num_classes):
                resized[c] = cv2.resize(
                    tile_spatial[c], (w_slice, h_slice),
                    interpolation=cv2.INTER_LINEAR
                )

            # Feathering: per-pixel spatial weight (raised cosine, center=1,
            # edges→0). Eliminates sharp count_map step boundaries that produce
            # visible tile-grid artifacts in the fused heatmap.
            # shape: (h_slice, w_slice), values in [0, 1]
            feather = self._get_feather_window(h_slice, w_slice)

            # Class 0: unpenalized accumulation
            w_bg = bg_channel_weight * feather  # (h_slice, w_slice)
            heatmap_accum[0, y1:y2, x1:x2] += resized[0] * w_bg
            count_map[0, y1:y2, x1:x2] += w_bg
            # Disease channels: bg-penalized accumulation
            if num_classes > 1:
                w_dis = disease_weight * feather  # (h_slice, w_slice)
                heatmap_accum[1:num_classes, y1:y2, x1:x2] += resized[1:] * w_dis
                count_map[1:num_classes, y1:y2, x1:x2] += w_dis

            # Scale diagnostics: uniform-weight side accumulation (no feather/bg-aware)
            # so each scale's heatmap reflects raw model confidence at that receptive field.
            if self._diag_accum is not None:
                self._diag_accum[:, y1:y2, x1:x2] += resized
                self._diag_count[:, y1:y2, x1:x2] += 1.0


def extract_detections(
    heatmap: np.ndarray,
    image_rgb: np.ndarray,
    top_k: int = 5,
    conf_threshold: float = 0.05,
    class_names: Optional[Dict[int, str]] = None,
    min_area: int = 100,
    percentile: float = 85.0,
    return_raw: bool = False,
    bg_region_max: float = 0.7,
) -> "List[Dict] | Tuple[List[Dict], List[Dict]]":
    """
    Extract disease detections from multi-class heatmap.

    Strategy: Per-class peak detection + spatial merging.
    Instead of thresholding the aggregate disease_map (which fails when
    model is uncertain everywhere), we find peaks in each individual class
    heatmap and merge nearby same-class detections.

    Steps:
        1. For each disease class (1..C-1): threshold at class-specific
           percentile to find "hot" regions
        2. Morphological close+open to clean up
        3. CCA to find connected regions
        4. Merge overlapping detections across classes (IoU > 0.3)
        5. Sort by weighted score (area * confidence)

    Args:
        heatmap: (num_classes, H, W) probability heatmap in [0, 1]
        image_rgb: (H, W, 3) original RGB image
        top_k: Maximum number of detections to return
        conf_threshold: Minimum mean confidence within region
        class_names: Optional class ID → name mapping
        min_area: Minimum component area in pixels to keep
        percentile: Percentile threshold (GUI slider 70-95).
            Controls sensitivity: lower = more detections, higher = fewer.
        return_raw: If True, also return raw detections before NMS as second element.
        bg_region_max: Max allowed mean class-0 probability within a detection region.
            Regions with heatmap[0].mean() > bg_region_max are discarded as
            background-dominated even if a disease channel passed the percentile
            threshold. Set to 1.0 to disable. Default: 0.7.

    Returns:
        If return_raw=False (default): List of detection dicts (after NMS + top_k).
        If return_raw=True: (final_detections, raw_detections_before_nms).
        Detection dict keys: score, class_id, class_name, coords (y1, y2, x1, x2),
            area, tile_rgb, top2_class, confidence_gap
    """
    num_classes = heatmap.shape[0]
    h_orig, w_orig = image_rgb.shape[:2]
    total_pixels = h_orig * w_orig
    class_names = class_names or {}

    if num_classes <= 1:
        return []

    # Disease map for diagnostics
    disease_map = 1.0 - heatmap[0]  # (H, W)
    dm_mean = float(disease_map.mean())
    dm_std = float(disease_map.std())
    print(f"[Detection] disease_map: mean={dm_mean:.4f} std={dm_std:.4f} max={disease_map.max():.4f}")

    # Morph kernel sizes based on image resolution
    short_edge = min(h_orig, w_orig)
    blur_k = max(3, int(short_edge * 0.008))
    if blur_k % 2 == 0:
        blur_k += 1
    close_k = max(7, int(short_edge * 0.03))
    if close_k % 2 == 0:
        close_k += 1
    open_k = max(3, close_k // 3)
    if open_k % 2 == 0:
        open_k += 1

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))

    # Dynamic min_area: at least 0.1% of total image
    dynamic_min_area = max(min_area, int(total_pixels * 0.001))

    # Per-class detection
    all_detections = []

    for class_id in range(1, num_classes):
        class_map = heatmap[class_id]  # (H, W)
        cm_max = float(class_map.max())

        # Skip classes with negligible presence
        if cm_max < 0.05:
            continue

        # Smooth to reduce grid artifacts
        class_smooth = cv2.GaussianBlur(
            class_map.astype(np.float32), (blur_k, blur_k), 0
        )

        # Per-class adaptive threshold:
        # Use percentile of non-zero values within this class
        nonzero = class_smooth[class_smooth > 0.02]
        if len(nonzero) < 50:
            continue

        thresh = float(np.percentile(nonzero, percentile))
        # Also enforce absolute minimum based on class max
        thresh = max(thresh, cm_max * 0.3)

        binary = np.uint8(class_smooth > thresh) * 255

        # Morph: close gaps, remove noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

        # CCA
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        for lid in range(1, n_labels):
            area = int(stats[lid, cv2.CC_STAT_AREA])
            if area < dynamic_min_area:
                continue

            x1 = int(stats[lid, cv2.CC_STAT_LEFT])
            y1 = int(stats[lid, cv2.CC_STAT_TOP])
            bw = int(stats[lid, cv2.CC_STAT_WIDTH])
            bh = int(stats[lid, cv2.CC_STAT_HEIGHT])
            x2, y2 = x1 + bw, y1 + bh

            comp_mask = (labels == lid)
            region_conf = float(class_map[comp_mask].mean())

            if region_conf < conf_threshold:
                continue

            # Also compute disease_map confidence for this region
            disease_conf = float(disease_map[comp_mask].mean())

            all_detections.append({
                'class_id': class_id,
                'class_score': region_conf,
                'disease_conf': disease_conf,
                'coords': (y1, y2, x1, x2),
                'area': area,
                'comp_mask': comp_mask,
            })

    print(f"[Detection] Per-class scan: {len(all_detections)} raw detections "
          f"(morph close={close_k} open={open_k}, min_area={dynamic_min_area})")

    # Build raw detection list (for visualization, before NMS)
    raw_detections_for_vis = []
    if return_raw:
        for det in sorted(all_detections, key=lambda d: d['class_score'], reverse=True):
            y1, y2, x1, x2 = det['coords']
            cid = det['class_id']
            raw_detections_for_vis.append({
                'score': det['class_score'],
                'class_id': cid,
                'class_name': class_names.get(cid, f"Class {cid}"),
                'class_score': det['class_score'],
                'coords': (y1, y2, x1, x2),
                'area': det['area'],
                'tile_rgb': image_rgb[y1:y2, x1:x2].copy(),
                'top2_class': None,
                'confidence_gap': det['class_score'],
            })

    # Merge overlapping detections (keep the one with higher class_score)
    # Sort by class_score descending for greedy NMS
    all_detections.sort(key=lambda d: d['class_score'], reverse=True)

    kept = []
    suppressed = set()
    for i, det_i in enumerate(all_detections):
        if i in suppressed:
            continue
        kept.append(det_i)
        yi1, yi2, xi1, xi2 = det_i['coords']
        area_i = float((yi2 - yi1) * (xi2 - xi1))

        for j in range(i + 1, len(all_detections)):
            if j in suppressed:
                continue
            yj1, yj2, xj1, xj2 = all_detections[j]['coords']

            # IoU computation
            inter_y1 = max(yi1, yj1)
            inter_y2 = min(yi2, yj2)
            inter_x1 = max(xi1, xj1)
            inter_x2 = min(xi2, xj2)
            inter_area = max(0, inter_y2 - inter_y1) * max(0, inter_x2 - inter_x1)

            area_j = float((yj2 - yj1) * (xj2 - xj1))
            union = area_i + area_j - inter_area
            iou = inter_area / union if union > 0 else 0

            if iou > 0.3:
                suppressed.add(j)

    print(f"[Detection] After NMS: {len(kept)} detections")

    # BG-region filter: discard detections where the region is still dominated
    # by background (class 0), even after the BG-aware accumulation upstream.
    # This catches cases where multiple low-weight bg tiles still accumulated
    # enough signal to form a CCA component.
    if bg_region_max < 1.0:
        bg_channel = heatmap[0]  # (H, W) class-0 probability map
        pre_filter = len(kept)
        kept = [
            det for det in kept
            if float(bg_channel[det['comp_mask']].mean()) <= bg_region_max
        ]
        n_filtered = pre_filter - len(kept)
        if n_filtered > 0:
            print(f"[Detection] BG-region filter removed {n_filtered} background-dominated boxes "
                  f"(bg_region_max={bg_region_max:.2f})")

    # Build final detection dicts
    detections = []
    for det in kept:
        y1, y2, x1, x2 = det['coords']
        class_id = det['class_id']
        comp_mask = det['comp_mask']

        # Compute per-class means for top-2 analysis
        class_means = []
        for c in range(1, num_classes):
            class_means.append(float(heatmap[c][comp_mask].mean()))

        sorted_indices = np.argsort(class_means)[::-1]
        best_class_id = int(sorted_indices[0]) + 1
        best_class_score = class_means[sorted_indices[0]]

        # Top-2 class + gap
        top2_class = None
        confidence_gap = best_class_score
        if len(sorted_indices) > 1:
            second_class_id = int(sorted_indices[1]) + 1
            second_class_score = class_means[sorted_indices[1]]
            confidence_gap = best_class_score - second_class_score
            top2_class = {
                'class_id': second_class_id,
                'class_name': class_names.get(second_class_id, f"Class {second_class_id}"),
                'score': second_class_score,
            }

        tile_rgb = image_rgb[y1:y2, x1:x2].copy()

        # Weighted score: area × confidence (favors large, confident regions)
        weighted_score = det['area'] * best_class_score

        detections.append({
            'score': det['disease_conf'],
            'class_id': best_class_id,
            'class_name': class_names.get(best_class_id, f"Class {best_class_id}"),
            'class_score': best_class_score,
            'coords': (y1, y2, x1, x2),
            'area': det['area'],
            'tile_rgb': tile_rgb,
            'top2_class': top2_class,
            'confidence_gap': confidence_gap,
            '_weighted_score': weighted_score,
        })

    # Sort by weighted score (area × confidence)
    detections.sort(key=lambda d: d['_weighted_score'], reverse=True)
    result = detections[:top_k]

    # Clean up internal keys
    for d in result:
        d.pop('_weighted_score', None)

    if result:
        for i, d in enumerate(result):
            y1, y2, x1, x2 = d['coords']
            area_pct = d['area'] / total_pixels * 100
            print(f"[Detection] #{i+1} {d['class_name']}: area={d['area']:,} ({area_pct:.2f}%) "
                  f"bbox=({x1},{y1})-({x2},{y2}) conf={d['score']:.3f} "
                  f"class_score={d['class_score']:.3f} gap={d['confidence_gap']:.3f}")
    else:
        print(f"[Detection] No detections (conf_threshold={conf_threshold}, "
              f"min_area={dynamic_min_area})")

    if return_raw:
        return result, raw_detections_for_vis
    return result


def _extract_top_tiles_legacy(
    heatmap: np.ndarray,
    image_rgb: np.ndarray,
    top_k: int = 5,
    conf_threshold: float = 0.3,
    tile_size: int = 384,
    class_names: Optional[Dict[int, str]] = None,
) -> List[Dict]:
    """
    DEPRECATED: Legacy peak detection method. Use extract_detections() instead.

    Problem: kernel_size = tile_size // 2 = 192px causes peak drift to background.
    Kept for backward compatibility only.
    """
    num_classes = heatmap.shape[0]
    h_orig, w_orig = image_rgb.shape[:2]
    class_names = class_names or {}

    candidates = []

    for class_id in range(1, num_classes):
        class_heatmap = heatmap[class_id]

        kernel_size = max(tile_size // 2, 3)
        kernel = np.ones((kernel_size, kernel_size))
        dilated = cv2.dilate(class_heatmap, kernel)
        peaks = (class_heatmap == dilated) & (class_heatmap > conf_threshold)

        peak_coords = np.where(peaks)
        for y, x in zip(peak_coords[0], peak_coords[1]):
            score = class_heatmap[y, x]

            y1 = max(0, y - tile_size // 2)
            x1 = max(0, x - tile_size // 2)
            y2 = min(h_orig, y1 + tile_size)
            x2 = min(w_orig, x1 + tile_size)

            tile_rgb = image_rgb[y1:y2, x1:x2].copy()

            candidates.append({
                'score': float(score),
                'class_id': int(class_id),
                'class_name': class_names.get(class_id, f"Class {class_id}"),
                'coords': (y1, y2, x1, x2),
                'center': (y, x),
                'tile_rgb': tile_rgb,
            })

    candidates.sort(key=lambda c: c['score'], reverse=True)

    selected = []
    for cand in candidates:
        is_duplicate = False
        cy, cx = cand['center']
        for sel in selected:
            sy, sx = sel['center']
            dist = np.sqrt((cy - sy) ** 2 + (cx - sx) ** 2)
            if dist < tile_size * 0.5:
                is_duplicate = True
                break
        if not is_duplicate:
            selected.append(cand)
            if len(selected) >= top_k:
                break

    return selected
