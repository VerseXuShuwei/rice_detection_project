"""
Unit tests for Positive Tile Pool.

Tests:
    - tile_processing extensions (pad_narrow_image, is_valid_tile)
    - spatial_nms functions (bbox_iou, spatial_nms)
    - TileInfo dataclass
    - PositiveTilePool basic functionality

Usage:
    python -m pytest tests/test_positive_pool.py -v
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

# Import modules to test
from src.data.tile_processing import (
    pad_narrow_image,
    is_valid_tile,
    generate_multiscale_tiles_with_coords
)
from src.data.spatial_nms import (
    bbox_iou,
    spatial_nms,
    center_distance_nms
)
from src.data.positive_pool import TileInfo


class TestPadNarrowImage:
    """Tests for pad_narrow_image function."""

    def test_no_padding_needed(self):
        """Image already large enough should not be padded."""
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        padded, padding = pad_narrow_image(image, min_short_edge=512)

        assert padded.shape == (600, 800, 3)
        assert padding == (0, 0, 0, 0)

    def test_vertical_padding(self):
        """Narrow horizontal image should be padded vertically."""
        image = np.zeros((300, 1000, 3), dtype=np.uint8)
        padded, padding = pad_narrow_image(image, min_short_edge=512)

        assert padded.shape[0] == 512  # Height padded to 512
        assert padded.shape[1] == 1000  # Width unchanged
        assert padding[0] + padding[1] == 212  # Total vertical padding

    def test_horizontal_padding(self):
        """Narrow vertical image should be padded horizontally."""
        image = np.zeros((1000, 300, 3), dtype=np.uint8)
        padded, padding = pad_narrow_image(image, min_short_edge=512)

        assert padded.shape[0] == 1000  # Height unchanged
        assert padded.shape[1] == 512  # Width padded to 512
        assert padding[2] + padding[3] == 212  # Total horizontal padding

    def test_symmetric_padding(self):
        """Padding should be roughly symmetric (center padding)."""
        image = np.zeros((300, 1000, 3), dtype=np.uint8)
        padded, padding = pad_narrow_image(image, min_short_edge=512)

        pad_top, pad_bottom, _, _ = padding
        assert abs(pad_top - pad_bottom) <= 1  # Symmetric within 1 pixel


class TestIsValidTile:
    """Tests for is_valid_tile function."""

    def test_pure_black_tile(self):
        """Pure black tile should be invalid."""
        tile = np.zeros((384, 384, 3), dtype=np.uint8)
        assert is_valid_tile(tile) is False

    def test_pure_white_tile(self):
        """Pure white (low saturation) tile should be invalid."""
        tile = np.full((384, 384, 3), 250, dtype=np.uint8)
        assert is_valid_tile(tile) is False

    def test_gray_tile(self):
        """Gray tile with content should be valid."""
        tile = np.full((384, 384, 3), 128, dtype=np.uint8)
        assert is_valid_tile(tile) is True

    def test_colored_tile(self):
        """Colored tile should be valid."""
        tile = np.zeros((384, 384, 3), dtype=np.uint8)
        tile[:, :, 0] = 100  # Red channel
        tile[:, :, 1] = 150  # Green channel
        tile[:, :, 2] = 50   # Blue channel
        assert is_valid_tile(tile) is True

    def test_mixed_tile(self):
        """Tile with some valid and some invalid pixels."""
        tile = np.zeros((384, 384, 3), dtype=np.uint8)
        # Half is gray (valid), half is black (invalid)
        tile[:192, :, :] = 128
        # Should still be valid (50% > 30% threshold)
        assert is_valid_tile(tile, valid_pixel_threshold=0.3) is True

    def test_custom_threshold(self):
        """Custom threshold should work."""
        tile = np.zeros((384, 384, 3), dtype=np.uint8)
        tile[:100, :, :] = 128  # ~26% valid

        assert is_valid_tile(tile, valid_pixel_threshold=0.2) is True
        assert is_valid_tile(tile, valid_pixel_threshold=0.3) is False


class TestBboxIou:
    """Tests for bbox_iou function."""

    def test_identical_boxes(self):
        """Identical boxes should have IoU = 1.0."""
        bbox = (0, 0, 100, 100)
        iou = bbox_iou(bbox, bbox)
        assert abs(iou - 1.0) < 1e-6

    def test_no_overlap(self):
        """Non-overlapping boxes should have IoU = 0.0."""
        bbox1 = (0, 0, 100, 100)
        bbox2 = (200, 200, 300, 300)
        iou = bbox_iou(bbox1, bbox2)
        assert iou == 0.0

    def test_partial_overlap(self):
        """Partially overlapping boxes."""
        bbox1 = (0, 0, 100, 100)
        bbox2 = (50, 50, 150, 150)
        iou = bbox_iou(bbox1, bbox2)
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU: 2500 / 17500 ≈ 0.143
        assert abs(iou - 2500/17500) < 1e-6

    def test_contained_box(self):
        """One box contains another."""
        bbox1 = (0, 0, 100, 100)
        bbox2 = (25, 25, 75, 75)
        iou = bbox_iou(bbox1, bbox2)
        # Intersection: 50x50 = 2500
        # Union: 10000 (larger box area)
        # IoU: 2500 / 10000 = 0.25
        assert abs(iou - 0.25) < 1e-6


class TestSpatialNms:
    """Tests for spatial_nms function."""

    def test_no_overlap(self):
        """Non-overlapping boxes should all be kept."""
        bboxes = [(0, 0, 100, 100), (200, 200, 300, 300), (400, 400, 500, 500)]
        scores = np.array([0.9, 0.7, 0.8])
        selected = spatial_nms(bboxes, scores, iou_threshold=0.5)
        assert len(selected) == 3

    def test_overlapping_boxes(self):
        """Overlapping boxes should suppress lower scores."""
        bboxes = [(0, 0, 100, 100), (50, 50, 150, 150), (200, 200, 300, 300)]
        scores = np.array([0.9, 0.7, 0.8])
        selected = spatial_nms(bboxes, scores, iou_threshold=0.1)
        # First and third should be selected (second suppressed by first)
        assert len(selected) == 2
        assert 0 in selected
        assert 2 in selected

    def test_score_ordering(self):
        """Higher score should be selected first."""
        bboxes = [(0, 0, 100, 100), (50, 50, 150, 150)]
        scores = np.array([0.5, 0.9])  # Second has higher score
        selected = spatial_nms(bboxes, scores, iou_threshold=0.1)
        # Second should be selected (higher score), first suppressed
        assert len(selected) == 1
        assert selected[0] == 1

    def test_max_output(self):
        """max_output should limit number of selections."""
        bboxes = [(i*200, 0, i*200+100, 100) for i in range(10)]
        scores = np.random.rand(10)
        selected = spatial_nms(bboxes, scores, iou_threshold=0.5, max_output=3)
        assert len(selected) == 3

    def test_empty_input(self):
        """Empty input should return empty list."""
        selected = spatial_nms([], np.array([]), iou_threshold=0.5)
        assert selected == []


class TestCenterDistanceNms:
    """Tests for center_distance_nms function."""

    def test_far_centers(self):
        """Far centers should all be kept."""
        centers = [(100, 100), (500, 500), (900, 900)]
        scores = np.array([0.9, 0.7, 0.8])
        selected = center_distance_nms(centers, scores, distance_threshold=100)
        assert len(selected) == 3

    def test_close_centers(self):
        """Close centers should suppress lower scores."""
        centers = [(100, 100), (120, 110), (500, 500)]
        scores = np.array([0.9, 0.7, 0.8])
        selected = center_distance_nms(centers, scores, distance_threshold=50)
        # First and third should be selected (second too close to first)
        assert len(selected) == 2
        assert 0 in selected
        assert 2 in selected


class TestTileInfo:
    """Tests for TileInfo dataclass."""

    def test_to_dict(self):
        """to_dict should return proper dictionary."""
        info = TileInfo(
            tile_id=0,
            source_image="/path/to/image.jpg",
            class_id=1,
            split='train',
            original_scale=1024,
            center_x=500,
            center_y=600,
            bbox=(0, 0, 1024, 1024)
        )
        d = info.to_dict()
        assert d['tile_id'] == 0
        assert d['source_image'] == "/path/to/image.jpg"
        assert d['bbox'] == (0, 0, 1024, 1024)

    def test_from_dict(self):
        """from_dict should create TileInfo from dictionary."""
        d = {
            'tile_id': 1,
            'source_image': "/path/to/image.jpg",
            'class_id': 2,
            'split': 'val',
            'original_scale': 512,
            'center_x': 256,
            'center_y': 256,
            'bbox': [0, 0, 512, 512]  # List should be converted to tuple
        }
        info = TileInfo.from_dict(d)
        assert info.tile_id == 1
        assert info.class_id == 2
        assert info.bbox == (0, 0, 512, 512)
        assert isinstance(info.bbox, tuple)


class TestGenerateMultiscaleTilesWithCoords:
    """Tests for generate_multiscale_tiles_with_coords function."""

    def test_basic_generation(self):
        """Basic multi-scale tile generation."""
        image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        tiles_info = generate_multiscale_tiles_with_coords(
            image,
            scales=[256, 512],
            overlaps=[64, 128]
        )

        assert len(tiles_info) > 0
        for ti in tiles_info:
            assert 'tile' in ti
            assert 'original_scale' in ti
            assert 'bbox' in ti
            assert 'center_x' in ti
            assert 'center_y' in ti
            assert ti['original_scale'] in [256, 512]

    def test_coordinate_validity(self):
        """Coordinates should be within image bounds."""
        image = np.random.randint(0, 255, (1000, 1200, 3), dtype=np.uint8)
        tiles_info = generate_multiscale_tiles_with_coords(
            image,
            scales=[256],
            overlaps=[64]
        )

        for ti in tiles_info:
            x1, y1, x2, y2 = ti['bbox']
            assert 0 <= x1 < 1200
            assert 0 <= y1 < 1000
            assert x1 <= x2 <= 1200
            assert y1 <= y2 <= 1000
            assert 0 <= ti['center_x'] <= 1200
            assert 0 <= ti['center_y'] <= 1000

    def test_with_padding(self):
        """Coordinates should account for padding."""
        image = np.random.randint(0, 255, (600, 1000, 3), dtype=np.uint8)
        padding = (100, 100, 0, 0)  # Top and bottom padding

        # Create padded image
        padded = np.pad(image, ((100, 100), (0, 0), (0, 0)), mode='reflect')

        tiles_info = generate_multiscale_tiles_with_coords(
            padded,
            scales=[256],
            overlaps=[64],
            padding=padding
        )

        # Coordinates should be in original (pre-padding) space
        for ti in tiles_info:
            x1, y1, x2, y2 = ti['bbox']
            assert 0 <= y1 < 600  # Original height
            assert 0 <= y2 <= 600


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
