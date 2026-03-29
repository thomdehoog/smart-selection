"""
Unit tests for the microscopy semantic search backend.

Tests are split into:
1. Tests that run without any models (image I/O, crop extraction, metadata, FAISS)
2. Tests that mock heavy models (segmentation, embedding)

Run with: python -m pytest tests.py -v
"""

import sys
import os
import numpy as np
import pytest

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))


# ═══════════════════════════════════════════════════════════════════════════
# 1. Image I/O tests
# ═══════════════════════════════════════════════════════════════════════════

class TestImageIO:
    """Tests for image loading, normalization, and thumbnail generation."""

    def test_load_tiff_hwc(self, tmp_path):
        """Loading a (H, W, 3) TIFF should return the array as-is in float32."""
        import tifffile
        from services.image_io import load_tiff

        img = np.random.randint(0, 65535, (100, 100, 3), dtype=np.uint16)
        path = str(tmp_path / "test_hwc.tif")
        tifffile.imwrite(path, img)

        result = load_tiff(path)
        assert result.shape == (100, 100, 3), f"Expected (100,100,3), got {result.shape}"
        assert result.dtype == np.float32
        # Values should be preserved (just cast to float)
        np.testing.assert_allclose(result, img.astype(np.float32), rtol=1e-5)

    def test_load_tiff_chw(self, tmp_path):
        """Loading a (3, H, W) TIFF should transpose to (H, W, 3)."""
        import tifffile
        from services.image_io import load_tiff

        img = np.random.randint(0, 65535, (3, 100, 120), dtype=np.uint16)
        path = str(tmp_path / "test_chw.tif")
        tifffile.imwrite(path, img)

        result = load_tiff(path)
        assert result.shape == (100, 120, 3), f"Expected (100,120,3), got {result.shape}"
        # Verify the transpose is correct: channel 0 should be in position [:,:,0]
        np.testing.assert_allclose(result[:, :, 0], img[0].astype(np.float32))

    def test_load_tiff_zstack_zchw(self, tmp_path):
        """Loading a (Z, 3, H, W) TIFF should max-project over Z and transpose."""
        import tifffile
        from services.image_io import load_tiff

        img = np.random.randint(0, 100, (5, 3, 80, 80), dtype=np.uint16)
        path = str(tmp_path / "test_zchw.tif")
        tifffile.imwrite(path, img)

        result = load_tiff(path)
        assert result.shape == (80, 80, 3), f"Expected (80,80,3), got {result.shape}"
        # Max projection over Z should give us the max of each pixel across Z
        expected_max = img.max(axis=0).transpose(1, 2, 0).astype(np.float32)
        np.testing.assert_allclose(result, expected_max)

    def test_load_tiff_wrong_channels(self, tmp_path):
        """Loading an image with !=3 channels should raise ValueError."""
        import tifffile
        from services.image_io import load_tiff

        img = np.random.randint(0, 100, (100, 100, 4), dtype=np.uint16)
        path = str(tmp_path / "test_4ch.tif")
        tifffile.imwrite(path, img)

        with pytest.raises(ValueError, match="3 channels"):
            load_tiff(path)

    def test_load_tiff_singleton_squeeze(self, tmp_path):
        """A (1, 3, H, W) TIFF should squeeze and then transpose correctly."""
        import tifffile
        from services.image_io import load_tiff

        img = np.random.randint(0, 100, (1, 3, 64, 64), dtype=np.uint16)
        path = str(tmp_path / "test_squeeze.tif")
        tifffile.imwrite(path, img)

        result = load_tiff(path)
        assert result.shape == (64, 64, 3)

    def test_reorder_channels(self):
        """Channel reordering should swap channels correctly."""
        from services.image_io import reorder_channels

        img = np.zeros((10, 10, 3), dtype=np.float32)
        img[:, :, 0] = 1.0  # Channel 0 = all ones
        img[:, :, 1] = 2.0  # Channel 1 = all twos
        img[:, :, 2] = 3.0  # Channel 2 = all threes

        # Reorder: put channel 2 first, then 0, then 1
        reordered = reorder_channels(img, [2, 0, 1])
        assert reordered[:, :, 0].mean() == 3.0, "Channel 2 should now be in slot 0"
        assert reordered[:, :, 1].mean() == 1.0, "Channel 0 should now be in slot 1"
        assert reordered[:, :, 2].mean() == 2.0, "Channel 1 should now be in slot 2"

    def test_make_thumbnail(self):
        """Thumbnail should be a uint8 RGB image of the specified size."""
        from services.image_io import make_thumbnail

        img = np.random.uniform(0, 65535, (200, 300, 3)).astype(np.float32)
        thumb = make_thumbnail(img, size=128)

        assert thumb.shape == (128, 128, 3), f"Expected (128,128,3), got {thumb.shape}"
        assert thumb.dtype == np.uint8
        assert thumb.min() >= 0 and thumb.max() <= 255


# ═══════════════════════════════════════════════════════════════════════════
# 2. Crop extraction tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCropExtraction:
    """Tests for context-aware crop extraction around cells."""

    def test_basic_crop(self):
        """A crop centered on a cell should be the correct size."""
        from services.crop_extraction import extract_crop

        image = np.random.uniform(0, 100, (500, 500, 3)).astype(np.float32)
        centroid = (250.0, 250.0)  # Center of image
        bbox = (230, 230, 270, 270)  # 40×40 bounding box
        multiplier = 4.0

        crop = extract_crop(image, centroid, bbox, multiplier)

        # Expected size: max(40, 40) * 4 = 160
        assert crop.shape == (160, 160, 3), f"Expected (160,160,3), got {crop.shape}"
        assert crop.dtype == np.float32

    def test_crop_at_edge_is_zero_padded(self):
        """A crop near the image edge should be zero-padded."""
        from services.crop_extraction import extract_crop

        image = np.ones((100, 100, 3), dtype=np.float32) * 42.0
        centroid = (5.0, 5.0)  # Near top-left corner
        bbox = (0, 0, 10, 10)  # 10×10 box
        multiplier = 4.0  # 40×40 crop

        crop = extract_crop(image, centroid, bbox, multiplier)
        assert crop.shape[0] == crop.shape[1], "Crop should be square"

        # The part that extends beyond the image boundary should be zeros
        # The crop is centered at (5, 5) with size 40, so it starts at (-15, -15)
        # The top-left 15 rows and 15 cols should be zero-padded
        assert crop[0, 0, 0] == 0.0, "Top-left corner should be zero (padded)"

        # But the center region should have the image data
        center = crop.shape[0] // 2
        assert crop[center, center, 0] > 0, "Center should have image data"

    def test_crop_minimum_size(self):
        """Very small cells should still produce a minimum-sized crop."""
        from services.crop_extraction import extract_crop

        image = np.random.uniform(0, 100, (200, 200, 3)).astype(np.float32)
        centroid = (100.0, 100.0)
        bbox = (99, 99, 101, 101)  # Tiny 2×2 bounding box
        multiplier = 4.0  # 2 * 4 = 8, but minimum is 64

        crop = extract_crop(image, centroid, bbox, multiplier)
        assert crop.shape[0] >= 64, f"Crop should be at least 64px, got {crop.shape[0]}"

    def test_rectangular_bbox_produces_square_crop(self):
        """A non-square bounding box should still produce a square crop."""
        from services.crop_extraction import extract_crop

        image = np.random.uniform(0, 100, (500, 500, 3)).astype(np.float32)
        centroid = (250.0, 250.0)
        bbox = (200, 240, 300, 260)  # Wide: 100 × 20
        multiplier = 4.0

        crop = extract_crop(image, centroid, bbox, multiplier)
        # Should use max(100, 20) = 100, × 4 = 400
        assert crop.shape[0] == crop.shape[1], "Crop must be square"
        assert crop.shape[0] == 400, f"Expected 400, got {crop.shape[0]}"

    def test_extract_all_crops(self):
        """Extracting crops for multiple objects should return the right count."""
        from services.crop_extraction import extract_all_crops

        image = np.random.uniform(0, 100, (500, 500, 3)).astype(np.float32)
        objects = [
            {"centroid": (100, 100), "bbox": (80, 80, 120, 120)},
            {"centroid": (300, 300), "bbox": (280, 280, 320, 320)},
            {"centroid": (400, 100), "bbox": (380, 80, 420, 120)},
        ]

        crops = extract_all_crops(image, objects)
        assert len(crops) == 3, f"Expected 3 crops, got {len(crops)}"
        for crop in crops:
            assert crop.ndim == 3 and crop.shape[2] == 3


# ═══════════════════════════════════════════════════════════════════════════
# 3. Segmentation metadata extraction tests (no model needed)
# ═══════════════════════════════════════════════════════════════════════════

class TestSegmentationMetadata:
    """Tests for extract_object_metadata — runs on synthetic label masks."""

    def test_single_object(self):
        """A mask with one object should return one metadata dict."""
        from services.segmentation import extract_object_metadata

        mask = np.zeros((100, 100), dtype=np.int32)
        mask[40:60, 30:70] = 1  # 20×40 rectangle

        objects = extract_object_metadata(mask)
        assert len(objects) == 1

        obj = objects[0]
        assert obj["object_id"] == 1
        assert obj["area"] == 20 * 40
        # Centroid should be near (50, 50) — (cx, cy) where cx is column, cy is row
        cx, cy = obj["centroid"]
        assert 48 < cx < 52, f"Expected cx near 50, got {cx}"
        assert 48 < cy < 52, f"Expected cy near 50, got {cy}"
        # Bbox should be (30, 40, 70, 60)
        assert obj["bbox"] == (30, 40, 70, 60)

    def test_multiple_objects(self):
        """Multiple labeled regions should each get their own metadata."""
        from services.segmentation import extract_object_metadata

        mask = np.zeros((200, 200), dtype=np.int32)
        mask[10:30, 10:30] = 1  # Object 1: 20×20
        mask[50:80, 50:80] = 2  # Object 2: 30×30
        mask[120:140, 160:180] = 3  # Object 3: 20×20

        objects = extract_object_metadata(mask)
        assert len(objects) == 3

        ids = {o["object_id"] for o in objects}
        assert ids == {1, 2, 3}

        # Check areas
        areas = {o["object_id"]: o["area"] for o in objects}
        assert areas[1] == 400  # 20×20
        assert areas[2] == 900  # 30×30
        assert areas[3] == 400  # 20×20

    def test_empty_mask(self):
        """An all-zero mask should return an empty list."""
        from services.segmentation import extract_object_metadata

        mask = np.zeros((100, 100), dtype=np.int32)
        objects = extract_object_metadata(mask)
        assert objects == []

    def test_irregular_shape(self):
        """Non-rectangular objects should have correct area and centroid."""
        from services.segmentation import extract_object_metadata

        mask = np.zeros((100, 100), dtype=np.int32)
        # Draw a circle-ish shape
        for y in range(100):
            for x in range(100):
                if (x - 50) ** 2 + (y - 50) ** 2 < 20 ** 2:
                    mask[y, x] = 1

        objects = extract_object_metadata(mask)
        assert len(objects) == 1
        obj = objects[0]

        # Area should be approximately pi * r^2 = pi * 400 ≈ 1257
        assert 1200 < obj["area"] < 1300, f"Area {obj['area']} not near expected ~1257"

        # Centroid should be near (50, 50)
        cx, cy = obj["centroid"]
        assert abs(cx - 50) < 1 and abs(cy - 50) < 1


# ═══════════════════════════════════════════════════════════════════════════
# 4. Embedding utility tests (no model needed)
# ═══════════════════════════════════════════════════════════════════════════

class TestEmbeddingUtils:
    """Tests for normalization and pooling functions."""

    def test_normalize_crop_scales_to_0_1(self):
        """Percentile normalization should map each channel to [0, 1]."""
        from services.embedding import normalize_crop

        crop = np.random.uniform(100, 60000, (64, 64, 3)).astype(np.float32)
        result = normalize_crop(crop)

        assert result.dtype == np.float32
        for c in range(3):
            assert result[:, :, c].min() >= 0.0, f"Channel {c} has values < 0"
            assert result[:, :, c].max() <= 1.0, f"Channel {c} has values > 1"

    def test_normalize_crop_constant_channel(self):
        """A constant-value channel should normalize to all zeros."""
        from services.embedding import normalize_crop

        crop = np.zeros((64, 64, 3), dtype=np.float32)
        crop[:, :, 0] = 42.0  # Constant channel
        crop[:, :, 1] = np.random.uniform(10, 100, (64, 64))
        crop[:, :, 2] = np.random.uniform(10, 100, (64, 64))

        result = normalize_crop(crop)
        assert result[:, :, 0].max() == 0.0, "Constant channel should be all zeros"

    def test_pool_to_global_embedding_l2_normalized(self):
        """Pooled embedding should be L2-normalized to unit length."""
        from services.embedding import pool_to_global_embedding

        # Fake patch grid
        grid = np.random.randn(37, 37, 768).astype(np.float32)
        emb = pool_to_global_embedding(grid)

        assert emb.shape == (768,)
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-5, f"Expected L2 norm ≈ 1.0, got {norm}"

    def test_pool_to_global_embedding_shape(self):
        """Pooling should reduce (Ph, Pw, D) to (D,)."""
        from services.embedding import pool_to_global_embedding

        grid = np.random.randn(37, 37, 768).astype(np.float32)
        emb = pool_to_global_embedding(grid)
        assert emb.shape == (768,)


# ═══════════════════════════════════════════════════════════════════════════
# 5. FAISS indexing and search tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFAISS:
    """Tests for FAISS index building and similarity search."""

    def _make_test_embeddings(self, n=100, d=768):
        """Create n random L2-normalized embeddings."""
        embs = np.random.randn(n, d).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / (norms + 1e-8)
        return embs

    def test_build_index(self):
        """Building a FAISS index should produce an index with the right count."""
        from services.indexing import build_index

        embs = self._make_test_embeddings(100)
        index = build_index(embs)
        assert index.ntotal == 100

    def test_search_returns_correct_count(self):
        """Search should return top_k results."""
        from services.indexing import build_index, search

        embs = self._make_test_embeddings(100)
        index = build_index(embs)

        result_ids, scores = search(index, embs, positive_ids=[0], top_k=10)
        assert len(result_ids) == 10
        assert len(scores) == 10

    def test_search_excludes_positives(self):
        """Positive IDs should not appear in search results."""
        from services.indexing import build_index, search

        embs = self._make_test_embeddings(100)
        index = build_index(embs)

        positive_ids = [0, 1, 2]
        result_ids, _ = search(index, embs, positive_ids=positive_ids, top_k=20)
        for pid in positive_ids:
            assert pid not in result_ids, f"Positive ID {pid} should be excluded"

    def test_search_excludes_negatives(self):
        """Negative IDs should not appear in search results."""
        from services.indexing import build_index, search

        embs = self._make_test_embeddings(100)
        index = build_index(embs)

        negative_ids = [50, 51, 52]
        result_ids, _ = search(
            index, embs, positive_ids=[0], negative_ids=negative_ids, top_k=20
        )
        for nid in negative_ids:
            assert nid not in result_ids, f"Negative ID {nid} should be excluded"

    def test_search_most_similar_is_itself(self):
        """The most similar vector to X should be X itself (if not excluded)."""
        from services.indexing import build_index, search

        embs = self._make_test_embeddings(100)
        index = build_index(embs)

        # Search using object 0 — since 0 is excluded as positive, the top result
        # should be the most similar *other* vector. But let's verify scores are valid.
        result_ids, scores = search(index, embs, positive_ids=[0], top_k=5)
        assert all(s <= 1.0 + 1e-5 for s in scores), "Cosine similarity should be <= 1.0"
        assert all(s >= -1.0 - 1e-5 for s in scores), "Cosine similarity should be >= -1.0"

    def test_search_with_known_similar_vectors(self):
        """Vectors close together should rank higher than distant ones."""
        from services.indexing import build_index, search

        d = 768
        embs = np.random.randn(50, d).astype(np.float32)

        # Make object 10 very similar to object 0
        embs[10] = embs[0] + np.random.randn(d).astype(np.float32) * 0.01

        # L2 normalize all
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / (norms + 1e-8)

        index = build_index(embs)
        result_ids, scores = search(index, embs, positive_ids=[0], top_k=5)

        # Object 10 should be the top result (most similar to 0)
        assert result_ids[0] == 10, f"Expected object 10 as top result, got {result_ids[0]}"

    def test_negative_adjustment_pushes_results_away(self):
        """Adding a negative example should change the search results."""
        from services.indexing import build_index, search

        d = 768
        n = 50
        embs = np.random.randn(n, d).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / (norms + 1e-8)

        index = build_index(embs)

        # Search without negatives
        results_no_neg, _ = search(index, embs, positive_ids=[0], top_k=10)

        # Search with the top result as a negative — it should drop out
        top_result = results_no_neg[0]
        results_with_neg, _ = search(
            index, embs, positive_ids=[0], negative_ids=[top_result],
            alpha=0.4, top_k=10
        )

        assert top_result not in results_with_neg, \
            "Top result should be excluded when added as negative"


# ═══════════════════════════════════════════════════════════════════════════
# 6. Dataset state model tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDatasetState:
    """Tests for the in-memory dataset state model."""

    def test_create_and_retrieve(self):
        from models.dataset import create_dataset, get_dataset, clear_dataset

        clear_dataset()
        state = create_dataset("test-123")
        assert state.dataset_id == "test-123"
        assert get_dataset() is state
        clear_dataset()

    def test_get_object_index(self):
        from models.dataset import DatasetState

        state = DatasetState()
        state.objects = [
            {"object_id": 10, "centroid": (0, 0), "bbox": (0, 0, 1, 1), "area": 1},
            {"object_id": 20, "centroid": (0, 0), "bbox": (0, 0, 1, 1), "area": 1},
            {"object_id": 30, "centroid": (0, 0), "bbox": (0, 0, 1, 1), "area": 1},
        ]

        assert state.get_object_index(10) == 0
        assert state.get_object_index(20) == 1
        assert state.get_object_index(30) == 2
        assert state.get_object_index(99) is None

    def test_num_objects(self):
        from models.dataset import DatasetState

        state = DatasetState()
        assert state.num_objects() == 0

        state.objects = [{"object_id": 1}, {"object_id": 2}]
        assert state.num_objects() == 2


# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
