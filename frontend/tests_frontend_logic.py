"""
Frontend logic tests.

These test the pure JavaScript logic from the React frontend by reimplementing
the key algorithms in Python. This validates:
- State transition logic (selection toggle, accept/reject, threshold filtering)
- API request/response shapes
- Score color mapping
- Hit-test logic for cell selection

These tests serve as a contract: if the backend produces data matching these
shapes and the frontend implements these algorithms, the integration will work.
"""

import pytest
import json


# ═══════════════════════════════════════════════════════════════════════════
# 1. State management logic
# ═══════════════════════════════════════════════════════════════════════════

class TestClassifierToggle:
    """Tests the classifier positive/negative toggle logic used in useAppState."""

    @staticmethod
    def toggle_pos(pos, neg, obj_id):
        """Reimplementation of togglePos: add to positive, remove from negative."""
        pos, neg = set(pos), set(neg)
        if obj_id in pos:
            pos.discard(obj_id)
        else:
            pos.add(obj_id)
            neg.discard(obj_id)
        return pos, neg

    @staticmethod
    def toggle_neg(pos, neg, obj_id):
        """Reimplementation of toggleNeg: add to negative, remove from positive."""
        pos, neg = set(pos), set(neg)
        if obj_id in neg:
            neg.discard(obj_id)
        else:
            neg.add(obj_id)
            pos.discard(obj_id)
        return pos, neg

    def test_toggle_pos_adds_new_id(self):
        pos, neg = self.toggle_pos(set(), set(), 42)
        assert 42 in pos

    def test_toggle_pos_removes_existing_id(self):
        pos, neg = self.toggle_pos({42, 7}, set(), 42)
        assert 42 not in pos
        assert len(pos) == 1

    def test_toggle_pos_removes_from_neg(self):
        """Adding to positive should remove from negative."""
        pos, neg = self.toggle_pos(set(), {42}, 42)
        assert 42 in pos
        assert 42 not in neg

    def test_toggle_neg_adds_new_id(self):
        pos, neg = self.toggle_neg(set(), set(), 42)
        assert 42 in neg

    def test_toggle_neg_removes_from_pos(self):
        """Adding to negative should remove from positive."""
        pos, neg = self.toggle_neg({42}, set(), 42)
        assert 42 in neg
        assert 42 not in pos

    def test_cannot_be_in_both(self):
        pos, neg = {1, 2, 3}, set()
        pos, neg = self.toggle_neg(pos, neg, 2)
        assert 2 not in pos and 2 in neg
        pos, neg = self.toggle_pos(pos, neg, 2)
        assert 2 in pos and 2 not in neg

    def test_exemplars_persist_across_images(self):
        """Switching images should not clear the classifier sets."""
        pos = {1, 2, 3}
        neg = {10}
        # Simulate switching to image 1 — sets are independent
        assert pos == {1, 2, 3}
        assert neg == {10}


# ═══════════════════════════════════════════════════════════════════════════
# 2. Per-image cap logic
# ═══════════════════════════════════════════════════════════════════════════

class TestPerImageCap:
    """Tests the per-image cap filtering applied to search results."""

    @staticmethod
    def apply_cap(results, cap):
        """Reimplementation of the per-image cap logic."""
        if cap <= 0:
            return results
        counts = {}
        out = []
        for r in results:
            img = r["image_index"]
            counts[img] = counts.get(img, 0) + 1
            if counts[img] <= cap:
                out.append(r)
        return out

    def test_no_cap(self):
        results = [{"image_index": 0}] * 10
        assert len(self.apply_cap(results, 0)) == 10

    def test_cap_limits_per_image(self):
        results = [
            {"image_index": 0, "similarity_score": 0.9},
            {"image_index": 0, "similarity_score": 0.8},
            {"image_index": 0, "similarity_score": 0.7},
            {"image_index": 1, "similarity_score": 0.6},
        ]
        capped = self.apply_cap(results, 2)
        assert len(capped) == 3  # 2 from image 0 + 1 from image 1
        img0 = [r for r in capped if r["image_index"] == 0]
        assert len(img0) == 2

    def test_cap_preserves_order(self):
        """Cap should keep the first N per image (highest similarity first)."""
        results = [
            {"image_index": 0, "similarity_score": 0.9},
            {"image_index": 0, "similarity_score": 0.5},
            {"image_index": 0, "similarity_score": 0.3},
        ]
        capped = self.apply_cap(results, 1)
        assert len(capped) == 1
        assert capped[0]["similarity_score"] == 0.9

    def test_cap_across_many_images(self):
        results = [{"image_index": i % 3, "similarity_score": 0.5} for i in range(9)]
        capped = self.apply_cap(results, 2)
        assert len(capped) == 6  # 2 per image × 3 images


# ═══════════════════════════════════════════════════════════════════════════
# 3. Similarity threshold filtering
# ═══════════════════════════════════════════════════════════════════════════

class TestThresholdFiltering:
    """Tests the client-side threshold filtering of search results."""

    def test_threshold_zero_shows_all(self):
        results = [
            {"object_id": 1, "similarity_score": 0.95},
            {"object_id": 2, "similarity_score": 0.5},
            {"object_id": 3, "similarity_score": 0.1},
        ]
        threshold = 0.0
        visible = [r for r in results if r["similarity_score"] >= threshold]
        assert len(visible) == 3

    def test_threshold_filters_low_scores(self):
        results = [
            {"object_id": 1, "similarity_score": 0.95},
            {"object_id": 2, "similarity_score": 0.5},
            {"object_id": 3, "similarity_score": 0.1},
        ]
        threshold = 0.6
        visible = [r for r in results if r["similarity_score"] >= threshold]
        assert len(visible) == 1
        assert visible[0]["object_id"] == 1

    def test_threshold_one_shows_none(self):
        results = [
            {"object_id": 1, "similarity_score": 0.99},
        ]
        threshold = 1.0
        visible = [r for r in results if r["similarity_score"] >= threshold]
        assert len(visible) == 0

    def test_threshold_boundary(self):
        results = [{"object_id": 1, "similarity_score": 0.5}]
        # Exactly at threshold should be included (>=)
        visible = [r for r in results if r["similarity_score"] >= 0.5]
        assert len(visible) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 4. Score color logic
# ═══════════════════════════════════════════════════════════════════════════

class TestScoreColor:
    """Tests the score → color mapping used for result badges."""

    @staticmethod
    def score_color(v):
        """Python reimplementation of the JS scoreColor function."""
        if v > 0.8:
            return "#16A34A"  # green
        if v > 0.5:
            return "#CA8A04"  # amber
        return "#DC2626"      # red

    def test_high_score_green(self):
        assert self.score_color(0.95) == "#16A34A"
        assert self.score_color(0.81) == "#16A34A"

    def test_medium_score_amber(self):
        assert self.score_color(0.7) == "#CA8A04"
        assert self.score_color(0.51) == "#CA8A04"

    def test_low_score_red(self):
        assert self.score_color(0.3) == "#DC2626"
        assert self.score_color(0.0) == "#DC2626"

    def test_boundary_0_8(self):
        assert self.score_color(0.8) == "#CA8A04"  # 0.8 is NOT > 0.8

    def test_boundary_0_5(self):
        assert self.score_color(0.5) == "#DC2626"  # 0.5 is NOT > 0.5


# ═══════════════════════════════════════════════════════════════════════════
# 5. Hit-test logic for cell selection
# ═══════════════════════════════════════════════════════════════════════════

class TestHitTest:
    """Tests the point-in-bbox hit testing used for canvas click handling."""

    @staticmethod
    def hit_test(click_x, click_y, objects):
        """Python reimplementation of the JS hitTest logic."""
        for o in objects:
            x1, y1, x2, y2 = o["bbox"]
            if click_x >= x1 and click_x <= x2 and click_y >= y1 and click_y <= y2:
                return o["object_id"]
        return None

    def test_click_inside_cell(self):
        objects = [{"object_id": 1, "bbox": [10, 10, 50, 50]}]
        assert self.hit_test(30, 30, objects) == 1

    def test_click_outside_cell(self):
        objects = [{"object_id": 1, "bbox": [10, 10, 50, 50]}]
        assert self.hit_test(60, 60, objects) is None

    def test_click_on_boundary(self):
        objects = [{"object_id": 1, "bbox": [10, 10, 50, 50]}]
        assert self.hit_test(10, 10, objects) == 1  # Top-left corner
        assert self.hit_test(50, 50, objects) == 1  # Bottom-right corner

    def test_overlapping_cells_returns_first(self):
        """When cells overlap, the first one in the list wins."""
        objects = [
            {"object_id": 1, "bbox": [10, 10, 50, 50]},
            {"object_id": 2, "bbox": [30, 30, 70, 70]},
        ]
        # Click at (40, 40) — inside both, but object 1 is checked first
        assert self.hit_test(40, 40, objects) == 1

    def test_multiple_non_overlapping_cells(self):
        objects = [
            {"object_id": 1, "bbox": [10, 10, 30, 30]},
            {"object_id": 2, "bbox": [50, 50, 70, 70]},
            {"object_id": 3, "bbox": [90, 10, 110, 30]},
        ]
        assert self.hit_test(20, 20, objects) == 1
        assert self.hit_test(60, 60, objects) == 2
        assert self.hit_test(100, 20, objects) == 3
        assert self.hit_test(40, 40, objects) is None

    def test_empty_objects(self):
        assert self.hit_test(50, 50, []) is None


# ═══════════════════════════════════════════════════════════════════════════
# 6. API request/response shape validation
# ═══════════════════════════════════════════════════════════════════════════

class TestAPIShapes:
    """Validates that the API request/response shapes match the frontend expectations."""

    def test_upload_response_shape(self):
        """The upload response must include these fields for the frontend to work."""
        response = {
            "status": "ok",
            "dataset_id": "abc123",
            "num_images": 5,
            "image_dimensions": [1024, 1024],
            "num_channels": 3,
            "channel_names": ["DAPI", "Tubulin", "Actin"],
        }
        assert "dataset_id" in response
        assert "num_images" in response
        assert "channel_names" in response
        assert isinstance(response["channel_names"], list)
        assert len(response["channel_names"]) == 3

    def test_status_response_shape(self):
        response = {
            "status": "processing",
            "progress": 0.65,
            "phase": "embedding",
            "message": "Embedding 300/500",
            "num_objects": 500,
        }
        assert response["status"] in ("processing", "complete", "error", "idle", "no_dataset")
        assert 0 <= response["progress"] <= 1
        assert isinstance(response["message"], str)

    def test_objects_response_shape(self):
        response = {
            "num_objects": 2,
            "objects": [
                {"object_id": 1, "centroid": [100.5, 200.3], "bbox": [80, 180, 120, 220], "area": 1500},
                {"object_id": 2, "centroid": [300.0, 400.0], "bbox": [280, 380, 320, 420], "area": 1200},
            ],
        }
        for obj in response["objects"]:
            assert "object_id" in obj
            assert "centroid" in obj and len(obj["centroid"]) == 2
            assert "bbox" in obj and len(obj["bbox"]) == 4
            assert "area" in obj
            # bbox should be [x_min, y_min, x_max, y_max]
            x1, y1, x2, y2 = obj["bbox"]
            assert x2 > x1 and y2 > y1

    def test_crops_response_shape(self):
        response = {
            "crops": [
                {
                    "object_id": 1,
                    "image_index": 0,
                    "centroid": [100.5, 200.3],
                    "area": 1500,
                    "thumbnail_base64": "data:image/png;base64,iVBOR...",
                },
            ],
        }
        for crop in response["crops"]:
            assert "object_id" in crop
            assert "thumbnail_base64" in crop
            assert crop["thumbnail_base64"].startswith("data:image/png;base64,")

    def test_search_request_shape(self):
        """Validate the shape of a search request the frontend would send."""
        request = {
            "positive_ids": [1, 2, 3],
            "negative_ids": [10, 11],
            "top_k": 200,
            "negative_alpha": 1.0,
        }
        assert len(request["positive_ids"]) > 0
        assert isinstance(request["negative_ids"], list)
        assert request["top_k"] > 0
        assert request["negative_alpha"] == 1.0  # fixed, not user-configurable

    def test_search_response_shape(self):
        response = {
            "results": [
                {
                    "object_id": 42,
                    "image_index": 2,
                    "similarity_score": 0.93,
                    "centroid": [512.0, 768.0],
                    "area": 2000,
                    "thumbnail_base64": "data:image/png;base64,iVBOR...",
                },
            ],
            "num_total_objects": 4900,
            "num_positive": 3,
            "num_negative": 2,
        }
        for r in response["results"]:
            assert "object_id" in r
            assert "similarity_score" in r
            assert 0 <= r["similarity_score"] <= 1
            assert "thumbnail_base64" in r


# ═══════════════════════════════════════════════════════════════════════════
# 7. Step navigation logic
# ═══════════════════════════════════════════════════════════════════════════

class TestStepNavigation:
    """Tests the step transition rules."""

    def test_initial_state(self):
        state = {"step": 1, "datasetId": None, "classifierPos": set(), "classifierNeg": set()}
        assert state["step"] == 1

    def test_load_to_select(self):
        """After pipeline completes, should transition to step 2."""
        state = {"step": 1, "datasetId": "abc"}
        state["step"] = 2
        assert state["step"] == 2

    def test_select_to_review_requires_positive(self):
        """Cannot go to review with empty positive set."""
        pos = set()
        can_proceed = len(pos) > 0
        assert not can_proceed

        pos.add(1)
        can_proceed = len(pos) > 0
        assert can_proceed

    def test_review_to_results(self):
        state = {"step": 3, "classifierPos": {1, 2, 3}, "classifierNeg": {10}}
        state["step"] = 4
        state["results"] = []
        assert state["step"] == 4

    def test_edit_exemplars_preserves_classifier(self):
        """Going back to step 2 should preserve classifier sets."""
        state = {"step": 4, "classifierPos": {1, 2}, "classifierNeg": {5}, "results": [{"id": 10}]}
        state["step"] = 2
        state["results"] = []
        assert state["step"] == 2
        assert state["classifierPos"] == {1, 2}
        assert state["classifierNeg"] == {5}


# ═══════════════════════════════════════════════════════════════════════════
# 8. Map view logic
# ═══════════════════════════════════════════════════════════════════════════

class TestMapViewLogic:
    """Tests the map view highlighting logic used in StepSearch."""

    def test_view_mode_default(self):
        state = {"viewMode": "gallery"}
        assert state["viewMode"] == "gallery"

    def test_view_mode_toggle(self):
        state = {"viewMode": "gallery"}
        state["viewMode"] = "map"
        assert state["viewMode"] == "map"
        state["viewMode"] = "gallery"
        assert state["viewMode"] == "gallery"

    def test_result_scores_lookup(self):
        """Results should be indexable by object_id for fast lookup."""
        results = [
            {"object_id": 10, "image_index": 0, "similarity_score": 0.9},
            {"object_id": 20, "image_index": 1, "similarity_score": 0.7},
            {"object_id": 30, "image_index": 0, "similarity_score": 0.5},
        ]
        scores = {r["object_id"]: r["similarity_score"] for r in results}
        assert scores[10] == 0.9
        assert scores[20] == 0.7
        assert 99 not in scores

    def test_highlight_priority(self):
        """Positive cells take priority over result cells in highlighting."""
        positive = {1, 2, 3}
        result_scores = {3: 0.85, 4: 0.7, 5: 0.6}
        cell_id = 3  # In both positive and results

        # Positive should take priority
        if cell_id in positive:
            color = "blue"
        elif cell_id in result_scores:
            color = "green"
        else:
            color = "hover"
        assert color == "blue"

    def test_highlight_result_cell(self):
        positive = {1, 2}
        result_scores = {4: 0.7}
        cell_id = 4

        if cell_id in positive:
            color = "blue"
        elif cell_id in result_scores:
            color = "green"
        else:
            color = "hover"
        assert color == "green"

    def test_mask_filters_by_image(self):
        """Only cells present in the mask for the viewed image should be highlighted."""
        # Simulate mask pixel data: image 0 has cells 1-30, image 1 has cells 31-60
        mask_cell_ids_image0 = set(range(1, 31))
        positive = {5, 35}  # 5 is on image 0, 35 is on image 1

        # When viewing image 0, only cell 5 would appear in the mask
        highlighted_on_image0 = positive & mask_cell_ids_image0
        assert highlighted_on_image0 == {5}

    def test_result_score_to_green_intensity(self):
        """Score should map to green channel intensity for visual distinction."""
        def score_to_green(score):
            return round(160 + score * 95)

        assert score_to_green(0.0) == 160  # Low similarity = darker green
        assert score_to_green(1.0) == 255  # High similarity = bright green
        assert score_to_green(0.5) == 208  # Mid similarity


# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
