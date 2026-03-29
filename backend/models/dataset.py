"""
Dataset state model.

Holds all computed data for a single dataset in memory: raw images,
segmentation masks, crops, embeddings, and the FAISS index.
This is the central data structure that all Flask routes interact with.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import faiss
import io
import base64
from PIL import Image


@dataclass
class DatasetState:
    """In-memory state for one dataset. Lives for the duration of the session."""

    dataset_id: str = ""
    channel_names: list = field(default_factory=lambda: ["DAPI", "Ch2", "Ch3"])

    # Raw image data — list of (H, W, 3) float32 arrays, all held in memory
    images: list = field(default_factory=list)

    # Segmentation results — one mask per image
    cell_masks: list = field(default_factory=list)          # (H, W) int32 label masks

    # Per-object data — flat lists across all images
    objects: list = field(default_factory=list)              # List of metadata dicts
    crops: list = field(default_factory=list)                # (crop_H, crop_W, 3) per object
    patch_grids: list = field(default_factory=list)          # (37, 37, D) per object
    global_embeddings: Optional[np.ndarray] = None           # (N, D) float32
    thumbnails: list = field(default_factory=list)           # base64 PNG strings

    # FAISS index
    faiss_index: Optional[faiss.Index] = None

    # Processing state
    processing: bool = False
    progress: float = 0.0
    progress_phase: str = ""
    progress_message: str = ""
    error: Optional[str] = None

    def num_images(self) -> int:
        return len(self.images)

    def num_objects(self) -> int:
        return len(self.objects)

    def image_dimensions(self) -> tuple:
        if self.images:
            return self.images[0].shape[:2]
        return (0, 0)

    def get_object_by_id(self, object_id: int) -> Optional[dict]:
        """Look up an object by its ID. Returns None if not found."""
        for obj in self.objects:
            if obj["object_id"] == object_id:
                return obj
        return None

    def get_object_index(self, object_id: int) -> Optional[int]:
        """Get the flat index of an object (for indexing into crops/embeddings arrays)."""
        for i, obj in enumerate(self.objects):
            if obj["object_id"] == object_id:
                return i
        return None

    def crop_to_thumbnail_base64(self, crop: np.ndarray, size: int = 256, max_crop_size: int = None) -> str:
        """Convert a float32 crop to a base64-encoded PNG thumbnail string."""
        from services.image_io import make_thumbnail
        thumb = make_thumbnail(crop, size=size, max_crop_size=max_crop_size)
        pil_img = Image.fromarray(thumb)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"


# Global dataset state — single-user, single-dataset for Phase 1
# In Phase 2+, this would be keyed by dataset_id for multi-user support
_current_dataset: Optional[DatasetState] = None


def get_dataset() -> Optional[DatasetState]:
    return _current_dataset


def create_dataset(dataset_id: str) -> DatasetState:
    global _current_dataset
    _current_dataset = DatasetState(dataset_id=dataset_id)
    return _current_dataset


def clear_dataset():
    global _current_dataset
    _current_dataset = None
