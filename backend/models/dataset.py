"""
Dataset state model.

Holds all computed data for a single dataset in memory: raw images,
segmentation masks, crops, embeddings, and the FAISS index.
This is the central data structure that all Flask routes interact with.

Projects can be saved to disk and loaded back, skipping the expensive
segmentation and embedding pipeline on subsequent sessions.
"""

from __future__ import annotations

import json
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
from datetime import datetime

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
    source_dir: str = ""  # original image directory path

    # Pipeline config (for reproducibility)
    crop_mode: str = "single_cell"
    size_invariant: bool = True
    rotation_invariant: bool = True

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

    def save_project(self, project_dir: str, classifier_pos=None, classifier_neg=None,
                     search_mode="weighted", threshold=0.0, per_image_cap=50):
        """Save project to disk for reproducibility and fast reload."""
        os.makedirs(project_dir, exist_ok=True)

        # 1. Project manifest
        manifest = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "dataset_id": self.dataset_id,
            "source_dir": self.source_dir,
            "channel_names": self.channel_names,
            "num_images": self.num_images(),
            "num_objects": self.num_objects(),
            "image_dimensions": list(self.image_dimensions()),
            "pipeline": {
                "crop_mode": self.crop_mode,
                "size_invariant": self.size_invariant,
                "rotation_invariant": self.rotation_invariant,
            },
        }
        with open(os.path.join(project_dir, "project.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        # 2. Objects metadata
        with open(os.path.join(project_dir, "objects.json"), "w") as f:
            json.dump(self.objects, f)

        # 3. Segmentation masks
        seg_dir = os.path.join(project_dir, "segmentation")
        os.makedirs(seg_dir, exist_ok=True)
        for i, mask in enumerate(self.cell_masks):
            np.save(os.path.join(seg_dir, f"mask_{i:03d}.npy"), mask)

        # 4. Embeddings
        emb_dir = os.path.join(project_dir, "embeddings")
        os.makedirs(emb_dir, exist_ok=True)
        if self.global_embeddings is not None:
            np.save(os.path.join(emb_dir, "embeddings.npy"), self.global_embeddings)

        # 5. Thumbnails (base64 strings)
        with open(os.path.join(emb_dir, "thumbnails.json"), "w") as f:
            json.dump(self.thumbnails, f)

        # 6. Classifier state (if provided)
        if classifier_pos is not None:
            classifier = {
                "positive_ids": list(classifier_pos),
                "negative_ids": list(classifier_neg or []),
                "search_mode": search_mode,
                "threshold": threshold,
                "per_image_cap": per_image_cap,
            }
            with open(os.path.join(project_dir, "classifier.json"), "w") as f:
                json.dump(classifier, f, indent=2)

            # 7. Auto-generated methods summary
            n_pos = len(classifier["positive_ids"])
            n_neg = len(classifier["negative_ids"])
            neg_text = f" and {n_neg} negative" if n_neg > 0 else ""
            mode_text = "variance-weighted" if search_mode == "weighted" else "cosine"
            inv_parts = []
            if self.size_invariant:
                inv_parts.append("size")
            if self.rotation_invariant:
                inv_parts.append("rotation")
            inv_text = " and ".join(inv_parts) + " invariance" if inv_parts else "no invariance"

            summary = (
                f"Cell selection was performed using Microscopy Search (v1.0). "
                f"Images were segmented using Cellpose-SAM, and DINOv2 embeddings "
                f"were extracted per cell ({self.crop_mode} crop mode, {inv_text}). "
                f"A classifier was defined using {n_pos} positive{neg_text} exemplars. "
                f"{mode_text.capitalize()} similarity search identified cells across "
                f"{self.num_images()} images (similarity threshold > {threshold:.2f}, "
                f"max {per_image_cap} per image)."
            )
            with open(os.path.join(project_dir, "summary.txt"), "w") as f:
                f.write(summary + "\n")

        return project_dir

    @classmethod
    def load_project(cls, project_dir: str) -> "DatasetState":
        """Load a saved project, restoring masks, embeddings, and metadata."""
        with open(os.path.join(project_dir, "project.json")) as f:
            manifest = json.load(f)

        state = cls(
            dataset_id=manifest["dataset_id"],
            channel_names=manifest.get("channel_names", ["DAPI", "Ch2", "Ch3"]),
            source_dir=manifest.get("source_dir", ""),
            crop_mode=manifest.get("pipeline", {}).get("crop_mode", "single_cell"),
            size_invariant=manifest.get("pipeline", {}).get("size_invariant", True),
            rotation_invariant=manifest.get("pipeline", {}).get("rotation_invariant", True),
        )

        # Objects metadata
        objects_path = os.path.join(project_dir, "objects.json")
        if os.path.exists(objects_path):
            with open(objects_path) as f:
                state.objects = json.load(f)

        # Segmentation masks
        seg_dir = os.path.join(project_dir, "segmentation")
        if os.path.isdir(seg_dir):
            mask_files = sorted(f for f in os.listdir(seg_dir) if f.endswith(".npy"))
            state.cell_masks = [np.load(os.path.join(seg_dir, f)) for f in mask_files]

        # Embeddings
        emb_dir = os.path.join(project_dir, "embeddings")
        emb_path = os.path.join(emb_dir, "embeddings.npy")
        if os.path.exists(emb_path):
            state.global_embeddings = np.load(emb_path)

        # Thumbnails
        thumb_path = os.path.join(emb_dir, "thumbnails.json")
        if os.path.exists(thumb_path):
            with open(thumb_path) as f:
                state.thumbnails = json.load(f)

        # Rebuild FAISS index from embeddings
        if state.global_embeddings is not None:
            from services.indexing import build_index
            state.faiss_index = build_index(state.global_embeddings)

        # Load raw images from source directory (needed for image viewer)
        if state.source_dir and os.path.isdir(state.source_dir):
            from services.image_io import load_bbbc021_first_n
            images, _ = load_bbbc021_first_n(state.source_dir, manifest.get("num_images", 5))
            state.images = images

        return state

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
