"""
Pipeline orchestration.

Ties together the full Phase 1 pipeline: segmentation → crop extraction →
embedding → FAISS indexing. Runs as a background job and updates the
DatasetState with progress info that the frontend can poll.
"""

import threading
import traceback
import numpy as np
from models.dataset import DatasetState
from services.segmentation import segment_cells, free_cellpose_model
from services.crop_extraction import extract_all_crops
from services.embedding import embed_all_objects
from services.indexing import build_index


def run_pipeline(state: DatasetState):
    """
    Run the full Phase 1 pipeline on a dataset.

    This function is intended to run in a background thread. It updates
    state.progress, state.progress_phase, and state.progress_message
    as it proceeds, so the frontend can poll for status.

    The pipeline:
    1. Segment each image with Cellpose → cell masks + per-cell metadata
    2. Extract crops around each cell → list of (H, W, 3) arrays in memory
    3. Free Cellpose from GPU → make room for DINOv2
    4. Embed all crops with DINOv2 → patch grids + global embeddings
    5. Build FAISS index → ready for search

    Args:
        state: The DatasetState to populate. Must already have images loaded.
    """
    state.processing = True
    state.error = None

    try:
        total_images = state.num_images()

        # ─── Phase 1: Segmentation ───────────────────────────────────────
        state.progress_phase = "segmentation"
        state.progress_message = "Running cell segmentation..."
        print(f"\n=== Segmentation ({total_images} images) ===")

        all_objects = []    # Flat list of per-cell metadata across all images
        object_id_counter = 0

        for img_idx, image in enumerate(state.images):
            state.progress = (img_idx / total_images) * 0.3  # Segmentation = 0-30%
            state.progress_message = f"Segmenting image {img_idx + 1}/{total_images}..."
            print(f"  Segmenting image {img_idx + 1}/{total_images}...")

            result = segment_cells(image)
            state.cell_masks.append(result["masks"])

            # Assign globally unique object IDs and tag with image index
            for obj in result["objects"]:
                obj["mask_label"] = obj["object_id"]  # preserve original mask label
                object_id_counter += 1
                obj["object_id"] = object_id_counter
                obj["image_index"] = img_idx
                all_objects.append(obj)

            print(f"    Found {result['num_cells']} cells")

        state.objects = all_objects
        print(f"  Total cells across all images: {len(all_objects)}")

        # ─── Phase 2: Crop extraction ────────────────────────────────────
        state.progress_phase = "cropping"
        state.progress_message = "Extracting cell crops..."
        state.progress = 0.3
        print(f"\n=== Crop extraction ===")

        # Group objects by image index for efficient cropping
        for img_idx, image in enumerate(state.images):
            img_objects = [o for o in all_objects if o["image_index"] == img_idx]
            crop_mode = getattr(state, 'crop_mode', 'single_cell')
            img_crops = extract_all_crops(image, img_objects, masks=state.cell_masks[img_idx], crop_mode=crop_mode)
            state.crops.extend(img_crops)

            pct = 0.3 + (img_idx / total_images) * 0.1  # Cropping = 30-40%
            state.progress = pct
            print(f"  Image {img_idx + 1}: extracted {len(img_crops)} crops")

        # Compute max cell extent (tight bounding box of non-zero pixels) for relative sizing
        def _cell_extent(crop):
            mask = np.any(crop > 0, axis=2)
            if not mask.any():
                return max(crop.shape[0], crop.shape[1])
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            h = np.where(rows)[0][-1] - np.where(rows)[0][0] + 1
            w = np.where(cols)[0][-1] - np.where(cols)[0][0] + 1
            return max(h, w)

        max_crop_size = max(_cell_extent(c) for c in state.crops)
        for crop in state.crops:
            thumb_b64 = state.crop_to_thumbnail_base64(crop, max_crop_size=max_crop_size)
            state.thumbnails.append(thumb_b64)

        print(f"  Total crops in memory: {len(state.crops)}")

        # ─── Phase 3: Free Cellpose, load DINOv2 ────────────────────────
        state.progress_phase = "loading_model"
        state.progress_message = "Loading DINOv2 model..."
        state.progress = 0.4
        print(f"\n=== Model swap: Cellpose → DINOv2 ===")

        free_cellpose_model()

        # ─── Phase 4: DINOv2 embedding ───────────────────────────────────
        state.progress_phase = "embedding"
        state.progress_message = "Computing DINOv2 embeddings..."
        state.progress = 0.45
        print(f"\n=== DINOv2 embedding ({len(state.crops)} crops) ===")

        global_embeddings, patch_grids = embed_all_objects(state.crops, batch_size=16)
        state.global_embeddings = global_embeddings
        state.patch_grids = patch_grids
        state.progress = 0.9

        # ─── Phase 5: FAISS index ────────────────────────────────────────
        state.progress_phase = "indexing"
        state.progress_message = "Building search index..."
        print(f"\n=== Building FAISS index ===")

        state.faiss_index = build_index(global_embeddings)
        state.progress = 1.0
        state.progress_phase = "complete"
        state.progress_message = "Ready for search!"
        state.processing = False

        print(f"\n=== Pipeline complete ===")
        print(f"  {state.num_objects()} objects indexed")
        print(f"  Embedding shape: {global_embeddings.shape}")
        print(f"  FAISS index size: {state.faiss_index.ntotal}")

    except Exception as e:
        state.error = str(e)
        state.processing = False
        state.progress_phase = "error"
        state.progress_message = f"Error: {str(e)}"
        print(f"\n!!! Pipeline error: {e}")
        traceback.print_exc()


def run_pipeline_async(state: DatasetState):
    """Launch the pipeline in a background thread so Flask stays responsive."""
    thread = threading.Thread(target=run_pipeline, args=(state,), daemon=True)
    thread.start()
    return thread
