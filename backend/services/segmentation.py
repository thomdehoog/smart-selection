"""
Segmentation service.

Wraps Cellpose-SAM (cpsam) to perform cell segmentation on 3-channel images.

Key cpsam behaviors (Cellpose 4):
- Default model is 'cpsam' (not cyto3 or nuclei — those are Cellpose 3 models)
- Channel-order invariant: just pass the first 3 channels, no channels= param needed
- Size invariant: diameter is optional (trained on 7.5-120px diameter range)
- Returns (masks, flows, styles) for a single image
- The old models.Cellpose class is removed — use models.CellposeModel only

For nuclear-only segmentation (Phase 2):
- Pass only channel 1 (nuclei), with the other 2 channels set to zero
"""

import numpy as np

# Lazy import — cellpose is only needed when actually running segmentation,
# not when importing helper functions like extract_object_metadata
models = None

def _ensure_cellpose():
    global models
    if models is None:
        from cellpose import models as _models
        globals()['models'] = _models


# Module-level model cache — loaded once, reused across calls
_cellpose_model = None


def get_cellpose_model(gpu: bool = True):
    """Load or return cached Cellpose-SAM model."""
    global _cellpose_model
    if _cellpose_model is None:
        _ensure_cellpose()
        print("Loading Cellpose-SAM (cpsam) model...")
        _cellpose_model = models.CellposeModel(gpu=gpu)
        print("Cellpose-SAM model loaded.")
    return _cellpose_model


def segment_cells(image: np.ndarray, gpu: bool = True) -> dict:
    """
    Segment cells in a 3-channel fluorescence image using Cellpose-SAM.

    cpsam is channel-order invariant — it takes the first 3 channels as-is.
    No channels= parameter is needed. We transpose to (3, H, W) and pass
    channel_axis=0 to be explicit about the layout.

    Args:
        image: (H, W, 3) float32 array. All 3 channels are used.
        gpu: Whether to use GPU for segmentation.

    Returns:
        dict with:
            'masks': (H, W) int32 label mask (0=background, 1,2,...=cells)
            'num_cells': number of detected cells
            'objects': list of per-cell metadata dicts
    """
    model = get_cellpose_model(gpu=gpu)

    # Transpose from (H, W, 3) to (3, H, W) for explicit channel_axis=0
    img_input = np.transpose(image, (2, 0, 1))  # (3, H, W)

    # cpsam: no channels= param, no diameter needed (auto-detect)
    masks, flows, styles = model.eval(
        img_input,
        channel_axis=0,
        diameter=None,
    )

    masks = masks.astype(np.int32)
    masks = remove_border_objects(masks)
    objects = extract_object_metadata(masks)

    return {
        "masks": masks,
        "num_cells": len(objects),
        "objects": objects,
    }


def segment_nuclei(image: np.ndarray, gpu: bool = True) -> dict:
    """
    Segment nuclei using only channel 0 (the nuclear stain).

    Phase 2 feature — not used in Phase 1.
    """
    model = get_cellpose_model(gpu=gpu)

    nuclei_ch = image[:, :, 0]
    img_input = np.stack([nuclei_ch, np.zeros_like(nuclei_ch), np.zeros_like(nuclei_ch)])

    masks, flows, styles = model.eval(
        img_input,
        channel_axis=0,
        diameter=None,
    )

    masks = masks.astype(np.int32)
    objects = extract_object_metadata(masks)

    return {
        "masks": masks,
        "num_cells": len(objects),
        "objects": objects,
    }


def remove_border_objects(masks: np.ndarray) -> np.ndarray:
    """Remove objects that touch any edge of the image."""
    border_ids = set()
    border_ids.update(np.unique(masks[0, :]))    # top
    border_ids.update(np.unique(masks[-1, :]))   # bottom
    border_ids.update(np.unique(masks[:, 0]))    # left
    border_ids.update(np.unique(masks[:, -1]))   # right
    border_ids.discard(0)

    if border_ids:
        masks = masks.copy()
        for obj_id in border_ids:
            masks[masks == obj_id] = 0

    return masks


def extract_object_metadata(masks: np.ndarray) -> list:
    """
    Extract centroid, bounding box, and area for each labeled object.

    Args:
        masks: (H, W) int32 label mask. 0=background, >0=cell IDs.

    Returns:
        List of dicts with: object_id, centroid (x, y), bbox, area
    """
    from scipy import ndimage

    object_ids = np.unique(masks)
    object_ids = object_ids[object_ids > 0]

    objects = []
    for obj_id in object_ids:
        binary_mask = (masks == obj_id)
        area = int(binary_mask.sum())
        cy, cx = ndimage.center_of_mass(binary_mask)

        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        objects.append({
            "object_id": int(obj_id),
            "centroid": (float(cx), float(cy)),
            "bbox": (int(x_min), int(y_min), int(x_max + 1), int(y_max + 1)),
            "area": area,
        })

    return objects


def free_cellpose_model():
    """Free the Cellpose model from GPU memory."""
    global _cellpose_model
    if _cellpose_model is not None:
        del _cellpose_model
        _cellpose_model = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Cellpose model freed from memory.")
