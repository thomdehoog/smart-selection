"""
Crop extraction service.

Extracts context-aware crops around each segmented cell. The crop is a square
centered on the cell's centroid, sized at CONTEXT_WINDOW_MULTIPLIER × the cell's
bounding box (using the max of width and height to keep it square, so we don't
distort morphology when resizing to DINOv2's fixed input resolution).
"""

import numpy as np
from config import CONTEXT_WINDOW_MULTIPLIER, CROP_MODE


def extract_crop(
    image: np.ndarray,
    centroid: tuple,
    bbox: tuple,
    multiplier: float = CONTEXT_WINDOW_MULTIPLIER,
) -> np.ndarray:
    """
    Extract a square crop around a cell from the full image.

    The crop size is `multiplier` × max(bbox_width, bbox_height) in each dimension,
    centered on the cell's centroid. The crop is zero-padded if it extends beyond
    the image boundaries.

    Args:
        image: (H, W, 3) float32 full image
        centroid: (cx, cy) center of the cell
        bbox: (x_min, y_min, x_max, y_max) tight bounding box of the cell
        multiplier: How much to expand the bounding box (default: 4×)

    Returns:
        crop: (crop_size, crop_size, 3) float32 array
    """
    H, W = image.shape[:2]
    cx, cy = centroid
    x_min, y_min, x_max, y_max = bbox

    # Compute crop size as multiplier × max dimension of the bounding box
    # Use max to keep the crop square — avoids aspect ratio distortion
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    crop_size = int(max(bbox_w, bbox_h) * multiplier)

    # Ensure minimum crop size (avoid tiny crops from very small cells)
    crop_size = max(crop_size, 64)

    half = crop_size // 2

    # Compute crop bounds centered on the centroid
    cx_int, cy_int = int(round(cx)), int(round(cy))
    y1 = cy_int - half
    x1 = cx_int - half
    y2 = y1 + crop_size
    x2 = x1 + crop_size

    # Handle boundary: compute source and destination regions for zero-padded crop
    # Source region (in the original image)
    src_y1 = max(y1, 0)
    src_x1 = max(x1, 0)
    src_y2 = min(y2, H)
    src_x2 = min(x2, W)

    # Destination region (in the crop array)
    dst_y1 = src_y1 - y1
    dst_x1 = src_x1 - x1
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x2 = dst_x1 + (src_x2 - src_x1)

    # Create zero-padded crop
    crop = np.zeros((crop_size, crop_size, 3), dtype=np.float32)
    crop[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

    return crop


def extract_crop_masked(
    image: np.ndarray,
    masks: np.ndarray,
    object_id: int,
    centroid: tuple,
    bbox: tuple,
    multiplier: float = CONTEXT_WINDOW_MULTIPLIER,
) -> np.ndarray:
    """
    Extract a crop with non-cell pixels zeroed out using the segmentation mask.

    Same crop region as extract_crop(), but pixels where masks != object_id
    are set to zero. This isolates the cell's morphology from neighbors.

    Args:
        image: (H, W, 3) float32 full image
        masks: (H, W) int32 label mask
        object_id: The cell's label in the mask
        centroid: (cx, cy) center of the cell
        bbox: (x_min, y_min, x_max, y_max) tight bounding box
        multiplier: How much to expand the bounding box

    Returns:
        crop: (crop_size, crop_size, 3) float32 array with non-cell pixels zeroed
    """
    H, W = image.shape[:2]
    cx, cy = centroid
    x_min, y_min, x_max, y_max = bbox

    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    crop_size = int(max(bbox_w, bbox_h) * multiplier)
    crop_size = max(crop_size, 64)
    half = crop_size // 2

    cx_int, cy_int = int(round(cx)), int(round(cy))
    y1 = cy_int - half
    x1 = cx_int - half
    y2 = y1 + crop_size
    x2 = x1 + crop_size

    src_y1 = max(y1, 0)
    src_x1 = max(x1, 0)
    src_y2 = min(y2, H)
    src_x2 = min(x2, W)

    dst_y1 = src_y1 - y1
    dst_x1 = src_x1 - x1
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x2 = dst_x1 + (src_x2 - src_x1)

    # Extract image crop
    crop = np.zeros((crop_size, crop_size, 3), dtype=np.float32)
    crop[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

    # Extract mask crop and zero out non-cell pixels
    mask_crop = np.zeros((crop_size, crop_size), dtype=np.int32)
    mask_crop[dst_y1:dst_y2, dst_x1:dst_x2] = masks[src_y1:src_y2, src_x1:src_x2]
    cell_mask = (mask_crop == object_id)
    crop[~cell_mask] = 0.0

    return crop


def extract_all_crops(
    image: np.ndarray,
    objects: list,
    masks: np.ndarray = None,
    crop_mode: str = CROP_MODE,
    multiplier: float = CONTEXT_WINDOW_MULTIPLIER,
) -> list:
    """
    Extract crops for all detected objects in an image.

    Args:
        image: (H, W, 3) float32 full image
        objects: List of dicts with 'centroid', 'bbox', and 'object_id' keys
        masks: (H, W) int32 label mask (required for single_cell mode)
        crop_mode: "single_cell" or "neighborhood"
        multiplier: Context window multiplier

    Returns:
        crops: List of (crop_size, crop_size, 3) float32 arrays
    """
    crops = []
    for obj in objects:
        if crop_mode == "single_cell" and masks is not None:
            mask_label = obj.get("mask_label", obj["object_id"])
            crop = extract_crop_masked(
                image, masks, mask_label, obj["centroid"], obj["bbox"], multiplier
            )
        else:
            crop = extract_crop(image, obj["centroid"], obj["bbox"], multiplier)
        crops.append(crop)
    return crops
