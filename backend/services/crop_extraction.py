"""
Crop extraction service.

Extracts context-aware crops around each segmented cell. The crop is a square
centered on the cell's centroid, sized at CONTEXT_WINDOW_MULTIPLIER × the cell's
bounding box (using the max of width and height to keep it square, so we don't
distort morphology when resizing to DINOv2's fixed input resolution).
"""

import numpy as np
from scipy.spatial import ConvexHull
from skimage.transform import rotate
from config import CONTEXT_WINDOW_MULTIPLIER, CROP_MODE, SIZE_INVARIANT, ROTATION_INVARIANT


def align_crop_rotation(crop: np.ndarray) -> np.ndarray:
    """
    Rotate a masked crop so the cell's major axis aligns to 45 degrees
    (top-left to bottom-right diagonal).

    Uses the convex hull of non-zero pixels to find the major axis via PCA,
    then rotates the crop to align it. Background (zero) pixels remain zero.
    """
    mask = np.any(crop > 0, axis=2)
    if mask.sum() < 10:
        return crop

    # Get coordinates of cell pixels
    ys, xs = np.where(mask)
    coords = np.column_stack([xs, ys]).astype(np.float64)

    # Convex hull to get the outline, then PCA on hull points for major axis
    try:
        hull = ConvexHull(coords)
        hull_pts = coords[hull.vertices]
    except Exception:
        hull_pts = coords

    # PCA: find major axis angle
    centroid = hull_pts.mean(axis=0)
    centered = hull_pts - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Major axis = eigenvector with largest eigenvalue
    major_axis = eigenvectors[:, np.argmax(eigenvalues)]

    # Angle of major axis relative to horizontal
    angle_rad = np.arctan2(major_axis[1], major_axis[0])
    angle_deg = np.degrees(angle_rad)

    # Target: 45 degrees (top-left to bottom-right diagonal)
    rotation_deg = 45 - angle_deg

    # Rotate the crop around its center
    rotated = rotate(crop, -rotation_deg, resize=True, preserve_range=True, order=1)

    # Re-crop to square centered on the cell
    # Find new bounding box of non-zero pixels
    new_mask = np.any(rotated > 0, axis=2)
    if not new_mask.any():
        return crop

    rows = np.any(new_mask, axis=1)
    cols = np.any(new_mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    # Square crop centered on the cell
    cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
    half = max(y2 - y1, x2 - x1) // 2 + 2  # small padding
    H, W = rotated.shape[:2]

    sy1 = max(cy - half, 0)
    sx1 = max(cx - half, 0)
    sy2 = min(cy + half, H)
    sx2 = min(cx + half, W)

    result = np.zeros((2 * half, 2 * half, 3), dtype=np.float32)
    dy1 = sy1 - (cy - half)
    dx1 = sx1 - (cx - half)
    dy2 = dy1 + (sy2 - sy1)
    dx2 = dx1 + (sx2 - sx1)
    result[dy1:dy2, dx1:dx2] = rotated[sy1:sy2, sx1:sx2]

    result = result.astype(np.float32)

    # Resolve 180° flip ambiguity: complex/curved side should be on the right
    result = _resolve_flip(result)

    return result


def _resolve_flip(crop: np.ndarray) -> np.ndarray:
    """
    Resolve the 180-degree flip ambiguity after major axis alignment.

    Splits the cell contour into two halves along the minor axis (perpendicular
    to the diagonal). Measures perimeter complexity of each half. If the more
    complex half is on the left, rotates 180 degrees so it ends up on the right.
    """
    mask = np.any(crop > 0, axis=2)
    if mask.sum() < 10:
        return crop

    H, W = mask.shape
    cy, cx = H / 2, W / 2

    # The minor axis is perpendicular to the 45° diagonal,
    # i.e., it runs from top-right to bottom-left (135°).
    # Points "right of the diagonal" satisfy: (x - cx) + (y - cy) > 0
    # Points "left of the diagonal" satisfy: (x - cx) + (y - cy) < 0

    # Find contour pixels (edge pixels of the mask)
    from scipy.ndimage import binary_erosion
    interior = binary_erosion(mask)
    contour = mask & ~interior
    if contour.sum() < 4:
        return crop

    ys, xs = np.where(contour)

    # Split contour into right half and left half relative to the minor axis
    # Minor axis perpendicular to 45° diagonal: direction is (1, -1)
    # Project onto minor axis: dot with (1, -1) / sqrt(2)
    proj = (xs - cx) - (ys - cy)  # positive = right side, negative = left side

    right_count = np.sum(proj > 0)
    left_count = np.sum(proj < 0)

    # More contour pixels on a side = more complex perimeter on that side
    # If left side is more complex, flip 180°
    if left_count > right_count:
        crop = np.rot90(crop, 2)

    return crop


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


def extract_crop_fixed(
    image: np.ndarray,
    centroid: tuple,
    fixed_size: int,
    masks: np.ndarray = None,
    object_id: int = None,
    crop_mode: str = "neighborhood",
) -> np.ndarray:
    """
    Extract a fixed-size square crop centered on the cell.

    All cells get the same crop size, so small cells appear small and large
    cells appear large — DINOv2 sees the actual scale differences.
    """
    H, W = image.shape[:2]
    half = fixed_size // 2
    cx_int, cy_int = int(round(centroid[0])), int(round(centroid[1]))

    y1 = cy_int - half
    x1 = cx_int - half
    y2 = y1 + fixed_size
    x2 = x1 + fixed_size

    src_y1, src_x1 = max(y1, 0), max(x1, 0)
    src_y2, src_x2 = min(y2, H), min(x2, W)
    dst_y1, dst_x1 = src_y1 - y1, src_x1 - x1
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x2 = dst_x1 + (src_x2 - src_x1)

    crop = np.zeros((fixed_size, fixed_size, 3), dtype=np.float32)
    crop[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

    if crop_mode == "single_cell" and masks is not None and object_id is not None:
        mask_crop = np.zeros((fixed_size, fixed_size), dtype=np.int32)
        mask_crop[dst_y1:dst_y2, dst_x1:dst_x2] = masks[src_y1:src_y2, src_x1:src_x2]
        crop[mask_crop != object_id] = 0.0

    return crop


def extract_all_crops(
    image: np.ndarray,
    objects: list,
    masks: np.ndarray = None,
    crop_mode: str = CROP_MODE,
    multiplier: float = CONTEXT_WINDOW_MULTIPLIER,
    size_invariant: bool = SIZE_INVARIANT,
    rotation_invariant: bool = ROTATION_INVARIANT,
    fixed_size: int = None,
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
        mask_label = obj.get("mask_label", obj["object_id"])
        if not size_invariant and fixed_size is not None:
            crop = extract_crop_fixed(
                image, obj["centroid"], fixed_size,
                masks=masks, object_id=mask_label, crop_mode=crop_mode,
            )
        elif crop_mode == "single_cell" and masks is not None:
            crop = extract_crop_masked(
                image, masks, mask_label, obj["centroid"], obj["bbox"], multiplier
            )
        else:
            crop = extract_crop(image, obj["centroid"], obj["bbox"], multiplier)
        if rotation_invariant:
            crop = align_crop_rotation(crop)
        crops.append(crop)
    return crops
