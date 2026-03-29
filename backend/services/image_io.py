"""
Image I/O service.

Handles loading microscopy images from TIFF files and the BBBC021 dataset format.
All images are normalized to (H, W, 3) float32 arrays with channel 1 = nuclei.
"""

import os
import numpy as np
import tifffile


def load_tiff(path: str) -> np.ndarray:
    """
    Load a 3-channel microscopy TIFF and normalize to (H, W, 3) float32.

    Handles common axis orders:
    - (H, W, 3) — already correct
    - (3, H, W) — channels-first
    - (Z, 3, H, W) or (Z, H, W, 3) — Z-stack, take max projection
    - Singleton dimensions are squeezed out

    Returns:
        (H, W, 3) float32 array

    Raises:
        ValueError: If the image doesn't have exactly 3 channels
    """
    img = tifffile.imread(path)
    img = img.squeeze()
    img = img.astype(np.float32)

    if img.ndim == 3:
        if img.shape[2] == 3:
            pass  # Already (H, W, 3)
        elif img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        else:
            raise ValueError(
                f"Cannot determine channel axis. Shape: {img.shape}. "
                f"Expected exactly 3 channels."
            )

    elif img.ndim == 4:
        if img.shape[1] == 3:
            img = img.max(axis=0)
            img = np.transpose(img, (1, 2, 0))
        elif img.shape[3] == 3:
            img = img.max(axis=0)
        else:
            raise ValueError(
                f"Cannot determine channel axis in 4D image. Shape: {img.shape}."
            )
    else:
        raise ValueError(f"Unexpected dimensions: {img.ndim}. Expected 3D or 4D.")

    if img.shape[2] != 3:
        raise ValueError(f"Must have exactly 3 channels. Got {img.shape[2]}.")

    return img


def load_bbbc021_field(image_dir: str, channel_files: dict) -> np.ndarray:
    """
    Load a single BBBC021 field of view from separate per-channel TIFFs.

    Channel mapping: w1=DAPI (→ slot 1), w2=Tubulin (→ slot 2), w4=Actin (→ slot 3).

    Args:
        image_dir: Directory containing the TIFF files
        channel_files: Dict mapping channel key ('w1','w2','w4') to filename

    Returns:
        (H, W, 3) float32 array
    """
    ch1 = tifffile.imread(os.path.join(image_dir, channel_files['w1']))  # DAPI
    ch2 = tifffile.imread(os.path.join(image_dir, channel_files['w2']))  # Tubulin
    ch3 = tifffile.imread(os.path.join(image_dir, channel_files['w4']))  # Actin

    return np.stack([ch1, ch2, ch3], axis=-1).astype(np.float32)


def _parse_bbbc021_files(image_dir: str) -> dict:
    """
    Parse BBBC021 filenames and group by field of view.

    Handles both simple names (Week1_22123_B02_s1_w1.tif) and UUID-suffixed
    names (Week1_150607_B02_s1_w1<UUID>.tif).

    Returns:
        Dict mapping field base name to {'w1': filename, 'w2': filename, 'w4': filename}
    """
    import re
    fields = {}
    for f in os.listdir(image_dir):
        if not f.endswith('.tif'):
            continue
        # Match pattern: ..._{well}_{site}_w{channel}{optional-uuid}.tif
        m = re.match(r'^(.+_s\d+)_w(\d)', f)
        if m:
            base = m.group(1)
            channel = f'w{m.group(2)}'
            if base not in fields:
                fields[base] = {}
            fields[base][channel] = f
    return fields


def load_bbbc021_first_n(image_dir: str, n: int = 5) -> list:
    """
    Load the first N fields of view from a BBBC021 plate directory.

    Discovers fields by scanning for matching channel triplets (w1, w2, w4),
    then loads all 3 channels for each. Returns a list of (H, W, 3) float32 arrays.
    """
    fields = _parse_bbbc021_files(image_dir)

    # Keep only fields that have all 3 required channels
    complete = {k: v for k, v in sorted(fields.items())
                if all(ch in v for ch in ('w1', 'w2', 'w4'))}

    images = []
    for base, channel_files in list(complete.items())[:n]:
        image = load_bbbc021_field(image_dir, channel_files)
        images.append(image)
        print(f"  Loaded {base}: shape={image.shape}")

    print(f"  Total: {len(images)} fields of view loaded")
    return images


def reorder_channels(image: np.ndarray, channel_order: list) -> np.ndarray:
    """
    Reorder channels so that channel 1 is always nuclei.

    Args:
        image: (H, W, 3) array
        channel_order: List of 3 ints specifying the new order,
                       e.g., [2, 0, 1] means physical channel 2 becomes slot 1

    Returns:
        (H, W, 3) reordered array
    """
    assert len(channel_order) == 3
    return image[:, :, channel_order]


def make_thumbnail(image: np.ndarray, size: int = 256, max_crop_size: int = None) -> np.ndarray:
    """
    Create an RGB thumbnail from a 3-channel float32 image.

    Each channel is independently percentile-normalized to [0, 255] and
    mapped to R, G, B for display.

    Args:
        image: (H, W, 3) float32 array (raw intensities)
        size: Output thumbnail size (square)
        max_crop_size: If provided, scale relative to this size to preserve
                       relative cell sizes across thumbnails.

    Returns:
        (size, size, 3) uint8 RGB array
    """
    from skimage.transform import resize

    # Percentile normalize each channel to [0, 1]
    # Use only non-zero pixels for percentile calculation (handles masked crops)
    nonzero_mask = np.any(image > 0, axis=2)
    normalized = np.zeros_like(image)
    for c in range(3):
        ch = image[:, :, c]
        vals = ch[nonzero_mask] if nonzero_mask.any() else ch.ravel()
        p_low, p_high = np.percentile(vals, [1, 99.5])
        if p_high > p_low:
            normalized[:, :, c] = np.clip((ch - p_low) / (p_high - p_low), 0, 1)

    if max_crop_size and max_crop_size > 0:
        # Tight-crop to non-zero pixels first, then scale relative to max
        if nonzero_mask.any():
            rows = np.any(nonzero_mask, axis=1)
            cols = np.any(nonzero_mask, axis=0)
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            tight = normalized[y1:y2+1, x1:x2+1]
            tight_size = max(tight.shape[0], tight.shape[1])
        else:
            tight = normalized
            tight_size = max(normalized.shape[0], normalized.shape[1])

        scale = tight_size / max_crop_size
        cell_size = max(int(size * scale), 1)
        cell_size = min(cell_size, size)

        # Resize the tight crop to fit, keeping aspect ratio
        th, tw = tight.shape[:2]
        aspect = tw / th if th > 0 else 1
        if aspect >= 1:
            rw = cell_size
            rh = max(int(cell_size / aspect), 1)
        else:
            rh = cell_size
            rw = max(int(cell_size * aspect), 1)

        resized_cell = resize(tight, (rh, rw, 3), anti_aliasing=True)

        # Dark canvas — natural for fluorescence microscopy
        canvas = np.zeros((size, size, 3), dtype=np.float64)
        y_off = (size - rh) // 2
        x_off = (size - rw) // 2
        canvas[y_off:y_off + rh, x_off:x_off + rw] = resized_cell
        return (canvas * 255).astype(np.uint8)
    else:
        resized = resize(normalized, (size, size, 3), anti_aliasing=True)
        return (resized * 255).astype(np.uint8)
