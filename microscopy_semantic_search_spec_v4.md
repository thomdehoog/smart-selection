# Microscopy Semantic Search Tool — Technical Specification (v4)

## 1. Project Overview

### 1.1 Purpose

This tool enables interactive, embedding-based similarity search across microscopy imaging data. Rather than relying on a pre-trained neural network classifier with fixed categories, users explore their data by selecting examples of structures they find interesting and letting the system find more instances. The system learns what the user means by "interesting" through an iterative feedback loop — no upfront labeling taxonomy is required.

The tool is general-purpose: it works on cells, nuclei, tissue regions, tumor microenvironments, or any biological structure that can be segmented or sampled from a 3-channel fluorescence image.

### 1.2 Core Concept

The workflow is query-by-example with iterative refinement. The user segments an overview scan (or samples it with a grid or random points), browses the results, and clicks on structures they find interesting. The system computes a query embedding from the selected examples and searches for similar structures using FAISS. The user accepts or rejects candidates, and the system adjusts the query vector in real time — no retraining needed.

### 1.3 Key Design Principles

**No upfront feature specification required.** The user does not need to decide whether they care about shape, color, texture, or neighborhood context. DINOv3 patch features capture appearance, and the appended segmentation masks encode shape. The refinement loop implicitly learns which features matter.

**Segmentation is decoupled from search.** Segmentation provides centroids and masks. The masks serve double duty: they define where to crop, and they contribute shape features appended to the DINOv3 patch embeddings.

**Interactive and fast.** All heavy computation happens once upfront. The interactive search loop operates on precomputed vectors in milliseconds.

**Browser-based.** React frontend, Flask backend, Python doing the heavy lifting.

**Simple pipeline.** One segmentation model (Cellpose-SAM). One embedding model (DINOv3). Three input channels map directly to RGB — a single forward pass per crop. Mask overlap fractions are appended to each patch vector, fusing shape and appearance at the patch level.

### 1.4 Technology Stack

| Layer             | Technology                              | Rationale                                                                                  |
|-------------------|-----------------------------------------|--------------------------------------------------------------------------------------------|
| Frontend          | React (JavaScript/TypeScript)           | Fast iteration, excellent dev tools, responsive UI                                         |
| Backend           | Flask (Python)                          | Lightweight, direct access to Python ML ecosystem                                          |
| Segmentation      | Cellpose-SAM (cpsam)                    | Superhuman generalization, channel-order invariant, size invariant, robust to noise/blur    |
| Embedding         | DINOv3 (Meta, ViT-L/14, frozen)        | State-of-the-art SSL vision model; rich patch-level features; 3-channel input = direct RGB  |
| Search Index      | FAISS (Facebook AI Similarity Search)   | Millisecond nearest-neighbor search over millions of vectors                               |

---

## 2. Architecture Overview

### 2.1 High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           REACT FRONTEND                                │
│                                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌────────────────┐  │
│  │  Part 1   │──▶│    Part 2     │──▶│  Part 3  │──▶│    Part 4      │  │
│  │  Data     │   │  Segmentation │   │  Gallery  │   │  Search &      │  │
│  │  Import   │   │  & Selection  │   │  Review   │   │  Refinement    │  │
│  └──────────┘   └──────────────┘   └──────────┘   └────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
        │                │                  │                │
        ▼                ▼                  ▼                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           FLASK BACKEND                                 │
│                                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────┐  ┌────────────┐  │
│  │  Image    │   │  Cellpose-SAM │   │  DINOv3        │  │  FAISS     │  │
│  │  Storage  │   │  (nuclei +    │   │  3ch → RGB     │  │  Index &   │  │
│  │  (3ch)   │   │   cells)      │   │  + Mask Append  │  │  Query     │  │
│  └──────────┘   └──────────────┘   └───────────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 The Embedding Pipeline at a Glance

The entire embedding pipeline for one object is five steps, each simple:

1. **Crop** a 3-channel patch around the object's centroid (4× or 8× the cell bounding box). Channel 1 (nuclei) maps to R, channel 2 to G, channel 3 to B.
2. **Normalize** intensities (percentile clip to [0,1], then ImageNet stats) and **resize** to 518×518.
3. **Forward pass** through frozen DINOv3 → get (37, 37, 1024) patch feature grid. All 3 channels are seen simultaneously — DINOv3 naturally encodes cross-channel spatial relationships (e.g., colocalization patterns).
4. **Append** nuclear and cell mask overlap fractions to each patch → (37, 37, 1026) enriched grid.
5. **Pool** patches inside the cell mask → single 1026-dim L2-normalized vector for FAISS.

No channel mixing, no second model, no separate shape pass.

---

## 3. Part 1 — Data Import and Configuration

### 3.1 User Interface

**File Picker.** User selects one or more image files (TIFF, ND2, CZI). The system validates that every file has exactly 3 channels — uploads with fewer or more are rejected with a clear error message asking the user to prepare a 3-channel image.

**Channel Configuration.** The image must have exactly 3 channels. The system convention is: channel 1 is always the nuclear stain, channels 2 and 3 are the other signals. After upload, the user sees the 3 detected channels and can name each one (e.g., "DAPI", "GFP", "mCherry") and assign a display color (pseudo-color LUT). A dropdown per slot lets the user reassign which physical channel maps to which position — so if the file has DAPI in position 3, the user simply maps it to slot 1 via the dropdown. The backend reorders the channel array at upload time and everything downstream assumes the fixed convention. No further channel selection is needed anywhere in the pipeline.

**Segmentation Strategy Dropdown.**

- `Segment Nuclei + Cells` (default) — Runs cpsam twice: once for nuclei, once for whole cells. Produces two complementary mask sets per object.
- `Segment Nuclei Only` — Runs cpsam once using only channel 1 (the nuclear stain). Faster, sufficient when cytoplasm boundaries are not needed.
- `Regular Grid` — No segmentation. The image is tiled into overlapping patches of a user-specified pixel size.
- `Random Sampling` — No segmentation. N random positions scattered across the image. User specifies sample count (1,000–50,000) and optionally enables stratified sampling for uniform spatial coverage.

**Context Window Size.** Two presets for segmentation modes:

- `4×` — Crop is 4 times the cell bounding box in each dimension. Captures cell plus immediate neighbors.
- `8×` — Crop is 8 times the cell bounding box. Captures wide tissue-level context.

For grid/random modes, an absolute pixel size (e.g., 256×256 or 512×512).

**Invariance Options.** Two toggleable checkboxes:

- `Rotation Invariant` — Each crop is embedded at 8 orientations (4 rotations × 2 flips). Patch grids are un-rotated before averaging. 8× forward passes per crop. Important for tissue where structures appear at arbitrary angles.
- `Scale Invariant` — Each crop is embedded at 3 scales (0.7×, 1.0×, 1.4× zoom). 3× forward passes per crop. Important when similar structures vary in physical size.

These stack multiplicatively (both on = 24× passes). The UI displays an estimated precomputation time.

### 3.2 Backend: `POST /api/upload`

**Request payload:**

- `files`: One or more 3-channel image files (channel 1 = nuclei, channels 2–3 = other signals)
- `channel_names`: `["DAPI", "GFP", "mCherry"]`
- `channel_colors`: `["#0000FF", "#00FF00", "#FF0000"]`
- `segmentation_strategy`: `"nuclei_and_cells"` | `"nuclei_only"` | `"grid"` | `"random"`
- `context_window`: `"4x"` | `"8x"` | integer pixel size
- `num_random_samples`: Integer (random mode only)
- `stratified_sampling`: Boolean (random mode only)
- `rotation_invariant`: Boolean
- `scale_invariant`: Boolean

**Response:**

```json
{
  "status": "ok",
  "dataset_id": "uuid-string",
  "num_images": 5,
  "image_dimensions": [2048, 2048],
  "num_channels": 3,
  "channel_names": ["DAPI", "GFP", "mCherry"],
  "estimated_precompute_minutes": 3.2
}
```

**Backend behavior:** Parse and store raw images using `tifffile`. Generate composite RGB thumbnails. Compute estimated precomputation time.

**Image loading with tifffile.** Microscopy TIFFs come in many axis orders and dtypes. The loader must normalize everything to a consistent `(H, W, 3)` float32 array before anything else touches it.

```python
import tifffile
import numpy as np

def load_image(path: str) -> np.ndarray:
    """
    Load a 3-channel microscopy TIFF and normalize to (H, W, 3) float32.

    Handles common axis orders from microscopy software:
    - (H, W, 3) — already correct
    - (3, H, W) — channels-first, common in ImageJ/FIJI exports
    - (Z, 3, H, W) or (Z, H, W, 3) — Z-stack, take max projection
    - (1, 3, H, W) — singleton dimension, squeeze it

    Handles dtypes: uint8, uint16, float32, float64.
    Output is always float32 with raw intensity values preserved.

    Args:
        path: Path to the TIFF file

    Returns:
        image: (H, W, 3) float32 array

    Raises:
        ValueError: If the image does not contain exactly 3 channels
    """
    img = tifffile.imread(path)

    # Squeeze out singleton dimensions (e.g., (1, 3, H, W) → (3, H, W))
    img = img.squeeze()

    # Convert to float32 for consistent downstream processing
    img = img.astype(np.float32)

    # Determine axis order based on shape
    if img.ndim == 3:
        # Could be (H, W, 3) or (3, H, W)
        if img.shape[2] == 3:
            # Already (H, W, 3)
            pass
        elif img.shape[0] == 3:
            # (3, H, W) → transpose to (H, W, 3)
            img = np.transpose(img, (1, 2, 0))
        else:
            raise ValueError(
                f"Cannot determine channel axis. Shape: {img.shape}. "
                f"Expected exactly 3 channels."
            )

    elif img.ndim == 4:
        # Z-stack: could be (Z, 3, H, W) or (Z, H, W, 3)
        if img.shape[1] == 3:
            # (Z, 3, H, W) → max project over Z, then transpose
            img = img.max(axis=0)               # (3, H, W)
            img = np.transpose(img, (1, 2, 0))  # (H, W, 3)
        elif img.shape[3] == 3:
            # (Z, H, W, 3) → max project over Z
            img = img.max(axis=0)               # (H, W, 3)
        else:
            raise ValueError(
                f"Cannot determine channel axis in 4D image. Shape: {img.shape}. "
                f"Expected exactly 3 channels."
            )

    else:
        raise ValueError(
            f"Unexpected number of dimensions: {img.ndim}. "
            f"Expected 3D (H, W, 3) or 4D (Z, H, W, 3) image."
        )

    # Final validation: must be exactly 3 channels
    if img.shape[2] != 3:
        raise ValueError(
            f"Image must have exactly 3 channels. Got {img.shape[2]}."
        )

    return img
```

---

## 4. Part 2 — Segmentation, Embedding, and Interactive Selection

### 4.1 Segmentation — Cellpose-SAM

Cellpose-SAM (cpsam) is the segmentation model. It uses a ViT-L encoder adapted from SAM with Cellpose's flow-field prediction framework. It takes up to 3 channels as input and is trained to be channel-order invariant. It is robust to cell size, noise, blur, and downsampling. Specifying cell diameter is optional.

**For `nuclei_and_cells` mode, cpsam runs twice per image:**

Pass 1 — Nuclear segmentation: Only channel 1 (the nuclear stain) is provided, replicated or zero-padded to fill the 3-channel input that cpsam expects. Output: integer label mask where each integer = one nucleus.

Pass 2 — Cell segmentation: All 3 channels are provided as-is. cpsam is channel-order invariant and uses all available signal to find cell boundaries. Output: integer label mask for whole cells including cytoplasm.

Nuclear and cell masks are matched by spatial overlap — each nucleus is associated with its enclosing cell.

**Per-object metadata stored:**

- `object_id`: Unique integer
- `image_index`: Which image in the dataset
- `centroid`: (x, y) center of mass of the nuclear mask
- `nuclear_bbox`: (x_min, y_min, x_max, y_max) tight nuclear bounding box
- `cell_bbox`: (x_min, y_min, x_max, y_max) tight cell bounding box (if available)
- `nuclear_mask`: Binary mask of the nucleus
- `cell_mask`: Binary mask of the whole cell (if available)
- `nuclear_area`: Pixel count
- `cell_area`: Pixel count (if available)

For `grid` and `random` modes: no masks are produced. Centroids are generated directly.

### 4.2 Crop Extraction

For each object, extract a 3-channel crop from the raw image. The crop is centered on the centroid and sized by the context window (4× or 8× the cell bbox, or absolute pixel size). Zero-padded at image edges.

Result: `(crop_H, crop_W, 3)` intensity crop, plus `(crop_H, crop_W)` nuclear mask crop and `(crop_H, crop_W)` cell mask crop at the same spatial extent.

**Critical: all crops must be held in memory before embedding begins.** After segmentation completes, the system extracts every crop and mask crop in a single pass over the raw images and stores them as numpy arrays in RAM. The subsequent DINOv3 embedding loop then indexes directly into these in-memory arrays. This avoids disk I/O during inference, which would otherwise stall the GPU and severely degrade throughput. For large datasets where all crops do not fit in RAM, crops should be written to a memory-mapped file (e.g., numpy memmap) during extraction and accessed from there — still much faster than random file reads.

### 4.3 Embedding — DINOv3 with Mask Enrichment

Since the image has exactly 3 channels in a fixed order (nuclei, signal 2, signal 3), the crop maps directly to an RGB image for DINOv3 — channel 1 → R, channel 2 → G, channel 3 → B. This means DINOv3 sees all three channels simultaneously in a single forward pass and naturally encodes their spatial relationships (e.g., where nuclear signal colocalizes with cytoplasmic signal). One forward pass per crop (or more if invariance augmentations are enabled).

**Step 1: Intensity normalization.** Per channel, clip to 1st–99.5th percentile and scale to [0, 1].

**Step 2: Resize** to 518×518 (bilinear interpolation).

**Step 3: ImageNet normalization.** Apply mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].

**Step 4: Forward pass** through frozen DINOv3 in eval mode, no gradients. Extract patch tokens (excluding CLS). For ViT-L/14 at 518 input: (37, 37, 1024) grid. Each patch vector encodes the visual content of its local region across all 3 channels simultaneously.

**Step 5: Invariance augmentation** (if enabled). For rotation invariance, repeat steps 1–4 at 8 orientations (0°/90°/180°/270° × no-flip/flip). Un-rotate each grid back to canonical orientation. For scale invariance, repeat at 3 zoom levels (0.7×, 1.0×, 1.4×). Average all grids into a single (37, 37, 1024).

**Step 6: Append mask information.** For each patch position (i, j) in the grid, compute the fraction of that patch's spatial region that overlaps with the nuclear mask and the cell mask. Append these two floats to the patch vector: (37, 37, 1024) → (37, 37, 1026). For grid/random modes without masks, this step is skipped and the features remain at 1024 dims.

**Step 7: Pool to global embedding.** Average only the patches where cell_overlap > 0.5 (i.e., mostly inside the target cell). L2-normalize. Result: a single 1026-dimensional vector per object. For grid/random modes, average all patches and the vector is 1024-dimensional.

### 4.4 FAISS Index

Build an index over all global embeddings. `IndexFlatIP` for <100K objects (exact cosine similarity), `IndexIVFFlat` for larger datasets. Hold in memory.

### 4.5 Progress Polling: `GET /api/job_status/<job_id>`

```json
{
  "status": "processing|complete|error",
  "progress": 0.65,
  "phase": "embedding",
  "message": "Embedding objects: 3200 / 4900",
  "num_objects_found": 4900
}
```

Frontend polls every 1–2 seconds for a progress bar.

### 4.6 User Interface — Interactive Selection

Once segmentation completes (embedding can still run in background), the user sees the overview scan with interactive overlays.

**Main viewport.** Multi-channel composite with per-channel toggle and brightness/contrast. Segmentation overlay (togglable): nuclear outlines in one color, cell outlines in another. On hover, the context bounding box (the actual crop region) highlights. Click to select/deselect. Selected objects get a yellow highlight.

**Navigation.** Thumbnail strip for multi-image datasets. Selection state persists across images. Pan and zoom for large images.

**Selection panel.** Persistent sidebar: selected count, scrollable thumbnails of selections, "Clear Selection," and "Proceed to Gallery" button.

### 4.7 Backend Endpoints

**`GET /api/get_image/<dataset_id>/<image_index>`** — Composite image with overlays.

**`GET /api/get_objects/<dataset_id>`** — Object metadata (centroids, bboxes, areas). Optional `image_index` filter.

---

## 5. Part 3 — Gallery Review

### 5.1 Purpose

Confirmation step. The user reviews all selections in a gallery, removes mistakes, and verifies the set captures their phenotype of interest.

### 5.2 `POST /api/get_crops`

**Request:** `{ "dataset_id": "...", "object_ids": [12, 45, 67, 89, 102] }`

**Response:** Array with `object_id`, `image_index`, `thumbnail_base64` (composite RGB, 128×128 or 256×256), `centroid`, `nuclear_area`, `cell_area`.

### 5.3 User Interface

Grid of thumbnail crops. Each has "×" to remove. Optional per-channel sub-thumbnails. "Find Similar" button below.

---

## 6. Part 4 — Search and Iterative Refinement

### 6.1 `POST /api/search`

**Request:**

```json
{
  "dataset_id": "uuid-string",
  "positive_ids": [12, 45, 67, 89, 102],
  "negative_ids": [],
  "top_k": 50,
  "negative_alpha": 0.4
}
```

**Backend behavior:**

1. Retrieve global embeddings for all `positive_ids`.
2. Query vector = mean of positive embeddings.
3. If negatives exist: subtract `alpha * mean(negative_embeddings)`.
4. L2-normalize.
5. FAISS search for top_k neighbors, excluding positives and negatives.
6. Return results with thumbnails and scores.

**Response:**

```json
{
  "results": [
    {
      "object_id": 203,
      "image_index": 2,
      "similarity_score": 0.93,
      "thumbnail_base64": "data:image/png;base64,...",
      "centroid": [1024, 768]
    }
  ]
}
```

### 6.2 User Interface

**Top: "Your Selections" gallery.** Positive selections with "×" to remove.

**Bottom: "Similar Structures Found" gallery.** Results ranked by similarity. Each has Accept (✓) and Reject (✗) buttons. "Recompute" triggers a new search with updated positives/negatives. Similarity threshold slider. Negative strength slider. "Show on Image" button per result.

**Refinement loop.** Review → accept/reject → recompute → review. Milliseconds per cycle. Iterate until satisfied.

### 6.3 `POST /api/export_results`

Exports accepted objects as CSV/JSON: object ID, image index, centroid, bboxes, areas, mean intensity per channel within masks, similarity score. Compatible with ImageJ, CellProfiler, napari, QuPath.

---

## 7. Embedding Pipeline — Complete Detail

This is the core of the system. It is intentionally simple: one model, one forward pass (per augmentation), mask append, pool.

### 7.1 Preprocessing

```python
import numpy as np
from PIL import Image
from torchvision import transforms

# ImageNet normalization — applied after percentile rescaling to [0,1]
preprocess = transforms.Compose([
    transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),  # Converts [0,1] float to (3, H, W) tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def normalize_crop(crop: np.ndarray) -> np.ndarray:
    """
    Percentile-normalize a 3-channel crop to [0, 1].

    Args:
        crop: (H, W, 3) raw intensity values (any dtype)

    Returns:
        normalized: (H, W, 3) float32 in [0, 1]
    """
    crop = crop.astype(np.float32)
    for c in range(3):
        ch = crop[:, :, c]
        p_low, p_high = np.percentile(ch, [1, 99.5])
        if p_high > p_low:
            crop[:, :, c] = np.clip((ch - p_low) / (p_high - p_low), 0, 1)
        else:
            crop[:, :, c] = 0.0
    return crop
```

### 7.2 DINOv3 Patch Feature Extraction

```python
import torch

# Load model once at server startup — keep in memory
dinov3_model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl14', pretrained=True)
dinov3_model.eval().cuda()

def extract_patch_features(crop_normalized: np.ndarray) -> np.ndarray:
    """
    Extract DINOv3 patch-level features from a 3-channel crop.

    Args:
        crop_normalized: (H, W, 3) float32 in [0, 1]

    Returns:
        patch_grid: (37, 37, 1024) patch feature grid
    """
    # Convert to PIL for torchvision transforms
    pil_img = Image.fromarray((crop_normalized * 255).astype(np.uint8))
    tensor = preprocess(pil_img).unsqueeze(0).cuda()

    with torch.no_grad():
        features = dinov3_model.forward_features(tensor)
        # Patch tokens excluding CLS — shape (1, 1369, 1024) for 37×37 grid
        patch_tokens = features["x_norm_patchtokens"]

    num_patches = int(patch_tokens.shape[1] ** 0.5)  # 37
    grid = patch_tokens[0].reshape(num_patches, num_patches, -1).cpu().numpy()
    return grid
```

### 7.3 Invariance Augmentation

```python
from scipy.ndimage import rotate as ndimage_rotate

def extract_with_invariance(
    crop_normalized: np.ndarray,
    rotation_invariant: bool = False,
    scale_invariant: bool = False
) -> np.ndarray:
    """
    Extract patch features with optional rotation and scale invariance.

    Processes the crop at multiple orientations and/or scales,
    un-rotates each grid back to canonical orientation, and averages.

    Args:
        crop_normalized: (H, W, 3) float32 in [0, 1]
        rotation_invariant: Apply 4 rotations × 2 flips = 8 augmentations
        scale_invariant: Apply 3 scales (0.7×, 1.0×, 1.4×)

    Returns:
        averaged_grid: (37, 37, 1024) averaged patch feature grid
    """
    rotations = [0, 90, 180, 270] if rotation_invariant else [0]
    flips = [False, True] if rotation_invariant else [False]
    scales = [0.7, 1.0, 1.4] if scale_invariant else [1.0]

    all_grids = []

    for scale in scales:
        # Apply zoom from center
        if scale != 1.0:
            scaled = _apply_center_zoom(crop_normalized, scale)
        else:
            scaled = crop_normalized

        for rot in rotations:
            for flip in flips:
                aug = scaled.copy()
                if flip:
                    aug = np.fliplr(aug)
                if rot > 0:
                    aug = ndimage_rotate(aug, rot, axes=(0, 1), reshape=False, order=1)

                # Extract patch features for this augmented view
                grid = extract_patch_features(aug)

                # Un-rotate and un-flip the grid back to canonical orientation
                if flip:
                    grid = np.flip(grid, axis=1).copy()
                if rot > 0:
                    grid = np.rot90(grid, k=-rot // 90, axes=(0, 1)).copy()

                all_grids.append(grid)

    # Average all grids into a single (37, 37, 1024) grid
    return np.mean(all_grids, axis=0)


def _apply_center_zoom(img: np.ndarray, scale: float) -> np.ndarray:
    """Zoom in or out from the center of the image, maintaining original dimensions."""
    from skimage.transform import resize
    h, w = img.shape[:2]
    # Compute the region to crop (zoom in) or the region to place (zoom out)
    new_h, new_w = int(h / scale), int(w / scale)
    cy, cx = h // 2, w // 2

    if scale > 1.0:
        # Zoom in: crop a smaller region and resize up
        y1 = max(cy - new_h // 2, 0)
        x1 = max(cx - new_w // 2, 0)
        cropped = img[y1:y1 + new_h, x1:x1 + new_w]
        return resize(cropped, (h, w), anti_aliasing=True).astype(np.float32)
    else:
        # Zoom out: resize down and pad
        resized = resize(img, (new_h, new_w), anti_aliasing=True).astype(np.float32)
        padded = np.zeros_like(img)
        y1 = (h - new_h) // 2
        x1 = (w - new_w) // 2
        padded[y1:y1 + new_h, x1:x1 + new_w] = resized
        return padded
```

### 7.4 Mask Enrichment

```python
def append_masks_to_patches(
    patch_grid: np.ndarray,
    nuclear_mask_crop: np.ndarray,
    cell_mask_crop: np.ndarray,
    crop_shape: tuple
) -> np.ndarray:
    """
    Append segmentation mask overlap fractions to each patch vector.

    For each patch in the grid, computes what fraction of its spatial
    region overlaps with the nuclear mask and the cell mask. These two
    floats are concatenated to the 1024-dim DINOv3 feature vector,
    producing a 1026-dim enriched vector per patch.

    Args:
        patch_grid: (Ph, Pw, 1024) DINOv3 patch features
        nuclear_mask_crop: (H, W) binary mask of target nucleus in crop space
        cell_mask_crop: (H, W) binary mask of target cell in crop space
        crop_shape: (H, W) original crop dimensions

    Returns:
        enriched_grid: (Ph, Pw, 1026) features with mask overlap appended
    """
    Ph, Pw, D = patch_grid.shape
    H, W = crop_shape
    patch_h, patch_w = H / Ph, W / Pw

    nuclear_overlaps = np.zeros((Ph, Pw), dtype=np.float32)
    cell_overlaps = np.zeros((Ph, Pw), dtype=np.float32)

    for i in range(Ph):
        for j in range(Pw):
            y1, y2 = int(i * patch_h), min(int((i + 1) * patch_h), H)
            x1, x2 = int(j * patch_w), min(int((j + 1) * patch_w), W)
            area = (y2 - y1) * (x2 - x1)
            if area > 0:
                nuclear_overlaps[i, j] = nuclear_mask_crop[y1:y2, x1:x2].sum() / area
                cell_overlaps[i, j] = cell_mask_crop[y1:y2, x1:x2].sum() / area

    mask_features = np.stack([nuclear_overlaps, cell_overlaps], axis=-1)  # (Ph, Pw, 2)
    return np.concatenate([patch_grid, mask_features], axis=-1)            # (Ph, Pw, 1026)
```

### 7.5 Pooling to Global Embedding

```python
def pool_to_global_embedding(
    enriched_grid: np.ndarray,
    mask_threshold: float = 0.5
) -> np.ndarray:
    """
    Pool mask-enriched patch features to a single global embedding.

    Averages only patches that are mostly inside the target cell
    (cell_overlap > threshold), then L2-normalizes.

    Args:
        enriched_grid: (Ph, Pw, D+2) enriched patch features
        mask_threshold: Minimum cell_overlap to include a patch

    Returns:
        embedding: (D+2,) L2-normalized global embedding
    """
    # Cell overlap is the last dimension
    cell_overlaps = enriched_grid[:, :, -1]
    mask = cell_overlaps > mask_threshold

    if mask.sum() == 0:
        # Fallback: use all patches if none pass threshold
        selected = enriched_grid.reshape(-1, enriched_grid.shape[-1])
    else:
        selected = enriched_grid[mask]

    embedding = selected.mean(axis=0)
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    return embedding
```

### 7.6 Complete Per-Object Pipeline

```python
def embed_object(
    crop: np.ndarray,
    nuclear_mask_crop: np.ndarray = None,
    cell_mask_crop: np.ndarray = None,
    rotation_invariant: bool = False,
    scale_invariant: bool = False
) -> tuple:
    """
    Complete embedding pipeline for a single object.

    Args:
        crop: (H, W, 3) raw 3-channel intensity crop
        nuclear_mask_crop: (H, W) binary nuclear mask (None for grid/random)
        cell_mask_crop: (H, W) binary cell mask (None for grid/random)
        rotation_invariant: Whether to apply rotation augmentation
        scale_invariant: Whether to apply scale augmentation

    Returns:
        global_embedding: (1026,) or (1024,) L2-normalized vector for FAISS
        enriched_grid: (Ph, Pw, 1026) or (Ph, Pw, 1024) full grid (for storage)
    """
    crop_shape = crop.shape[:2]

    # Step 1: Normalize intensities to [0, 1]
    crop_norm = normalize_crop(crop)

    # Steps 2–5: Extract DINOv3 patch features (with optional invariance)
    patch_grid = extract_with_invariance(crop_norm, rotation_invariant, scale_invariant)

    # Step 6: Append mask information (if masks available)
    if nuclear_mask_crop is not None and cell_mask_crop is not None:
        enriched_grid = append_masks_to_patches(
            patch_grid, nuclear_mask_crop, cell_mask_crop, crop_shape
        )
    else:
        enriched_grid = patch_grid  # No masks — raw 1024-dim features

    # Step 7: Pool to global embedding
    if cell_mask_crop is not None:
        global_embedding = pool_to_global_embedding(enriched_grid)
    else:
        flat = enriched_grid.reshape(-1, enriched_grid.shape[-1])
        global_embedding = flat.mean(axis=0)
        global_embedding = global_embedding / (np.linalg.norm(global_embedding) + 1e-8)

    return global_embedding, enriched_grid
```

### 7.7 Batch Processing

```python
def embed_all_objects(
    crops: list,
    nuclear_masks: list,
    cell_masks: list,
    rotation_invariant: bool = False,
    scale_invariant: bool = False,
    progress_callback=None
) -> tuple:
    """
    Embed all objects in the dataset.

    Args:
        crops: List of (H, W, 3) arrays
        nuclear_masks: List of (H, W) binary masks (or None per object)
        cell_masks: List of (H, W) binary masks (or None per object)
        progress_callback: Optional function(current, total) for progress reporting

    Returns:
        global_embeddings: (N, D) numpy array (D=1026 with masks, 1024 without)
        enriched_grids: List of (Ph, Pw, D) arrays
    """
    embeddings = []
    grids = []

    for i, (crop, nuc_mask, cell_mask) in enumerate(zip(crops, nuclear_masks, cell_masks)):
        emb, grid = embed_object(
            crop, nuc_mask, cell_mask, rotation_invariant, scale_invariant
        )
        embeddings.append(emb)
        grids.append(grid)

        if progress_callback:
            progress_callback(i + 1, len(crops))

    return np.stack(embeddings), grids
```

---

## 8. FAISS Integration

### 8.1 Index Construction

```python
import faiss
import numpy as np

def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = True) -> faiss.Index:
    """
    Build a FAISS index over L2-normalized embeddings.

    Uses exact inner product search (= cosine similarity on normalized vectors)
    for datasets up to 100K objects, and IVF for larger datasets.
    """
    N, D = embeddings.shape

    if N < 100_000:
        index = faiss.IndexFlatIP(D)
    else:
        nlist = min(int(np.sqrt(N)), 1024)
        quantizer = faiss.IndexFlatIP(D)
        index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.nprobe = 32

    if use_gpu and faiss.get_num_gpus() > 0:
        gpu_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

    index.add(embeddings)
    return index
```

### 8.2 Search with Refinement

```python
def search_similar(
    index: faiss.Index,
    embeddings: np.ndarray,
    positive_ids: list,
    negative_ids: list = None,
    alpha: float = 0.4,
    top_k: int = 50
) -> tuple:
    """
    Search for similar objects with positive/negative feedback.

    The query vector is the mean of positive embeddings, adjusted away
    from negatives. This is pure vector arithmetic — no retraining.
    """
    query = embeddings[positive_ids].mean(axis=0)

    if negative_ids and len(negative_ids) > 0:
        neg_vec = embeddings[negative_ids].mean(axis=0)
        query = query - alpha * neg_vec

    query = query / (np.linalg.norm(query) + 1e-8)
    query = query.reshape(1, -1).astype(np.float32)

    exclude = set(positive_ids) | set(negative_ids or [])
    search_k = top_k + len(exclude) + 10
    scores, indices = index.search(query, search_k)

    results = [
        (int(idx), float(score))
        for idx, score in zip(indices[0], scores[0])
        if idx not in exclude and idx >= 0
    ]

    return [r[0] for r in results[:top_k]], [r[1] for r in results[:top_k]]
```

---

## 9. Flask Backend

### 9.1 Project Structure

```
backend/
├── app.py                       # Flask app entry, DINOv3 model loading at startup
├── config.py                    # Defaults, hardware detection
├── routes/
│   ├── upload.py                # POST /api/upload
│   ├── segmentation.py          # POST /api/segment_and_embed, GET /api/job_status
│   ├── images.py                # GET /api/get_image, GET /api/get_objects
│   ├── gallery.py               # POST /api/get_crops
│   ├── search.py                # POST /api/search
│   └── export.py                # POST /api/export_results
├── services/
│   ├── image_io.py              # Image loading, 3-channel validation
│   ├── segmentation.py          # Cellpose-SAM wrapper (nuclei + cell passes)
│   ├── crop_extraction.py       # Context-aware cropping with masks
│   ├── embedding.py             # DINOv3 forward pass + invariance augmentation
│   ├── mask_enrichment.py       # Append masks to patches + pooling
│   ├── indexing.py              # FAISS build + search
│   └── preprocessing.py         # Percentile normalization
├── models/
│   └── dataset.py               # In-memory dataset state
└── requirements.txt
```

### 9.2 In-Memory State

```python
class DatasetState:
    dataset_id: str
    config: dict                               # All Part 1 settings
    raw_images: list[np.ndarray]               # (H, W, 3) per image
    nuclear_masks: list[np.ndarray]            # (H, W) integer labels per image
    cell_masks: list[np.ndarray]               # (H, W) integer labels per image (or None)
    objects: list[dict]                        # Per-object metadata
    crops: list[np.ndarray]                    # (crop_H, crop_W, 3) per object
    nuclear_mask_crops: list[np.ndarray]       # (crop_H, crop_W) binary per object
    cell_mask_crops: list[np.ndarray]          # (crop_H, crop_W) binary per object (or None)
    enriched_grids: list[np.ndarray]           # (37, 37, 1026) per object
    global_embeddings: np.ndarray              # (N, 1026) for FAISS
    faiss_index: faiss.Index
    thumbnails: list[bytes]                    # Pre-rendered PNGs for gallery
```

### 9.3 Endpoint Reference

| Method | Endpoint                                   | Purpose                                    |
|--------|---------------------------------------------|--------------------------------------------|
| POST   | `/api/upload`                               | Upload 3-channel images, configure dataset |
| POST   | `/api/segment_and_embed`                    | Trigger segmentation + embedding           |
| GET    | `/api/job_status/<job_id>`                  | Poll progress                              |
| GET    | `/api/get_image/<dataset_id>/<image_index>` | Composite image with overlays              |
| GET    | `/api/get_objects/<dataset_id>`             | Object metadata                            |
| POST   | `/api/get_crops`                            | Thumbnails for selected objects            |
| POST   | `/api/search`                               | Similarity search with refinement          |
| POST   | `/api/export_results`                       | Export as CSV/JSON                         |

---

## 10. React Frontend

### 10.1 Structure

```
frontend/src/
├── App.tsx
├── store/useDatasetStore.ts
├── components/
│   ├── steps/
│   │   ├── DataImport.tsx
│   │   ├── SegmentAndSelect.tsx
│   │   ├── GalleryReview.tsx
│   │   └── SearchRefine.tsx
│   ├── viewer/
│   │   ├── ImageViewer.tsx
│   │   ├── MaskOverlay.tsx
│   │   ├── HoverHighlight.tsx
│   │   └── ChannelControls.tsx
│   ├── gallery/
│   │   ├── CropGallery.tsx
│   │   ├── CropCard.tsx
│   │   └── SimilarityBadge.tsx
│   └── controls/
│       ├── NegativeStrengthSlider.tsx
│       ├── SimilarityThreshold.tsx
│       ├── ProgressBar.tsx
│       └── StepNavigation.tsx
├── api/client.ts
└── utils/
    ├── imageProcessing.ts
    └── colorMaps.ts
```

### 10.2 Zustand Store

```typescript
interface DatasetStore {
  datasetId: string | null;
  numImages: number;
  channelNames: [string, string, string];
  channelColors: [string, string, string];

  currentStep: 1 | 2 | 3 | 4;
  currentImageIndex: number;

  // Selection — persists across images
  selectedObjectIds: Set<number>;
  toggleSelection: (objectId: number) => void;
  clearSelection: () => void;

  // Search state
  positiveIds: number[];
  negativeIds: number[];
  searchResults: SearchResult[];
  acceptResult: (objectId: number) => void;
  rejectResult: (objectId: number) => void;

  // Controls
  negativeAlpha: number;
  similarityThreshold: number;

  // Display
  activeChannels: [boolean, boolean, boolean];
  showNuclearMask: boolean;
  showCellMask: boolean;
  showPatchBounds: boolean;
  brightness: [number, number, number];
  contrast: [number, number, number];
}
```

---

## 11. Implementation Roadmap

### Phase 1: End-to-End MVP (1–2 weeks)

The simplest possible version: segment cells, crop around them, embed, search.

Backend: TIFF upload (3-channel, validated). Channel reorder dropdown. Run cpsam once for cell segmentation using all 3 channels. Extract a fixed-size crop around each cell's bounding box (start with 4× hardcoded). Load all crops into memory. Run DINOv3 ViT-B on each crop (3 channels straight to RGB, single forward pass, no invariance augmentation). Store raw patch features — no mask enrichment yet (pure 1024-dim). Pool all patches per crop to a single global vector (simple mean, no masked pooling). FAISS `IndexFlatIP` over all vectors. Search endpoint with positive/negative refinement.

Frontend: Simple image viewer showing the 3-channel composite. Cell mask overlay. Click cells to select/deselect. Basic gallery of selected crops. Search results gallery with accept/reject buttons and recompute. That's it.

### Phase 2: Full Pipeline (2–3 weeks)

Backend: Add nuclear segmentation (cpsam second pass on channel 1 only). Mask-enriched patch features (append nuclear + cell overlap → 1026-dim). Masked pooling (average only patches inside cell mask). 4×/8× context window toggle. Multi-format image loading (ND2, CZI). Export endpoint.

Frontend: Channel toggles and brightness/contrast. Hover-to-highlight context boxes. Multi-image navigation with persistent selection. Progress bar. Similarity threshold slider. Negative strength slider. "Show on Image" button.

### Phase 3: Invariance and Advanced Features (2–4 weeks)

Backend: Rotation and scale invariance augmentation. Grid mode. Random sampling mode with stratified option. Estimated precompute time display.

Frontend: Invariance toggles with time estimate. Sampling configuration UI. Persistent sessions (save/load to disk). Multiple simultaneous search queries.

---

## 12. Test Dataset — BBBC021

For Phase 1 development and testing, use BBBC021 from the Broad Bioimage Benchmark Collection. This dataset contains MCF7 human breast cancer cells labeled with DAPI (nuclei), phalloidin/F-actin (cytoskeleton), and anti-tubulin/β-tubulin (microtubules). Exactly 3 channels, exactly the kind of phenotypic variation the tool is designed to find — cells treated with different compounds show distinct morphological changes in their tubulin and actin networks while sharing the same nuclear stain.

The images are stored as separate single-channel 16-bit TIFFs per field of view, following the naming convention `<plate>_<well>_<site>_w<channel>.tif`. To keep the test small, download only the first plate ZIP and use the first 5 fields of view.

### 12.1 Download

```bash
# Download a single plate (~750 MB) — only need one for testing
wget https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22123.zip
unzip BBBC021_v1_images_Week1_22123.zip -d bbbc021_raw/
```

### 12.2 Loading Helper

The channels are separate files that need to be stacked. Channel mapping: w1 = DAPI (nuclei → slot 1), w2 = Tubulin (→ slot 2), w4 = Actin (→ slot 3). Note: w3 does not exist in this dataset.

```python
import os
import re
import numpy as np
import tifffile

def load_bbbc021_field(image_dir: str, plate: str, well: str, site: str) -> np.ndarray:
    """
    Load a single BBBC021 field of view as a (H, W, 3) float32 array.

    Channel mapping:
      - Channel 1 (nuclei): *_w1.tif  (DAPI)
      - Channel 2:          *_w2.tif  (Tubulin)
      - Channel 3:          *_w4.tif  (Actin)

    Args:
        image_dir: Path to the extracted image directory
        plate: Plate ID, e.g., "Week1_22123"
        well: Well ID, e.g., "B02"
        site: Site number, e.g., "s1"

    Returns:
        image: (H, W, 3) float32 array
    """
    prefix = f"{plate}_{well}_{site}"

    ch1 = tifffile.imread(os.path.join(image_dir, f"{prefix}_w1.tif"))  # DAPI
    ch2 = tifffile.imread(os.path.join(image_dir, f"{prefix}_w2.tif"))  # Tubulin
    ch3 = tifffile.imread(os.path.join(image_dir, f"{prefix}_w4.tif"))  # Actin

    # Stack to (H, W, 3) and convert to float32
    image = np.stack([ch1, ch2, ch3], axis=-1).astype(np.float32)
    return image


def load_bbbc021_first_n(image_dir: str, n: int = 5) -> list:
    """
    Load the first N fields of view from a BBBC021 plate directory.

    Discovers available fields by scanning for *_w1.tif files (DAPI channel),
    sorts them, and loads the first N as 3-channel images.

    Args:
        image_dir: Path to the extracted image directory
        n: Number of fields of view to load (default: 5)

    Returns:
        images: List of (H, W, 3) float32 arrays
    """
    # Find all DAPI files to discover available fields
    dapi_files = sorted([
        f for f in os.listdir(image_dir)
        if f.endswith("_w1.tif")
    ])

    images = []
    for dapi_file in dapi_files[:n]:
        # Parse the filename to extract plate, well, site
        # Format: Week1_22123_B02_s1_w1.tif
        base = dapi_file.replace("_w1.tif", "")
        parts = base.rsplit("_", 2)  # Split from right: [plate, well, site]
        plate = parts[0]             # e.g., "Week1_22123"
        well = parts[1]              # e.g., "B02"
        site = parts[2]              # e.g., "s1"

        # Verify all 3 channels exist for this field
        w2_path = os.path.join(image_dir, f"{base}_w2.tif")
        w4_path = os.path.join(image_dir, f"{base}_w4.tif")
        if not os.path.exists(w2_path) or not os.path.exists(w4_path):
            print(f"Skipping {base}: missing channels")
            continue

        image = load_bbbc021_field(image_dir, plate, well, site)
        images.append(image)
        print(f"Loaded {base}: shape={image.shape}, dtype={image.dtype}")

    print(f"\nLoaded {len(images)} fields of view")
    return images
```

### 12.3 Quick Sanity Check

```python
# Load 5 test images
images = load_bbbc021_first_n("bbbc021_raw/Week1_22123/", n=5)

# Verify dimensions and channel convention
for i, img in enumerate(images):
    assert img.ndim == 3 and img.shape[2] == 3, f"Image {i} wrong shape: {img.shape}"
    print(f"Image {i}: {img.shape}, "
          f"DAPI range [{img[:,:,0].min():.0f}, {img[:,:,0].max():.0f}], "
          f"Tubulin range [{img[:,:,1].min():.0f}, {img[:,:,1].max():.0f}], "
          f"Actin range [{img[:,:,2].min():.0f}, {img[:,:,2].max():.0f}]")
```

This gives you a small, well-understood dataset to validate the entire Phase 1 pipeline end-to-end: upload → segmentation → crop → embed → index → search → refine.

---

## 13. Dependencies

### Python

```
flask>=3.0
flask-cors>=4.0
torch>=2.1
torchvision>=0.16
numpy>=1.24
scipy>=1.11
scikit-image>=0.21
tifffile>=2023.7
cellpose>=4.0            # cpsam model
faiss-gpu>=1.7.4         # or faiss-cpu
Pillow>=10.0
```

### Frontend

```json
{
  "dependencies": {
    "react": "^18.2",
    "react-dom": "^18.2",
    "zustand": "^4.4",
    "axios": "^1.6",
    "tailwindcss": "^3.4"
  }
}
```

### Hardware

Minimum: Any NVIDIA GPU ≥4 GB VRAM (ViT-B), 16 GB RAM. Slow but works.

Recommended: NVIDIA GPU ≥8 GB VRAM (ViT-L), 32 GB RAM, SSD.

Optimal: NVIDIA GPU ≥16 GB VRAM, 64 GB RAM. Enables invariance augmentation with fast throughput.

---

## 14. Future Extensions

**Multi-channel support (>3 channels).** Extend to N channels using combinatorial RGB triplets through DINOv3, averaging the resulting patch grids. For ≤6 channels use all C(N,3) combinations; for >6 use a reference-anchored strategy. The mask enrichment and FAISS search remain identical.

**DINOv2 shape embedding.** Add DINOv2's linear CLS embedding as a complementary shape-focused vector, concatenated with the DINOv3 appearance embedding. Controllable via a Shape vs Appearance slider.

**Separate shape triplet.** Render masks as a 3-channel image (nuclear, cell, neighbor) and pass through DINOv3 for a richer shape representation than the 2-float mask overlap append.

**EWC-FAISS multi-model ensemble.** Combine DINOv3 with SAM or CLIP embeddings using entropy-weighted combination for improved robustness on difficult phenotypes.

**Active learning.** Use accumulated positive/negative examples to train a lightweight linear classifier on the embeddings for automated screening.

**Batch effect correction.** Apply Harmony or ComBat to the embedding space to reduce technical variation across imaging sessions.

**Patch-level similarity visualization.** Show which patches matched between query and result as a heatmap overlay for interpretability.
