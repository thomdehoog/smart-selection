"""
DINOv2 embedding service.

Extracts patch-level features from 3-channel cell crops using a frozen DINOv2 backbone.
The 3 channels map directly to RGB — single forward pass per crop, no channel mixing.

Phase 1: simple mean pooling across all patches for the global embedding.
Phase 2: masked pooling using cell/nuclear masks.
"""

from __future__ import annotations  # Makes all type annotations lazy — avoids torch import at parse time

import numpy as np
from config import DINOV2_MODEL, DINOV2_INPUT_SIZE, DINOV2_NUM_PATCHES, DINOV2_EMBED_DIM

# Torch and torchvision are lazy-imported — only needed when running inference,
# not when using utility functions like normalize_crop or pool_to_global_embedding
torch = None
transforms = None
Image = None


# Module-level model cache — loaded once, reused across all embedding calls
_dinov2_model = None
_device = None

def _ensure_torch():
    """Lazily import torch, torchvision, and PIL when needed for model inference."""
    global torch, transforms, Image
    if torch is None:
        import torch as _torch
        from torchvision import transforms as _transforms
        from PIL import Image as _Image
        torch = _torch
        transforms = _transforms
        Image = _Image


# Preprocessing pipeline — created lazily after torch is imported
_preprocess = None


def _get_preprocess():
    global _preprocess
    if _preprocess is None:
        _ensure_torch()
        _preprocess = transforms.Compose([
            transforms.Resize(
                (DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _preprocess


def get_dinov2_model():
    """Load or return cached DINOv2 model. Uses GPU if available."""
    global _dinov2_model, _device
    _ensure_torch()

    if _dinov2_model is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            _device = torch.device("mps")
        else:
            _device = torch.device("cpu")
        print(f"Loading DINOv2 ({DINOV2_MODEL}) on {_device}...")

        _dinov2_model = torch.hub.load(
            "facebookresearch/dinov2", DINOV2_MODEL, pretrained=True
        )
        _dinov2_model.eval()
        _dinov2_model.to(_device)

        print(f"DINOv2 loaded. Embedding dim: {DINOV2_EMBED_DIM}, "
              f"Patch grid: {DINOV2_NUM_PATCHES}×{DINOV2_NUM_PATCHES}")

    return _dinov2_model, _device


def normalize_crop(crop: np.ndarray) -> np.ndarray:
    """
    Percentile-normalize a 3-channel crop to [0, 1] range.

    This maps the dynamic range of fluorescence images (which can be 16-bit
    with most values concentrated in a narrow range) to the [0, 1] range that
    DINOv2 expects before ImageNet normalization is applied.

    Args:
        crop: (H, W, 3) float32 raw intensities

    Returns:
        (H, W, 3) float32 in [0, 1]
    """
    result = crop.copy()
    nonzero_mask = np.any(crop > 0, axis=2)
    for c in range(3):
        ch = result[:, :, c]
        vals = ch[nonzero_mask] if nonzero_mask.any() else ch.ravel()
        p_low, p_high = np.percentile(vals, [1, 99.5])
        if p_high > p_low:
            result[:, :, c] = np.clip((ch - p_low) / (p_high - p_low), 0, 1)
        else:
            result[:, :, c] = 0.0
    return result


def crop_to_tensor(crop_normalized: np.ndarray) -> torch.Tensor:
    """
    Convert a normalized [0,1] crop to a preprocessed PyTorch tensor.

    Applies resize to DINOv2 input size and ImageNet normalization.

    Args:
        crop_normalized: (H, W, 3) float32 in [0, 1]

    Returns:
        (1, 3, 518, 518) tensor on the model's device
    """
    # Convert to uint8 PIL image for torchvision transforms
    _ensure_torch()
    pil_img = Image.fromarray((crop_normalized * 255).astype(np.uint8))
    tensor = _get_preprocess()(pil_img)
    return tensor


def extract_patch_features(crop: np.ndarray) -> np.ndarray:
    """
    Extract DINOv2 patch-level features from a single 3-channel crop.

    The crop is percentile-normalized, resized to 518×518, ImageNet-normalized,
    and passed through the frozen DINOv2 backbone. Returns the spatial grid
    of patch token features (excluding the CLS token).

    Args:
        crop: (H, W, 3) float32 raw intensity crop

    Returns:
        patch_grid: (37, 37, 768) numpy array of patch features for ViT-B
    """
    model, device = get_dinov2_model()

    # Normalize intensities to [0, 1] and convert to tensor
    crop_norm = normalize_crop(crop)
    tensor = crop_to_tensor(crop_norm).unsqueeze(0).to(device)

    with torch.no_grad():
        # DINOv2 forward_features returns a dict with 'x_norm_patchtokens'
        # Shape: (1, num_patches^2, embed_dim) = (1, 1369, 768) for ViT-B at 518 input
        features = model.forward_features(tensor)
        patch_tokens = features["x_norm_patchtokens"]  # (1, 1369, 768)

    # Reshape to spatial grid
    grid = patch_tokens[0].reshape(
        DINOV2_NUM_PATCHES, DINOV2_NUM_PATCHES, DINOV2_EMBED_DIM
    ).cpu().numpy()

    return grid


def extract_patch_features_batch(crops: list, batch_size: int = 16) -> list:
    """
    Extract patch features for a list of crops in batches.

    Much more efficient than one-by-one because the GPU processes multiple
    crops simultaneously. All crops must already be in memory.

    Args:
        crops: List of (H, W, 3) float32 raw intensity arrays
        batch_size: Number of crops per GPU batch

    Returns:
        grids: List of (37, 37, 768) numpy arrays, one per crop
    """
    model, device = get_dinov2_model()
    grids = []

    for i in range(0, len(crops), batch_size):
        batch_crops = crops[i:i + batch_size]

        # Normalize and convert each crop to a tensor
        tensors = []
        for crop in batch_crops:
            crop_norm = normalize_crop(crop)
            tensors.append(crop_to_tensor(crop_norm))

        # Stack into a batch tensor and move to GPU
        batch_tensor = torch.stack(tensors).to(device)  # (B, 3, 518, 518)

        with torch.no_grad():
            features = model.forward_features(batch_tensor)
            patch_tokens = features["x_norm_patchtokens"]  # (B, 1369, 768)

        # Reshape each to spatial grid and move to CPU
        for j in range(len(batch_crops)):
            grid = patch_tokens[j].reshape(
                DINOV2_NUM_PATCHES, DINOV2_NUM_PATCHES, DINOV2_EMBED_DIM
            ).cpu().numpy()
            grids.append(grid)

        print(f"  Embedded {min(i + batch_size, len(crops))}/{len(crops)} crops")

    return grids


def pool_to_global_embedding(patch_grid: np.ndarray) -> np.ndarray:
    """
    Pool a patch feature grid to a single global embedding vector.

    Phase 1: simple mean over all patches (no mask-based pooling yet).
    Phase 2: will add masked pooling using cell/nuclear masks.

    Args:
        patch_grid: (Ph, Pw, D) patch feature grid

    Returns:
        embedding: (D,) L2-normalized vector
    """
    # Flatten spatial dimensions and average all patch vectors
    flat = patch_grid.reshape(-1, patch_grid.shape[-1])
    embedding = flat.mean(axis=0)

    # L2 normalize — this makes cosine similarity = inner product in FAISS
    norm = np.linalg.norm(embedding)
    if norm > 1e-8:
        embedding = embedding / norm

    return embedding


def embed_all_objects(crops: list, batch_size: int = 16) -> tuple:
    """
    Complete embedding pipeline: extract patch features and pool to global vectors.

    This is the main entry point for the embedding phase. It processes all crops
    through DINOv2 in batches, then pools each to a global embedding for FAISS.

    Args:
        crops: List of (H, W, 3) float32 crops (must all be in memory)
        batch_size: GPU batch size

    Returns:
        global_embeddings: (N, D) numpy array, L2-normalized, ready for FAISS
        patch_grids: List of (37, 37, D) arrays (stored for future mask enrichment)
    """
    print(f"Embedding {len(crops)} crops with DINOv2 ({DINOV2_MODEL})...")

    # Step 1: Extract patch features in batches
    patch_grids = extract_patch_features_batch(crops, batch_size=batch_size)

    # Step 2: Pool each grid to a global embedding
    global_embeddings = []
    for grid in patch_grids:
        emb = pool_to_global_embedding(grid)
        global_embeddings.append(emb)

    global_embeddings = np.stack(global_embeddings).astype(np.float32)
    print(f"Embedding complete. Shape: {global_embeddings.shape}")

    return global_embeddings, patch_grids


def free_dinov2_model():
    """Free the DINOv2 model from GPU memory."""
    global _dinov2_model, _device
    if _dinov2_model is not None:
        del _dinov2_model
        _dinov2_model = None
        _ensure_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("DINOv2 model freed from memory.")
