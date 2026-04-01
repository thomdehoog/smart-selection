"""
Configuration defaults for the microscopy semantic search tool.
All settings for Phase 1 are hardcoded here — no user configuration needed yet.
"""

# DINOv2 model configuration
DINOV2_MODEL = "dinov2_vitb14"  # ViT-B/14 — good balance of speed and quality
DINOV2_EMBED_DIM = 768          # Embedding dimension for ViT-B
DINOV2_PATCH_SIZE = 14          # Patch size in pixels
DINOV2_INPUT_SIZE = 518         # Input resolution (must be divisible by patch_size: 518/14=37)
DINOV2_NUM_PATCHES = 37         # 518 / 14 = 37 patches per side

# Segmentation
CELLPOSE_MODEL = "cyto3"        # Phase 1: cell segmentation only (not cpsam yet for compatibility)
CELLPOSE_GPU = True

# Crop extraction
CONTEXT_WINDOW_MULTIPLIER = 1.5  # 1.5× bounding box — tight crop with some context
CROP_MODE = "single_cell"        # "single_cell" = mask out non-cell pixels; "neighborhood" = full context
SIZE_INVARIANT = True            # True = resize all crops to same input (ignore size); False = fixed crop window (size matters)
ROTATION_INVARIANT = True        # True = align major axis before embedding; False = keep original orientation

# FAISS
FAISS_USE_GPU = False            # CPU is fine for <100K objects in Phase 1

# Image normalization
PERCENTILE_LOW = 1.0
PERCENTILE_HIGH = 99.5

# Search defaults
DEFAULT_TOP_K = 200

# Server
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5050
UPLOAD_DIR = "uploads"
PROJECTS_DIR = "projects"
