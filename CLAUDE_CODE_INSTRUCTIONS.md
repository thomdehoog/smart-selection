# Claude Code Instructions — Microscopy Semantic Search

## Context

This project has been designed and scaffolded in Claude.ai. The backend code, frontend, mock server, and 65 unit/integration tests are all written and passing. However, nothing has run against real microscopy data with real GPU models yet. Your job is to get it working end-to-end on this machine.

The project spec is in `microscopy_semantic_search_spec_v4.md`. The full codebase is in the `microscopy-search/` directory.

## What's been validated so far

All 30 backend unit tests pass. These cover image I/O (tifffile axis detection, Z-stack projection, 3-channel validation), crop extraction (sizing, padding, square enforcement), segmentation metadata extraction (centroids, bounding boxes, areas from synthetic masks), embedding utilities (percentile normalization, L2-normalized pooling), FAISS search (index construction, top-k retrieval, positive/negative exclusion, ranking correctness), and dataset state management.

All 35 frontend logic tests pass. These cover selection toggle behavior, accept/reject state transitions, threshold filtering, score color mapping, hit-test for canvas clicks, API request/response shape contracts, and step navigation guards.

The mock server and integration test have been tested together — all 9 integration checks pass against the mock.

What has NOT been tested: Cellpose-SAM on real images, DINOv2 on real crops, the full pipeline end-to-end, and the Flask server under real load.

## Step-by-step instructions

### Step 1: Install dependencies

Run `pip install -r backend/requirements.txt`. This needs torch, torchvision, cellpose>=4.0, faiss-cpu (or faiss-gpu), flask, flask-cors, tifffile, numpy, scipy, scikit-image, and Pillow.

Verify everything is importable:

```python
python -c "
import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import cellpose; print(f'Cellpose {cellpose.__version__}')
import faiss; print(f'FAISS OK')
"
```

### Step 2: Download BBBC021 test data

```bash
wget https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22123.zip
unzip BBBC021_v1_images_Week1_22123.zip -d bbbc021_raw/
```

This is about 750 MB. Only one plate ZIP is needed for testing.

### Step 3: Run existing unit tests to confirm nothing is broken

```bash
cd backend && python -m pytest tests.py -v
```

All 30 should pass. If any fail, fix them before continuing.

### Step 4: Test image loading with real BBBC021 data

```python
from services.image_io import load_bbbc021_first_n
images = load_bbbc021_first_n('bbbc021_raw/Week1_22123/', n=2)
for i, img in enumerate(images):
    print(f'Image {i}: shape={img.shape}, dtype={img.dtype}, range=[{img.min():.0f}, {img.max():.0f}]')
```

If this fails, the BBBC021 filename parsing probably doesn't match the actual files. List the directory, check the naming pattern, and fix the `rsplit` logic in `image_io.py`. The expected pattern is `Week1_22123_B02_s1_w1.tif` where `_w1` is the channel suffix.

### Step 5: Test Cellpose-SAM segmentation on a real image

This is the highest-risk step because cpsam is new (Cellpose 4) and the API is different from older versions.

```python
from services.image_io import load_bbbc021_first_n
from services.segmentation import segment_cells
images = load_bbbc021_first_n('bbbc021_raw/Week1_22123/', n=1)
result = segment_cells(images[0], gpu=True)
print(f'Found {result["num_cells"]} cells')
print(f'Mask shape: {result["masks"].shape}')
for obj in result["objects"][:3]:
    print(f'  Cell {obj["object_id"]}: centroid={obj["centroid"]}, area={obj["area"]}')
```

Likely failure modes and how to fix them:

If Cellpose version is below 4.0, the cpsam model won't exist. Run `pip install cellpose --upgrade`.

If the model download fails (behind firewall), download manually from `https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam` and put it in `~/.cellpose/models/`.

If zero cells are detected, the auto-diameter detection might not work for BBBC021. Add `diameter=30` to the `model.eval()` call in `segmentation.py`.

If the channel axis is wrong (you get weird results), the code transposes from (H,W,3) to (3,H,W) and passes `channel_axis=0`. Try passing (H,W,3) with `channel_axis=2` instead — cpsam is channel-order invariant either way, it just needs to know which axis is channels.

If CUDA runs out of memory, cpsam's ViT-L encoder needs about 3-4 GB VRAM. Try `gpu=False` as a fallback.

### Step 6: Test DINOv2 embedding on a crop

```python
import numpy as np
from services.embedding import extract_patch_features, pool_to_global_embedding
crop = np.random.uniform(0, 65535, (256, 256, 3)).astype(np.float32)
grid = extract_patch_features(crop)
print(f'Patch grid: {grid.shape}')   # Expect (37, 37, 768)
emb = pool_to_global_embedding(grid)
print(f'Embedding: {emb.shape}')     # Expect (768,)
print(f'L2 norm: {np.linalg.norm(emb):.4f}')  # Expect ~1.0
```

The most likely failure here is the `forward_features` return value having different key names. If `features["x_norm_patchtokens"]` throws a KeyError, print `features.keys()` and update the key name in `embedding.py`. Possible alternatives are `"x_patchtokens"` or `"patchtokens"`.

### Step 7: Run the full pipeline end-to-end

```python
from services.image_io import load_bbbc021_first_n
from models.dataset import create_dataset
from pipeline import run_pipeline

state = create_dataset('test')
state.images = load_bbbc021_first_n('bbbc021_raw/Week1_22123/', n=5)
run_pipeline(state)  # This takes 2-5 minutes

print(f'Objects: {state.num_objects()}')
print(f'Embeddings: {state.global_embeddings.shape}')
print(f'FAISS index: {state.faiss_index.ntotal}')
```

The pipeline runs sequentially: segment all images → extract all crops into memory → free Cellpose from GPU → load DINOv2 → embed all crops in batches → build FAISS index. The GPU memory swap between Cellpose and DINOv2 is handled by `free_cellpose_model()`. If you still get OOM, reduce the batch size in `embedding.py` from 16 to 4 or 8.

### Step 8: Run the integration test against the real backend

```bash
python app.py &
sleep 5
python integration_test.py bbbc021_raw/Week1_22123/
```

This calls every API endpoint in sequence (upload → process → poll → image → objects → crops → search → search-with-rejection → export) and validates all response shapes. If all 9 checks pass, the backend is fully working.

### Step 9: Verify search quality

After the pipeline completes, run a manual search to check whether the DINOv2 embeddings capture meaningful cell phenotypes:

```python
from services.indexing import search
pos = [1, 2, 3]  # Pick any 3 cell IDs
result_ids, scores = search(state.faiss_index, state.global_embeddings, pos, top_k=10)
for rid, score in zip(result_ids, scores):
    obj = state.objects[rid]
    print(f'Cell {obj["object_id"]} (image {obj["image_index"]}): {score:.3f}')
```

If the top scores are all clustered near zero with no separation, the embeddings might not be capturing enough signal. Check whether the percentile normalization is appropriate — BBBC021 is 16-bit, so the raw values can be in the 0-65535 range. The `normalize_crop` function clips to the 1st-99.5th percentile, which should handle this, but verify the normalized crops look reasonable when converted to uint8 and saved as PNGs.

### Step 10: Connect the React frontend

The frontend is `frontend/MicroscopySearch.jsx`. It's a single React component that connects to `http://localhost:5000`. The simplest way to run it is to create a React app and drop it in:

```bash
npx create-react-app microscopy-ui
cp frontend/MicroscopySearch.jsx microscopy-ui/src/App.jsx
cd microscopy-ui && npm start
```

Make sure the Flask backend is running on port 5000 with CORS enabled (it is by default). The frontend should connect automatically.

## Files you may need to modify

`services/segmentation.py` — Most likely file to need fixes. The Cellpose-SAM API is new and might behave differently than documented. The key function is `segment_cells()`. Check the transpose, `channel_axis`, `diameter`, and return value unpacking.

`services/embedding.py` — Second most likely. The DINOv2 `forward_features()` return format might have different key names. Check `extract_patch_features()` and update the dict key if needed.

`services/image_io.py` — The BBBC021 filename parser might need adjustment. Check `load_bbbc021_first_n()` and the `rsplit` logic against actual filenames in your directory.

`config.py` — If you want to change model size (e.g., `dinov2_vits14` for faster testing or `dinov2_vitl14` for better quality), update `DINOV2_MODEL`, `DINOV2_EMBED_DIM`, and `DINOV2_NUM_PATCHES` together.

## What comes after this works

Once the full pipeline runs and search returns meaningful results, the next development phases from the spec are:

Phase 2 adds nuclear segmentation (second cpsam pass on channel 0 alone), mask-enriched patch features (two extra dimensions appended to each patch vector indicating nuclear and cell overlap), masked pooling (averaging only patches inside the cell mask instead of all patches), the 4×/8× context window toggle, and the export endpoint with per-channel intensity measurements.

Phase 3 adds rotation and scale invariance (test-time augmentation with 8 orientations and 3 scales), regular grid and random sampling modes as alternatives to segmentation, and persistent sessions so state survives server restarts.

The spec document has complete details on all of these including code sketches.
