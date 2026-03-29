"""
Flask application — main entry point for the microscopy semantic search backend.

Phase 1 MVP: all routes in a single file for simplicity.
Phase 2+: split into separate route modules under routes/.
"""

import os
import uuid
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from config import FLASK_HOST, FLASK_PORT, UPLOAD_DIR
from models.dataset import create_dataset, get_dataset, DatasetState
from services.image_io import load_bbbc021_first_n, make_thumbnail
from services.indexing import search, search_dissimilar
from pipeline import run_pipeline_async

import io
import base64
from PIL import Image

app = Flask(__name__)
CORS(app)  # Allow React frontend on localhost:3000 to call Flask on :5000

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ─── Upload & Configure ─────────────────────────────────────────────────────

@app.route("/api/upload_bbbc021", methods=["POST"])
def upload_bbbc021():
    """
    Load images from a BBBC021 plate directory on disk.

    Phase 1 shortcut: instead of actual file upload, point to a local directory
    containing BBBC021 TIFF files. This avoids multipart upload complexity.

    Request JSON:
        image_dir: str — path to the extracted BBBC021 plate directory
        n: int — number of fields of view to load (default: 5)
        channel_names: optional list of 3 strings (default: ["DAPI", "Tubulin", "Actin"])
    """
    data = request.json
    image_dir = data.get("image_dir")
    n = data.get("n", 5)
    channel_names = data.get("channel_names", ["DAPI", "Tubulin", "Actin"])

    if not image_dir or not os.path.isdir(image_dir):
        return jsonify({"error": f"Invalid image directory: {image_dir}"}), 400

    # Create a new dataset
    dataset_id = str(uuid.uuid4())[:8]
    state = create_dataset(dataset_id)
    state.channel_names = channel_names

    # Load images into memory
    print(f"\nLoading BBBC021 images from {image_dir} (n={n})...")
    images = load_bbbc021_first_n(image_dir, n=n)

    if not images:
        return jsonify({"error": "No valid 3-channel images found"}), 400

    state.images = images

    return jsonify({
        "status": "ok",
        "dataset_id": dataset_id,
        "num_images": len(images),
        "image_dimensions": list(images[0].shape[:2]),
        "num_channels": 3,
        "channel_names": channel_names,
    })


# ─── Segment & Embed ────────────────────────────────────────────────────────

@app.route("/api/segment_and_embed", methods=["POST"])
def segment_and_embed():
    """
    Trigger the full pipeline: segmentation → cropping → embedding → indexing.

    Runs asynchronously in a background thread. Poll /api/status for progress.
    """
    state = get_dataset()
    if state is None:
        return jsonify({"error": "No dataset loaded. Call /api/upload_bbbc021 first."}), 400

    if state.processing:
        return jsonify({"error": "Pipeline already running."}), 409

    if not state.images:
        return jsonify({"error": "No images in dataset."}), 400

    # Accept crop_mode from request
    data = request.json or {}
    crop_mode = data.get("crop_mode", "single_cell")
    state.crop_mode = crop_mode

    # Launch pipeline in background thread
    run_pipeline_async(state)

    return jsonify({
        "status": "processing",
        "dataset_id": state.dataset_id,
        "num_images": state.num_images(),
    })


@app.route("/api/status", methods=["GET"])
def get_status():
    """Poll pipeline progress. Returns current phase, progress (0–1), and message."""
    state = get_dataset()
    if state is None:
        return jsonify({"status": "no_dataset"})

    return jsonify({
        "status": "processing" if state.processing else (
            "error" if state.error else "complete" if state.faiss_index else "idle"
        ),
        "progress": state.progress,
        "phase": state.progress_phase,
        "message": state.progress_message,
        "num_objects": state.num_objects(),
        "error": state.error,
    })


# ─── Mask Data ─────────────────────────────────────────────────────────────

@app.route("/api/mask/<int:image_index>", methods=["GET"])
def get_mask(image_index: int):
    """
    Get the segmentation mask as a base64-encoded PNG with cell IDs in pixels.
    ID is encoded as: R = id & 0xFF, G = (id >> 8) & 0xFF, B = 0.
    """
    state = get_dataset()
    if state is None or image_index >= len(state.cell_masks):
        return jsonify({"error": "Mask not found"}), 404

    masks = state.cell_masks[image_index]

    # Build a mapping from original mask labels to global object IDs
    label_to_id = {}
    for obj in state.objects:
        if obj["image_index"] == image_index:
            label_to_id[obj.get("mask_label", obj["object_id"])] = obj["object_id"]

    # Encode mask: remap labels to global IDs, then encode in RGB
    h, w = masks.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, obj_id in label_to_id.items():
        cell_pixels = masks == label
        rgb[cell_pixels, 0] = obj_id & 0xFF
        rgb[cell_pixels, 1] = (obj_id >> 8) & 0xFF

    pil_img = Image.fromarray(rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({
        "image_index": image_index,
        "mask_base64": f"data:image/png;base64,{b64}",
    })


# ─── Image & Object Data ────────────────────────────────────────────────────

@app.route("/api/image/<int:image_index>", methods=["GET"])
def get_image(image_index: int):
    """
    Get a composite RGB thumbnail of an image for display in the viewer.

    Returns a base64-encoded PNG. The image is downsampled for web display.
    """
    state = get_dataset()
    if state is None or image_index >= state.num_images():
        return jsonify({"error": "Image not found"}), 404

    image = state.images[image_index]
    thumb = make_thumbnail(image, size=1024)  # Downsample for web

    # Encode as base64 PNG
    pil_img = Image.fromarray(thumb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({
        "image_index": image_index,
        "width": image.shape[1],
        "height": image.shape[0],
        "thumbnail_base64": f"data:image/png;base64,{b64}",
    })


@app.route("/api/objects", methods=["GET"])
def get_objects():
    """
    Get metadata for all detected objects (or filter by image_index).

    Used by the frontend to render clickable cell regions and hover highlights.
    """
    state = get_dataset()
    if state is None:
        return jsonify({"error": "No dataset loaded"}), 400

    image_index = request.args.get("image_index", type=int)

    objects = state.objects
    if image_index is not None:
        objects = [o for o in objects if o["image_index"] == image_index]

    return jsonify({
        "num_objects": len(objects),
        "objects": objects,
    })


# ─── Gallery ─────────────────────────────────────────────────────────────────

@app.route("/api/crops", methods=["POST"])
def get_crops():
    """
    Get thumbnail crops for a list of object IDs.

    Used by the gallery view to display selected cells and search results.
    """
    state = get_dataset()
    if state is None:
        return jsonify({"error": "No dataset loaded"}), 400

    data = request.json
    object_ids = data.get("object_ids", [])

    crops_data = []
    for obj_id in object_ids:
        idx = state.get_object_index(obj_id)
        if idx is not None:
            obj = state.objects[idx]
            crops_data.append({
                "object_id": obj_id,
                "image_index": obj["image_index"],
                "centroid": obj["centroid"],
                "area": obj["area"],
                "thumbnail_base64": state.thumbnails[idx],
            })

    return jsonify({"crops": crops_data})


# ─── Search & Refinement ────────────────────────────────────────────────────

@app.route("/api/search", methods=["POST"])
def search_similar():
    """
    Search for objects similar to positive examples, adjusted away from negatives.

    This is the core of the interactive refinement loop. The frontend sends
    the current positive and negative sets, and receives ranked results.

    Request JSON:
        positive_ids: list of object IDs the user selected as interesting
        negative_ids: list of object IDs the user rejected (optional)
        top_k: number of results to return (default: 50)
        negative_alpha: strength of negative adjustment (default: 0.4)
    """
    state = get_dataset()
    if state is None or state.faiss_index is None:
        return jsonify({"error": "No index available. Run pipeline first."}), 400

    data = request.json
    positive_ids = data.get("positive_ids", [])
    negative_ids = data.get("negative_ids", [])
    top_k = data.get("top_k", 50)
    alpha = data.get("negative_alpha", 0.4)

    if not positive_ids:
        return jsonify({"error": "At least one positive example required."}), 400

    # Convert object IDs to flat indices into the embeddings array
    pos_indices = [state.get_object_index(oid) for oid in positive_ids]
    neg_indices = [state.get_object_index(oid) for oid in negative_ids]
    pos_indices = [i for i in pos_indices if i is not None]
    neg_indices = [i for i in neg_indices if i is not None]

    if not pos_indices:
        return jsonify({"error": "None of the positive IDs were found."}), 400

    # Run FAISS search
    result_indices, result_scores = search(
        state.faiss_index,
        state.global_embeddings,
        pos_indices,
        neg_indices,
        alpha=alpha,
        top_k=top_k,
    )

    # Convert flat indices back to object IDs and include thumbnails
    results = []
    for idx, score in zip(result_indices, result_scores):
        obj = state.objects[idx]
        results.append({
            "object_id": obj["object_id"],
            "image_index": obj["image_index"],
            "similarity_score": round(score, 4),
            "centroid": obj["centroid"],
            "area": obj["area"],
            "thumbnail_base64": state.thumbnails[idx],
        })

    return jsonify({
        "results": results,
        "num_total_objects": state.num_objects(),
        "num_positive": len(pos_indices),
        "num_negative": len(neg_indices),
    })


@app.route("/api/search_dissimilar", methods=["POST"])
def search_dissimilar_endpoint():
    """Search for the most dissimilar objects to the positive examples."""
    state = get_dataset()
    if state is None or state.faiss_index is None:
        return jsonify({"error": "No index available."}), 400

    data = request.json
    positive_ids = data.get("positive_ids", [])
    top_k = data.get("top_k", 28)

    if not positive_ids:
        return jsonify({"error": "At least one positive example required."}), 400

    pos_indices = [state.get_object_index(oid) for oid in positive_ids]
    pos_indices = [i for i in pos_indices if i is not None]

    result_indices, result_scores = search_dissimilar(
        state.faiss_index, state.global_embeddings, pos_indices, top_k=top_k,
    )

    results = []
    for idx, score in zip(result_indices, result_scores):
        obj = state.objects[idx]
        results.append({
            "object_id": obj["object_id"],
            "image_index": obj["image_index"],
            "similarity_score": round(score, 4),
            "thumbnail_base64": state.thumbnails[idx],
        })

    return jsonify({"results": results})


# ─── Export ──────────────────────────────────────────────────────────────────

@app.route("/api/export", methods=["POST"])
def export_results():
    """
    Export accepted objects as JSON.

    Returns per-object metadata including centroid, bounding box, area,
    and mean intensity per channel within the bounding box.
    """
    state = get_dataset()
    if state is None:
        return jsonify({"error": "No dataset loaded"}), 400

    data = request.json
    accepted_ids = data.get("accepted_ids", [])

    export_data = []
    for obj_id in accepted_ids:
        idx = state.get_object_index(obj_id)
        if idx is not None:
            obj = state.objects[idx]
            crop = state.crops[idx]

            # Compute mean intensity per channel across the crop
            mean_intensities = [
                float(crop[:, :, c].mean()) for c in range(3)
            ]

            export_data.append({
                "object_id": obj_id,
                "image_index": obj["image_index"],
                "centroid_x": obj["centroid"][0],
                "centroid_y": obj["centroid"][1],
                "bbox_x_min": obj["bbox"][0],
                "bbox_y_min": obj["bbox"][1],
                "bbox_x_max": obj["bbox"][2],
                "bbox_y_max": obj["bbox"][3],
                "area": obj["area"],
                "mean_intensity_ch1": mean_intensities[0],
                "mean_intensity_ch2": mean_intensities[1],
                "mean_intensity_ch3": mean_intensities[2],
            })

    return jsonify({"exported_objects": export_data})


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Microscopy Semantic Search Tool — Phase 1 MVP")
    print("=" * 60)
    print(f"Starting Flask server on {FLASK_HOST}:{FLASK_PORT}")
    print(f"CORS enabled for frontend development")
    print()
    print("Workflow:")
    print("  1. POST /api/upload_bbbc021  — load test images")
    print("  2. POST /api/segment_and_embed — run pipeline")
    print("  3. GET  /api/status          — poll progress")
    print("  4. GET  /api/objects          — get cell metadata")
    print("  5. POST /api/crops            — get cell thumbnails")
    print("  6. POST /api/search           — find similar cells")
    print()

    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)
