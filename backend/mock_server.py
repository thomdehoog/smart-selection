"""
Mock backend server for frontend development.

Returns synthetic but correctly-shaped responses for every API endpoint.
Thumbnails are generated as colored rectangles with fake nuclei.
The search endpoint uses real cosine similarity on random embeddings,
so the accept/reject/recompute loop actually works.

Usage:
    python mock_server.py
    # Frontend connects to http://localhost:5000
"""

import io
import base64
import random
import time
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw

app = Flask(__name__)
CORS(app)

# ─── Synthetic dataset ───────────────────────────────────────────────────────

NUM_IMAGES = 5
CELLS_PER_IMAGE = 30
IMAGE_W, IMAGE_H = 1024, 1024
EMBED_DIM = 768

# Generate cell metadata for all images
ALL_OBJECTS = []
for img_idx in range(NUM_IMAGES):
    for cell_i in range(CELLS_PER_IMAGE):
        obj_id = img_idx * CELLS_PER_IMAGE + cell_i + 1
        cx = random.randint(60, IMAGE_W - 60)
        cy = random.randint(60, IMAGE_H - 60)
        w = random.randint(20, 50)
        h = random.randint(20, 50)
        ALL_OBJECTS.append({
            "object_id": obj_id,
            "image_index": img_idx,
            "centroid": [float(cx), float(cy)],
            "bbox": [cx - w, cy - h, cx + w, cy + h],
            "area": w * h * 4,
        })

# Random unit-vector embeddings — search uses real cosine similarity on these
EMBEDDINGS = np.random.randn(len(ALL_OBJECTS), EMBED_DIM).astype(np.float32)
EMBEDDINGS /= np.linalg.norm(EMBEDDINGS, axis=1, keepdims=True) + 1e-8

# Pipeline simulation state
pipeline_state = {"status": "idle", "progress": 0.0, "message": "", "started": 0}


# ─── Thumbnail generators ────────────────────────────────────────────────────

def make_cell_thumbnail(obj_id, size=128):
    """Colored rectangle with a darker ellipse (fake nucleus) and ID label."""
    r = (obj_id * 47) % 200 + 55
    g = (obj_id * 83) % 200 + 55
    b = (obj_id * 131) % 200 + 55
    img = Image.new("RGB", (size, size), (r, g, b))
    draw = ImageDraw.Draw(img)
    cx, cy, nr = size // 2, size // 2, size // 5
    draw.ellipse([cx - nr, cy - nr, cx + nr, cy + nr], fill=(r // 2, g // 2, b // 2))
    draw.text((4, 4), f"#{obj_id}", fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def make_overview_image(img_idx, size=1024):
    """Dark image with colored dots at cell locations."""
    img = Image.new("RGB", (size, size), (10, 10, 20))
    draw = ImageDraw.Draw(img)
    for obj in ALL_OBJECTS:
        if obj["image_index"] != img_idx:
            continue
        cx, cy = int(obj["centroid"][0]), int(obj["centroid"][1])
        rad = random.randint(8, 18)
        color = ((obj["object_id"] * 47) % 200 + 55,
                 (obj["object_id"] * 83) % 200 + 55,
                 (obj["object_id"] * 131) % 100 + 30)
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


# Pre-generate overview images (avoid regenerating on every request)
IMAGE_CACHE = {i: make_overview_image(i) for i in range(NUM_IMAGES)}


# ─── API Routes ──────────────────────────────────────────────────────────────

@app.route("/api/upload_bbbc021", methods=["POST"])
def upload():
    return jsonify({
        "status": "ok",
        "dataset_id": "mock-001",
        "num_images": NUM_IMAGES,
        "image_dimensions": [IMAGE_W, IMAGE_H],
        "num_channels": 3,
        "channel_names": ["DAPI", "Tubulin", "Actin"],
    })


@app.route("/api/segment_and_embed", methods=["POST"])
def segment():
    pipeline_state["status"] = "processing"
    pipeline_state["progress"] = 0.0
    pipeline_state["started"] = time.time()
    return jsonify({"status": "processing", "dataset_id": "mock-001", "num_images": NUM_IMAGES})


@app.route("/api/status", methods=["GET"])
def status():
    # Simulate a 5-second pipeline with phase transitions
    if pipeline_state["status"] == "processing":
        elapsed = time.time() - pipeline_state["started"]
        p = min(elapsed / 5.0, 1.0)
        pipeline_state["progress"] = p
        if p < 0.3:
            pipeline_state["message"] = "Segmenting cells..."
        elif p < 0.7:
            pipeline_state["message"] = f"Embedding crops ({int(p * 150)}/150)..."
        elif p < 1.0:
            pipeline_state["message"] = "Building FAISS index..."
        else:
            pipeline_state["status"] = "complete"
            pipeline_state["message"] = "Ready for search!"

    return jsonify({
        "status": pipeline_state["status"],
        "progress": pipeline_state["progress"],
        "phase": "mock",
        "message": pipeline_state["message"],
        "num_objects": len(ALL_OBJECTS),
        "error": None,
    })


@app.route("/api/image/<int:img_idx>", methods=["GET"])
def get_image(img_idx):
    if img_idx >= NUM_IMAGES:
        return jsonify({"error": "Not found"}), 404
    return jsonify({
        "image_index": img_idx,
        "width": IMAGE_W,
        "height": IMAGE_H,
        "thumbnail_base64": IMAGE_CACHE[img_idx],
    })


@app.route("/api/objects", methods=["GET"])
def get_objects():
    img_idx = request.args.get("image_index", type=int)
    objs = ALL_OBJECTS if img_idx is None else [o for o in ALL_OBJECTS if o["image_index"] == img_idx]
    return jsonify({"num_objects": len(objs), "objects": objs})


@app.route("/api/crops", methods=["POST"])
def get_crops():
    ids = request.json.get("object_ids", [])
    crops = []
    for oid in ids:
        obj = next((o for o in ALL_OBJECTS if o["object_id"] == oid), None)
        if obj:
            crops.append({
                "object_id": oid,
                "image_index": obj["image_index"],
                "centroid": obj["centroid"],
                "area": obj["area"],
                "thumbnail_base64": make_cell_thumbnail(oid),
            })
    return jsonify({"crops": crops})


@app.route("/api/search", methods=["POST"])
def search():
    data = request.json
    pos_ids = data.get("positive_ids", [])
    neg_ids = data.get("negative_ids", [])
    top_k = data.get("top_k", 50)
    alpha = data.get("negative_alpha", 0.4)

    # Map object IDs to array indices (IDs are 1-based)
    pos_idx = [oid - 1 for oid in pos_ids if 1 <= oid <= len(ALL_OBJECTS)]
    neg_idx = [oid - 1 for oid in neg_ids if 1 <= oid <= len(ALL_OBJECTS)]

    if not pos_idx:
        return jsonify({"error": "No valid positive IDs"}), 400

    # Real cosine similarity search on fake embeddings
    query = EMBEDDINGS[pos_idx].mean(axis=0)
    if neg_idx:
        query -= alpha * EMBEDDINGS[neg_idx].mean(axis=0)
    query /= np.linalg.norm(query) + 1e-8

    scores = EMBEDDINGS @ query
    exclude = set(pos_ids) | set(neg_ids)
    ranked = np.argsort(-scores)

    results = []
    for idx in ranked:
        oid = ALL_OBJECTS[idx]["object_id"]
        if oid in exclude:
            continue
        results.append({
            "object_id": oid,
            "image_index": ALL_OBJECTS[idx]["image_index"],
            "similarity_score": round(float(scores[idx]), 4),
            "centroid": ALL_OBJECTS[idx]["centroid"],
            "area": ALL_OBJECTS[idx]["area"],
            "thumbnail_base64": make_cell_thumbnail(oid),
        })
        if len(results) >= top_k:
            break

    return jsonify({
        "results": results,
        "num_total_objects": len(ALL_OBJECTS),
        "num_positive": len(pos_idx),
        "num_negative": len(neg_idx),
    })


@app.route("/api/export", methods=["POST"])
def export():
    ids = request.json.get("accepted_ids", [])
    return jsonify({
        "exported_objects": [
            {"object_id": oid, "image_index": o["image_index"],
             "centroid_x": o["centroid"][0], "centroid_y": o["centroid"][1], "area": o["area"]}
            for oid in ids for o in ALL_OBJECTS if o["object_id"] == oid
        ]
    })


if __name__ == "__main__":
    print("Mock Backend — Microscopy Semantic Search")
    print(f"Dataset: {NUM_IMAGES} images, {len(ALL_OBJECTS)} cells, {EMBED_DIM}d embeddings")
    print("Pipeline simulates 5s processing. Search uses real cosine similarity.")
    print("http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
