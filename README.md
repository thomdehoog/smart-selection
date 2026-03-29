# Microscopy Semantic Search Tool

Interactive similarity search for microscopy images. Select a few cells you find interesting, and the system finds more like them using DINOv2 embeddings and FAISS nearest-neighbor search. Refine results by accepting or rejecting candidates — no training required.

## Architecture

```
React Frontend (localhost:3000)     Flask Backend (localhost:5000)
┌─────────────────────────┐        ┌──────────────────────────────┐
│ Step 1: Load data        │───────▶│ Image loading (tifffile)     │
│ Step 2: Select cells     │◀──────▶│ Cellpose-SAM segmentation    │
│ Step 3: Review gallery   │───────▶│ DINOv2 patch embeddings      │
│ Step 4: Search & refine  │◀──────▶│ FAISS similarity search      │
└─────────────────────────┘        └──────────────────────────────┘
```

## Quick Start

### 1. Download test data

```bash
wget https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22123.zip
unzip BBBC021_v1_images_Week1_22123.zip -d bbbc021_raw/
```

### 2. Install and run the backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Server starts on http://localhost:5000. Model weights download automatically on first run.

### 3. Run the frontend

The frontend is a single React component (MicroscopySearch.jsx). For development without GPU models, use the mock backend:

```bash
cd backend
python mock_server.py   # Returns synthetic data on all endpoints
```

### 4. Run tests

```bash
cd backend && python -m pytest tests.py -v              # 30 backend tests
cd frontend && python -m pytest tests_frontend_logic.py -v  # 35 frontend logic tests
cd backend && python integration_test.py bbbc021_raw/Week1_22123/  # End-to-end smoke test
```

## Project Structure

```
microscopy-search/
├── backend/
│   ├── app.py                    # Flask server, all API routes
│   ├── config.py                 # Configuration defaults
│   ├── pipeline.py               # Segmentation → embedding → indexing orchestration
│   ├── mock_server.py            # Mock backend for frontend dev
│   ├── tests.py                  # Backend unit tests (30 tests)
│   ├── integration_test.py       # End-to-end API smoke test
│   ├── requirements.txt
│   ├── services/
│   │   ├── image_io.py           # TIFF loading, BBBC021 format, thumbnails
│   │   ├── segmentation.py       # Cellpose-SAM wrapper
│   │   ├── crop_extraction.py    # Context-aware crop extraction
│   │   ├── embedding.py          # DINOv2 patch features + pooling
│   │   └── indexing.py           # FAISS index + search
│   └── models/
│       └── dataset.py            # In-memory dataset state
├── frontend/
│   ├── MicroscopySearch.jsx      # Complete React frontend
│   └── tests_frontend_logic.py   # Frontend logic tests (35 tests)
└── README.md
```

## API Reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | /api/upload_bbbc021 | Load images from disk |
| POST | /api/segment_and_embed | Run pipeline (async) |
| GET | /api/status | Poll progress |
| GET | /api/image/\<index\> | Get image thumbnail |
| GET | /api/objects | Get cell metadata |
| POST | /api/crops | Get cell thumbnails |
| POST | /api/search | Similarity search |
| POST | /api/export | Export results |
