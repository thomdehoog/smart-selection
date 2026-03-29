from __future__ import annotations
"""
FAISS indexing service.

Builds a similarity search index over the global embeddings and handles
the query-by-example search with positive/negative refinement.

All vectors must be L2-normalized before indexing — this makes inner product
search equivalent to cosine similarity, which is what we want.
"""

import numpy as np
from config import DEFAULT_TOP_K, DEFAULT_NEGATIVE_ALPHA

# Lazy import — faiss segfaults on macOS if imported before torch/cellpose
faiss = None

def _ensure_faiss():
    global faiss
    if faiss is None:
        import faiss as _faiss
        globals()['faiss'] = _faiss


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index over L2-normalized embeddings.

    For Phase 1 (<100K objects), uses IndexFlatIP for exact search.
    Inner product on L2-normalized vectors = cosine similarity.

    Args:
        embeddings: (N, D) float32 array, L2-normalized

    Returns:
        FAISS index ready for search
    """
    _ensure_faiss()
    N, D = embeddings.shape
    print(f"Building FAISS index: {N} vectors, {D} dimensions")

    # Exact search — fast enough for Phase 1 datasets
    index = faiss.IndexFlatIP(D)
    index.add(embeddings)

    print(f"FAISS index built. Total vectors: {index.ntotal}")
    return index


def search(
    index: faiss.Index,
    embeddings: np.ndarray,
    positive_ids: list,
    negative_ids: list = None,
    alpha: float = DEFAULT_NEGATIVE_ALPHA,
    top_k: int = DEFAULT_TOP_K,
) -> tuple:
    """
    Search for objects similar to positive examples, adjusted away from negatives.

    The query vector is computed as:
        query = mean(positive_embeddings) - alpha * mean(negative_embeddings)
    then L2-normalized. This is pure vector arithmetic — no retraining.

    The positive and negative objects are excluded from results.

    Args:
        index: FAISS index
        embeddings: (N, D) all global embeddings
        positive_ids: List of object indices the user selected as interesting
        negative_ids: List of object indices the user rejected (or None)
        alpha: Strength of negative adjustment (0.0 = ignore, 1.0 = full subtraction)
        top_k: Number of results to return

    Returns:
        result_ids: List of object indices, ranked by descending similarity
        scores: List of corresponding similarity scores (cosine similarity)
    """
    negative_ids = negative_ids or []

    # Build query vector from positive examples
    pos_embeddings = embeddings[positive_ids]
    query = pos_embeddings.mean(axis=0)

    # Subtract negative influence
    if len(negative_ids) > 0:
        neg_embeddings = embeddings[negative_ids]
        neg_vec = neg_embeddings.mean(axis=0)
        query = query - alpha * neg_vec

    # L2 normalize the query — required for cosine similarity via inner product
    norm = np.linalg.norm(query)
    if norm > 1e-8:
        query = query / norm
    query = query.reshape(1, -1).astype(np.float32)

    # Search FAISS — request extra results to account for excluded IDs
    exclude = set(positive_ids) | set(negative_ids)
    search_k = top_k + len(exclude) + 10
    scores, indices = index.search(query, search_k)

    # Filter out excluded IDs and invalid indices
    results = [
        (int(idx), float(score))
        for idx, score in zip(indices[0], scores[0])
        if idx >= 0 and idx not in exclude
    ]

    result_ids = [r[0] for r in results[:top_k]]
    result_scores = [r[1] for r in results[:top_k]]

    return result_ids, result_scores


def search_dissimilar(
    index: faiss.Index,
    embeddings: np.ndarray,
    positive_ids: list,
    top_k: int = DEFAULT_TOP_K,
) -> tuple:
    """
    Search for the most dissimilar objects to the positive examples.

    Queries FAISS for ALL objects, then returns the bottom-k by similarity.
    """
    _ensure_faiss()

    pos_embeddings = embeddings[positive_ids]
    query = pos_embeddings.mean(axis=0)
    norm = np.linalg.norm(query)
    if norm > 1e-8:
        query = query / norm
    query = query.reshape(1, -1).astype(np.float32)

    # Search all objects
    n_total = index.ntotal
    scores, indices = index.search(query, n_total)

    # Exclude positives, take the bottom-k (least similar)
    exclude = set(positive_ids)
    results = [
        (int(idx), float(score))
        for idx, score in zip(indices[0], scores[0])
        if idx >= 0 and idx not in exclude
    ]

    # Bottom-k: least similar first
    bottom = results[-top_k:]
    bottom.reverse()

    result_ids = [r[0] for r in bottom]
    result_scores = [r[1] for r in bottom]

    return result_ids, result_scores
