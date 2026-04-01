from __future__ import annotations
"""
FAISS indexing service.

Builds a similarity search index over the global embeddings and handles
the query-by-example search with positive/negative refinement.

All vectors must be L2-normalized before indexing — this makes inner product
search equivalent to cosine similarity, which is what we want.

The search function uses variance-weighted similarity: dimensions where
positive exemplars agree (low variance) are upweighted, and dimensions where
positives and negatives diverge are further emphasized. This lets the system
automatically learn which visual features (shape, texture, intensity) the
user cares about from just a few examples.
"""

import numpy as np
from config import DEFAULT_TOP_K

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


def _compute_dimension_weights(pos_embeddings, neg_embeddings=None):
    """
    Compute per-dimension importance weights from exemplar embeddings.

    The weight for each dimension reflects how useful it is for distinguishing
    what the user is looking for. Two signals are combined:

    1. Positive consistency: dimensions where positives agree (low variance)
       are likely encoding the feature the user cares about. Weight is
       inversely proportional to variance among positives.

    2. Discriminative power (if negatives exist): dimensions where the positive
       and negative means are far apart are especially informative. This signal
       is added on top of the consistency signal.

    The result is a (D,) weight vector, normalized to sum to D (so that the
    average weight is 1.0 — preserving the overall scale of similarity scores).
    """
    D = pos_embeddings.shape[1]

    if pos_embeddings.shape[0] == 1:
        # Single positive: no variance info, use uniform weights
        return np.ones(D, dtype=np.float32)

    # Signal 1: inverse variance among positives (consistency)
    pos_var = pos_embeddings.var(axis=0) + 1e-8
    consistency = 1.0 / pos_var

    if neg_embeddings is not None and len(neg_embeddings) > 0:
        # Signal 2: squared difference of means (discriminative power)
        pos_mean = pos_embeddings.mean(axis=0)
        neg_mean = neg_embeddings.mean(axis=0)
        discrimination = (pos_mean - neg_mean) ** 2

        # Combine: consistency tells us "where positives agree",
        # discrimination tells us "where positives and negatives differ".
        # Both are useful — multiply them so a dimension must be consistent
        # AND discriminative to get high weight.
        weights = consistency * (1.0 + discrimination / (discrimination.mean() + 1e-8))
    else:
        weights = consistency

    # Normalize so weights sum to D (average weight = 1.0)
    weights = weights * (D / (weights.sum() + 1e-8))

    return weights.astype(np.float32)


def search(
    index: faiss.Index,
    embeddings: np.ndarray,
    positive_ids: list,
    negative_ids: list = None,
    alpha: float = 1.0,
    top_k: int = DEFAULT_TOP_K,
    mode: str = "weighted",
) -> tuple:
    """
    Search for objects similar to positive examples.

    Supports two modes:
    - "weighted": Variance-weighted similarity. Automatically learns which
      embedding dimensions matter based on exemplar statistics. Dimensions
      where positives agree are upweighted; dimensions where positives and
      negatives diverge are further emphasized.
    - "centroid": Simple centroid-based search. Averages positive embeddings,
      subtracts negatives, and ranks by cosine similarity. All dimensions
      treated equally.

    Args:
        index: FAISS index (used for centroid mode)
        embeddings: (N, D) all global embeddings, L2-normalized
        positive_ids: List of object indices the user selected as interesting
        negative_ids: List of object indices the user rejected (or None)
        alpha: Strength of negative adjustment (default 1.0)
        top_k: Number of results to return
        mode: "weighted" or "centroid"

    Returns:
        result_ids: List of object indices, ranked by descending similarity
        scores: List of corresponding similarity scores
    """
    negative_ids = negative_ids or []
    exclude = set(positive_ids) | set(negative_ids)

    pos_embeddings = embeddings[positive_ids]
    neg_embeddings = embeddings[negative_ids] if negative_ids else None

    # Build base query: centroid of positives, adjusted by negatives
    query = pos_embeddings.mean(axis=0)
    if neg_embeddings is not None and len(neg_embeddings) > 0:
        query = query - alpha * neg_embeddings.mean(axis=0)

    if mode == "centroid":
        # Simple cosine similarity — use FAISS if available
        norm = np.linalg.norm(query)
        if norm > 1e-8:
            query = query / norm
        query_2d = query.reshape(1, -1).astype(np.float32)

        if index is not None:
            search_k = top_k + len(exclude) + 10
            scores_arr, indices_arr = index.search(query_2d, search_k)
            result_ids = []
            result_scores = []
            for idx, score in zip(indices_arr[0], scores_arr[0]):
                if int(idx) < 0 or int(idx) in exclude:
                    continue
                result_ids.append(int(idx))
                result_scores.append(float(score))
                if len(result_ids) >= top_k:
                    break
            return result_ids, result_scores
        else:
            # Fallback: brute force
            scores = embeddings @ query
    else:
        # Variance-weighted similarity
        weights = _compute_dimension_weights(pos_embeddings, neg_embeddings)

        weighted_query = query * weights
        scores = embeddings @ weighted_query

        # Normalize to [0, 1] range (weighted cosine similarity)
        query_wnorm = np.sqrt(np.sum(query ** 2 * weights)) + 1e-8
        emb_wnorms = np.sqrt(np.sum(embeddings ** 2 * weights[np.newaxis, :], axis=1)) + 1e-8
        scores = scores / (query_wnorm * emb_wnorms)

    ranked = np.argsort(-scores)

    result_ids = []
    result_scores = []
    for idx in ranked:
        if int(idx) in exclude:
            continue
        result_ids.append(int(idx))
        result_scores.append(float(scores[idx]))
        if len(result_ids) >= top_k:
            break

    return result_ids, result_scores


def search_dissimilar(
    index: faiss.Index,
    embeddings: np.ndarray,
    positive_ids: list,
    top_k: int = DEFAULT_TOP_K,
) -> tuple:
    """Find objects most dissimilar to the positive examples."""
    exclude = set(positive_ids)
    pos_embeddings = embeddings[positive_ids]
    query = pos_embeddings.mean(axis=0)
    norm = np.linalg.norm(query)
    if norm > 1e-8:
        query = query / norm

    scores = embeddings @ query
    ranked = np.argsort(scores)  # ascending = most dissimilar first

    result_ids = []
    result_scores = []
    for idx in ranked:
        if int(idx) in exclude:
            continue
        result_ids.append(int(idx))
        result_scores.append(float(scores[idx]))
        if len(result_ids) >= top_k:
            break

    return result_ids, result_scores
