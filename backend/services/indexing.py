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
) -> tuple:
    """
    Search for objects similar to positive examples using variance-weighted
    similarity.

    Instead of treating all embedding dimensions equally, this computes
    per-dimension weights based on where the positive exemplars agree and
    where they differ from negatives. This lets the system automatically
    discover which visual features (shape, texture, intensity) matter for
    the user's current query.

    The weighted search works by:
    1. Computing dimension weights from exemplar statistics
    2. Applying sqrt(weights) to both query and all embeddings
    3. Running standard cosine similarity on the reweighted space

    This is mathematically equivalent to weighted cosine similarity but
    allows us to use FAISS for fast search.

    Args:
        index: FAISS index (used as fallback; bypassed for weighted search)
        embeddings: (N, D) all global embeddings, L2-normalized
        positive_ids: List of object indices the user selected as interesting
        negative_ids: List of object indices the user rejected (or None)
        alpha: Strength of negative adjustment (default 1.0)
        top_k: Number of results to return

    Returns:
        result_ids: List of object indices, ranked by descending similarity
        scores: List of corresponding similarity scores
    """
    negative_ids = negative_ids or []

    pos_embeddings = embeddings[positive_ids]
    neg_embeddings = embeddings[negative_ids] if negative_ids else None

    # Compute dimension weights from exemplar statistics
    weights = _compute_dimension_weights(pos_embeddings, neg_embeddings)
    sqrt_w = np.sqrt(weights)

    # Build query: weighted centroid of positives, adjusted by negatives
    query = pos_embeddings.mean(axis=0)
    if neg_embeddings is not None and len(neg_embeddings) > 0:
        query = query - alpha * neg_embeddings.mean(axis=0)

    # Apply weights to query and compute weighted similarity against all embeddings
    # This is equivalent to: sum(w_d * q_d * e_d) for each embedding e
    weighted_query = query * weights
    scores = embeddings @ weighted_query

    # Normalize scores to [0, 1] range for consistency with cosine similarity
    # by dividing by the weighted norm of the query and each embedding
    query_wnorm = np.sqrt(np.sum(query ** 2 * weights)) + 1e-8
    emb_wnorms = np.sqrt(np.sum(embeddings ** 2 * weights[np.newaxis, :], axis=1)) + 1e-8
    scores = scores / (query_wnorm * emb_wnorms)

    # Exclude exemplars from results
    exclude = set(positive_ids) | set(negative_ids)
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
