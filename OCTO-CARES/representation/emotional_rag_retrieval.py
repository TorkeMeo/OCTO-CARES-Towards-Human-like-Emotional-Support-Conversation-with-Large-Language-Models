"""EmotionalRAG retrieval rules, adapted to leave-one-post-out Reddit retrieval.

Source: BAI-LAB/EmotionalRAG get_response.py, commit
2f932e027a6edc247a222b50353490bf557623b5. No labels enter this module.
We use stable corpus-order tie breaking; upstream uses numpy's default sort.
Encoders are inputs, not silently chosen or normalized by this module.
"""
from __future__ import annotations

import numpy as np

UPSTREAM_COMMIT = "2f932e027a6edc247a222b50353490bf557623b5"
UPSTREAM_URL = f"https://github.com/BAI-LAB/EmotionalRAG/blob/{UPSTREAM_COMMIT}/get_response.py"
METHODS = ("OriginalRAG", "C-A", "C-M", "S-C", "S-S")
PROTOCOL = "emotional-rag-reddit-retrieval-v1"


def vectors(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty, finite 2-D array")
    return array


def distance_matrices(semantic, emotion):
    """Unscaled Euclidean semantic distance + cosine emotional distance.

    No min-max normalization, learned weights or label-based filtering.
    Reject zero emotion vectors instead of turning undefined cosine into zero.
    """
    semantic = vectors(semantic, "semantic vectors")
    emotion = vectors(emotion, "emotion vectors")
    if len(semantic) != len(emotion):
        raise ValueError("Semantic/emotion row counts differ")
    norms = np.linalg.norm(emotion, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("Zero emotion vector: cosine distance is undefined")
    # Direct row-wise subtraction avoids cancellation for identical vectors.
    semantic_dist = np.stack([np.linalg.norm(semantic - row, axis=1) for row in semantic])
    normalized = emotion / norms[:, None]
    emotion_dist = np.clip(1 - normalized @ normalized.T, 0, 2)
    np.fill_diagonal(emotion_dist, 0)
    return semantic_dist, emotion_dist


def rankings_from_distances(semantic_dist, emotion_dist, post_ids, pool_size=20, output_k=10):
    semantic_dist = vectors(semantic_dist, "semantic distances")
    emotion_dist = vectors(emotion_dist, "emotion distances")
    post_ids = np.asarray(post_ids, dtype=str)
    count = len(post_ids)
    if post_ids.ndim != 1 or any(not pid.strip() for pid in post_ids):
        raise ValueError("post_ids must be nonempty strings")
    if semantic_dist.shape != (count, count) or emotion_dist.shape != (count, count):
        raise ValueError("Distance matrices must match post_ids")
    if np.any(semantic_dist < 0) or np.any(emotion_dist < 0):
        raise ValueError("Distances must be nonnegative")
    if not 1 <= output_k <= pool_size:
        raise ValueError("Require 1 <= output_k <= pool_size")
    rankings = {method: [] for method in METHODS}
    for i, pid in enumerate(post_ids):
        # Exclude every same-ID candidate BEFORE either top-20 filtering pass.
        eligible = np.flatnonzero(post_ids != pid)
        if len(eligible) < output_k:
            raise ValueError(f"{pid}: fewer than {output_k} other-post candidates")
        sd, ed = semantic_dist[i], emotion_dist[i]
        for method, distances in (("OriginalRAG", sd), ("C-A", sd + ed), ("C-M", sd * ed)):
            order = np.argsort(distances[eligible], kind="stable")[:output_k]
            rankings[method].append(eligible[order])
        for method, first, second in (("S-C", sd, ed), ("S-S", ed, sd)):
            shortlist = eligible[np.argsort(first[eligible], kind="stable")[:pool_size]]
            order = np.argsort(second[shortlist], kind="stable")[:output_k]
            rankings[method].append(shortlist[order])
    return {method: np.stack(rows) for method, rows in rankings.items()}


def rank_scores(rankings, count):
    """Expose exact precomputed rankings to the existing retrieval metric code.

    These are ordinal scores, NOT distances. Only the returned top-10 entries
    are eligible for reported metrics. This is essential for two-stage rules:
    globally re-sorting by the second distance would undo top-20 filtering.
    """
    output = {}
    for name, rows in rankings.items():
        rows = np.asarray(rows)
        if rows.ndim != 2 or len(rows) != count or rows.shape[1] == 0:
            raise ValueError(f"Invalid ranking shape for {name}")
        if np.any(rows < 0) or np.any(rows >= count):
            raise ValueError(f"Out-of-range candidate in {name}")
        if any(len(set(row.tolist())) != len(row) for row in rows):
            raise ValueError(f"Duplicate candidate in {name}")
        score = np.full((count, count), -float(count + 1))
        for i, row in enumerate(rows):
            score[i, row] = -np.arange(len(row), dtype=float)
        output[f"EmotionalRAG_{name}"] = score
    return output
