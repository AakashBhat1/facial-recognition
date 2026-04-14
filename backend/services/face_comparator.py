from __future__ import annotations

from typing import Any

import numpy as np

from config import CENTROID_WEIGHT, MAX_SCORE_WEIGHT, SIMILARITY_THRESHOLD, MULTI_USER_MARGIN
from services.threshold_service import get_effective_threshold


def cosine_similarity(emb1: list[float], emb2: list[float]) -> float:
    """Return cosine similarity between two face embeddings (0.0 to 1.0)."""
    e1 = np.array(emb1)
    e2 = np.array(emb2)
    norm = np.linalg.norm(e1) * np.linalg.norm(e2)
    if norm == 0:
        return 0.0
    return float(np.dot(e1, e2) / norm)


def find_best_match(
    query_embedding: list[float],
    users: list  # list of User ORM objects
) -> tuple:
    """
    Compare query_embedding against all registered users.
    Returns (best_user | None, best_similarity_score).
    """
    best_user  = None
    best_score = 0.0

    for user in users:
        stored_embedding = user.get_embedding()
        score = cosine_similarity(query_embedding, stored_embedding)
        if score > best_score:
            best_score = score
            best_user  = user

    if best_score >= SIMILARITY_THRESHOLD:
        return best_user, best_score

    return None, best_score


# ─── Phase 1: Weighted similarity ────────────────────────────────────────────


def compute_weighted_score(
    query: list[float],
    embeddings: list[list[float]],
    centroid: list[float],
) -> tuple[float, float, float]:
    """Compute weighted similarity score.

    Returns ``(weighted_score, centroid_score, max_score)``.
    ``weighted_score = CENTROID_WEIGHT * centroid_score + MAX_SCORE_WEIGHT * max_score``
    """
    centroid_score = cosine_similarity(query, centroid)
    max_score = max(
        (cosine_similarity(query, emb) for emb in embeddings),
        default=0.0,
    )
    weighted = CENTROID_WEIGHT * centroid_score + MAX_SCORE_WEIGHT * max_score
    return weighted, centroid_score, max_score


def compute_confidence(score: float) -> float:
    """Return ``score - threshold``, clamped to [-1.0, 1.0]."""
    return max(-1.0, min(1.0, score - SIMILARITY_THRESHOLD))


def find_best_match_v2(
    query: list[float],
    users_with_embeddings: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, float, float]:
    """Find the best matching user using weighted centroid + max scoring.

    *users_with_embeddings* is a list of dicts with keys:
    ``{"user": User, "embeddings": list[list[float]], "centroid": list[float]}``.

    Returns ``(best_match_dict | None, best_weighted_score, confidence)``.
    """
    best_match: dict[str, Any] | None = None
    best_weighted = 0.0

    for entry in users_with_embeddings:
        weighted, _, _ = compute_weighted_score(
            query, entry["embeddings"], entry["centroid"]
        )
        if weighted > best_weighted:
            best_weighted = weighted
            best_match = entry

    confidence = compute_confidence(best_weighted)

    raw = getattr(best_match["user"], "personal_threshold", None) if best_match else None
    personal = raw if isinstance(raw, (int, float)) else None
    threshold = get_effective_threshold(personal)
    if best_weighted >= threshold:
        return best_match, best_weighted, confidence

    return None, best_weighted, confidence


# ─── Phase 4: Multi-user support ─────────────────────────────────────────────


def find_best_match_multi_user(
    query: list[float],
    users_with_embeddings: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, float, float]:
    """Find the best match with ambiguity rejection for multi-user lockers.

    When multiple users are enrolled on the same locker, this function
    rejects the match if the gap between the top-2 scores is smaller
    than ``MULTI_USER_MARGIN``, preventing misidentification.

    Returns ``(best_match_dict | None, best_weighted_score, confidence)``.
    """
    scores: list[tuple[float, dict[str, Any]]] = []

    for entry in users_with_embeddings:
        weighted, _, _ = compute_weighted_score(
            query, entry["embeddings"], entry["centroid"]
        )
        scores.append((weighted, entry))

    if not scores:
        return None, 0.0, compute_confidence(0.0)

    scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_entry = scores[0]
    confidence = compute_confidence(best_score)

    # Check threshold
    raw = getattr(best_entry["user"], "personal_threshold", None)
    personal = raw if isinstance(raw, (int, float)) else None
    threshold = get_effective_threshold(personal)

    if best_score < threshold:
        return None, best_score, confidence

    # Check margin against second-best (ambiguity rejection)
    if len(scores) >= 2:
        second_score = scores[1][0]
        if (best_score - second_score) < MULTI_USER_MARGIN:
            return None, best_score, confidence  # Ambiguous — reject

    return best_entry, best_score, confidence
