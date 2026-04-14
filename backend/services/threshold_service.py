"""Per-user threshold computation.

Computes a personal similarity threshold from the cross-similarities
of a user's enrolled embeddings: ``mean - k * std``, clamped so it
never exceeds ``SIMILARITY_THRESHOLD + MAX_PERSONAL_UPLIFT``.

Falls back to the global ``SIMILARITY_THRESHOLD`` when no personal value exists.
"""

from __future__ import annotations

import statistics

import numpy as np

import config

# Personal threshold must not exceed global + this uplift.
# Prevents uniform enrollment sessions (same lighting/angle) from
# producing an unreachable threshold like 0.94.
MAX_PERSONAL_UPLIFT = 0.20


def compute_personal_threshold(
    embeddings: list[list[float]],
    k: float = config.PERSONAL_THRESHOLD_K,
) -> float | None:
    """Compute a per-user threshold from pairwise cosine similarities.

    Returns ``None`` if fewer than 2 embeddings are provided.
    """
    if len(embeddings) < 2:
        return None

    scores: list[float] = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            e1 = np.array(embeddings[i])
            e2 = np.array(embeddings[j])
            norm = np.linalg.norm(e1) * np.linalg.norm(e2)
            sim = float(np.dot(e1, e2) / norm) if norm > 0 else 0.0
            scores.append(sim)

    if not scores:
        return None

    mean = statistics.mean(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    threshold = mean - k * std

    # Clamp: at least global threshold, at most global + uplift
    floor = config.SIMILARITY_THRESHOLD
    ceiling = config.SIMILARITY_THRESHOLD + MAX_PERSONAL_UPLIFT
    return max(floor, min(ceiling, threshold))


def get_effective_threshold(personal_threshold: float | None) -> float:
    """Return the per-user threshold if set, else the global default."""
    if personal_threshold is not None:
        return personal_threshold
    return config.SIMILARITY_THRESHOLD
