"""Embedding drift handling.

On each successful unlock, appends the new embedding to the user's collection,
evicts the oldest if over the cap, and recomputes the centroid and personal threshold.
"""

from __future__ import annotations

import json
import logging

import numpy as np
from sqlalchemy.orm import Session

import config
from models.database import FaceEmbedding, User
from services.threshold_service import compute_personal_threshold

logger = logging.getLogger(__name__)


def _normalize(vector: list[float]) -> list[float]:
    arr = np.array(vector, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return vector
    return (arr / norm).tolist()


def _compute_centroid(embeddings: list[list[float]]) -> list[float]:
    matrix = np.array(embeddings, dtype=np.float64)
    mean = matrix.mean(axis=0)
    return _normalize(mean.tolist())


def update_embeddings_on_unlock(
    user_id: int,
    new_embedding: list[float],
    db: Session,
) -> None:
    """Append a new embedding after successful unlock and recompute centroid.

    If the number of non-centroid embeddings exceeds ``EMBEDDING_DRIFT_CAP``,
    the oldest embedding is evicted.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        logger.warning("Drift update skipped: user %d not found", user_id)
        return

    # Get non-centroid rows ordered oldest-first
    non_centroid_rows = (
        db.query(FaceEmbedding)
        .filter(FaceEmbedding.user_id == user_id, FaceEmbedding.is_centroid == False)  # noqa: E712
        .order_by(FaceEmbedding.created_at.asc())
        .all()
    )

    # Evict oldest if at cap
    if len(non_centroid_rows) >= config.EMBEDDING_DRIFT_CAP:
        db.delete(non_centroid_rows[0])

    # Add new embedding
    db.add(FaceEmbedding(
        user_id=user_id,
        embedding=json.dumps(new_embedding),
        is_centroid=False,
        model_name="arcface",
    ))
    db.flush()

    # Reload all non-centroid embeddings for centroid recomputation
    all_rows = (
        db.query(FaceEmbedding)
        .filter(FaceEmbedding.user_id == user_id, FaceEmbedding.is_centroid == False)  # noqa: E712
        .all()
    )
    all_embeddings = [row.get_embedding() for row in all_rows]
    centroid = _compute_centroid(all_embeddings)

    # Update or create centroid row
    centroid_row = (
        db.query(FaceEmbedding)
        .filter(FaceEmbedding.user_id == user_id, FaceEmbedding.is_centroid == True)  # noqa: E712
        .first()
    )
    if centroid_row:
        centroid_row.embedding = json.dumps(centroid)
    else:
        db.add(FaceEmbedding(
            user_id=user_id,
            embedding=json.dumps(centroid),
            is_centroid=True,
            model_name="arcface",
        ))

    # Update user record for backward compat
    user.embedding = json.dumps(centroid)
    user.personal_threshold = compute_personal_threshold(all_embeddings)

    db.commit()
    logger.debug(
        "Drift update for user %d: %d embeddings, centroid recomputed",
        user_id, len(all_embeddings),
    )
