"""Embedding versioning service.

Tracks which model version was used to generate each embedding so that
model upgrades can be handled cleanly (stale embeddings → re-enrollment).
"""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

import config
from models.database import FaceEmbedding

logger = logging.getLogger(__name__)


def get_current_version() -> str:
    """Return the currently configured model version string."""
    return config.EMBEDDING_MODEL_VERSION


def needs_re_enrollment(user_id: int, db: Session) -> bool:
    """Check whether a user has any embeddings from an older model version.

    Returns True if *any* of the user's embeddings don't match the current
    version, meaning the user should re-enroll.
    """
    current = get_current_version()
    rows = (
        db.query(FaceEmbedding)
        .filter(FaceEmbedding.user_id == user_id)
        .all()
    )
    if not rows:
        return True  # No embeddings at all

    for row in rows:
        version = getattr(row, "model_version", None) or "unknown"
        if version != current:
            return True
    return False


def get_stale_user_ids(db: Session) -> list[int]:
    """Return user IDs that have embeddings from an outdated model version."""
    current = get_current_version()
    rows = (
        db.query(FaceEmbedding.user_id)
        .filter(FaceEmbedding.model_version != current)
        .distinct()
        .all()
    )
    return [r[0] for r in rows]


def stamp_version(user_id: int, db: Session) -> None:
    """Update all embeddings for a user to the current model version.

    Called after a successful re-enrollment.
    """
    current = get_current_version()
    (
        db.query(FaceEmbedding)
        .filter(FaceEmbedding.user_id == user_id)
        .update({FaceEmbedding.model_version: current})
    )
    db.commit()
    logger.info("Stamped user %d embeddings with version '%s'", user_id, current)
