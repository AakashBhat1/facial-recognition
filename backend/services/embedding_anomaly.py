"""Embedding-level anomaly detection using Isolation Forest.

Operates on raw face embeddings to reject corrupted or adversarial vectors
before similarity matching. Distinct from ``ml_anomaly_service.py`` which
scores access-log features (time, IP, etc.).
"""

from __future__ import annotations

import logging

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

import config

logger = logging.getLogger(__name__)

_model: IsolationForest | None = None


def train_embedding_anomaly_model(
    clean_embeddings: list[list[float]],
) -> IsolationForest:
    """Fit an Isolation Forest on clean enrollment embeddings."""
    X = np.array(clean_embeddings, dtype=np.float64)
    model = IsolationForest(
        contamination=config.EMBEDDING_ANOMALY_CONTAMINATION,
        random_state=42,
    )
    model.fit(X)
    logger.info("Trained embedding anomaly model on %d embeddings", len(clean_embeddings))
    return model


def save_model(model: IsolationForest, path: str | None = None) -> None:
    """Persist a trained model to disk."""
    dest = path or config.EMBEDDING_ANOMALY_MODEL_PATH
    joblib.dump(model, dest)
    logger.info("Saved embedding anomaly model to %s", dest)


def load_model(path: str | None = None) -> bool:
    """Load a trained model from disk into the module-level singleton.

    Returns ``True`` on success, ``False`` if the file is missing.
    """
    global _model
    src = path or config.EMBEDDING_ANOMALY_MODEL_PATH
    try:
        _model = joblib.load(src)
        logger.info("Loaded embedding anomaly model from %s", src)
        return True
    except FileNotFoundError:
        logger.info("No embedding anomaly model found at %s", src)
        _model = None
        return False


def is_anomalous(embedding: list[float]) -> bool:
    """Return ``True`` if the embedding is flagged as anomalous.

    If no model is loaded, returns ``False`` (permissive fallback).
    """
    if _model is None:
        return False
    prediction = _model.predict([embedding])
    return int(prediction[0]) == -1


def reset() -> None:
    """Clear the loaded model (for tests)."""
    global _model
    _model = None
