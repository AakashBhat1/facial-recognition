"""Runtime ML anomaly scoring service."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from config import ML_ANOMALY_THRESHOLD, ML_MODEL_PATH, ML_SCORING_ENABLED
from services.feature_extractor import FEATURE_NAMES, extract_features_from_db
from services.ml_model_scoring import get_score_mode, is_score_anomalous, score_samples

logger = logging.getLogger(__name__)

_model = None
_model_loaded: bool = False


def load_model(path: str | None = None) -> bool:
    """Load the trained anomaly model from disk."""
    global _model, _model_loaded  # noqa: PLW0603

    if not ML_SCORING_ENABLED:
        logger.info("ML scoring is disabled (ML_SCORING_ENABLED=false).")
        _model = None
        _model_loaded = False
        return False

    model_path = Path(path or ML_MODEL_PATH)
    if not model_path.exists():
        logger.warning("ML model file not found at %s - scoring disabled.", model_path)
        _model = None
        _model_loaded = False
        return False

    import joblib

    _model = joblib.load(model_path)
    _model_loaded = True
    logger.info("ML anomaly model loaded from %s (score_mode=%s)", model_path, get_score_mode(_model))
    return True


def is_available() -> bool:
    """Return True if the model is loaded and scoring is enabled."""
    return _model_loaded and _model is not None


def score_access_event(
    db: Session,
    user_id: int | None,
    ip_address: str,
    timestamp: datetime,
    *,
    result: str | None = None,
    confidence_score: float | None = None,
    similarity_score: float | None = None,
) -> float | None:
    """Score a single access event."""
    if not is_available():
        return None

    features = extract_features_from_db(
        db,
        user_id,
        ip_address,
        timestamp,
        result=result,
        confidence_score=confidence_score,
        similarity_score=similarity_score,
    )
    feature_vector = np.array([[features[name] for name in FEATURE_NAMES]])
    return float(score_samples(_model, feature_vector)[0])


def is_anomalous(score: float) -> bool:
    """Interpret a raw score using the active model's score direction."""
    return is_score_anomalous(_model, score, ML_ANOMALY_THRESHOLD)


def reset() -> None:
    """Reset module state for tests."""
    global _model, _model_loaded  # noqa: PLW0603
    _model = None
    _model_loaded = False
