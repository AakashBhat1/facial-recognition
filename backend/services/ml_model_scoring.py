"""Helpers for scoring ML anomaly models consistently across scripts and runtime."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


ScoreMode = Literal["probability", "decision_function"]


def get_score_mode(model: object) -> ScoreMode:
    """Infer how a model exposes anomaly scores."""
    if hasattr(model, "predict_proba"):
        return "probability"
    if hasattr(model, "decision_function"):
        return "decision_function"
    raise TypeError("Model must expose either predict_proba() or decision_function().")


def default_threshold_for(model: object) -> float:
    """Return a reasonable default threshold for the model type."""
    return 0.5 if get_score_mode(model) == "probability" else 0.0


def score_samples(model: object, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Return raw anomaly scores for the provided rows."""
    if get_score_mode(model) == "probability":
        probabilities = model.predict_proba(X)
        return np.asarray(probabilities, dtype=float)[:, 1]
    return np.asarray(model.decision_function(X), dtype=float)


def predict_with_threshold(
    model: object,
    X: pd.DataFrame | np.ndarray,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return anomaly labels and scores using model-appropriate thresholding."""
    scores = score_samples(model, X)
    effective_threshold = default_threshold_for(model) if threshold is None else threshold
    if get_score_mode(model) == "probability":
        predictions = np.where(scores >= effective_threshold, 1, 0)
    else:
        predictions = np.where(scores < effective_threshold, 1, 0)
    return predictions, scores


def is_score_anomalous(model: object | None, score: float, threshold: float) -> bool:
    """Interpret a single score using the model's score direction when known."""
    if model is not None:
        return score >= threshold if get_score_mode(model) == "probability" else score < threshold

    if 0.0 <= threshold <= 1.0 and 0.0 <= score <= 1.0:
        return score >= threshold
    return score < threshold
