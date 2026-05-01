"""Centralized model manager for InsightFace and anti-spoof models.

Provides a singleton interface so that every module shares the same
model instances and avoids redundant GPU/CPU memory allocation.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import config

logger = logging.getLogger(__name__)

_insightface_model: Any | None = None
_model_load_time: float = 0.0
_active_backend: str = ""
_active_model_name: str = ""


def _build_face_analysis(model_name: str, root: str | None = None) -> Any:
    """Build and prepare an InsightFace FaceAnalysis instance."""
    from insightface.app import FaceAnalysis

    kwargs: dict[str, Any] = {
        "name": model_name,
        "providers": ["CPUExecutionProvider"],
    }
    if root:
        kwargs["root"] = root

    model = FaceAnalysis(**kwargs)
    # ctx_id=-1 forces CPU. Deployment target is CPU-only kiosk hardware,
    # so this matches the configured CPUExecutionProvider explicitly.
    model.prepare(
        ctx_id=-1,
        det_thresh=config.FACE_DETECTION_THRESHOLD,
        det_size=(config.FACE_DET_SIZE, config.FACE_DET_SIZE),
    )
    return model


def get_insightface_model() -> Any:
    """Return the shared InsightFace FaceAnalysis model, loading on first call."""
    global _insightface_model, _model_load_time, _active_backend, _active_model_name

    if _insightface_model is not None:
        return _insightface_model

    t0 = time.perf_counter()
    if config.USE_CUSTOM_FACE_MODEL:
        model = _build_face_analysis(
            model_name=config.CUSTOM_FACE_MODEL_NAME,
            root=config.CUSTOM_FACE_MODEL_ROOT or None,
        )
        _active_backend = "custom"
        _active_model_name = config.CUSTOM_FACE_MODEL_NAME
    elif config.USE_BUFFALO_MODEL:
        model = _build_face_analysis(model_name=config.FACE_MODEL_NAME)
        _active_backend = "buffalo"
        _active_model_name = config.FACE_MODEL_NAME
    else:
        raise RuntimeError(
            "Invalid face backend selection: enable either Buffalo or custom model"
        )

    _model_load_time = time.perf_counter() - t0
    _insightface_model = model

    logger.info(
        "Face backend '%s' model '%s' loaded in %.2fs",
        _active_backend,
        _active_model_name,
        _model_load_time,
    )
    return _insightface_model


def get_model_load_time() -> float:
    """Return how long it took to load the model (seconds), or 0 if not loaded."""
    return _model_load_time


def get_active_face_backend() -> str:
    """Return active face backend label once loaded, else empty string."""
    return _active_backend


def get_active_face_model_name() -> str:
    """Return active face model name once loaded, else empty string."""
    return _active_model_name


def time_operation(name: str):
    """Context manager that logs execution time for a named operation."""
    class Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.t0
            logger.debug("%s took %.3fms", name, elapsed * 1000)
            self.elapsed_ms = elapsed * 1000

    return Timer()
