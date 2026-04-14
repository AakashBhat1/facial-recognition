"""Quality filtering with hard thresholds.

Rejects face images before embedding extraction when they are too blurry,
too dark/bright, too small, or at too extreme an angle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

import config
from services.face_pipeline import get_bbox, get_face_pose


@dataclass(frozen=True)
class QualityResult:
    """Immutable result of quality checks on a single face image."""

    passed: bool
    blur_score: float
    brightness: float
    face_width: float
    yaw: float
    pitch: float
    rejection_reasons: tuple[str, ...]


def check_blur(image: np.ndarray) -> tuple[bool, float]:
    """Check image sharpness via variance of Laplacian.

    Returns ``(passed, blur_score)``.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return score >= config.QUALITY_BLUR_THRESHOLD, score


def check_brightness(image: np.ndarray) -> tuple[bool, float]:
    """Check mean pixel intensity is within acceptable range.

    Returns ``(passed, mean_brightness)``.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mean_val = float(np.mean(gray))
    passed = config.QUALITY_BRIGHTNESS_MIN <= mean_val <= config.QUALITY_BRIGHTNESS_MAX
    return passed, mean_val


def check_face_size(bbox: tuple[float, float, float, float]) -> tuple[bool, float]:
    """Check face bounding box width meets minimum size.

    Returns ``(passed, width)``.
    """
    width = bbox[2] - bbox[0]
    return width >= config.QUALITY_MIN_FACE_WIDTH, float(width)


def check_yaw(yaw: float) -> tuple[bool, float]:
    """Check yaw angle is within tolerance."""
    return abs(yaw) <= config.QUALITY_MAX_YAW, yaw


def check_pitch(pitch: float) -> tuple[bool, float]:
    """Check pitch angle is within tolerance."""
    return abs(pitch) <= config.QUALITY_MAX_PITCH, pitch


def run_quality_checks(image: np.ndarray, face: Any) -> QualityResult:
    """Run all quality checks on an image + detected face.

    Returns a ``QualityResult`` with details on every check.
    """
    reasons: list[str] = []

    blur_ok, blur_score = check_blur(image)
    if not blur_ok:
        reasons.append("blur")

    bright_ok, brightness = check_brightness(image)
    if not bright_ok:
        reasons.append("brightness")

    bbox = get_bbox(face)
    size_ok, face_width = check_face_size(bbox)
    if not size_ok:
        reasons.append("face_size")

    yaw, pitch, _ = get_face_pose(face)

    yaw_ok, yaw_val = check_yaw(yaw)
    if not yaw_ok:
        reasons.append("yaw")

    pitch_ok, pitch_val = check_pitch(pitch)
    if not pitch_ok:
        reasons.append("pitch")

    return QualityResult(
        passed=len(reasons) == 0,
        blur_score=blur_score,
        brightness=brightness,
        face_width=face_width,
        yaw=yaw_val,
        pitch=pitch_val,
        rejection_reasons=tuple(reasons),
    )
