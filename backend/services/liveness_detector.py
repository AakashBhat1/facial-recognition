"""Liveness detection via blink and head movement.

Uses InsightFace landmarks to detect:
- Blinks via Eye Aspect Ratio (EAR) dips across frames
- Head movement via landmark displacement between frames

With 106-point landmarks, EAR is computed from the 6 eye corner points
per eye. With only 5-point landmarks, we rely on head movement only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config


@dataclass(frozen=True)
class LivenessResult:
    """Immutable result of liveness checks across frames."""

    passed: bool
    blink_detected: bool
    head_movement_detected: bool
    reason: str


# 106-point landmark indices for left and right eye (InsightFace 2d106det).
# Empirically verified against insightface buffalo_s on insightface/data/images/t1.jpg:
#   - Indices 33-42 (10 pts) cluster at the face's right eye.
#   - Indices 87-95 (9 pts) cluster at the face's left eye.
#   - Indices 43-51 and 96-105 are the eyebrows (NOT eyes — prior code used them
#     by mistake, which made blink detection essentially never fire).
_RIGHT_EYE_106 = list(range(33, 43))
_LEFT_EYE_106 = list(range(87, 96))


def _eye_aspect_ratio_106(landmarks: np.ndarray, eye_indices: list[int]) -> float:
    """EAR-equivalent: bounding-box height/width over all eye landmarks.

    Open eyes yield ~0.25-0.40, blinks drop below ~0.15. This avoids depending
    on a specific 6-point ordering, which the JD-AI 106-point layout does not
    document and which differs between the two eyes (10 vs 9 points).
    """
    pts = landmarks[eye_indices]
    if len(pts) < 4:
        return 1.0  # safe default — assume open
    width = float(pts[:, 0].max() - pts[:, 0].min())
    height = float(pts[:, 1].max() - pts[:, 1].min())
    if width < 1e-6:
        return 1.0
    return height / width


def compute_ear(landmarks: np.ndarray) -> float:
    """Compute average Eye Aspect Ratio from landmarks.

    Returns a value ~0.2-0.3 for open eyes, dropping below 0.2 during blinks.
    Returns ``-1.0`` if landmarks are insufficient for EAR (5-point only).
    """
    if len(landmarks) >= 106:
        left_ear = _eye_aspect_ratio_106(landmarks, _LEFT_EYE_106)
        right_ear = _eye_aspect_ratio_106(landmarks, _RIGHT_EYE_106)
        return (left_ear + right_ear) / 2.0
    # 5-point landmarks: cannot compute reliable EAR
    return -1.0


def detect_blink(ear_history: list[float]) -> bool:
    """Return ``True`` if a blink pattern is detected in the EAR history.

    A blink is: EAR drops below threshold for N consecutive frames, then
    recovers above threshold.
    """
    # Filter out invalid EAR values (from 5-point landmarks)
    valid = [e for e in ear_history if e >= 0]
    if len(valid) < config.LIVENESS_EAR_CONSEC_FRAMES + 1:
        return False

    below_count = 0
    recovered = False

    for ear in valid:
        if ear < config.LIVENESS_EAR_THRESHOLD:
            below_count += 1
        else:
            if below_count >= config.LIVENESS_EAR_CONSEC_FRAMES:
                recovered = True
                break
            below_count = 0

    return recovered


def compute_head_displacement(
    landmarks_prev: np.ndarray, landmarks_curr: np.ndarray,
) -> float:
    """Mean Euclidean displacement of landmarks between two frames."""
    n = min(len(landmarks_prev), len(landmarks_curr))
    if n == 0:
        return 0.0
    diffs = landmarks_curr[:n] - landmarks_prev[:n]
    distances = np.linalg.norm(diffs, axis=1)
    return float(np.mean(distances))


def detect_head_movement(displacement_history: list[float]) -> bool:
    """Return ``True`` if significant head movement is detected."""
    if not displacement_history:
        return False
    return max(displacement_history) >= config.LIVENESS_HEAD_MOVEMENT_THRESHOLD


def check_liveness(frames_landmarks: list[np.ndarray]) -> LivenessResult:
    """Run liveness checks across a sequence of frame landmarks.

    Requires **blink OR head movement** to pass.
    """
    if len(frames_landmarks) < 2:
        return LivenessResult(
            passed=False,
            blink_detected=False,
            head_movement_detected=False,
            reason="Insufficient frames for liveness check",
        )

    # Blink detection
    ear_history = [compute_ear(lm) for lm in frames_landmarks]
    blink = detect_blink(ear_history)

    # Head movement detection
    displacements = [
        compute_head_displacement(frames_landmarks[i], frames_landmarks[i + 1])
        for i in range(len(frames_landmarks) - 1)
    ]
    movement = detect_head_movement(displacements)

    passed = blink or movement

    if passed:
        reason = "Liveness confirmed"
    elif all(e < 0 for e in ear_history):
        reason = "Only 5-point landmarks available and no head movement detected"
    else:
        reason = "No blink or head movement detected"

    return LivenessResult(
        passed=passed,
        blink_detected=blink,
        head_movement_detected=movement,
        reason=reason,
    )
