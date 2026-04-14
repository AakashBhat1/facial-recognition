"""Face detection, alignment, and embedding extraction using InsightFace.

This module wraps InsightFace to provide a clean interface for the rest of the
application. No other module should import InsightFace directly.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

import config
from services import custom_pt_embedder
from services.model_manager import get_insightface_model

logger = logging.getLogger(__name__)


def _get_model() -> Any:
    """Return the shared InsightFace model via model_manager."""
    return get_insightface_model()


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes to a BGR numpy array.

    Raises ``ValueError`` if the bytes cannot be decoded as an image.
    """
    if not data:
        raise ValueError("Could not decode image from provided bytes")
    buf = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image from provided bytes")
    return image


def detect_faces(image: np.ndarray) -> list[Any]:
    """Run face detection on *image* and return a list of InsightFace Face objects.

    Raises ``ValueError`` when no face is found.
    """
    model = _get_model()
    faces = model.get(image)
    if not faces:
        raise ValueError("No face detected in the image")
    if len(faces) > 1:
        logger.warning("Multiple faces detected (%d) — using the largest", len(faces))
    return faces


def _largest_face(faces: list[Any]) -> Any:
    """Return the face with the largest bounding-box area."""
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def extract_embedding(face: Any, image: np.ndarray | None = None) -> list[float]:
    """Extract a normalised embedding vector.

    Uses the custom PyTorch checkpoint embedder when enabled; otherwise
    reads InsightFace's native ``normed_embedding`` from the face object.

    Raises ``ValueError`` if the embedding dimension does not match the
    configured ``EMBEDDING_DIM``.
    """
    if custom_pt_embedder.is_enabled():
        if image is None:
            raise ValueError("Image is required for custom checkpoint embedding")
        embedding = np.asarray(
            custom_pt_embedder.extract_embedding(image=image, bbox=get_bbox(face)),
            dtype=np.float32,
        )
    else:
        embedding = np.asarray(face.normed_embedding, dtype=np.float32)

    if embedding.shape[0] != config.EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: expected {config.EMBEDDING_DIM}, "
            f"got {embedding.shape[0]}"
        )
    return embedding.tolist()


def get_bbox(face: Any) -> tuple[float, float, float, float]:
    """Return the face bounding box as ``(x1, y1, x2, y2)``."""
    b = face.bbox
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])


def get_face_landmarks(face: Any) -> np.ndarray:
    """Return face landmarks as an (N, 2) array.

    Prefers the 106-point landmarks (``landmark_2d_106``) for accuracy.
    Falls back to the 5-point key-points (``kps``) if 106 are unavailable.
    """
    lm106 = getattr(face, "landmark_2d_106", None)
    if lm106 is not None and len(lm106) > 0:
        return np.asarray(lm106, dtype=np.float32)
    kps = getattr(face, "kps", None)
    if kps is not None and len(kps) > 0:
        return np.asarray(kps, dtype=np.float32)
    raise ValueError("No landmarks available on the face object")


def get_face_pose(face: Any) -> tuple[float, float, float]:
    """Return ``(yaw, pitch, roll)`` in degrees.

    Uses InsightFace's built-in pose if available, otherwise estimates
    from 5-point landmarks using eye-nose geometry.
    """
    pose = getattr(face, "pose", None)
    if pose is not None and len(pose) >= 3:
        return float(pose[0]), float(pose[1]), float(pose[2])

    # Fallback: estimate from 5-point key-points (left_eye, right_eye, nose)
    kps = getattr(face, "kps", None)
    if kps is None or len(kps) < 3:
        return 0.0, 0.0, 0.0

    left_eye = kps[0]
    right_eye = kps[1]
    nose = kps[2]

    eye_center = (left_eye + right_eye) / 2
    eye_dist = np.linalg.norm(right_eye - left_eye)
    if eye_dist < 1e-6:
        return 0.0, 0.0, 0.0

    # Yaw: how far nose deviates horizontally from eye center
    nose_offset_x = (nose[0] - eye_center[0]) / eye_dist
    yaw = float(np.degrees(np.arctan2(nose_offset_x, 1.0)))

    # Pitch: vertical distance ratio between nose and eye center
    nose_offset_y = (nose[1] - eye_center[1]) / eye_dist
    pitch = float(np.degrees(np.arctan2(nose_offset_y - 0.65, 1.0)))  # 0.65 is typical ratio

    # Roll: angle of the eye line
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    roll = float(np.degrees(np.arctan2(dy, dx)))

    return yaw, pitch, roll


def process_image(image: np.ndarray) -> tuple[list[float], float]:
    """Convenience function: detect → pick largest face → extract embedding.

    Returns ``(embedding, detection_confidence)``.
    """
    faces = detect_faces(image)
    face = _largest_face(faces)
    embedding = extract_embedding(face, image=image)
    detection_confidence = float(face.det_score)
    return embedding, detection_confidence


def process_image_full(image: np.ndarray) -> tuple[list[float], float, Any]:
    """Like :func:`process_image` but also returns the InsightFace face object.

    Returns ``(embedding, detection_confidence, face_object)``.
    The caller can pass ``face_object`` to :func:`quality_filter.run_quality_checks`.
    """
    faces = detect_faces(image)
    face = _largest_face(faces)
    embedding = extract_embedding(face, image=image)
    detection_confidence = float(face.det_score)
    return embedding, detection_confidence, face
