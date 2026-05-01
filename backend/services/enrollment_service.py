"""Enrollment service — multi-image enrollment with augmentation.

Handles the full enrollment pipeline: image decoding, face detection,
embedding extraction, augmentation, centroid computation, and persistence.
"""

from __future__ import annotations

import json
import logging

import numpy as np
from sqlalchemy.orm import Session

import config
from models.database import FaceEmbedding, LockerState, User, locker_id_for_user
from services.embedding_crypto import encrypt_embedding, decrypt_embedding
from services.face_pipeline import extract_embedding, load_image_from_bytes, process_image
from services.image_augmentation import augment_image
from services.threshold_service import compute_personal_threshold
from services.version_service import get_current_version, stamp_version

logger = logging.getLogger(__name__)


def _normalize(vector: list[float]) -> list[float]:
    """L2-normalise a vector."""
    arr = np.array(vector, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return vector
    return (arr / norm).tolist()


def _compute_centroid(embeddings: list[list[float]]) -> list[float]:
    """Compute the L2-normalised centroid (mean) of a set of embeddings."""
    matrix = np.array(embeddings, dtype=np.float64)
    mean = matrix.mean(axis=0)
    return _normalize(mean.tolist())


def enroll_user(
    name: str,
    images: list[bytes],
    db: Session,
    model_name: str = "arcface",
) -> tuple[User, int]:
    """Enroll a new user from a set of face images.

    1. Decode each image and extract its embedding.
    2. Augment each image and extract embeddings from augmented copies.
    3. Compute a centroid from all embeddings.
    4. Persist User + FaceEmbedding rows.

    Returns ``(user, embedding_count)``.

    Raises ``ValueError`` if fewer than ``MIN_ENROLLMENT_IMAGES`` images
    yield a valid face.
    """
    if not (config.MIN_ENROLLMENT_IMAGES <= len(images) <= config.MAX_ENROLLMENT_IMAGES):
        raise ValueError(
            f"Expected {config.MIN_ENROLLMENT_IMAGES}-{config.MAX_ENROLLMENT_IMAGES} "
            f"images, got {len(images)}"
        )

    all_embeddings: list[list[float]] = []
    successful_images = 0

    for idx, image_bytes in enumerate(images):
        try:
            image = load_image_from_bytes(image_bytes)
            embedding, _ = process_image(image)
            all_embeddings.append(embedding)
            successful_images += 1

            # Augment only if few images provided (≥7 images have enough diversity)
            if len(images) < 7:
                for aug_image in augment_image(image):
                    try:
                        from services.face_pipeline import detect_faces, _largest_face

                        faces = detect_faces(aug_image)
                        face = _largest_face(faces)
                        aug_embedding = extract_embedding(face, image=aug_image)
                        all_embeddings.append(aug_embedding)
                    except ValueError:
                        logger.debug("Augmented image %d failed face detection, skipping", idx)

        except ValueError as exc:
            logger.warning("Image %d failed: %s", idx, exc)
            continue

    if successful_images < config.MIN_ENROLLMENT_IMAGES:
        raise ValueError(
            f"Only {successful_images} of {len(images)} images had a detectable face. "
            f"Need at least {config.MIN_ENROLLMENT_IMAGES}."
        )

    centroid = _compute_centroid(all_embeddings)
    personal_threshold = compute_personal_threshold(all_embeddings)

    # Create User row (centroid in legacy embedding field for backward compat)
    user = User(
        name=name,
        embedding=json.dumps(centroid),
        personal_threshold=personal_threshold,
    )
    db.add(user)
    db.flush()  # get user.id

    # Auto-assign locker by user id (first user → L001, second → L002, …)
    user.assigned_locker_id = locker_id_for_user(user.id)
    if not db.query(LockerState).filter_by(locker_id=user.assigned_locker_id).first():
        db.add(LockerState(locker_id=user.assigned_locker_id, status="LOCKED"))

    logger.info(
        "User '%s' personal threshold: %s, locker: %s",
        name,
        f"{personal_threshold:.4f}" if personal_threshold else "N/A",
        user.assigned_locker_id,
    )

    # Create FaceEmbedding rows (encrypted if enabled)
    model_version = get_current_version()
    for emb in all_embeddings:
        db.add(FaceEmbedding(
            user_id=user.id,
            embedding=encrypt_embedding(emb),
            is_centroid=False,
            model_name=model_name,
            model_version=model_version,
        ))

    db.add(FaceEmbedding(
        user_id=user.id,
        embedding=encrypt_embedding(centroid),
        is_centroid=True,
        model_name=model_name,
        model_version=model_version,
    ))

    db.commit()
    db.refresh(user)

    total_count = len(all_embeddings) + 1  # +1 for centroid
    logger.info(
        "Enrolled user '%s' (id=%d) with %d embeddings",
        name, user.id, total_count,
    )
    return user, total_count


def get_user_embeddings(user_id: int, db: Session) -> dict[str, list[list[float]]]:
    """Load all FaceEmbedding rows for a user.

    Returns ``{"embeddings": [...], "centroid": [...]}``.
    """
    rows = db.query(FaceEmbedding).filter(FaceEmbedding.user_id == user_id).all()

    embeddings: list[list[float]] = []
    centroid: list[float] = []

    for row in rows:
        emb = decrypt_embedding(row.embedding)
        if row.is_centroid:
            centroid = emb
        else:
            embeddings.append(emb)

    return {"embeddings": embeddings, "centroid": centroid}


def get_all_users_with_embeddings(db: Session) -> list[dict]:
    """Load all users that have FaceEmbedding rows.

    Returns a list of dicts suitable for ``find_best_match_v2``:
    ``[{"user": User, "embeddings": [...], "centroid": [...]}, ...]``
    """
    users = db.query(User).all()
    result: list[dict] = []

    for user in users:
        data = get_user_embeddings(user.id, db)
        if data["embeddings"] and data["centroid"]:
            result.append({
                "user": user,
                "embeddings": data["embeddings"],
                "centroid": data["centroid"],
            })

    return result


def _extract_all_embeddings(images: list[bytes]) -> tuple[list[list[float]], int]:
    """Extract embeddings from images + augmented copies.

    Shared by ``enroll_user`` and ``re_enroll_user``.
    """
    all_embeddings: list[list[float]] = []
    successful = 0

    for idx, image_bytes in enumerate(images):
        try:
            image = load_image_from_bytes(image_bytes)
            embedding, _ = process_image(image)
            all_embeddings.append(embedding)
            successful += 1

            if len(images) < 7:
                for aug_image in augment_image(image):
                    try:
                        from services.face_pipeline import detect_faces, _largest_face

                        faces = detect_faces(aug_image)
                        face = _largest_face(faces)
                        aug_embedding = extract_embedding(face, image=aug_image)
                        all_embeddings.append(aug_embedding)
                    except ValueError:
                        logger.debug("Augmented image %d failed face detection, skipping", idx)

        except ValueError as exc:
            logger.warning("Image %d failed: %s", idx, exc)

    return all_embeddings, successful


def re_enroll_user(
    user_id: int,
    images: list[bytes],
    db: Session,
    model_name: str = "arcface",
) -> tuple[User, int]:
    """Re-enroll an existing user by replacing all embeddings.

    Atomically deletes old embeddings and creates new ones from the
    provided images. Recomputes centroid and personal threshold.

    Raises ``ValueError`` if user not found or if too few images have faces.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise ValueError(f"User with id={user_id} not found")

    if not (config.MIN_ENROLLMENT_IMAGES <= len(images) <= config.MAX_ENROLLMENT_IMAGES):
        raise ValueError(
            f"Expected {config.MIN_ENROLLMENT_IMAGES}-{config.MAX_ENROLLMENT_IMAGES} "
            f"images, got {len(images)}"
        )

    all_embeddings, successful = _extract_all_embeddings(images)

    if successful < config.MIN_ENROLLMENT_IMAGES:
        raise ValueError(
            f"Only {successful} of {len(images)} images had a detectable face. "
            f"Need at least {config.MIN_ENROLLMENT_IMAGES}."
        )

    # Delete old embeddings in same transaction
    db.query(FaceEmbedding).filter(FaceEmbedding.user_id == user_id).delete()

    centroid = _compute_centroid(all_embeddings)
    personal_threshold = compute_personal_threshold(all_embeddings)

    # Update user record
    user.embedding = json.dumps(centroid)
    user.personal_threshold = personal_threshold

    # Insert new embeddings (encrypted if enabled)
    model_version = get_current_version()
    for emb in all_embeddings:
        db.add(FaceEmbedding(
            user_id=user.id,
            embedding=encrypt_embedding(emb),
            is_centroid=False,
            model_name=model_name,
            model_version=model_version,
        ))
    db.add(FaceEmbedding(
        user_id=user.id,
        embedding=encrypt_embedding(centroid),
        is_centroid=True,
        model_name=model_name,
        model_version=model_version,
    ))

    db.commit()
    db.refresh(user)

    total_count = len(all_embeddings) + 1
    logger.info(
        "Re-enrolled user '%s' (id=%d) with %d embeddings, threshold=%s",
        user.name, user.id, total_count,
        f"{personal_threshold:.4f}" if personal_threshold else "N/A",
    )
    return user, total_count
