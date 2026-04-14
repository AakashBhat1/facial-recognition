"""Enrollment API — accepts face images and creates a user with embeddings."""

import logging
from datetime import datetime
from typing import Annotated, List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

import config
from models.database import get_db
from models.schemas import EnrollResponse, ReEnrollResponse
from services.enrollment_service import enroll_user, re_enroll_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/enroll", tags=["Enroll"])

_MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB total


@router.post("", response_model=EnrollResponse)
async def enroll(
    name: Annotated[str, Form(description="Name of the person being enrolled")],
    images: Annotated[List[UploadFile], File(description="5-10 face images captured from the device camera")],
    db: Session = Depends(get_db),
) -> EnrollResponse:
    """Enroll a new user by uploading 5-10 face images captured from the device camera."""
    if not (config.MIN_ENROLLMENT_IMAGES <= len(images) <= config.MAX_ENROLLMENT_IMAGES):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected {config.MIN_ENROLLMENT_IMAGES}-{config.MAX_ENROLLMENT_IMAGES} "
                f"images, got {len(images)}"
            ),
        )

    # Validate content types
    for img in images:
        if not img.content_type or not img.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{img.filename}' is not an image (got {img.content_type})",
            )

    # Read image bytes (sequentially to limit memory usage)
    image_bytes_list: list[bytes] = []
    total_size = 0
    for img in images:
        data = await img.read()
        total_size += len(data)
        if total_size > _MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=400, detail="Total upload size exceeds 50 MB")
        image_bytes_list.append(data)

    try:
        user, embedding_count = enroll_user(name, image_bytes_list, db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return EnrollResponse(
        user_id=user.id,
        user_name=user.name,
        embedding_count=embedding_count,
        assigned_locker_id=user.assigned_locker_id,
        created_at=user.created_at,
    )


@router.put("/{user_id}/re-enroll", response_model=ReEnrollResponse)
async def re_enroll(
    user_id: int,
    images: Annotated[List[UploadFile], File(description="5-10 face images captured from the device camera")],
    db: Session = Depends(get_db),
) -> ReEnrollResponse:
    """Re-enroll an existing user by replacing all face embeddings with images captured from the device camera."""
    if not (config.MIN_ENROLLMENT_IMAGES <= len(images) <= config.MAX_ENROLLMENT_IMAGES):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected {config.MIN_ENROLLMENT_IMAGES}-{config.MAX_ENROLLMENT_IMAGES} "
                f"images, got {len(images)}"
            ),
        )

    for img in images:
        if not img.content_type or not img.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="All uploaded files must be images",
            )

    image_bytes_list: list[bytes] = []
    total_size = 0
    for img in images:
        data = await img.read()
        total_size += len(data)
        if total_size > _MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=400, detail="Total upload size exceeds 50 MB")
        image_bytes_list.append(data)

    try:
        user, embedding_count = re_enroll_user(user_id, image_bytes_list, db)
    except ValueError as exc:
        detail = str(exc)
        status = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=status, detail=detail)

    return ReEnrollResponse(
        user_id=user.id,
        user_name=user.name,
        embedding_count=embedding_count,
        personal_threshold=user.personal_threshold,
        assigned_locker_id=user.assigned_locker_id,
        updated_at=datetime.utcnow(),
    )
