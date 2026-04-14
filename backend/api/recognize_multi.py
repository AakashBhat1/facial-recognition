"""Multi-frame recognition API — accepts multiple face images for robust matching."""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

import config
from config import RATE_LIMIT_RECOGNIZE_MAX, RATE_LIMIT_RECOGNIZE_WINDOW_SECONDS
from middleware.rate_limiter import SlidingWindowLimiter
from models.database import AccessLog, LockerState, User, get_db
from models.schemas import (
    FrameResultResponse,
    LivenessResultResponse,
    PromptResponse,
    RecognizeMultiFrameResponse,
)
from services import anomaly_detector, enrollment_service, locker_controller
from services.face_pipeline import load_image_from_bytes
from services.logger_service import log_access_event
from services.multi_frame_recognizer import recognize_multi_frame

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Auth"])

_multi_limiter = SlidingWindowLimiter(
    max_requests=RATE_LIMIT_RECOGNIZE_MAX,
    window_seconds=RATE_LIMIT_RECOGNIZE_WINDOW_SECONDS,
)

_MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB total
_MAX_PER_IMAGE_SIZE = 5 * 1024 * 1024  # 5 MB per frame


def _check_rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    if not _multi_limiter.is_allowed(ip):
        retry_after = _multi_limiter.get_retry_after(ip) or 1
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )


def _validate_locker_id(locker_id: str, db: Session) -> None:
    """Validate locker_id exists in the database."""
    locker = db.query(LockerState).filter(LockerState.locker_id == locker_id).first()
    if locker is None:
        raise HTTPException(status_code=400, detail=f"Unknown locker_id: {locker_id}")


def _resolve_user_and_locker(
    user_id: int, locker_id: str | None, db: Session
) -> tuple[User, str]:
    """Look up the claimed user and resolve which locker to open.

    The recognize-multi endpoint is now a 1:1 verify: the caller claims a user_id
    and proves it with face frames. If locker_id is omitted we use the user's
    assigned locker (set at enrollment time).
    """
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail=f"User id={user_id} not found")
    resolved_locker = locker_id or user.assigned_locker_id
    if not resolved_locker:
        raise HTTPException(
            status_code=400,
            detail="No locker_id provided and user has no assigned locker",
        )
    _validate_locker_id(resolved_locker, db)
    return user, resolved_locker


@router.post(
    "/recognize-multi",
    response_model=RecognizeMultiFrameResponse,
    dependencies=[Depends(_check_rate_limit)],
)
async def recognize_multi(
    images: list[UploadFile],
    request: Request,
    user_id: int,
    locker_id: str | None = None,
    check_liveness: bool = True,
    db: Session = Depends(get_db),
) -> RecognizeMultiFrameResponse:
    """1:1 face verification — caller claims a user_id and proves it with frames.

    The face frames are matched ONLY against the claimed user's enrolled
    embeddings. If ``locker_id`` is omitted, the user's assigned locker is used.
    """
    user, locker_id = _resolve_user_and_locker(user_id, locker_id, db)

    if len(images) < config.MULTI_FRAME_MIN_REQUIRED:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {config.MULTI_FRAME_MIN_REQUIRED} images, got {len(images)}",
        )

    # Validate and read images
    decoded_images = []
    total_size = 0
    for img in images:
        if not img.content_type or not img.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not an image",
            )
        data = await img.read()
        if len(data) > _MAX_PER_IMAGE_SIZE:
            raise HTTPException(status_code=400, detail="Single image exceeds 5 MB limit")
        total_size += len(data)
        if total_size > _MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=400, detail="Total upload size exceeds 50 MB")
        try:
            decoded_images.append(load_image_from_bytes(data))
        except ValueError:
            logger.warning("Could not decode image '%s', skipping", img.filename)

    if len(decoded_images) < config.MULTI_FRAME_MIN_REQUIRED:
        raise HTTPException(
            status_code=400,
            detail=f"Only {len(decoded_images)} images could be decoded, need {config.MULTI_FRAME_MIN_REQUIRED}",
        )

    # 1:1 verify — restrict candidate set to only the claimed user's embeddings.
    user_data = enrollment_service.get_user_embeddings(user.id, db)
    if not user_data["embeddings"] or not user_data["centroid"]:
        raise HTTPException(
            status_code=400,
            detail=f"User id={user.id} has no enrolled face embeddings",
        )
    users_with_emb = [{
        "user": user,
        "embeddings": user_data["embeddings"],
        "centroid": user_data["centroid"],
    }]
    result = recognize_multi_frame(decoded_images, users_with_emb, check_liveness)

    ip = request.client.host if request.client else "unknown"
    action = "OPEN" if result.recognized else "ACCESS_DENIED"
    log_result = "SUCCESS" if result.recognized else "FAILURE"

    log_entry = AccessLog(
        user_id=result.user_id,
        user_name=result.user_name,
        action=action,
        result=log_result,
        confidence_score=result.confidence,
        similarity_score=result.final_score,
        locker_id=locker_id,
        ip_address=ip,
        anomaly_flag=False,
        timestamp=datetime.utcnow(),
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)

    flagged = anomaly_detector.run_all_checks(
        db, ip, result.user_id, log_result, log_entry_id=log_entry.id,
    )
    if flagged:
        log_entry.anomaly_flag = True
        db.commit()

    log_access_event(
        log_id=log_entry.id,
        user_id=result.user_id,
        user_name=result.user_name,
        action=action,
        result=log_result,
        confidence_score=result.confidence,
        similarity_score=result.final_score,
        locker_id=locker_id,
        ip_address=ip,
        anomaly_flag=flagged,
    )

    if result.recognized:
        current = locker_controller.get_status(db, locker_id)
        if current.status != "UNLOCKED":
            locker_controller.open_locker(db, locker_id, result.user_id)

    # Build response
    frame_responses = [
        FrameResultResponse(
            frame_index=fr.frame_index,
            quality_passed=fr.quality_passed,
            rejection_reasons=list(fr.quality_result.rejection_reasons) if fr.quality_result else [],
            antispoof_passed=fr.antispoof_passed,
            anomaly_passed=fr.anomaly_passed,
            score=fr.score,
        )
        for fr in result.frame_results
    ]

    liveness_response = None
    if result.liveness_result is not None:
        liveness_response = LivenessResultResponse(
            passed=result.liveness_result.passed,
            blink_detected=result.liveness_result.blink_detected,
            head_movement_detected=result.liveness_result.head_movement_detected,
            reason=result.liveness_result.reason,
        )

    prompt_response = None
    if result.prompt is not None:
        prompt_response = PromptResponse(
            message=result.prompt.message,
            category=result.prompt.category,
        )

    return RecognizeMultiFrameResponse(
        access_granted=result.recognized,
        user_id=result.user_id,
        user_name=result.user_name,
        locker_action=action,
        final_score=result.final_score,
        confidence=result.confidence,
        liveness=liveness_response,
        prompt=prompt_response,
        frame_results=frame_responses,
        frames_processed=result.frames_processed,
        frames_passed_quality=result.frames_passed_quality,
        frames_passed_anomaly=result.frames_passed_anomaly,
        log_id=log_entry.id,
        anomaly_flag=flagged,
    )
