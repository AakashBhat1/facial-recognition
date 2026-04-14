"""PIN-based fallback authentication.

Provides endpoints for setting and verifying a PIN when face recognition
fails. PINs are stored as bcrypt hashes.
"""
from __future__ import annotations

import hashlib
import hmac
import logging

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from config import RATE_LIMIT_COOLDOWN_SECONDS
from middleware.rate_limiter import SlidingWindowLimiter
from models.database import get_db, User, AccessLog
from models.schemas import PinAuthRequest, PinSetRequest, PinAuthResponse
from services.logger_service import log_access_event
from services import locker_controller
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Auth"])
users_router = APIRouter(prefix="/api/users", tags=["Users"])

# Strict rate limiting for PIN: 3 attempts per 60 seconds
_pin_limiter = SlidingWindowLimiter(max_requests=3, window_seconds=60)

# bcrypt work factor: 12 gives ~250ms per hash on modest CPU, strong against
# brute-force against a 10k-100M PIN keyspace if the DB is ever exfiltrated.
_BCRYPT_ROUNDS = 12


def _hash_pin(pin: str) -> str:
    """Hash a PIN using bcrypt. Returns the hash string."""
    return bcrypt.hashpw(pin.encode("utf-8"), bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)).decode("ascii")


def _verify_pin(pin: str, stored_hash: str) -> bool:
    """Verify a PIN against a stored hash. Supports legacy ``salt$sha256`` format."""
    if not stored_hash:
        return False
    # Legacy SHA-256 format (pre-bcrypt migration)
    if "$" in stored_hash and not stored_hash.startswith("$2"):
        salt, expected = stored_hash.split("$", 1)
        h = hashlib.sha256(f"{salt}{pin}".encode("utf-8")).hexdigest()
        return hmac.compare_digest(h, expected)
    # bcrypt format
    try:
        return bcrypt.checkpw(pin.encode("utf-8"), stored_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


@users_router.put("/{user_id}/pin")
def set_pin(
    user_id: int,
    payload: PinSetRequest,
    db: Session = Depends(get_db),
):
    """Set or update a user's fallback PIN.

    First-time set: send only ``{"pin": "1234"}``.
    Update: send ``{"old_pin": "1234", "pin": "5678"}`` — the old PIN is verified before the new one is stored.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not payload.pin.isdigit():
        raise HTTPException(status_code=400, detail="PIN must contain only digits")

    if user.pin_hash:
        if not payload.old_pin:
            raise HTTPException(
                status_code=400,
                detail="old_pin is required to update an existing PIN",
            )
        if not _verify_pin(payload.old_pin, user.pin_hash):
            raise HTTPException(status_code=401, detail="old_pin is incorrect")
        action = "updated"
    else:
        action = "created"

    user.pin_hash = _hash_pin(payload.pin)
    db.commit()
    return {"message": f"PIN {action} successfully", "user_id": user_id}


@router.post("/pin", response_model=PinAuthResponse)
def pin_authenticate(
    payload: PinAuthRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Authenticate via PIN fallback after face recognition failure."""
    ip = request.client.host if request.client else "unknown"

    if not _pin_limiter.is_allowed(ip):
        retry_after = _pin_limiter.get_retry_after(ip) or 1
        raise HTTPException(
            status_code=429,
            detail="Too many PIN attempts. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )

    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not user.pin_hash:
        raise HTTPException(
            status_code=400,
            detail="No PIN set for this user. Use PUT /api/users/{user_id}/pin to set one.",
        )

    locker_id = payload.locker_id or user.assigned_locker_id or "L001"

    access_granted = _verify_pin(payload.pin, user.pin_hash)
    action = "OPEN" if access_granted else "ACCESS_DENIED"
    result = "SUCCESS" if access_granted else "FAILURE"

    log_entry = AccessLog(
        user_id=user.id,
        user_name=user.name,
        action=action,
        result=result,
        confidence_score=1.0 if access_granted else 0.0,
        similarity_score=0.0,
        locker_id=locker_id,
        ip_address=ip,
        anomaly_flag=False,
        timestamp=datetime.utcnow(),
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)

    log_access_event(
        log_id=log_entry.id,
        user_id=user.id,
        user_name=user.name,
        action=action,
        result=result,
        confidence_score=1.0 if access_granted else 0.0,
        similarity_score=0.0,
        locker_id=locker_id,
        ip_address=ip,
        anomaly_flag=False,
    )

    if access_granted:
        current = locker_controller.get_status(db, locker_id)
        if current.status != "UNLOCKED":
            locker_controller.open_locker(db, locker_id, user.id)

    return PinAuthResponse(
        access_granted=access_granted,
        user_id=user.id,
        user_name=user.name,
        locker_action=action,
        method="PIN",
        log_id=log_entry.id,
    )
