from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal, Optional


# ─── Enroll (Phase 1) ─────────────────────────────────────────────────────────

class EnrollResponse(BaseModel):
    user_id: int
    user_name: str
    embedding_count: int
    assigned_locker_id: Optional[str]
    created_at: datetime


class ReEnrollResponse(BaseModel):
    user_id: int
    user_name: str
    embedding_count: int
    personal_threshold: Optional[float]
    assigned_locker_id: Optional[str]
    updated_at: datetime


# ─── Multi-Frame Recognition (Phase 2) ────────────────────────────────────────

class FrameResultResponse(BaseModel):
    frame_index: int
    quality_passed: bool
    rejection_reasons: list[str]
    antispoof_passed: bool = True
    anomaly_passed: bool
    score: Optional[float]


class LivenessResultResponse(BaseModel):
    passed: bool
    blink_detected: bool
    head_movement_detected: bool
    reason: str


class PromptResponse(BaseModel):
    message: str
    category: str


class RecognizeMultiFrameResponse(BaseModel):
    access_granted: bool
    user_id: Optional[int]
    user_name: str
    locker_action: Literal["OPEN", "ACCESS_DENIED"]
    final_score: float
    confidence: float
    liveness: Optional[LivenessResultResponse]
    prompt: Optional[PromptResponse]
    frame_results: list[FrameResultResponse]
    frames_processed: int
    frames_passed_quality: int
    frames_passed_anomaly: int
    log_id: int
    anomaly_flag: bool


# ─── Users ─────────────────────────────────────────────────────────────────────

class UserResponse(BaseModel):
    id: int
    name: str
    assigned_locker_id: Optional[str] = None
    has_pin: bool = False
    created_at: datetime

    class Config:
        from_attributes = True


class DeleteUserResponse(BaseModel):
    user_id: int
    user_name: str
    embeddings_deleted: int
    access_logs_deleted: int
    locker_freed: Optional[str]
    message: str


# ─── Locker ────────────────────────────────────────────────────────────────────

class LockerStatusResponse(BaseModel):
    locker_id: str
    status: str
    last_user_id: Optional[int]
    updated_at: datetime

    class Config:
        from_attributes = True


# ─── PIN Auth (Phase 4) ───────────────────────────────────────────────────────

class PinSetRequest(BaseModel):
    pin: str = Field(..., min_length=4, max_length=8, description="New 4-8 digit PIN")
    old_pin: Optional[str] = Field(
        default=None,
        min_length=4,
        max_length=8,
        description="Required when updating an existing PIN; omit on first-time set",
    )


class PinAuthRequest(BaseModel):
    user_id: int = Field(..., description="ID of the user to authenticate")
    pin: str = Field(..., min_length=4, max_length=8, description="4-8 digit PIN")
    locker_id: Optional[str] = Field(
        default=None,
        description="Locker to open. If omitted, uses the user's assigned locker.",
    )


class PinAuthResponse(BaseModel):
    access_granted: bool
    user_id: int
    user_name: str
    locker_action: Literal["OPEN", "ACCESS_DENIED"]
    method: str = "PIN"
    log_id: int
