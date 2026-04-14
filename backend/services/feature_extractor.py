"""Shared feature engineering for ML anomaly detection.

Single source of truth for the feature vector used by the data generator,
model trainer, and runtime scorer. Keeping all feature logic here prevents
train/serve skew.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from config import (
    BRUTE_FORCE_WINDOW_SECONDS,
    OFF_HOURS_END,
    OFF_HOURS_START,
    RAPID_ACCESS_WINDOW_MINUTES,
)
from models.database import AccessLog

FEATURE_NAMES: list[str] = [
    "hour",
    "day_of_week",
    "is_weekend",
    "is_off_hours",
    "seconds_since_last_access",
    "rolling_failure_rate",
    "recent_ip_failures_60s",
    "recent_unknown_count_5m",
    "recent_user_access_count_10m",
    "is_known_user",
    "is_usual_ip_for_user",
    "access_frequency_zscore",
    "confidence_score",
    "similarity_score",
]

_DEFAULT_SECONDS_SINCE_LAST = 86400.0  # 24 h when no prior access exists


def extract_features_from_db(
    db: Session,
    user_id: int | None,
    ip_address: str,
    timestamp: datetime,
    *,
    result: str | None = None,
    confidence_score: float | None = None,
    similarity_score: float | None = None,
) -> dict[str, float]:
    """Extract the feature vector from live DB context (runtime scoring)."""
    hour = float(timestamp.hour)
    day_of_week = float(timestamp.weekday())
    is_weekend = 1.0 if timestamp.weekday() >= 5 else 0.0
    is_off_hours = 1.0 if _is_off_hours(timestamp) else 0.0
    is_known_user = 1.0 if user_id is not None else 0.0

    seconds_since_last = _seconds_since_last_access(db, user_id, timestamp)
    rolling_failure_rate = _rolling_failure_rate(db, ip_address, timestamp)
    recent_ip_failures_60s = _recent_ip_failures_60s(db, ip_address, timestamp)
    recent_unknown_count_5m = _recent_unknown_count_5m(db, ip_address, timestamp)
    recent_user_access_count_10m = _recent_user_access_count_10m(db, user_id, timestamp)
    is_usual_ip_for_user = _is_usual_ip_for_user(db, user_id, ip_address, timestamp)
    access_freq_zscore = _access_frequency_zscore(db, user_id, timestamp)

    return {
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_off_hours": is_off_hours,
        "seconds_since_last_access": seconds_since_last,
        "rolling_failure_rate": rolling_failure_rate,
        "recent_ip_failures_60s": recent_ip_failures_60s,
        "recent_unknown_count_5m": recent_unknown_count_5m,
        "recent_user_access_count_10m": recent_user_access_count_10m,
        "is_known_user": is_known_user,
        "is_usual_ip_for_user": is_usual_ip_for_user,
        "access_frequency_zscore": access_freq_zscore,
        "confidence_score": float(confidence_score or 0.0),
        "similarity_score": float(similarity_score or 0.0),
    }


def extract_features_from_row(row: dict[str, Any]) -> list[float]:
    """Convert a pre-computed feature dict/row to the canonical ordered list."""
    return [float(row[name]) for name in FEATURE_NAMES]


# ─── Private helpers ──────────────────────────────────────────────────────────

def _is_off_hours(timestamp: datetime) -> bool:
    hour = timestamp.hour
    return OFF_HOURS_START <= hour or hour < OFF_HOURS_END


def _seconds_since_last_access(
    db: Session, user_id: int | None, timestamp: datetime,
) -> float:
    if user_id is None:
        return _DEFAULT_SECONDS_SINCE_LAST

    prev = (
        db.query(AccessLog.timestamp)
        .filter(AccessLog.user_id == user_id, AccessLog.timestamp < timestamp)
        .order_by(AccessLog.timestamp.desc())
        .first()
    )
    if prev is None:
        return _DEFAULT_SECONDS_SINCE_LAST
    delta = (timestamp - prev[0]).total_seconds()
    return max(delta, 0.0)


def _rolling_failure_rate(
    db: Session, ip_address: str, timestamp: datetime,
) -> float:
    """Failure count / total in the last 20 access attempts for this IP."""
    recent = (
        db.query(AccessLog.result)
        .filter(AccessLog.ip_address == ip_address, AccessLog.timestamp <= timestamp)
        .order_by(AccessLog.timestamp.desc())
        .limit(20)
        .all()
    )
    if not recent:
        return 0.0
    failures = sum(1 for (r,) in recent if r == "FAILURE")
    return failures / len(recent)


def _recent_ip_failures_60s(
    db: Session, ip_address: str, timestamp: datetime,
) -> float:
    window_start = timestamp - timedelta(seconds=BRUTE_FORCE_WINDOW_SECONDS)
    count = (
        db.query(func.count(AccessLog.id))
        .filter(
            AccessLog.ip_address == ip_address,
            AccessLog.result == "FAILURE",
            AccessLog.timestamp >= window_start,
            AccessLog.timestamp <= timestamp,
        )
        .scalar()
    ) or 0
    return float(count)


def _recent_unknown_count_5m(
    db: Session, ip_address: str, timestamp: datetime,
) -> float:
    window_start = timestamp - timedelta(minutes=5)
    count = (
        db.query(func.count(AccessLog.id))
        .filter(
            AccessLog.ip_address == ip_address,
            AccessLog.user_id.is_(None),
            AccessLog.timestamp >= window_start,
            AccessLog.timestamp <= timestamp,
        )
        .scalar()
    ) or 0
    return float(count)


def _recent_user_access_count_10m(
    db: Session, user_id: int | None, timestamp: datetime,
) -> float:
    if user_id is None:
        return 0.0

    window_start = timestamp - timedelta(minutes=RAPID_ACCESS_WINDOW_MINUTES)
    count = (
        db.query(func.count(AccessLog.id))
        .filter(
            AccessLog.user_id == user_id,
            AccessLog.timestamp >= window_start,
            AccessLog.timestamp <= timestamp,
        )
        .scalar()
    ) or 0
    return float(count)


def _is_usual_ip_for_user(
    db: Session, user_id: int | None, ip_address: str, timestamp: datetime,
) -> float:
    if user_id is None:
        return 0.0

    prior_accesses = (
        db.query(func.count(AccessLog.id))
        .filter(
            AccessLog.user_id == user_id,
            AccessLog.timestamp < timestamp,
        )
        .scalar()
    ) or 0
    if prior_accesses == 0:
        return 1.0

    seen_this_ip = (
        db.query(func.count(AccessLog.id))
        .filter(
            AccessLog.user_id == user_id,
            AccessLog.ip_address == ip_address,
            AccessLog.timestamp < timestamp,
        )
        .scalar()
    ) or 0
    return 1.0 if seen_this_ip > 0 else 0.0


def _access_frequency_zscore(
    db: Session, user_id: int | None, timestamp: datetime,
) -> float:
    """Z-score of today's access count vs the user's trailing 7-day daily mean."""
    if user_id is None:
        return 0.0

    today_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    today_count = (
        db.query(func.count(AccessLog.id))
        .filter(
            AccessLog.user_id == user_id,
            AccessLog.timestamp >= today_start,
            AccessLog.timestamp <= timestamp,
        )
        .scalar()
    ) or 0

    # Daily counts for trailing 7 days (excluding today)
    daily_counts: list[int] = []
    for days_ago in range(1, 8):
        day_start = today_start - timedelta(days=days_ago)
        day_end = day_start + timedelta(days=1)
        count = (
            db.query(func.count(AccessLog.id))
            .filter(
                AccessLog.user_id == user_id,
                AccessLog.timestamp >= day_start,
                AccessLog.timestamp < day_end,
            )
            .scalar()
        ) or 0
        daily_counts.append(count)

    if not daily_counts:
        return 0.0

    mean = sum(daily_counts) / len(daily_counts)
    variance = sum((c - mean) ** 2 for c in daily_counts) / len(daily_counts)
    std = max(variance ** 0.5, 1.0)  # floor at 1.0 to avoid div-by-zero
    return (today_count - mean) / std
