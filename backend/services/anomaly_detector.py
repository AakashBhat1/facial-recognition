from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from models.database import AccessLog, Alert
from services.logger_service import log_alert
from services import ml_anomaly_service
from config import (
    BRUTE_FORCE_LIMIT, BRUTE_FORCE_WINDOW_SECONDS,
    OFF_HOURS_START, OFF_HOURS_END,
    RAPID_ACCESS_LIMIT, RAPID_ACCESS_WINDOW_MINUTES,
)


def _create_alert(db: Session, alert_type: str, severity: str, description: str, user_id: int | None = None) -> Alert:
    alert = Alert(
        type=alert_type,
        severity=severity,
        user_id=user_id,
        description=description,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    log_alert(alert_type, severity, description)
    return alert


def run_all_checks(
    db:           Session,
    ip_address:   str,
    user_id:      int | None,
    result:       str,
    now:          datetime | None = None,
    log_entry_id: int | None = None,
) -> bool:
    """
    Run all anomaly rules after every access attempt.
    Returns True if any anomaly was detected.
    """
    now = now or datetime.utcnow()
    flagged = False

    flagged |= _check_brute_force(db, ip_address, now)
    flagged |= _check_off_hours(db, user_id, now)
    if user_id:
        flagged |= _check_rapid_access(db, user_id, now)
    if not user_id and result == "FAILURE":
        flagged |= _check_repeated_unknown(db, ip_address, now)

    flagged |= _check_ml_anomaly(db, user_id, ip_address, now, log_entry_id)

    return flagged


# ─── Rule 1: Brute Force ───────────────────────────────────────────────────────
def _check_brute_force(db: Session, ip_address: str, now: datetime) -> bool:
    window_start = now - timedelta(seconds=BRUTE_FORCE_WINDOW_SECONDS)
    recent_failures = (
        db.query(AccessLog)
        .filter(
            AccessLog.ip_address == ip_address,
            AccessLog.result == "FAILURE",
            AccessLog.timestamp >= window_start,
        )
        .count()
    )
    if recent_failures >= BRUTE_FORCE_LIMIT:
        _create_alert(
            db,
            alert_type="BRUTE_FORCE",
            severity="HIGH",
            description=f"{recent_failures} failed attempts from IP {ip_address} in the last {BRUTE_FORCE_WINDOW_SECONDS}s.",
        )
        return True
    return False


# ─── Rule 2: Off-Hours Access ─────────────────────────────────────────────────
def _check_off_hours(db: Session, user_id: int | None, now: datetime) -> bool:
    hour = now.hour
    if OFF_HOURS_START <= hour or hour < OFF_HOURS_END:
        _create_alert(
            db,
            alert_type="OFF_HOURS",
            severity="MEDIUM",
            user_id=user_id,
            description=f"Access attempt at {now.strftime('%H:%M')} UTC (off-hours window: {OFF_HOURS_START}:00–{OFF_HOURS_END}:00).",
        )
        return True
    return False


# ─── Rule 3: Rapid Repeated Access ────────────────────────────────────────────
def _check_rapid_access(db: Session, user_id: int, now: datetime) -> bool:
    window_start = now - timedelta(minutes=RAPID_ACCESS_WINDOW_MINUTES)
    recent_count = (
        db.query(AccessLog)
        .filter(
            AccessLog.user_id == user_id,
            AccessLog.result  == "SUCCESS",
            AccessLog.timestamp >= window_start,
        )
        .count()
    )
    if recent_count >= RAPID_ACCESS_LIMIT:
        _create_alert(
            db,
            alert_type="RAPID_ACCESS",
            severity="LOW",
            user_id=user_id,
            description=f"User {user_id} accessed the locker {recent_count} times in {RAPID_ACCESS_WINDOW_MINUTES} minutes.",
        )
        return True
    return False


# ─── Rule 4: Repeated Unknown Face ────────────────────────────────────────────
def _check_repeated_unknown(db: Session, ip_address: str, now: datetime) -> bool:
    window_start = now - timedelta(minutes=5)
    unknown_count = (
        db.query(AccessLog)
        .filter(
            AccessLog.ip_address == ip_address,
            AccessLog.user_id    == None,
            AccessLog.timestamp  >= window_start,
        )
        .count()
    )
    if unknown_count >= 2:
        _create_alert(
            db,
            alert_type="REPEATED_UNKNOWN",
            severity="HIGH",
            description=f"Unknown face detected {unknown_count} times from IP {ip_address} in the last 5 minutes.",
        )
        return True
    return False


# ─── Rule 5: ML Anomaly (Isolation Forest) ──────────────────────────────────
def _check_ml_anomaly(
    db: Session,
    user_id: int | None,
    ip_address: str,
    now: datetime,
    log_entry_id: int | None,
) -> bool:
    log_entry = None
    if log_entry_id is not None:
        log_entry = db.query(AccessLog).filter(AccessLog.id == log_entry_id).first()

    score = ml_anomaly_service.score_access_event(
        db,
        user_id,
        ip_address,
        now,
        result=log_entry.result if log_entry else None,
        confidence_score=log_entry.confidence_score if log_entry else None,
        similarity_score=log_entry.similarity_score if log_entry else None,
    )
    if score is None:
        return False

    # Persist the score on the access log entry
    if log_entry is not None:
        log_entry.ml_anomaly_score = score
        db.commit()

    if ml_anomaly_service.is_anomalous(score):
        _create_alert(
            db,
            alert_type="ML_ANOMALY",
            severity="MEDIUM",
            user_id=user_id,
            description=f"ML model flagged anomalous access (score={score:.4f}) from IP {ip_address}.",
        )
        return True
    return False
