from datetime import datetime
from sqlalchemy.orm import Session
from models.database import LockerState


def get_status(db: Session, locker_id: str = "L001") -> LockerState:
    locker = db.query(LockerState).filter_by(locker_id=locker_id).first()
    if not locker:
        locker = LockerState(locker_id=locker_id, status="LOCKED")
        db.add(locker)
        db.commit()
        db.refresh(locker)
    return locker


def open_locker(db: Session, locker_id: str = "L001", user_id: int | None = None) -> LockerState:
    locker = get_status(db, locker_id)
    locker.status       = "UNLOCKED"
    locker.last_user_id = user_id
    locker.updated_at   = datetime.utcnow()
    db.commit()
    db.refresh(locker)
    return locker


def close_locker(db: Session, locker_id: str = "L001") -> LockerState:
    locker = get_status(db, locker_id)
    locker.status     = "LOCKED"
    locker.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(locker)
    return locker
