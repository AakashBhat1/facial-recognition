import json
import logging
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DATABASE_URL

logger = logging.getLogger(__name__)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def locker_id_for_user(user_id: int) -> str:
    """Convention: each user owns the locker named L<id padded to 3 digits>."""
    return f"L{user_id:03d}"


# ─── Tables ────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String, nullable=False)
    embedding  = Column(Text, nullable=False)   # JSON string of 128-dim float list
    created_at = Column(DateTime, default=datetime.utcnow)
    personal_threshold = Column(Float, nullable=True, default=None)
    pin_hash   = Column(String, nullable=True, default=None)
    assigned_locker_id = Column(String, nullable=True, index=True)

    def get_embedding(self) -> list[float]:
        return json.loads(self.embedding)


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, nullable=False, index=True)
    embedding    = Column(Text, nullable=False)   # JSON string of 512-dim float list
    is_centroid  = Column(Boolean, default=False)
    model_name   = Column(String, default="arcface")  # "arcface" or "mobilefacenet"
    model_version = Column(String, default="buffalo_l_v1")  # Phase 4 versioning
    created_at   = Column(DateTime, default=datetime.utcnow)

    def get_embedding(self) -> list[float]:
        return json.loads(self.embedding)


class AccessLog(Base):
    __tablename__ = "access_logs"

    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, nullable=True)   # null if unknown
    user_name        = Column(String, default="UNKNOWN")
    action           = Column(String, nullable=False)   # OPEN, CLOSE, ACCESS_DENIED
    result           = Column(String, nullable=False)   # SUCCESS, FAILURE
    confidence_score = Column(Float, default=0.0)
    similarity_score = Column(Float, default=0.0)
    locker_id        = Column(String, default="L001")
    ip_address       = Column(String, default="unknown")
    anomaly_flag     = Column(Boolean, default=False)
    ml_anomaly_score = Column(Float, nullable=True, default=None)
    timestamp        = Column(DateTime, default=datetime.utcnow)


class Alert(Base):
    __tablename__ = "alerts"

    id          = Column(Integer, primary_key=True, index=True)
    type        = Column(String, nullable=False)    # BRUTE_FORCE, OFF_HOURS, RAPID_ACCESS, REPEATED_UNKNOWN, ML_ANOMALY
    severity    = Column(String, nullable=False)    # LOW, MEDIUM, HIGH
    user_id     = Column(Integer, nullable=True)
    description = Column(Text, nullable=False)
    resolved    = Column(Boolean, default=False)
    timestamp   = Column(DateTime, default=datetime.utcnow)


class LockerState(Base):
    __tablename__ = "locker_state"

    locker_id    = Column(String, primary_key=True)
    status       = Column(String, default="LOCKED")  # LOCKED, UNLOCKED
    last_user_id = Column(Integer, nullable=True)
    updated_at   = Column(DateTime, default=datetime.utcnow)


# ─── Init ──────────────────────────────────────────────────────────────────────

def _migrate_assigned_locker_id() -> None:
    """Idempotent SQLite migration: add users.assigned_locker_id if missing."""
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return
    columns = {col["name"] for col in inspector.get_columns("users")}
    if "assigned_locker_id" in columns:
        return
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE users ADD COLUMN assigned_locker_id VARCHAR"))
    logger.info("Added users.assigned_locker_id column")


def _backfill_assigned_lockers() -> None:
    """Assign L<id> to any user missing assigned_locker_id and seed LockerState rows."""
    db = SessionLocal()
    try:
        users = db.query(User).filter(
            (User.assigned_locker_id.is_(None)) | (User.assigned_locker_id == "")
        ).order_by(User.id.asc()).all()
        for user in users:
            user.assigned_locker_id = locker_id_for_user(user.id)
        if users:
            db.commit()
            logger.info("Backfilled assigned_locker_id for %d users", len(users))

        all_users = db.query(User).all()
        for user in all_users:
            locker_id = user.assigned_locker_id or locker_id_for_user(user.id)
            existing = db.query(LockerState).filter_by(locker_id=locker_id).first()
            if not existing:
                db.add(LockerState(locker_id=locker_id, status="LOCKED"))
        db.commit()
    finally:
        db.close()


def init_db():
    """Create all tables, run idempotent migrations, and seed default state."""
    Base.metadata.create_all(bind=engine)
    _migrate_assigned_locker_id()
    db = SessionLocal()
    try:
        if not db.query(LockerState).filter_by(locker_id="L001").first():
            db.add(LockerState(locker_id="L001", status="LOCKED"))
            db.commit()
    finally:
        db.close()
    _backfill_assigned_lockers()


def get_db():
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
