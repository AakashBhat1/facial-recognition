import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from models.database import (
    AccessLog,
    FaceEmbedding,
    LockerState,
    User,
    get_db,
)
from models.schemas import DeleteUserResponse, UserResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/users", tags=["Users"])


def _to_user_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        name=user.name,
        assigned_locker_id=user.assigned_locker_id,
        has_pin=bool(user.pin_hash),
        created_at=user.created_at,
    )


@router.get("", response_model=list[UserResponse])
def list_users(db: Session = Depends(get_db)) -> list[UserResponse]:
    """Return all enrolled users (without embeddings). Used by demo to find user_id."""
    users = db.query(User).order_by(User.id.asc()).all()
    return [_to_user_response(u) for u in users]


@router.delete("/{user_id}", response_model=DeleteUserResponse)
def delete_user(user_id: int, db: Session = Depends(get_db)) -> DeleteUserResponse:
    """Delete a user along with their face embeddings, access logs, and assigned locker.

    This is a destructive operation: the user, every ``FaceEmbedding`` row,
    every ``AccessLog`` row owned by them, and the matching ``LockerState``
    row are removed in a single transaction.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User id={user_id} not found")

    user_name = user.name
    locker_id = user.assigned_locker_id

    embeddings_deleted = (
        db.query(FaceEmbedding).filter(FaceEmbedding.user_id == user_id).delete()
    )
    access_logs_deleted = (
        db.query(AccessLog).filter(AccessLog.user_id == user_id).delete()
    )

    locker_freed: str | None = None
    if locker_id:
        deleted_lockers = (
            db.query(LockerState).filter(LockerState.locker_id == locker_id).delete()
        )
        if deleted_lockers:
            locker_freed = locker_id

    db.delete(user)
    db.commit()

    logger.info(
        "Deleted user '%s' (id=%d): %d embeddings, %d access logs, locker=%s",
        user_name, user_id, embeddings_deleted, access_logs_deleted, locker_freed,
    )

    return DeleteUserResponse(
        user_id=user_id,
        user_name=user_name,
        embeddings_deleted=embeddings_deleted,
        access_logs_deleted=access_logs_deleted,
        locker_freed=locker_freed,
        message=f"User '{user_name}' (id={user_id}) deleted with all associated data.",
    )
