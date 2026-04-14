from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from models.database import get_db
from models.schemas import LockerStatusResponse
from services import locker_controller

router = APIRouter(prefix="/api/locker", tags=["Locker"])


@router.get("/status", response_model=LockerStatusResponse)
def locker_status(locker_id: str = "L001", db: Session = Depends(get_db)):
    """Get the current state of a locker. Read-only."""
    return locker_controller.get_status(db, locker_id)
