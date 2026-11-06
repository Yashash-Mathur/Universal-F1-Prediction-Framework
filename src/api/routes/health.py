from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from src.api.dependencies import get_db
from src.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "error"
    return HealthResponse(status="ok", db=db_status)
