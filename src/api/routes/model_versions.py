from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.schemas import ModelMetadataResponse
from src.db.models import ModelVersion

router = APIRouter(prefix="/api/v1/models", tags=["Models"])


@router.get("/latest", response_model=ModelMetadataResponse)
def get_latest_model(db: Session = Depends(get_db)):
    mv = db.query(ModelVersion).order_by(ModelVersion.id.desc()).first()
    if not mv:
        raise HTTPException(status_code=404, detail="No model versions found.")
    return mv
