from fastapi import APIRouter, Depends, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from src.api.dependencies import get_api_key, get_db
from src.api.schemas import RaceResponse
from src.db.models import ApiKey, Race

router = APIRouter(prefix="/api/v1/races", tags=["Races"])

# shortcut: limiter imported here to apply decorators; app.state.limiter owns the shared instance
_limiter = Limiter(key_func=get_remote_address)


@router.get("", response_model=list[RaceResponse])
@_limiter.limit("30/minute")
def list_races(
    request: Request,
    year: int | None = Query(default=None),
    db: Session = Depends(get_db),
    api_key: ApiKey | None = Depends(get_api_key),
):
    q = db.query(Race)
    if year:
        q = q.filter(Race.year == year)
    return q.order_by(Race.year, Race.round_number).all()


@router.get("/{race_id}", response_model=RaceResponse)
def get_race(
    race_id: int,
    db: Session = Depends(get_db),
    api_key: ApiKey | None = Depends(get_api_key),
):
    race = db.get(Race, race_id)
    if not race:
        raise HTTPException(status_code=404, detail="Race not found.")
    return race
