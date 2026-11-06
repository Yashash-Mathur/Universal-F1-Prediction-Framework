from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session, joinedload

from src.api.dependencies import get_api_key, get_db
from src.api.schemas import DriverProbability, RacePredictionResponse
from src.db.models import ApiKey, Prediction, Race

router = APIRouter(prefix="/api/v1/predictions", tags=["Predictions"])


def _to_driver_prob(p: Prediction) -> DriverProbability:
    return DriverProbability(
        rank=p.predicted_rank,
        code=p.driver.code,
        full_name=p.driver.full_name,
        team_name=p.driver.team_name,
        podium_probability=round(p.podium_probability, 4),
        predicted_podium=p.predicted_podium,
    )


_limiter = Limiter(key_func=get_remote_address)


@router.get("/{race_id}", response_model=RacePredictionResponse)
@_limiter.limit("20/minute")
def get_prediction(
    race_id: int,
    request: Request,
    db: Session = Depends(get_db),
    api_key: ApiKey | None = Depends(get_api_key),
):
    race = db.get(Race, race_id)
    if not race:
        raise HTTPException(status_code=404, detail="Race not found.")

    preds = (
        db.query(Prediction)
        .options(joinedload(Prediction.driver), joinedload(Prediction.model_version))
        .filter(Prediction.race_id == race_id)
        .order_by(Prediction.predicted_rank)
        .all()
    )
    if not preds:
        raise HTTPException(status_code=404, detail="No predictions found for this race.")

    full_grid = [_to_driver_prob(p) for p in preds]
    top_3 = [p for p in full_grid if p.predicted_podium][:3]
    version_tag = preds[0].model_version.version_tag if preds else "unknown"
    stage = preds[0].stage if preds else "unknown"

    return RacePredictionResponse(
        race_id=race_id,
        race_name=race.race_name,
        model_version=version_tag,
        stage=stage,
        top_3=top_3,
        full_grid=full_grid,
    )
