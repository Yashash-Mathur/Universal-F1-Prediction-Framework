"""
Pydantic v2 response schemas.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    db: str
    version: str = "v6-catboost-300"


class RaceResponse(BaseModel):
    id: int
    year: int
    round_number: int
    race_name: str
    circuit_type: str | None = None
    event_date: str | None = None

    model_config = {"from_attributes": True}


class DriverProbability(BaseModel):
    rank: int
    code: str
    full_name: str
    team_name: str | None
    podium_probability: float
    predicted_podium: bool

    model_config = {"from_attributes": True}


class RacePredictionResponse(BaseModel):
    race_id: int
    race_name: str
    model_version: str
    stage: str
    top_3: list[DriverProbability]
    full_grid: list[DriverProbability]


class ModelMetadataResponse(BaseModel):
    version_tag: str
    training_date: str | None
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1_score: float | None
    mean_podiums: float | None
    perfect_podiums: int | None
    races_evaluated: int | None

    model_config = {"from_attributes": True}
