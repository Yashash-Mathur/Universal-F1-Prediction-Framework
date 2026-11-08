"""
SQLAlchemy ORM models.

Minimal schema covering what the API actually serves:
  - Races (calendar + circuit info)
  - Drivers (identity only)
  - ModelVersions (walk-forward metrics)
  - Predictions (podium probability per driver per race)
  - ApiKeys (hashed credentials + tier for rate-limit policy)
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utc_now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Race(Base):
    __tablename__ = "races"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    race_name: Mapped[str] = mapped_column(String, nullable=False)
    circuit_type: Mapped[str | None] = mapped_column(String)   # street / permanent / high_speed / high_downforce
    event_date: Mapped[str | None] = mapped_column(String)     # ISO date string

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="race")

    __table_args__ = (UniqueConstraint("year", "round_number", name="uq_race_year_round"),)


class Driver(Base):
    __tablename__ = "drivers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(3), unique=True, nullable=False)   # e.g. NOR
    full_name: Mapped[str] = mapped_column(String, nullable=False)
    team_name: Mapped[str | None] = mapped_column(String)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="driver")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version_tag: Mapped[str] = mapped_column(String, unique=True, nullable=False)   # e.g. v6-catboost-300
    training_date: Mapped[str | None] = mapped_column(String)
    # Walk-forward metrics
    accuracy: Mapped[float | None] = mapped_column(Float)
    precision: Mapped[float | None] = mapped_column(Float)
    recall: Mapped[float | None] = mapped_column(Float)
    f1_score: Mapped[float | None] = mapped_column(Float)
    mean_podiums: Mapped[float | None] = mapped_column(Float)
    perfect_podiums: Mapped[int | None] = mapped_column(Integer)
    races_evaluated: Mapped[int | None] = mapped_column(Integer)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model_version")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.id"), nullable=False)
    driver_id: Mapped[int] = mapped_column(ForeignKey("drivers.id"), nullable=False)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), nullable=False)
    stage: Mapped[str] = mapped_column(String, default="post_quali")  # pre_quali | post_quali
    podium_probability: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_podium: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)

    race: Mapped["Race"] = relationship(back_populates="predictions")
    driver: Mapped["Driver"] = relationship(back_populates="predictions")
    model_version: Mapped["ModelVersion"] = relationship(back_populates="predictions")

    __table_args__ = (
        UniqueConstraint("race_id", "driver_id", "stage", name="uq_prediction_race_driver_stage"),
    )


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key_hash: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    client_name: Mapped[str] = mapped_column(String, nullable=False)
    tier: Mapped[str] = mapped_column(String, default="free")     # free | pro | admin
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)

    usage_logs: Mapped[list["ApiUsage"]] = relationship(back_populates="api_key")


class ApiUsage(Base):
    __tablename__ = "api_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    api_key_id: Mapped[int | None] = mapped_column(ForeignKey("api_keys.id"), nullable=True)
    client_ip: Mapped[str] = mapped_column(String, nullable=False)
    endpoint: Mapped[str] = mapped_column(String, nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)

    api_key: Mapped["ApiKey | None"] = relationship(back_populates="usage_logs")
