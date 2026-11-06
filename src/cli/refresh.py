"""
Ingestion CLI — refresh predictions in the database.

Usage:
    python -m src.cli.refresh --features data/f1_2020_2026_features_v6.csv \
        --target data/<race>_prediction_fixed.csv \
        --train-year 2026 --train-through-round 11 \
        --race-id 12

This script:
  1. Runs CatBoost inference via src.prediction.predict
  2. Atomically upserts predictions into the database

It never runs live FastF1 collection — supply a pre-built target CSV.
Data collection (FastF1) is a separate manual step using src.data_collection.collector.
"""

import argparse
import hashlib
import tempfile
import os

import pandas as pd

from src.db.session import SessionLocal, init_db
from src.db.models import Driver, ModelVersion, Prediction, Race
from src.prediction.predict import predict as run_inference


def refresh(features_path: str, target_path: str,
            train_year: int, train_through_round: int, race_id: int) -> None:
    init_db()
    db = SessionLocal()
    try:
        race = db.get(Race, race_id)
        if not race:
            raise ValueError(f"Race id={race_id} not found in database. Run seed first.")

        mv = db.query(ModelVersion).order_by(ModelVersion.id.desc()).first()
        if not mv:
            raise ValueError("No model version found. Run seed first.")

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        ranking = run_inference(
            features_path=features_path,
            target_path=target_path,
            train_year=train_year,
            train_through_round=train_through_round,
            output_path=tmp_path,
        )

        # Upsert predictions
        for _, row in ranking.iterrows():
            code = row["Abbreviation"]
            d = db.query(Driver).filter_by(code=code).first()
            if not d:
                d = Driver(code=code, full_name=row.get("FullName", code),
                           team_name=row.get("TeamName"))
                db.add(d)
                db.flush()

            existing = db.query(Prediction).filter_by(
                race_id=race_id, driver_id=d.id, stage="post_quali"
            ).first()

            if existing:
                existing.podium_probability = float(row["PodiumProbability"])
                existing.predicted_rank = int(row["PredictedRank"])
                existing.predicted_podium = bool(row["PredictedPodium"])
            else:
                db.add(Prediction(
                    race_id=race_id,
                    driver_id=d.id,
                    model_version_id=mv.id,
                    stage="post_quali",
                    podium_probability=float(row["PodiumProbability"]),
                    predicted_rank=int(row["PredictedRank"]),
                    predicted_podium=bool(row["PredictedPodium"]),
                ))

        db.commit()
        print(f"Refreshed predictions for race_id={race_id} ({race.race_name}).")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="Refresh race predictions in the database.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--train-year", type=int, required=True)
    parser.add_argument("--train-through-round", type=int, required=True)
    parser.add_argument("--race-id", type=int, required=True)
    args = parser.parse_args()
    refresh(args.features, args.target, args.train_year, args.train_through_round, args.race_id)


if __name__ == "__main__":
    main()
