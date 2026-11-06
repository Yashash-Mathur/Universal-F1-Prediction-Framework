"""
Database seeder.

Populates the database with:
  - 2026 race calendar (from f1_2020_2026_features_v6.csv)
  - Known drivers (from same CSV)
  - Model v6-catboost-300 walk-forward metrics (recorded results)
  - Dutch GP predictions (from data/dutch_gp_catboost_prediction_final.csv)
  - Default dev API key

Usage:
    python -m src.db.seed
"""

import hashlib
import os
import sys

import pandas as pd

from src.db.session import SessionLocal, init_db
from src.db.models import ApiKey, Driver, ModelVersion, Prediction, Race

FEATURES_CSV = "data/f1_2020_2026_features_v6.csv"
DUTCH_PRED_CSV = "data/dutch_gp_catboost_prediction_final.csv"
DEV_KEY = os.getenv("DEFAULT_ADMIN_KEY", "f1_dev_key_2026")


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def seed() -> None:
    init_db()
    db = SessionLocal()
    try:
        # ── Model version ──────────────────────────────────────────────────
        mv = db.query(ModelVersion).filter_by(version_tag="v6-catboost-300").first()
        if not mv:
            mv = ModelVersion(
                version_tag="v6-catboost-300",
                training_date="2026-11-06",
                accuracy=0.8926,
                precision=0.6061,
                recall=0.5455,
                f1_score=0.5619,
                mean_podiums=1.82,
                perfect_podiums=2,
                races_evaluated=11,
            )
            db.add(mv)
            db.flush()
        print(f"Model version: {mv.version_tag} (id={mv.id})")

        # ── Races & Drivers from features CSV ──────────────────────────────
        df = pd.read_csv(FEATURES_CSV)
        df_2026 = df[df["Year"] == 2026][["RoundNumber", "RaceName"]].drop_duplicates()

        race_map: dict[tuple, Race] = {}
        for _, row in df_2026.iterrows():
            key = (2026, int(row["RoundNumber"]))
            race = db.query(Race).filter_by(year=key[0], round_number=key[1]).first()
            if not race:
                race = Race(year=2026, round_number=int(row["RoundNumber"]), race_name=row["RaceName"])
                db.add(race)
                db.flush()
            race_map[key] = race
        print(f"Races seeded: {len(race_map)}")

        driver_rows = df[["Abbreviation", "FullName", "TeamName"]].drop_duplicates("Abbreviation")
        driver_map: dict[str, Driver] = {}
        for _, row in driver_rows.iterrows():
            code = row["Abbreviation"]
            d = db.query(Driver).filter_by(code=code).first()
            if not d:
                d = Driver(code=code, full_name=row["FullName"], team_name=row["TeamName"])
                db.add(d)
                db.flush()
            driver_map[code] = d
        print(f"Drivers seeded: {len(driver_map)}")

        # ── Dutch GP Predictions ───────────────────────────────────────────
        dutch_race = db.query(Race).filter_by(year=2026, round_number=12).first()
        if not dutch_race:
            dutch_race = Race(year=2026, round_number=12, race_name="Dutch Grand Prix")
            db.add(dutch_race)
            db.flush()
        race_map[(2026, 12)] = dutch_race

        if os.path.exists(DUTCH_PRED_CSV):
            pred_df = pd.read_csv(DUTCH_PRED_CSV)
            existing = db.query(Prediction).filter_by(race_id=dutch_race.id, stage="post_quali").count()
            if existing == 0:
                for _, row in pred_df.iterrows():
                    code = row["Abbreviation"]
                    if code not in driver_map:
                        d = Driver(code=code, full_name=row.get("FullName", code),
                                   team_name=row.get("TeamName"))
                        db.add(d)
                        db.flush()
                        driver_map[code] = d
                    p = Prediction(
                        race_id=dutch_race.id,
                        driver_id=driver_map[code].id,
                        model_version_id=mv.id,
                        stage="post_quali",
                        podium_probability=float(row["PodiumProbability"]),
                        predicted_rank=int(row["PredictedRank"]),
                        predicted_podium=bool(row["PredictedPodium"]),
                    )
                    db.add(p)
                print(f"Dutch GP predictions seeded: {len(pred_df)}")
            else:
                print("Dutch GP predictions already present — skipped.")

        # ── Dev API key ────────────────────────────────────────────────────
        h = _hash(DEV_KEY)
        if not db.query(ApiKey).filter_by(key_hash=h).first():
            db.add(ApiKey(key_hash=h, client_name="dev", tier="admin"))
            print(f"Dev API key seeded: {DEV_KEY}")
        else:
            print("Dev API key already present.")

        db.commit()
        print("Seed complete.")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed()
