"""
Automated race refresh worker.

Monitors the current season calendar and triggers prediction updates
as soon as new session data (qualifying or race) becomes available.

Usage:
    # Run once (e.g. from cron):
    python -m src.cli.scheduler --once

    # Run continuously (e.g. background container):
    python -m src.cli.scheduler --interval 3600
"""

import argparse
import logging
import time
from datetime import datetime, timezone

import fastf1
import pandas as pd

from src.db.session import SessionLocal, init_db
from src.db.models import Race, Prediction
from src.cli.refresh import refresh

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("f1_scheduler")


def check_and_update(year: int = 2026, features_path: str = "data/f1_2020_2026_features_v6.csv") -> None:
    """Check if any race in the calendar needs prediction refresh."""
    init_db()
    db = SessionLocal()
    try:
        races = db.query(Race).filter(Race.year == year).order_by(Race.round_number).all()
        logger.info(f"Checking {len(races)} scheduled races for season {year}...")

        for race in races:
            existing_preds = (
                db.query(Prediction)
                .filter(Prediction.race_id == race.id, Prediction.stage == "post_quali")
                .count()
            )

            # Target fixed file if exists (e.g. Dutch GP or newly injected race)
            target_path = f"data/f1_2020_2026_{race.race_name.lower().replace(' ', '_')}_prediction_fixed.csv"

            # Check special naming convention fallback
            if "dutch" in race.race_name.lower():
                target_path = "data/f1_2020_2026_dutch_gp_prediction_fixed.csv"

            import os
            if os.path.exists(target_path) and existing_preds == 0:
                logger.info(f"Target data found for {race.race_name} (Round {race.round_number}). Refreshing predictions...")
                refresh(
                    features_path=features_path,
                    target_path=target_path,
                    train_year=year,
                    train_through_round=max(1, race.round_number - 1),
                    race_id=race.id,
                )
                logger.info(f"Successfully refreshed {race.race_name}.")

    except Exception as e:
        logger.error(f"Error during check and update: {e}")
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Automated F1 prediction refresh scheduler.")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--once", action="store_true", help="Run a single check and exit.")
    parser.add_argument("--interval", type=int, default=3600, help="Poll interval in seconds (default: 1 hour).")
    args = parser.parse_args()

    if args.once:
        check_and_update(year=args.year)
    else:
        logger.info(f"Starting F1 prediction scheduler (polling every {args.interval}s)...")
        while True:
            check_and_update(year=args.year)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
