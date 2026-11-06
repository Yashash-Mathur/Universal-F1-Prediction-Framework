"""
Minimal FastF1 data collector.

Pulls only race and qualifying results without telemetry/weather bloat.
"""

import os
import pandas as pd
import fastf1

fastf1.Cache.enable_cache("cache")


def collect_race_session(year: int, race_name: str, event_date: str = None) -> pd.DataFrame:
    """
    Load qualifying and race results for a single Grand Prix event.
    Loads results without heavy telemetry, weather, or lap data.
    """
    quali = fastf1.get_session(year, race_name, "Q")
    quali.load(telemetry=False, weather=False, messages=False)

    race = fastf1.get_session(year, race_name, "R")
    race.load(telemetry=False, weather=False, messages=False)

    # Best qualifying time across Q1, Q2, Q3
    bestqualitime = quali.results[["Q3", "Q2", "Q1"]].min(axis=1)
    quali.results["bestqualitime"] = bestqualitime
    pole_time = bestqualitime.min()
    quali.results["gaptopole_bestquali"] = (bestqualitime - pole_time).dt.total_seconds()

    quali_cols = [
        "DriverNumber", "Abbreviation", "DriverId", "FullName",
        "TeamName", "TeamId", "Q1", "Q2", "Q3",
        "bestqualitime", "gaptopole_bestquali"
    ]
    race_cols = ["Abbreviation", "GridPosition", "Position", "Points"]

    quali_df = quali.results[[c for c in quali_cols if c in quali.results.columns]]
    race_df = race.results[[c for c in race_cols if c in race.results.columns]]

    race_dataset = pd.merge(quali_df, race_df, on="Abbreviation", how="inner")
    race_dataset["Year"] = year
    race_dataset["RoundNumber"] = race.event["RoundNumber"]
    race_dataset["RaceName"] = race_name
    if event_date is not None:
        race_dataset["EventDate"] = pd.to_datetime(event_date)
    else:
        race_dataset["EventDate"] = pd.to_datetime(race.event.get("EventDate", pd.Timestamp.today()))

    return race_dataset
