"""
Circuit-type feature engineering.

All circuit performance features are strictly pre-race.

Historical performance uses:
    - races from 2023 onward
    - only races strictly before the current race
"""

import numpy as np
import pandas as pd


STREET_CIRCUITS = {
    "Australian Grand Prix",
    "Azerbaijan Grand Prix",
    "Canadian Grand Prix",
    "Miami Grand Prix",
    "Monaco Grand Prix",
    "Saudi Arabian Grand Prix",
    "Singapore Grand Prix",
    "Las Vegas Grand Prix",
}

HIGH_SPEED_CIRCUITS = {
    "Australian Grand Prix",
    "Austrian Grand Prix",
    "Azerbaijan Grand Prix",
    "Belgian Grand Prix",
    "British Grand Prix",
    "Canadian Grand Prix",
    "Italian Grand Prix",
    "Las Vegas Grand Prix",
    "Mexico City Grand Prix",
    "Miami Grand Prix",
    "Saudi Arabian Grand Prix",
    "Styrian Grand Prix",
    "70th Anniversary Grand Prix",
    "Sakhir Grand Prix",
}

HIGH_DOWNFORCE_CIRCUITS = {
    "Monaco Grand Prix",
    "Singapore Grand Prix",
    "Hungarian Grand Prix",
    "Dutch Grand Prix",
    "Japanese Grand Prix",
    "Spanish Grand Prix",
    "Qatar Grand Prix",
    "British Grand Prix",
}


FEATURE_MAP = {
    "StreetCircuitPerformance": (
        STREET_CIRCUITS,
        "HasStreetCircuitHistory",
    ),
    "PermanentCircuitPerformance": (
        None,
        "HasPermanentCircuitHistory",
    ),
    "HighSpeedCircuitPerformance": (
        HIGH_SPEED_CIRCUITS,
        "HasHighSpeedCircuitHistory",
    ),
    "HighDownforceCircuitPerformance": (
        HIGH_DOWNFORCE_CIRCUITS,
        "HasHighDownforceCircuitHistory",
    ),
}


def add_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add strictly pre-race circuit-type performance features.

    Historical information is restricted to races from 2023 onward
    and excludes the current race entirely.
    """

    df = df.copy()

    permanent_circuits = (
        set(df["RaceName"].unique()) - STREET_CIRCUITS
    )

    # ----------------------------------------------------------
    # Historical races only
    # ----------------------------------------------------------

    history = df.loc[
        df["Year"] >= 2023,
        ["FullName", "Year", "RoundNumber", "RaceName", "Position"],
    ].copy()

    # One record per driver/race.
    # This is the temporal unit of the feature.
    race_keys = ["FullName", "Year", "RoundNumber"]

    race_history = (
        history.groupby(race_keys, sort=False)
        .agg(
            RaceName=("RaceName", "first"),
            PositionSum=("Position", "sum"),
            PositionCount=("Position", "count"),
        )
        .reset_index()
        .sort_values(race_keys)
        .reset_index(drop=True)
    )



    # ----------------------------------------------------------
    # Calculate cumulative pre-race history for each category
    # ----------------------------------------------------------

    for performance_col, (circuits, history_col) in FEATURE_MAP.items():

        if circuits is None:
            category_mask = race_history["RaceName"].isin(
                permanent_circuits
            )
        else:
            category_mask = race_history["RaceName"].isin(
                circuits
            )

        category_sum = race_history["PositionSum"].where(
            category_mask,
            0.0,
        )

        category_count = race_history["PositionCount"].where(
            category_mask,
            0,
        )

        cumulative_sum = (
            category_sum
            .groupby(race_history["FullName"])
            .cumsum()
        )

        cumulative_count = (
            category_count
            .groupby(race_history["FullName"])
            .cumsum()
        )

        # Shift by one complete race.
        # Therefore the current race can NEVER enter its own feature.
        previous_sum = (
            cumulative_sum
            .groupby(race_history["FullName"])
            .shift(1)
        )

        previous_count = (
            cumulative_count
            .groupby(race_history["FullName"])
            .shift(1)
        )

        race_history[performance_col] = (
            previous_sum / previous_count
        )

        race_history[history_col] = (
            (previous_count > 0)
            .astype(float)
        )

    # ----------------------------------------------------------
    # Merge race-level historical features back to every driver
    # row.
    # ----------------------------------------------------------

    feature_columns = [
        "StreetCircuitPerformance",
        "PermanentCircuitPerformance",
        "HighSpeedCircuitPerformance",
        "HighDownforceCircuitPerformance",
        "HasStreetCircuitHistory",
        "HasPermanentCircuitHistory",
        "HasHighSpeedCircuitHistory",
        "HasHighDownforceCircuitHistory",
    ]

    df = df.merge(
        race_history[
            race_keys + feature_columns
        ],
        on=race_keys,
        how="left",
        sort=False,
    )
    # No prior history means 0, matching the original feature definition.
    history_columns = [
        "HasStreetCircuitHistory",
        "HasPermanentCircuitHistory",
        "HasHighSpeedCircuitHistory",
        "HasHighDownforceCircuitHistory",
    ]

    df[history_columns] = df[history_columns].fillna(0)
    return df