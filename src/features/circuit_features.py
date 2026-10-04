"""
Circuit-type feature engineering.

All circuit performance features are strictly pre-race.

Historical performance is calculated using:
    - races from 2023 onward
    - only races before the current race
"""

import numpy as np
import pandas as pd


# ==========================================================
# CIRCUIT CLASSIFICATIONS
# ==========================================================

STREET_CIRCUITS = {
    "Australian Grand Prix",
    "Azerbaijan Grand Prix",
    "Canadian Grand Prix",
    "Miami Grand Prix",
    "Monaco Grand Prix",
    "Saudi Arabian Grand Prix",
    "Singapore Grand Prix",
    "Las Vegas Grand Prix"
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
    "Sakhir Grand Prix"
}


HIGH_DOWNFORCE_CIRCUITS = {
    "Monaco Grand Prix",
    "Singapore Grand Prix",
    "Hungarian Grand Prix",
    "Dutch Grand Prix",
    "Japanese Grand Prix",
    "Spanish Grand Prix",
    "Qatar Grand Prix",
    "British Grand Prix"
}


def add_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add circuit-type performance and history features.

    Features added:
        - StreetCircuitPerformance
        - PermanentCircuitPerformance
        - HighSpeedCircuitPerformance
        - HighDownforceCircuitPerformance
        - HasStreetCircuitHistory
        - HasPermanentCircuitHistory
        - HasHighSpeedCircuitHistory
        - HasHighDownforceCircuitHistory

    Historical data is restricted to:
        - Year >= 2023
        - races strictly before the current race
    """

    df = df.copy()

    # ==========================================================
    # PERMANENT CIRCUITS
    # ==========================================================

    permanent_circuits = (
        set(df["RaceName"].unique())
        - STREET_CIRCUITS
    )

    # ==========================================================
    # INITIALIZE FEATURES
    # ==========================================================

    circuit_columns = [
        "StreetCircuitPerformance",
        "PermanentCircuitPerformance",
        "HighSpeedCircuitPerformance",
        "HighDownforceCircuitPerformance",
        "HasStreetCircuitHistory",
        "HasPermanentCircuitHistory",
        "HasHighSpeedCircuitHistory",
        "HasHighDownforceCircuitHistory"
    ]

    for col in circuit_columns:
        df[col] = np.nan

    # ==========================================================
    # CIRCUIT PERFORMANCE
    # HISTORY FROM 2023 ONWARDS
    # STRICTLY PRE-RACE
    # ==========================================================

    print("Building circuit performance features...")

    for idx in range(len(df)):

        if idx % 500 == 0:
            print(
                f"Processed circuit features: "
                f"{idx}/{len(df)}"
            )

        row = df.iloc[idx]

        driver = row["FullName"]
        current_year = row["Year"]
        current_round = row["RoundNumber"]

        history = df[
            (df["FullName"] == driver)
            &
            (
                (
                    (df["Year"] >= 2023)
                    &
                    (df["Year"] < current_year)
                )
                |
                (
                    (df["Year"] == current_year)
                    &
                    (df["Year"] >= 2023)
                    &
                    (df["RoundNumber"] < current_round)
                )
            )
        ]

        # ------------------------------------------------------
        # STREET
        # ------------------------------------------------------

        street_history = history[
            history["RaceName"].isin(
                STREET_CIRCUITS
            )
        ]

        if len(street_history) > 0:

            df.at[
                idx,
                "StreetCircuitPerformance"
            ] = street_history["Position"].mean()

            df.at[
                idx,
                "HasStreetCircuitHistory"
            ] = 1

        else:

            df.at[
                idx,
                "HasStreetCircuitHistory"
            ] = 0

        # ------------------------------------------------------
        # PERMANENT
        # ------------------------------------------------------

        permanent_history = history[
            history["RaceName"].isin(
                permanent_circuits
            )
        ]

        if len(permanent_history) > 0:

            df.at[
                idx,
                "PermanentCircuitPerformance"
            ] = permanent_history["Position"].mean()

            df.at[
                idx,
                "HasPermanentCircuitHistory"
            ] = 1

        else:

            df.at[
                idx,
                "HasPermanentCircuitHistory"
            ] = 0

        # ------------------------------------------------------
        # HIGH SPEED
        # ------------------------------------------------------

        high_speed_history = history[
            history["RaceName"].isin(
                HIGH_SPEED_CIRCUITS
            )
        ]

        if len(high_speed_history) > 0:

            df.at[
                idx,
                "HighSpeedCircuitPerformance"
            ] = high_speed_history["Position"].mean()

            df.at[
                idx,
                "HasHighSpeedCircuitHistory"
            ] = 1

        else:

            df.at[
                idx,
                "HasHighSpeedCircuitHistory"
            ] = 0

        # ------------------------------------------------------
        # HIGH DOWNFORCE
        # ------------------------------------------------------

        high_downforce_history = history[
            history["RaceName"].isin(
                HIGH_DOWNFORCE_CIRCUITS
            )
        ]

        if len(high_downforce_history) > 0:

            df.at[
                idx,
                "HighDownforceCircuitPerformance"
            ] = high_downforce_history["Position"].mean()

            df.at[
                idx,
                "HasHighDownforceCircuitHistory"
            ] = 1

        else:

            df.at[
                idx,
                "HasHighDownforceCircuitHistory"
            ] = 0

    return df