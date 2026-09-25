"""
Recent-form feature engineering.

All features are strictly pre-race:
the current race is excluded using shift(1).
"""

import pandas as pd


def add_recent_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add driver and constructor recent-form features.

    Features added:
        - AverageFinishLast5
        - AverageFinishLast3
        - AverageGridLast3
        - ConstructorAverageFinishLast3

    The implementation preserves the validated feature-engineering
    behavior and original column ordering from 60th_real_pipeline.py.
    """

    df = df.copy()

    # ----------------------------------------------------------
    # DRIVER: AVERAGE FINISH LAST 5
    # ----------------------------------------------------------

    average_finish_last5 = (
        df.groupby("FullName")["Position"]
        .transform(
            lambda s:
            s.shift(1)
            .rolling(
                window=5,
                min_periods=1
            )
            .mean()
        )
        .fillna(0)
    )

    # Preserve the original location of AverageFinishLast5:
    # immediately before TeammateQualifyingGap.
    insert_position = df.columns.get_loc("TeammateQualifyingGap")

    df.insert(
        insert_position,
        "AverageFinishLast5",
        average_finish_last5
    )

    # ----------------------------------------------------------
    # DRIVER: AVERAGE FINISH LAST 3
    # ----------------------------------------------------------

    df["AverageFinishLast3"] = (
        df.groupby("FullName")["Position"]
        .transform(
            lambda s:
            s.shift(1)
            .rolling(
                window=3,
                min_periods=1
            )
            .mean()
        )
    )

    # ----------------------------------------------------------
    # DRIVER: AVERAGE GRID LAST 3
    # ----------------------------------------------------------

    df["AverageGridLast3"] = (
        df.groupby("FullName")["GridPosition"]
        .transform(
            lambda s:
            s.shift(1)
            .replace(0, 24)
            .rolling(
                window=3,
                min_periods=1
            )
            .mean()
        )
    )

    # ----------------------------------------------------------
    # CONSTRUCTOR: AVERAGE FINISH LAST 3
    # ----------------------------------------------------------

    team_races = (
        df.groupby(
            [
                "Year",
                "RoundNumber",
                "TeamName"
            ],
            as_index=False
        )["Position"]
        .mean()
        .rename(
            columns={
                "Position":
                "ConstructorRaceAverageFinish"
            }
        )
        .sort_values(
            [
                "TeamName",
                "Year",
                "RoundNumber"
            ]
        )
    )

    team_races["ConstructorAverageFinishLast3"] = (
        team_races
        .groupby("TeamName")["ConstructorRaceAverageFinish"]
        .transform(
            lambda s:
            s.shift(1)
            .rolling(
                window=3,
                min_periods=1
            )
            .mean()
        )
    )

    df = df.merge(
        team_races[
            [
                "Year",
                "RoundNumber",
                "TeamName",
                "ConstructorAverageFinishLast3"
            ]
        ],
        on=[
            "Year",
            "RoundNumber",
            "TeamName"
        ],
        how="left"
    )

    return df