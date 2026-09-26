"""
Championship feature engineering.

All championship features are strictly pre-race.
Current-race points are excluded using shift(1).
"""

import pandas as pd


def add_championship_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add strictly pre-race championship features.

    Features added:
        - DriverChampionshipPoints
        - ConstructorChampionshipPoints
        - DriverChampionshipPosition
        - ConstructorChampionshipPosition
    """

    df = df.copy()

    # ==========================================================
    # DRIVER CHAMPIONSHIP POINTS
    # ==========================================================

    df["DriverChampionshipPoints"] = (
        df.groupby(
            ["Year", "FullName"]
        )["Points"]
        .transform(
            lambda s:
            s.cumsum().shift(1).fillna(0)
        )
    )

    # ==========================================================
    # CONSTRUCTOR RACE POINTS
    # ==========================================================

    team_race_points = (
        df.groupby(
            [
                "Year",
                "RoundNumber",
                "TeamName"
            ],
            as_index=False
        )["Points"]
        .sum()
        .sort_values(
            [
                "Year",
                "TeamName",
                "RoundNumber"
            ]
        )
    )

    # ==========================================================
    # CONSTRUCTOR CHAMPIONSHIP POINTS
    # ==========================================================

    team_race_points[
        "ConstructorChampionshipPoints"
    ] = (
        team_race_points
        .groupby(
            ["Year", "TeamName"]
        )["Points"]
        .transform(
            lambda s:
            s.cumsum().shift(1).fillna(0)
        )
    )

    # ==========================================================
    # DRIVER CHAMPIONSHIP POSITION
    # ==========================================================

    driver_standings = (
        df[
            [
                "Year",
                "RoundNumber",
                "FullName",
                "DriverChampionshipPoints"
            ]
        ]
        .drop_duplicates()
    )

    driver_standings[
        "DriverChampionshipPosition"
    ] = (
        driver_standings
        .groupby(
            ["Year", "RoundNumber"]
        )["DriverChampionshipPoints"]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )

    driver_position_map = (
        driver_standings
        .set_index(
            [
                "Year",
                "RoundNumber",
                "FullName"
            ]
        )["DriverChampionshipPosition"]
    )

    driver_index = pd.MultiIndex.from_frame(
        df[
            [
                "Year",
                "RoundNumber",
                "FullName"
            ]
        ]
    )

    df["DriverChampionshipPosition"] = (
        driver_index.map(driver_position_map)
    )

    # ==========================================================
    # CONSTRUCTOR CHAMPIONSHIP POSITION
    # ==========================================================

    constructor_standings = (
        team_race_points[
            [
                "Year",
                "RoundNumber",
                "TeamName",
                "ConstructorChampionshipPoints"
            ]
        ]
        .drop_duplicates()
    )

    constructor_standings[
        "ConstructorChampionshipPosition"
    ] = (
        constructor_standings
        .groupby(
            ["Year", "RoundNumber"]
        )["ConstructorChampionshipPoints"]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )

    constructor_position_map = (
        constructor_standings
        .set_index(
            [
                "Year",
                "RoundNumber",
                "TeamName"
            ]
        )["ConstructorChampionshipPosition"]
    )

    constructor_index = pd.MultiIndex.from_frame(
        df[
            [
                "Year",
                "RoundNumber",
                "TeamName"
            ]
        ]
    )

    df["ConstructorChampionshipPosition"] = (
        constructor_index.map(
            constructor_position_map
        )
    )

    # ==========================================================
    # CONSTRUCTOR CHAMPIONSHIP POINTS
    # ==========================================================

    constructor_points_map = (
        team_race_points
        .set_index(
            [
                "Year",
                "RoundNumber",
                "TeamName"
            ]
        )["ConstructorChampionshipPoints"]
    )

    df["ConstructorChampionshipPoints"] = (
        constructor_index.map(
            constructor_points_map
        )
    )

    return df