"""
Qualifying feature engineering.

Contains qualifying-time derived features used by the F1
prediction pipeline.
"""

import numpy as np
import pandas as pd


def add_qualifying_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add qualifying and teammate-gap features.

    Features added:
        - TeammateQualifyingGap
        - HasGapToPole
        - HasTeammateGap
    """

    df = df.copy()

    # ==========================================================
    # TEAMMATE QUALIFYING GAP
    # ==========================================================

    team_group = df.groupby(
        [
            "Year",
            "RoundNumber",
            "TeamName"
        ]
    )

    team_quali_count = (
        team_group["bestqualitime"]
        .transform("count")
    )

    team_quali_sum = (
        team_group["bestqualitime"]
        .transform("sum")
    )

    teammate_time = (
        team_quali_sum
        - df["bestqualitime"]
    )

    df["TeammateQualifyingGap"] = np.nan

    valid_teammate_gap = (
        (team_quali_count == 2)
        &
        df["bestqualitime"].notna()
    )

    df.loc[
        valid_teammate_gap,
        "TeammateQualifyingGap"
    ] = (
        df.loc[
            valid_teammate_gap,
            "bestqualitime"
        ]
        - teammate_time.loc[
            valid_teammate_gap
        ]
    ).dt.total_seconds()

    # ==========================================================
    # MISSINGNESS INDICATORS
    # ==========================================================

    df["HasGapToPole"] = (
        df["gaptopole_bestquali"].notna()
    ).astype(int)

    df["HasTeammateGap"] = (
        df["TeammateQualifyingGap"].notna()
    ).astype(int)

    return df