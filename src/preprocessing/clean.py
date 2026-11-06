"""
Raw dataset cleaning.

Converts column dtypes, drops rows missing a finishing position,
sets the Podium target, and handles pit-lane starts.
"""

import pandas as pd


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw F1 dataset and add the Podium target and PitLaneStart flag.

    Input: raw dataframe from f1_2020_2026_raw_v2.csv (or equivalent).
    Output: sorted, cleaned dataframe ready for feature engineering.
    """
    df = df.dropna(subset=["Position"]).copy()

    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df["GridPosition"] = pd.to_numeric(df["GridPosition"], errors="coerce")
    df["Points"] = pd.to_numeric(df["Points"], errors="coerce")
    df["bestqualitime"] = pd.to_timedelta(df["bestqualitime"], errors="coerce")

    df = df.sort_values(["Year", "RoundNumber", "Abbreviation"]).reset_index(drop=True)

    df["Podium"] = (df["Position"] <= 3).astype(int)
    df["PitLaneStart"] = (df["GridPosition"] == 0).astype(int)
    df.loc[df["GridPosition"] == 0, "GridPosition"] = 21
    df["HasQualiTime"] = df["bestqualitime"].notna().astype(int)

    return df
