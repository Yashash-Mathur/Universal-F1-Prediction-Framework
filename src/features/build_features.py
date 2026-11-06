"""
Feature pipeline orchestrator.

Calls each feature module in the correct order and returns
the complete feature dataframe. This is the single entry point
for building features from a cleaned raw dataframe.
"""

import pandas as pd

from src.features.championship import add_championship_features
from src.features.qualifying import add_qualifying_features
from src.features.circuit_features import add_circuit_features
from src.features.recent_form import add_recent_form_features


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature modules to a cleaned dataframe.

    Input: output of src.preprocessing.clean.clean_raw()
    Output: dataframe with all model features, sorted by
            Year / RoundNumber / GridPosition / Abbreviation.
    """
    df = add_championship_features(df)
    df = add_qualifying_features(df)
    df = add_circuit_features(df)
    df = add_recent_form_features(df)

    df = df.sort_values(
        ["Year", "RoundNumber", "GridPosition", "Abbreviation"]
    ).reset_index(drop=True)

    return df
