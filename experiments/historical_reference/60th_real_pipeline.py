import pandas as pd
import numpy as np

from src.features.championship import add_championship_features
from src.features.qualifying import add_qualifying_features
from src.features.circuit_features import add_circuit_features
from src.features.recent_form import add_recent_form_features


# ==========================================================
# CONFIG
# ==========================================================

RAW_DATASET = "f1_2020_2026_raw_v2.csv"
OUTPUT_DATASET = "f1_2020_2026_features_v6.csv"


# ==========================================================
# LOAD
# ==========================================================

df = pd.read_csv(RAW_DATASET)

print("=" * 70)
print("LOADING RAW DATASET")
print("=" * 70)

print("Raw shape:", df.shape)


# ==========================================================
# BASIC CLEANING
# ==========================================================

df = df.dropna(subset=["Position"]).copy()

df["Position"] = pd.to_numeric(
    df["Position"],
    errors="coerce"
)

df["GridPosition"] = pd.to_numeric(
    df["GridPosition"],
    errors="coerce"
)

df["Points"] = pd.to_numeric(
    df["Points"],
    errors="coerce"
)

df["bestqualitime"] = pd.to_timedelta(
    df["bestqualitime"],
    errors="coerce"
)

df = df.sort_values(
    ["Year", "RoundNumber", "Abbreviation"]
).reset_index(drop=True)


# ==========================================================
# TARGET + BASIC FEATURES
# ==========================================================

df["Podium"] = (
    df["Position"] <= 3
).astype(int)

df["PitLaneStart"] = (
    df["GridPosition"] == 0
).astype(int)

df.loc[
    df["GridPosition"] == 0,
    "GridPosition"
] = 21

df["HasQualiTime"] = (
    df["bestqualitime"].notna()
).astype(int)


# ==========================================================
# CHAMPIONSHIP FEATURES
# STRICTLY PRE-RACE
# ==========================================================

print("\nBuilding championship features...")

df = add_championship_features(df)


# ==========================================================
# QUALIFYING FEATURES
# ==========================================================

print("Building qualifying features...")

df = add_qualifying_features(df)


# ==========================================================
# CIRCUIT FEATURES
# HISTORY FROM 2023 ONWARDS
# STRICTLY PRE-RACE
# ==========================================================

df = add_circuit_features(df)


# ==========================================================
# RECENT FORM
# ==========================================================

print("Building recent-form features...")

df = add_recent_form_features(df)


# ==========================================================
# FINAL SORT
# ==========================================================

df = df.sort_values(
    [
        "Year",
        "RoundNumber",
        "GridPosition",
        "Abbreviation"
    ]
).reset_index(drop=True)


# ==========================================================
# SAVE
# ==========================================================

df.to_csv(
    OUTPUT_DATASET,
    index=False
)


# ==========================================================
# VALIDATION
# ==========================================================

print("\n")
print("=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print("Final shape:", df.shape)

print("\n2026 races:")

print(
    df[df["Year"] == 2026]
    .groupby(
        ["RoundNumber", "RaceName"]
    )
    .size()
    .reset_index(name="Rows")
    .to_string(index=False)
)


feature_columns = [
    "ConstructorChampionshipPoints",
    "DriverChampionshipPoints",
    "ConstructorChampionshipPosition",
    "DriverChampionshipPosition",
    "AverageFinishLast5",
    "TeammateQualifyingGap",
    "StreetCircuitPerformance",
    "PermanentCircuitPerformance",
    "HighSpeedCircuitPerformance",
    "HighDownforceCircuitPerformance",
    "AverageFinishLast3",
    "AverageGridLast3",
    "ConstructorAverageFinishLast3",
    "HasGapToPole",
    "HasTeammateGap",
    "HasStreetCircuitHistory",
    "HasPermanentCircuitHistory",
    "HasHighSpeedCircuitHistory",
    "HasHighDownforceCircuitHistory"
]

print("\nFeature columns:")
print(feature_columns)

print("\nMissing values:")
print(
    df[feature_columns]
    .isna()
    .sum()
)

print("\nMissingness flags:")
print(
    df[
        [
            "HasGapToPole",
            "HasTeammateGap",
            "HasStreetCircuitHistory",
            "HasPermanentCircuitHistory",
            "HasHighSpeedCircuitHistory",
            "HasHighDownforceCircuitHistory"
        ]
    ].sum()
)

print("\nSaved:")
print(OUTPUT_DATASET)