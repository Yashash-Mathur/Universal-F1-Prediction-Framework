import pandas as pd
import numpy as np
import fastf1

fastf1.Cache.enable_cache("cache")


# ==========================================================
# FILES
# ==========================================================

RAW_DATASET = "f1_2020_2026_raw_v2.csv"
FEATURE_DATASET = "f1_2020_2026_features_v6.csv"
OUTPUT_DATASET = "f1_2020_2026_dutch_gp_prediction.csv"

YEAR = 2026
ROUND = 12
PREVIOUS_ROUND = 11
RACE_NAME = "Dutch Grand Prix"


# ==========================================================
# LOAD DATA
# ==========================================================

raw = pd.read_csv(RAW_DATASET)
features = pd.read_csv(FEATURE_DATASET)

raw["Position"] = pd.to_numeric(
    raw["Position"],
    errors="coerce"
)

raw["GridPosition"] = pd.to_numeric(
    raw["GridPosition"],
    errors="coerce"
)

raw["Points"] = pd.to_numeric(
    raw["Points"],
    errors="coerce"
)

raw = raw.dropna(
    subset=["Position"]
).copy()

raw = raw.sort_values(
    ["Year", "RoundNumber", "Abbreviation"]
).reset_index(drop=True)


# ==========================================================
# HISTORY AVAILABLE BEFORE DUTCH GP
# ==========================================================

history = raw[
    (
        (raw["Year"] < YEAR)
        |
        (
            (raw["Year"] == YEAR)
            &
            (raw["RoundNumber"] < ROUND)
        )
    )
].copy()

current_season = history[
    history["Year"] == YEAR
].copy()


# ==========================================================
# CURRENT 2026 GRID
# USE HUNGARIAN GP AS LATEST KNOWN DRIVER ROSTER
# ==========================================================

latest = history[
    (history["Year"] == YEAR)
    &
    (history["RoundNumber"] == PREVIOUS_ROUND)
].copy()

if len(latest) != 22:
    raise ValueError(
        f"Expected 22 drivers from Round {PREVIOUS_ROUND}, "
        f"found {len(latest)}."
    )


# ==========================================================
# DRIVER CHAMPIONSHIP
# ==========================================================

driver_points = (
    current_season
    .groupby("Abbreviation")["Points"]
    .sum()
)

driver_standings = (
    driver_points
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


# ==========================================================
# CONSTRUCTOR CHAMPIONSHIP
# ==========================================================

constructor_points = (
    current_season
    .groupby("TeamName")["Points"]
    .sum()
)

constructor_standings = (
    constructor_points
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


# ==========================================================
# BUILD DUTCH GP DATASET
# ==========================================================

dutch = latest[
    [
        "Abbreviation",
        "FullName",
        "TeamName"
    ]
].copy()

dutch["Year"] = YEAR
dutch["RoundNumber"] = ROUND
dutch["RaceName"] = RACE_NAME


# ==========================================================
# CHAMPIONSHIP FEATURES
# ==========================================================

dutch["DriverChampionshipPoints"] = (
    dutch["Abbreviation"]
    .map(driver_points)
    .fillna(0)
)

dutch["DriverChampionshipPosition"] = (
    dutch["Abbreviation"]
    .map(driver_standings)
    .fillna(99)
    .astype(int)
)

dutch["ConstructorChampionshipPoints"] = (
    dutch["TeamName"]
    .map(constructor_points)
    .fillna(0)
)

dutch["ConstructorChampionshipPosition"] = (
    dutch["TeamName"]
    .map(constructor_standings)
    .fillna(99)
    .astype(int)
)


# ==========================================================
# DRIVER RECENT FORM
# STRICTLY PRE-DUTCH GP
# ==========================================================

driver_history = (
    history
    .sort_values(
        ["Abbreviation", "Year", "RoundNumber"]
    )
)

avg_finish_5 = (
    driver_history
    .groupby("Abbreviation")["Position"]
    .rolling(
        window=5,
        min_periods=1
    )
    .mean()
    .groupby(level=0)
    .last()
)

avg_finish_3 = (
    driver_history
    .groupby("Abbreviation")["Position"]
    .rolling(
        window=3,
        min_periods=1
    )
    .mean()
    .groupby(level=0)
    .last()
)

avg_grid_3 = (
    driver_history
    .groupby("Abbreviation")["GridPosition"]
    .rolling(
        window=3,
        min_periods=1
    )
    .mean()
    .groupby(level=0)
    .last()
)

dutch["AverageFinishLast5"] = (
    dutch["Abbreviation"]
    .map(avg_finish_5)
)

dutch["AverageFinishLast3"] = (
    dutch["Abbreviation"]
    .map(avg_finish_3)
)

dutch["AverageGridLast3"] = (
    dutch["Abbreviation"]
    .map(avg_grid_3)
)


# ==========================================================
# CONSTRUCTOR RECENT FORM
# ==========================================================

constructor_race = (
    history
    .groupby(
        [
            "Year",
            "RoundNumber",
            "TeamName"
        ]
    )["Position"]
    .mean()
    .reset_index()
    .sort_values(
        [
            "TeamName",
            "Year",
            "RoundNumber"
        ]
    )
)

constructor_form = (
    constructor_race
    .groupby("TeamName")
    .tail(3)
    .groupby("TeamName")["Position"]
    .mean()
)

dutch["ConstructorAverageFinishLast3"] = (
    dutch["TeamName"]
    .map(constructor_form)
)


# ==========================================================
# CIRCUIT PERFORMANCE
# USE ONLY INFORMATION BEFORE DUTCH GP
# ==========================================================

previous_features = features[
    (
        (features["Year"] < YEAR)
        |
        (
            (features["Year"] == YEAR)
            &
            (features["RoundNumber"] < ROUND)
        )
    )
].copy()

previous_features = previous_features.sort_values(
    ["Year", "RoundNumber"]
)

for feature in [
    "StreetCircuitPerformance",
    "PermanentCircuitPerformance",
    "HighSpeedCircuitPerformance",
    "HighDownforceCircuitPerformance"
]:

    latest_circuit_value = (
        previous_features
        .groupby("Abbreviation")[feature]
        .last()
    )

    dutch[feature] = (
        dutch["Abbreviation"]
        .map(latest_circuit_value)
    )


# ==========================================================
# QUALIFYING FEATURES
# UNKNOWN BEFORE QUALIFYING
# ==========================================================

dutch["GridPosition"] = np.nan
dutch["gaptopole_bestquali"] = np.nan
dutch["TeammateQualifyingGap"] = np.nan

dutch["PitLaneStart"] = 0
dutch["HasQualiTime"] = 0


# ==========================================================
# TARGET
# UNKNOWN BEFORE RACE
# ==========================================================

dutch["Position"] = np.nan
dutch["Podium"] = np.nan


# ==========================================================
# SORT
# ==========================================================

dutch = dutch.sort_values(
    "DriverChampionshipPosition"
).reset_index(drop=True)


# ==========================================================
# SAVE
# ==========================================================

dutch.to_csv(
    OUTPUT_DATASET,
    index=False
)


# ==========================================================
# VALIDATION
# ==========================================================

print("=" * 70)
print("DUTCH GP PRE-QUALIFYING DATASET")
print("=" * 70)

print("Shape:", dutch.shape)

print("\nDrivers:")
print(dutch["Abbreviation"].tolist())

print("\nChampionship:")
print(
    dutch[
        [
            "Abbreviation",
            "DriverChampionshipPoints",
            "DriverChampionshipPosition",
            "ConstructorChampionshipPoints",
            "ConstructorChampionshipPosition"
        ]
    ].to_string(index=False)
)

print("\nUnknown qualifying features:")

print(
    dutch[
        [
            "GridPosition",
            "gaptopole_bestquali",
            "TeammateQualifyingGap"
        ]
    ].isna().sum()
)

print("\nUnknown race target:")

print(
    dutch[
        [
            "Position",
            "Podium"
        ]
    ].isna().sum()
)

print("\nSaved:", OUTPUT_DATASET)