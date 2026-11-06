import pandas as pd
import numpy as np
import fastf1


# ==========================================================
# CONFIG
# ==========================================================

INPUT_DATASET = "f1_2020_2026_dutch_gp_prediction.csv"
OUTPUT_DATASET = "f1_2020_2026_dutch_gp_prediction_fixed.csv"

RAW_DATASET = "f1_2020_2026_raw_v2.csv"
FEATURE_DATASET = "f1_2020_2026_features_v6.csv"

YEAR = 2026
ROUND = 12
SESSION = "Q"

RACE_NAME = "Dutch Grand Prix"


# ==========================================================
# CACHE
# ==========================================================

fastf1.Cache.enable_cache("cache")


# ==========================================================
# HEADER
# ==========================================================

print("=" * 70)
print("DUTCH GP DATA FIX - FINAL VERSION")
print("=" * 70)


# ==========================================================
# LOAD DATASETS
# ==========================================================

dutch = pd.read_csv(INPUT_DATASET)
raw = pd.read_csv(RAW_DATASET)
features = pd.read_csv(FEATURE_DATASET)

print("\nInput Dutch GP shape:", dutch.shape)


# ==========================================================
# NORMALIZE RAW DATA
# ==========================================================

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


# ==========================================================
# HISTORY BEFORE DUTCH GP
# ==========================================================

history = raw[
    (
        raw["Year"] < YEAR
    )
    |
    (
        (raw["Year"] == YEAR)
        &
        (raw["RoundNumber"] < ROUND)
    )
].copy()


current_season = history[
    history["Year"] == YEAR
].copy()


# ==========================================================
# DRIVER CHAMPIONSHIP
# ==========================================================

driver_points = (
    current_season
    .groupby("Abbreviation")["Points"]
    .sum()
)


# ==========================================================
# CONSTRUCTOR CHAMPIONSHIP
# ==========================================================

constructor_points = (
    current_season
    .groupby("TeamName")["Points"]
    .sum()
)


# ==========================================================
# GET DRIVER CHAMPIONSHIP POSITION
# ==========================================================

driver_standings = (
    driver_points
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


# ==========================================================
# GET CONSTRUCTOR CHAMPIONSHIP POSITION
# ==========================================================

constructor_standings = (
    constructor_points
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


# ==========================================================
# LOAD DUTCH GP QUALIFYING
# ==========================================================

print("\nLoading Dutch GP qualifying session...")

session = fastf1.get_session(
    YEAR,
    ROUND,
    SESSION
)


# ----------------------------------------------------------
# IMPORTANT:
# Only request the data required for qualifying.
# ----------------------------------------------------------

session.load(
    laps=False,
    telemetry=False,
    weather=False,
    messages=False,
    livedata=False
)


# ==========================================================
# QUALIFYING RESULTS
# ==========================================================

results = session.results.copy()

print(
    "\nFastF1 qualifying rows:",
    len(results)
)


if len(results) != 22:

    raise ValueError(
        f"Expected 22 Dutch GP drivers, "
        f"but FastF1 returned {len(results)}."
    )


# ==========================================================
# NORMALIZE IDENTIFIERS
# ==========================================================

results["Abbreviation"] = (
    results["Abbreviation"]
    .astype(str)
    .str.upper()
    .str.strip()
)

results["FullName"] = (
    results["FullName"]
    .astype(str)
    .str.strip()
)

results["TeamName"] = (
    results["TeamName"]
    .astype(str)
    .str.strip()
)

results["DriverNumber"] = (
    results["DriverNumber"]
    .astype(str)
    .str.strip()
)


# ==========================================================
# PRINT ACTUAL DUTCH GP ROSTER
# ==========================================================

print("\n")
print("=" * 70)
print("ACTUAL DUTCH GP QUALIFYING ROSTER")
print("=" * 70)

print(
    results[
        [
            "Position",
            "DriverNumber",
            "Abbreviation",
            "FullName",
            "TeamName"
        ]
    ]
    .sort_values("Position")
    .to_string(index=False)
)


# ==========================================================
# SANITY CHECK:
# HADJAR MUST NOT BE IN DUTCH GP
# ==========================================================

if "HAD" in set(results["Abbreviation"]):

    raise ValueError(
        "ERROR: Hadjar is present in Dutch GP "
        "qualifying results. This should not happen."
    )


# ==========================================================
# SANITY CHECK:
# LAWSON MUST BE RED BULL
# ==========================================================

lawson_row = results[
    results["Abbreviation"] == "LAW"
]

if len(lawson_row) != 1:

    raise ValueError(
        "Could not uniquely identify Lawson."
    )

lawson_team = (
    lawson_row.iloc[0]["TeamName"]
)


print(
    "\nLawson Dutch GP team:",
    lawson_team
)


# ==========================================================
# SANITY CHECK:
# TSUNODA MUST BE PRESENT
# ==========================================================

if "TSU" not in set(results["Abbreviation"]):

    raise ValueError(
        "Yuki Tsunoda is missing from the Dutch GP "
        "qualifying roster."
    )


# ==========================================================
# DRIVER RECENT FORM
# ==========================================================

driver_history = (
    history
    .sort_values(
        [
            "Abbreviation",
            "Year",
            "RoundNumber"
        ]
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


# ==========================================================
# CIRCUIT FEATURES
# ==========================================================

previous_features = features[
    (
        features["Year"] < YEAR
    )
    |
    (
        (features["Year"] == YEAR)
        &
        (features["RoundNumber"] < ROUND)
    )
].copy()


previous_features = previous_features.sort_values(
    [
        "Year",
        "RoundNumber"
    ]
)


circuit_features = [
    "StreetCircuitPerformance",
    "PermanentCircuitPerformance",
    "HighSpeedCircuitPerformance",
    "HighDownforceCircuitPerformance"
]


circuit_maps = {}


for feature in circuit_features:

    circuit_maps[feature] = (
        previous_features
        .groupby("Abbreviation")[feature]
        .last()
    )


# ==========================================================
# BUILD DUTCH GP DATASET
#
# IMPORTANT:
# THE QUALIFYING SESSION IS THE AUTHORITATIVE ROSTER.
# ==========================================================

dutch = results[
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
# DRIVER CHAMPIONSHIP FEATURES
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


# ==========================================================
# CONSTRUCTOR CHAMPIONSHIP FEATURES
#
# IMPORTANT:
# USE THE TEAM THE DRIVER IS ACTUALLY RACING FOR
# AT THE DUTCH GP.
# ==========================================================

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
# DRIVER FORM
# ==========================================================

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
# CONSTRUCTOR FORM
#
# USE CURRENT DUTCH GP TEAM.
# ==========================================================

dutch["ConstructorAverageFinishLast3"] = (
    dutch["TeamName"]
    .map(constructor_form)
)


# ==========================================================
# CIRCUIT HISTORY
# ==========================================================

for feature in circuit_features:

    dutch[feature] = (
        dutch["Abbreviation"]
        .map(
            circuit_maps[feature]
        )
    )


# ==========================================================
# CIRCUIT HISTORY FLAGS
#
# PRESERVE FROM FEATURE DATASET IF AVAILABLE.
# ==========================================================

history_flags = [
    "HasStreetCircuitHistory",
    "HasPermanentCircuitHistory",
    "HasHighSpeedCircuitHistory",
    "HasHighDownforceCircuitHistory"
]


for flag in history_flags:

    if flag in previous_features.columns:

        latest_flag = (
            previous_features
            .groupby("Abbreviation")[flag]
            .last()
        )

        dutch[flag] = (
            dutch["Abbreviation"]
            .map(latest_flag)
            .fillna(0)
            .astype(int)
        )

    else:

        dutch[flag] = 0


# ==========================================================
# QUALIFYING POSITION
# ==========================================================

dutch["GridPosition"] = (
    results["Position"]
    .astype(float)
    .values
)


# ==========================================================
# QUALIFYING TIMES
# ==========================================================

def get_final_quali_time(row):

    for column in ["Q3", "Q2", "Q1"]:

        if column not in row.index:
            continue

        value = row[column]

        if pd.notna(value):

            return value

    return pd.NaT


results["BestQualifyingTime"] = (
    results.apply(
        get_final_quali_time,
        axis=1
    )
)


# ==========================================================
# TIME -> SECONDS
# ==========================================================

def time_to_seconds(value):

    if pd.isna(value):
        return np.nan

    try:

        if isinstance(
            value,
            pd.Timedelta
        ):
            return value.total_seconds()

        if isinstance(
            value,
            np.timedelta64
        ):
            return (
                pd.to_timedelta(value)
                .total_seconds()
            )

        if isinstance(
            value,
            (int, float, np.integer, np.floating)
        ):
            return float(value)

        return (
            pd.to_timedelta(value)
            .total_seconds()
        )

    except Exception:

        return np.nan


results["BestQualifyingSeconds"] = (
    results["BestQualifyingTime"]
    .apply(time_to_seconds)
)


# ==========================================================
# POLE
# ==========================================================

pole_time = (
    results["BestQualifyingSeconds"]
    .dropna()
    .min()
)


if pd.isna(pole_time):

    raise ValueError(
        "Could not determine qualifying pole time."
    )


print(
    "\nPole time:",
    f"{pole_time:.3f}s"
)


# ==========================================================
# GAP TO POLE
# ==========================================================

results["GapToPole"] = (
    results["BestQualifyingSeconds"]
    - pole_time
)


# ==========================================================
# QUALIFYING FLAGS
# ==========================================================

results["HasQualiTime"] = (
    results["BestQualifyingSeconds"]
    .notna()
    .astype(int)
)


results["HasGapToPole"] = (
    results["GapToPole"]
    .notna()
    .astype(int)
)


# ==========================================================
# MAP QUALIFYING DATA
# ==========================================================

gap_map = (
    results
    .set_index("Abbreviation")
    ["GapToPole"]
    .to_dict()
)


quali_time_map = (
    results
    .set_index("Abbreviation")
    ["BestQualifyingSeconds"]
    .to_dict()
)


has_quali_map = (
    results
    .set_index("Abbreviation")
    ["HasQualiTime"]
    .to_dict()
)


has_gap_map = (
    results
    .set_index("Abbreviation")
    ["HasGapToPole"]
    .to_dict()
)


# ==========================================================
# APPLY QUALIFYING FEATURES
# ==========================================================

dutch["gaptopole_bestquali"] = (
    dutch["Abbreviation"]
    .map(gap_map)
)


dutch["HasQualiTime"] = (
    dutch["Abbreviation"]
    .map(has_quali_map)
    .fillna(0)
    .astype(int)
)


dutch["HasGapToPole"] = (
    dutch["Abbreviation"]
    .map(has_gap_map)
    .fillna(0)
    .astype(int)
)


# ==========================================================
# PIT LANE START
#
# Qualifying result alone cannot tell us a post-qualifying
# grid penalty / pit-lane start.
#
# For now preserve 0.
# ==========================================================

dutch["PitLaneStart"] = 0


# ==========================================================
# TEAMMATE QUALIFYING GAP
# ==========================================================

teammate_gap = {}


for team, group in results.groupby("TeamName"):

    drivers = (
        group["Abbreviation"]
        .tolist()
    )

    for driver in drivers:

        driver_gap = gap_map.get(
            driver,
            np.nan
        )

        teammate_values = []

        for teammate in drivers:

            if teammate == driver:
                continue

            teammate_gap_value = gap_map.get(
                teammate,
                np.nan
            )

            if (
                pd.notna(driver_gap)
                and
                pd.notna(teammate_gap_value)
            ):

                teammate_values.append(
                    driver_gap
                    - teammate_gap_value
                )

        if teammate_values:

            teammate_gap[driver] = (
                np.mean(teammate_values)
            )

        else:

            teammate_gap[driver] = np.nan


dutch["TeammateQualifyingGap"] = (
    dutch["Abbreviation"]
    .map(teammate_gap)
)


dutch["HasTeammateGap"] = (
    dutch["TeammateQualifyingGap"]
    .notna()
    .astype(int)
)


# ==========================================================
# UNKNOWN RACE TARGET
# ==========================================================

dutch["Position"] = np.nan
dutch["Podium"] = np.nan


# ==========================================================
# SORT BY QUALIFYING POSITION
# ==========================================================

dutch = (
    dutch
    .sort_values(
        "GridPosition"
    )
    .reset_index(drop=True)
)


# ==========================================================
# CRITICAL VALIDATION
# ==========================================================

print("\n")
print("=" * 70)
print("DUTCH GP FINAL DATA VALIDATION")
print("=" * 70)


print(
    "\nRoster:",
    len(dutch)
)


print(
    "\nDrivers:"
)

print(
    dutch[
        [
            "GridPosition",
            "Abbreviation",
            "FullName",
            "TeamName"
        ]
    ]
    .to_string(index=False)
)


# ----------------------------------------------------------
# HADJAR
# ----------------------------------------------------------

if "HAD" in set(
    dutch["Abbreviation"]
):

    raise ValueError(
        "HADJAR IS STILL PRESENT. "
        "Dutch GP roster is incorrect."
    )


# ----------------------------------------------------------
# LAWSON
# ----------------------------------------------------------

lawson_final_team = (
    dutch.loc[
        dutch["Abbreviation"] == "LAW",
        "TeamName"
    ]
    .iloc[0]
)


print(
    "\nLawson team:",
    lawson_final_team
)


if "Red Bull" not in lawson_final_team:

    raise ValueError(
        "Lawson is not assigned to Red Bull "
        "for the Dutch GP."
    )


# ----------------------------------------------------------
# TSUNODA
# ----------------------------------------------------------

tsu_team = (
    dutch.loc[
        dutch["Abbreviation"] == "TSU",
        "TeamName"
    ]
    .iloc[0]
)


print(
    "Tsunoda team:",
    tsu_team
)


# ----------------------------------------------------------
# QUALIFYING COMPLETENESS
# ----------------------------------------------------------

print(
    "\nGrid positions:",
    dutch["GridPosition"].notna().sum(),
    "/ 22"
)


print(
    "Gap-to-pole:",
    dutch["gaptopole_bestquali"].notna().sum(),
    "/ 22"
)


print(
    "Teammate gaps:",
    dutch["TeammateQualifyingGap"].notna().sum(),
    "/ 22"
)


if dutch["GridPosition"].isna().any():

    raise ValueError(
        "At least one driver is missing "
        "GridPosition."
    )


if dutch["gaptopole_bestquali"].isna().any():

    missing = (
        dutch.loc[
            dutch["gaptopole_bestquali"].isna(),
            "Abbreviation"
        ]
        .tolist()
    )

    raise ValueError(
        "Missing qualifying gap for: "
        + ", ".join(missing)
    )


if dutch["GridPosition"].nunique() != 22:

    raise ValueError(
        "Grid positions are not unique."
    )


# ==========================================================
# PRINT MODEL FEATURES
# ==========================================================

print("\n")
print("=" * 70)
print("QUALIFYING + MODEL FEATURES")
print("=" * 70)

print(
    dutch[
        [
            "GridPosition",
            "Abbreviation",
            "TeamName",
            "DriverChampionshipPoints",
            "DriverChampionshipPosition",
            "ConstructorChampionshipPoints",
            "ConstructorChampionshipPosition",
            "AverageFinishLast5",
            "AverageFinishLast3",
            "AverageGridLast3",
            "gaptopole_bestquali",
            "TeammateQualifyingGap",
            "HasQualiTime",
            "HasGapToPole",
            "HasTeammateGap"
        ]
    ]
    .to_string(index=False)
)


# ==========================================================
# SAVE
# ==========================================================

dutch.to_csv(
    OUTPUT_DATASET,
    index=False
)


print("\n")
print("=" * 70)
print("DUTCH GP DATA FIX COMPLETE")
print("=" * 70)

print(
    "\nSaved:",
    OUTPUT_DATASET
)

print(
    "Final shape:",
    dutch.shape
)