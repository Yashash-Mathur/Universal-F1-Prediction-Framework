import pandas as pd
import numpy as np


# ==========================================================
# FILES
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

# Remove rows without a valid race finishing position
df = df.dropna(subset=["Position"]).copy()

# Ensure numeric columns are numeric
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

# Convert qualifying time back to timedelta
df["bestqualitime"] = pd.to_timedelta(
    df["bestqualitime"],
    errors="coerce"
)

# Sort chronologically
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

# ----------------------------------------------------------
# DRIVER POINTS BEFORE CURRENT RACE
# ----------------------------------------------------------

df["DriverChampionshipPoints"] = (
    df.groupby(
        ["Year", "FullName"]
    )["Points"]
    .transform(
        lambda s:
        s.cumsum().shift(1).fillna(0)
    )
)

# ----------------------------------------------------------
# CONSTRUCTOR POINTS AT RACE LEVEL
# ----------------------------------------------------------

team_race_points = (
    df.groupby(
        ["Year", "RoundNumber", "TeamName"],
        as_index=False
    )["Points"]
    .sum()
    .sort_values(
        ["Year", "TeamName", "RoundNumber"]
    )
)

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

team_race_points = team_race_points[
    [
        "Year",
        "RoundNumber",
        "TeamName",
        "ConstructorChampionshipPoints"
    ]
]

df = df.merge(
    team_race_points,
    on=[
        "Year",
        "RoundNumber",
        "TeamName"
    ],
    how="left"
)


# ----------------------------------------------------------
# CHAMPIONSHIP POSITIONS
# ----------------------------------------------------------

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

df = df.merge(
    driver_standings[
        [
            "Year",
            "RoundNumber",
            "FullName",
            "DriverChampionshipPosition"
        ]
    ],
    on=[
        "Year",
        "RoundNumber",
        "FullName"
    ],
    how="left"
)


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

df = df.merge(
    constructor_standings[
        [
            "Year",
            "RoundNumber",
            "TeamName",
            "ConstructorChampionshipPosition"
        ]
    ],
    on=[
        "Year",
        "RoundNumber",
        "TeamName"
    ],
    how="left"
)


# ==========================================================
# AVERAGE FINISH LAST 5
# STRICTLY PRE-RACE
# ==========================================================

print("Building AverageFinishLast5...")

df["AverageFinishLast5"] = (
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


# ==========================================================
# TEAMMATE QUALIFYING GAP
# CURRENT QUALIFYING IS AVAILABLE PRE-RACE
# ==========================================================

print("Building TeammateQualifyingGap...")

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
# CIRCUIT CLASSIFICATIONS
# SAME DEFINITIONS AS EXISTING PIPELINE
# ==========================================================

street_circuits = {
    "Australian Grand Prix",
    "Azerbaijan Grand Prix",
    "Canadian Grand Prix",
    "Miami Grand Prix",
    "Monaco Grand Prix",
    "Saudi Arabian Grand Prix",
    "Singapore Grand Prix",
    "Las Vegas Grand Prix"
}

high_speed_circuits = {
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

high_downforce_circuits = {
    "Monaco Grand Prix",
    "Singapore Grand Prix",
    "Hungarian Grand Prix",
    "Dutch Grand Prix",
    "Japanese Grand Prix",
    "Spanish Grand Prix",
    "Qatar Grand Prix",
    "British Grand Prix"
}

permanent_circuits = (
    set(df["RaceName"].unique())
    - street_circuits
)


# ==========================================================
# CIRCUIT PERFORMANCE
# ONLY HISTORY FROM 2023 ONWARDS
# STRICTLY PRE-RACE
# ==========================================================

print("Building circuit performance features...")

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


# We deliberately process chronologically.
# History for a row is always created from previous rows only.

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

    # ------------------------------------------------------
    # PRE-RACE HISTORY
    # ------------------------------------------------------

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
            street_circuits
        )
    ]

    if len(street_history) > 0:

        df.at[
            idx,
            "StreetCircuitPerformance"
        ] = (
            street_history["Position"].mean()
        )

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
        ] = (
            permanent_history["Position"].mean()
        )

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
            high_speed_circuits
        )
    ]

    if len(high_speed_history) > 0:

        df.at[
            idx,
            "HighSpeedCircuitPerformance"
        ] = (
            high_speed_history["Position"].mean()
        )

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
            high_downforce_circuits
        )
    ]

    if len(high_downforce_history) > 0:

        df.at[
            idx,
            "HighDownforceCircuitPerformance"
        ] = (
            high_downforce_history["Position"].mean()
        )

        df.at[
            idx,
            "HasHighDownforceCircuitHistory"
        ] = 1

    else:

        df.at[
            idx,
            "HasHighDownforceCircuitHistory"
        ] = 0


# ==========================================================
# RECENT FORM — LAST 3
# STRICTLY PRE-RACE
# ==========================================================

print("Building recent-form features...")

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


# ==========================================================
# CONSTRUCTOR AVERAGE FINISH LAST 3
# STRICTLY PRE-RACE
# ==========================================================

print("Building constructor recent-form feature...")

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

team_races[
    "ConstructorAverageFinishLast3"
] = (
    team_races
    .groupby("TeamName")
    ["ConstructorRaceAverageFinish"]
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

print("\nFeature columns:")

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
    "ConstructorAverageFinishLast3"
]

print(feature_columns)

print("\nMissing values:")

print(
    df[feature_columns]
    .isna()
    .sum()
)

print("\nSaved:")
print(OUTPUT_DATASET)