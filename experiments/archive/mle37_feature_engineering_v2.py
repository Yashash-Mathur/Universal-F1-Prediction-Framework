import pandas as pd

# ==================================
# LOAD DATASET
# ==================================

df = pd.read_csv(
    "f1_2020_2026_cleaned_v2.csv"
)

print("Dataset loaded.")
print(df.shape)

# ==================================
# SORT DATA
# ==================================

df = df.sort_values(
    by=[
        "Year",
        "RoundNumber"
    ]
).reset_index(drop=True)

# ==================================
# NEW FEATURE COLUMNS
# ==================================

df["ConstructorChampionshipPoints"] = 0.0
df["DriverChampionshipPoints"] = 0.0

df["ConstructorChampionshipPosition"] = 0
df["DriverChampionshipPosition"] = 0

# ==================================
# PROCESS EACH SEASON
# ==================================

years = sorted(
    df["Year"].unique()
)

for year in years:

    print(f"\nProcessing season {year}")

    season_df = df[
        df["Year"] == year
    ]

    rounds = sorted(
        season_df["RoundNumber"].unique()
    )

    # ------------------------------
    # EACH ROUND
    # ------------------------------

    for current_round in rounds:

        # --------------------------
        # CURRENT RACE ROWS
        # --------------------------

        current_mask = (
            (df["Year"] == year)
            &
            (df["RoundNumber"] == current_round)
        )

        # --------------------------
        # PREVIOUS RACES ONLY
        # --------------------------

        previous_races = df[
            (df["Year"] == year)
            &
            (df["RoundNumber"] < current_round)
        ]

        # --------------------------
        # ROUND 1
        # --------------------------

        if len(previous_races) == 0:

            df.loc[
                current_mask,
                "ConstructorChampionshipPoints"
            ] = 0

            df.loc[
                current_mask,
                "DriverChampionshipPoints"
            ] = 0

            df.loc[
                current_mask,
                "ConstructorChampionshipPosition"
            ] = 0

            df.loc[
                current_mask,
                "DriverChampionshipPosition"
            ] = 0

            continue

        # --------------------------
        # CONSTRUCTOR POINTS
        # --------------------------

        constructor_points = (
            previous_races
            .groupby("TeamName")["Points"]
            .sum()
        )

        # --------------------------
        # DRIVER POINTS
        # --------------------------

        driver_points = (
            previous_races
            .groupby("FullName")["Points"]
            .sum()
        )

        # --------------------------
        # CONSTRUCTOR POSITIONS
        # --------------------------

        constructor_positions = (
            constructor_points
            .rank(
                ascending=False,
                method="min"
            )
            .astype(int)
        )

        # --------------------------
        # DRIVER POSITIONS
        # --------------------------

        driver_positions = (
            driver_points
            .rank(
                ascending=False,
                method="min"
            )
            .astype(int)
        )

        # --------------------------
        # ASSIGN FEATURES
        # --------------------------

        current_rows = df.loc[
            current_mask
        ]

        for idx in current_rows.index:

            team = df.loc[
                idx,
                "TeamName"
            ]

            driver = df.loc[
                idx,
                "FullName"
            ]

            # ----------------------
            # POINTS
            # ----------------------

            df.loc[
                idx,
                "ConstructorChampionshipPoints"
            ] = (
                constructor_points.get(
                    team,
                    0
                )
            )

            df.loc[
                idx,
                "DriverChampionshipPoints"
            ] = (
                driver_points.get(
                    driver,
                    0
                )
            )

            # ----------------------
            # POSITIONS
            # ----------------------

            df.loc[
                idx,
                "ConstructorChampionshipPosition"
            ] = (
                constructor_positions.get(
                    team,
                    0
                )
            )

            df.loc[
                idx,
                "DriverChampionshipPosition"
            ] = (
                driver_positions.get(
                    driver,
                    0
                )
            )

# ==================================
# SAVE
# ==================================

df.to_csv(
    "f1_2020_2026_features_v2.csv",
    index=False
)

print("\n====================")
print("FEATURE ENGINEERING COMPLETE")
print("====================")

print(df.shape)

print("\nNew Columns Added:")

print(
    [
        "ConstructorChampionshipPoints",
        "DriverChampionshipPoints",
        "ConstructorChampionshipPosition",
        "DriverChampionshipPosition"
    ]
)

print("\nMissing Values:")

print(
    df[
        [
            "ConstructorChampionshipPoints",
            "DriverChampionshipPoints",
            "ConstructorChampionshipPosition",
            "DriverChampionshipPosition"
        ]
    ].isnull().sum()
)