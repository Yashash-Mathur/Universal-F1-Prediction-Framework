import pandas as pd
import numpy as np

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("f1_2020_2026_features_v5.csv")

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)
print(df.shape)

# ==========================================================
# SORT FOR ANTI LEAKAGE
# ==========================================================

df = df.sort_values(
    by=["Year", "RoundNumber"]
).reset_index(drop=True)

# ==========================================================
# CREATE NEW COLUMNS
# ==========================================================

df["AverageFinishLast3"] = np.nan
df["AverageGridLast3"] = np.nan
df["ConstructorAverageFinishLast3"] = np.nan

# ==========================================================
# DRIVER FEATURES
# ==========================================================

print("\nBuilding driver form features...")

for idx in range(len(df)):

    if idx % 500 == 0:
        print(f"Processed {idx}/{len(df)}")

    row = df.iloc[idx]

    driver = row["FullName"]
    year = row["Year"]
    round_number = row["RoundNumber"]

    previous_driver_races = df[
        (df["FullName"] == driver)
        &
        (
            (df["Year"] < year)
            |
            (
                (df["Year"] == year)
                &
                (df["RoundNumber"] < round_number)
            )
        )
    ].sort_values(
        by=["Year", "RoundNumber"]
    )

    previous_driver_races = previous_driver_races.tail(3)

    # ======================================================
    # AverageFinishLast3
    # ======================================================

    if len(previous_driver_races) > 0:

        df.at[idx, "AverageFinishLast3"] = (
            previous_driver_races["Position"].mean()
        )

    # ======================================================
    # AverageGridLast3
    # ======================================================

    if len(previous_driver_races) > 0:

        grids = previous_driver_races["GridPosition"].copy()

        grids = grids.replace(0, 24)

        df.at[idx, "AverageGridLast3"] = grids.mean()

# ==========================================================
# CONSTRUCTOR FEATURE
# ==========================================================

print("\nBuilding constructor momentum feature...")

race_weekends = (
    df[
        ["Year", "RoundNumber"]
    ]
    .drop_duplicates()
    .sort_values(
        by=["Year", "RoundNumber"]
    )
    .reset_index(drop=True)
)

for idx in range(len(df)):

    if idx % 500 == 0:
        print(f"Constructor pass {idx}/{len(df)}")

    row = df.iloc[idx]

    team = row["TeamName"]
    year = row["Year"]
    round_number = row["RoundNumber"]

    previous_weekends = race_weekends[
        (
            race_weekends["Year"] < year
        )
        |
        (
            (race_weekends["Year"] == year)
            &
            (
                race_weekends["RoundNumber"]
                < round_number
            )
        )
    ]

    previous_weekends = previous_weekends.tail(3)

    if len(previous_weekends) == 0:
        continue

    constructor_history = pd.merge(
        previous_weekends,
        df,
        on=["Year", "RoundNumber"]
    )

    constructor_history = constructor_history[
        constructor_history["TeamName"] == team
    ]

    if len(constructor_history) > 0:

        df.at[
            idx,
            "ConstructorAverageFinishLast3"
        ] = (
            constructor_history["Position"]
            .mean()
        )

# ==========================================================
# SAVE
# ==========================================================

output_file = "f1_2020_2026_features_v6.csv"

df.to_csv(
    output_file,
    index=False
)

print("\n")
print("=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)

print("Saved as:")
print(output_file)

print("\nMissing Values:")

print(
    df[
        [
            "AverageFinishLast3",
            "AverageGridLast3",
            "ConstructorAverageFinishLast3"
        ]
    ]
    .isna()
    .sum()
)

print("\nSummary:")

print(
    df[
        [
            "AverageFinishLast3",
            "AverageGridLast3",
            "ConstructorAverageFinishLast3"
        ]
    ]
    .describe()
)