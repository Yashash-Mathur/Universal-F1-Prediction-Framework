import pandas as pd

# ==================================
# LOAD DATASET
# ==================================

df = pd.read_csv(
    "f1_2020_2026_features_v3.csv"
)

# Convert qualifying times back into timedeltas
df["bestqualitime"] = pd.to_timedelta(
    df["bestqualitime"],
    errors="coerce"
)

print("Dataset loaded.")
print(df.shape)

# ==================================
# NEW COLUMN
# ==================================

df["TeammateQualifyingGap"] = float("nan")

# ==================================
# PROCESS EACH RACE
# ==================================

race_groups = df.groupby(
    ["Year", "RoundNumber"]
)

for (year, round_number), race_df in race_groups:

    # ------------------------------
    # PROCESS EACH TEAM
    # ------------------------------

    team_groups = race_df.groupby(
        "TeamName"
    )

    for team_name, team_df in team_groups:

        # We expect exactly 2 drivers
        if len(team_df) != 2:
            continue

        idx1 = team_df.index[0]
        idx2 = team_df.index[1]

        time1 = df.loc[
            idx1,
            "bestqualitime"
        ]

        time2 = df.loc[
            idx2,
            "bestqualitime"
        ]

        # --------------------------
        # MISSING QUALI TIME
        # --------------------------

        if (
            pd.isna(time1)
            or
            pd.isna(time2)
        ):
            continue

        # --------------------------
        # DRIVER TIME - TEAMMATE TIME
        # --------------------------

        gap1 = (
            time1 - time2
        ).total_seconds()

        gap2 = (
            time2 - time1
        ).total_seconds()

        df.loc[
            idx1,
            "TeammateQualifyingGap"
        ] = gap1

        df.loc[
            idx2,
            "TeammateQualifyingGap"
        ] = gap2

print(
    "\nTeammateQualifyingGap created."
)

# ==================================
# SAVE DATASET
# ==================================

df.to_csv(
    "f1_2020_2026_features_v4.csv",
    index=False
)

print("\n====================")
print("FEATURE COMPLETE")
print("====================")

print(df.shape)

print()

print(
    "Missing Values:"
)

print(
    df[
        "TeammateQualifyingGap"
    ]
    .isnull()
    .sum()
)

print()

print(
    "Sample Values:"
)

print(
    df[
        [
            "FullName",
            "TeamName",
            "bestqualitime",
            "TeammateQualifyingGap"
        ]
    ]
    .head(20)
)

