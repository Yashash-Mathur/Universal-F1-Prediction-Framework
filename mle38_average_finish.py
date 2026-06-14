import pandas as pd

# ==================================
# LOAD DATASET
# ==================================

df = pd.read_csv(
    "f1_2020_2026_features_v2.csv"
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
# NEW COLUMN
# ==================================

df["AverageFinishLast5"] = 0.0

# ==================================
# PROCESS EACH DRIVER
# ==================================

drivers = df["FullName"].unique()

for driver in drivers:

    driver_mask = (
        df["FullName"] == driver
    )

    driver_rows = df[
        driver_mask
    ].sort_values(
        by=[
            "Year",
            "RoundNumber"
        ]
    )

    driver_indices = list(
        driver_rows.index
    )

    finishes = []

    for idx in driver_indices:

        # --------------------------
        # FEATURE VALUE
        # --------------------------

        if len(finishes) == 0:

            avg_finish = 0

        else:

            recent_finishes = finishes[-5:]

            avg_finish = (
                sum(recent_finishes)
                /
                len(recent_finishes)
            )

        df.loc[
            idx,
            "AverageFinishLast5"
        ] = avg_finish

        # --------------------------
        # ADD CURRENT RESULT
        # --------------------------

        current_finish = df.loc[
            idx,
            "Position"
        ]

        finishes.append(
            current_finish
        )

print(
    "\nAverageFinishLast5 created."
)

# ==================================
# SAVE
# ==================================

df.to_csv(
    "f1_2020_2026_features_v3.csv",
    index=False
)

print("\n====================")
print("FEATURE COMPLETE")
print("====================")

print(df.shape)

print()

print(
    df[
        [
            "FullName",
            "Year",
            "RoundNumber",
            "Position",
            "AverageFinishLast5"
        ]
    ].head(20)
)

print()

print(
    df["AverageFinishLast5"]
    .isnull()
    .sum()
)