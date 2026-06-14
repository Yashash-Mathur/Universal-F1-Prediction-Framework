# import pandas as pd

# # Load your repaired dataset
# df = pd.read_csv("f1_2020_2025_cleaned.csv")

# print("=" * 50)
# print("DATASET SHAPE")
# print("=" * 50)

# print(df.shape)

# print("\n" + "=" * 50)
# print("MISSING VALUES")
# print("=" * 50)

# print(df.isnull().sum())

# print("\n" + "=" * 50)
# print("PODIUM DISTRIBUTION")
# print("=" * 50)

# print(df["Podium"].value_counts())

# print("\nPercentages:")

# print(
#     df["Podium"]
#     .value_counts(normalize=True)
#     * 100
# )

# print("\n" + "=" * 50)
# print("PIT LANE STARTS")
# print("=" * 50)

# print(df["PitLaneStart"].value_counts())

# print("\nPercentages:")

# print(
#     df["PitLaneStart"]
#     .value_counts(normalize=True)
#     * 100
# )

# print("\n" + "=" * 50)
# print("HAS QUALIFYING TIME")
# print("=" * 50)

# print(df["HasQualiTime"].value_counts())

# print("\nPercentages:")

# print(
#     df["HasQualiTime"]
#     .value_counts(normalize=True)
#     * 100
# )

# print("\n" + "=" * 50)
# print("GRID POSITION RANGE")
# print("=" * 50)

# print(df["GridPosition"].describe())

# print("\n" + "=" * 50)
# print("GAP TO POLE RANGE")
# print("=" * 50)

# print(df["gaptopole_bestquali"].describe())

# print("\n" + "=" * 50)
# print("NEGATIVE GAPS REMAINING?")
# print("=" * 50)

# print(
#     (df["gaptopole_bestquali"] < 0).sum()
# )

# print("\n" + "=" * 50)
# print("ROWS WITH MISSING GAP TO POLE")
# print("=" * 50)

# print(
#     df[
#         df["gaptopole_bestquali"].isna()
#     ][
#         [
#             "Year",
#             "RaceName",
#             "Abbreviation",
#             "GridPosition",
#             "Position"
#         ]
#     ]
# )

import pandas as pd

df = pd.read_csv("f1_2020_2025_cleaned.csv")

print(
    df[df["gaptopole_bestquali"] < 0][
        [
            "Year",
            "RaceName",
            "Abbreviation",
            "gaptopole_bestquali"
        ]
    ]
    .head(30)
)