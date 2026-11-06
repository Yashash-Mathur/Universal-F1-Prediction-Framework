# import pandas as pd

# df = pd.read_csv("f1_2020_2025_dataset.csv")

# bad = df[df["gaptopole_bestquali"] < 0]

# print(
#     bad.groupby(
#         ["Year", "RaceName"]
#     )
#     .size()
#     .sort_values(
#         ascending=False
#     )
# )
DON'T RUN THIS YET !!!!!!!!!!!!!!!!
DON'T RUN THIS YET !!!!!!!!!!!!!!!!
DON'T RUN THIS YET !!!!!!!!!!!!!!!!
DON'T RUN THIS YET !!!!!!!!!!!!!!!!
DON'T RUN THIS YET !!!!!!!!!!!!!!!!
DON'T RUN THIS YET !!!!!!!!!!!!!!!!
DON'T RUN THIS YET !!!!!!!!!!!!!!!!
DON'T RUN THIS YET !!!!!!!!!!!!!!!!
import pandas as pd

df = pd.read_csv("f1_2020_2025_dataset.csv")

# ----------------------------------
# Remove rows with missing target
# ----------------------------------

df = df.dropna(subset=["Position"])

# ----------------------------------
# Podium target
# ----------------------------------

df["Podium"] = (
    df["Position"] <= 3
).astype(int)

# ----------------------------------
# Pit lane starts
# ----------------------------------

df["PitLaneStart"] = (
    df["GridPosition"] == 0
).astype(int)

# ----------------------------------
# Replace GridPosition 0
# ----------------------------------

df.loc[
    df["GridPosition"] == 0,
    "GridPosition"
] = 21

# ----------------------------------
# Negative GapToPole -> NaN
# ----------------------------------

df.loc[
    df["gaptopole_bestquali"] < 0,
    "gaptopole_bestquali"
] = pd.NA

# ----------------------------------
# HasQualiTime
# ----------------------------------

df["HasQualiTime"] = (
    df["gaptopole_bestquali"]
    .notna()
    .astype(int)
)

# ----------------------------------
# Check result
# ----------------------------------

print(df.isnull().sum())

print()

print(df["HasQualiTime"].value_counts())

# ----------------------------------
# Save cleaned dataset
# ----------------------------------

df.to_csv(
    "f1_2020_2025_cleaned.csv",
    index=False
)

print("\nSaved!") 
