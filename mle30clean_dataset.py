import pandas as pd

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv(
    "f1_2020_2025_dataset.csv"
)

# ==========================
# REMOVE MISSING TARGETS
# ==========================

df = df.dropna(
    subset=["Position"]
)

# ==========================
# PODIUM TARGET
# ==========================

df["Podium"] = (
    df["Position"] <= 3
).astype(int)

# ==========================
# PIT LANE START
# ==========================

df["PitLaneStart"] = (
    df["GridPosition"] == 0
).astype(int)

# ==========================
# REPLACE GRID 0
# ==========================

df.loc[
    df["GridPosition"] == 0,
    "GridPosition"
] = 21

# ==========================
# HAS QUALI TIME
# ==========================

df["HasQualiTime"] = (
    df["gaptopole_bestquali"]
    .notna()
    .astype(int)
)

# ==========================
# CHECKS
# ==========================

print("Shape:")
print(df.shape)

print()

print("Missing values:")
print(df.isnull().sum())

print()

print("Negative gaps:")
print(
    (df["gaptopole_bestquali"] < 0)
    .sum()
)

print()

print("Podium:")
print(
    df["Podium"]
    .value_counts()
)

print()

print("HasQualiTime:")
print(
    df["HasQualiTime"]
    .value_counts()
)

print()

print("PitLaneStart:")
print(
    df["PitLaneStart"]
    .value_counts()
)

# ==========================
# SAVE
# ==========================

df.to_csv(
    "f1_2020_2025_cleaned.csv",
    index=False
)

print()
print("Saved!")