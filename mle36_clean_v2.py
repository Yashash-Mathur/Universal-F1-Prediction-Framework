import pandas as pd

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv(
    "f1_2020_2026_raw_v2.csv"
)

print("Original Shape:")
print(df.shape)

# ==========================
# DROP MISSING POSITION
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
# HAS QUALIFYING TIME
# ==========================

df["HasQualiTime"] = (
    df["bestqualitime"]
    .notna()
).astype(int)

# ==========================
# SAVE CLEANED DATASET
# ==========================

df.to_csv(
    "f1_2020_2026_cleaned_v2.csv",
    index=False
)

print()
print("Cleaned Shape:")
print(df.shape)

print()
print("Missing Values:")
print(df.isnull().sum())

print()
print("Podium Distribution:")
print(
    df["Podium"]
    .value_counts()
)