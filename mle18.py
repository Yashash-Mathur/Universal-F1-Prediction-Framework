import pandas as pd

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv(
    "f1_2020_2025_dataset.csv"
)

print("Original shape:")
print(df.shape)

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
# PIT LANE START FEATURE
# ==========================

df["PitLaneStart"] = (
    df["GridPosition"] == 0
).astype(int)

# ==========================
# REPLACE GRIDPOSITION 0
# ==========================

df.loc[
    df["GridPosition"] == 0,
    "GridPosition"
] = 21

# ==========================
# QUALIFY FEATURE AVAILABLE?
# ==========================

df["HasQualiTime"] = (
    df["gaptopole_bestquali"]
    .notna()
    .astype(int)
)

# ==========================
# CHECK NEGATIVE VALUES
# ==========================

negative_rows = (
    df["gaptopole_bestquali"] < 0
).sum()

print("\nNegative values remaining:")
print(negative_rows)

# ==========================
# CHECK DATASET
# ==========================

print("\nMissing values:")
print(df.isnull().sum())

print("\nPodium distribution:")
print(
    df["Podium"]
    .value_counts()
)

print("\nPit lane starts:")
print(
    df["PitLaneStart"]
    .value_counts()
)

print("\nHasQualiTime:")
print(
    df["HasQualiTime"]
    .value_counts()
)

print("\nFinal shape:")
print(df.shape)

# ==========================
# SAVE
# ==========================

df.to_csv(
    "f1_2020_2025_cleaned.csv",
    index=False
)

print("\nSaved: f1_2020_2025_cleaned.csv")