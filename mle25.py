import pandas as pd

df = pd.read_csv("f1_2020_2025_dataset.csv")

# -------------------------
# Remove rows without target
# -------------------------

df = df.dropna(subset=["Position"])

# -------------------------
# Podium target
# -------------------------

df["Podium"] = (
    df["Position"] <= 3
).astype(int)

# -------------------------
# Has qualifying time
# -------------------------

df["HasQualiTime"] = (
    df["gaptopole_bestquali"]
    .notna()
    .astype(int)
)

# -------------------------
# Pit lane starts
# -------------------------

df["PitLaneStart"] = (
    df["GridPosition"] == 0
).astype(int)

# -------------------------
# Replace pit lane starts
# -------------------------

df.loc[
    df["GridPosition"] == 0,
    "GridPosition"
] = 21

# -------------------------
# Summary
# -------------------------

print("Shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nNegative gaps:")
print(
    (df["gaptopole_bestquali"] < 0).sum()
)

print("\nPodium:")
print(df["Podium"].value_counts())

print("\nHasQualiTime:")
print(df["HasQualiTime"].value_counts())

print("\nPitLaneStart:")
print(df["PitLaneStart"].value_counts())

# -------------------------
# Save
# -------------------------

df.to_csv(
    "f1_2020_2025_cleaned.csv",
    index=False
)

print("\nSaved!")