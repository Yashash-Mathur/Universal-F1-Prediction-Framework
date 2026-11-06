import pandas as pd

df = pd.read_csv("f1_2020_2025_dataset.csv")

# Create feature before changing anything
df["PitLaneStart"] = (
    df["GridPosition"] == 0
).astype(int)

# Replace 0 grid positions
df.loc[
    df["GridPosition"] == 0,
    "GridPosition"
] = 21

print(
    df["PitLaneStart"].value_counts()
)

print()

print(
    df["GridPosition"].describe()
)

print()

print(
    df[df["PitLaneStart"] == 1][
        [
            "Year",
            "RaceName",
            "Abbreviation",
            "GridPosition",
            "Position"
        ]
    ].head(20)
)   