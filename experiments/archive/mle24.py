import pandas as pd

df = pd.read_csv("f1_2020_2025_dataset.csv")
df = df.dropna(subset=["Position"])
print("Missing GridPosition")
print(
    df[df["GridPosition"].isna()][
        [
            "Year",
            "RaceName",
            "Abbreviation",
            "GridPosition",
            "Position"
        ]
    ]
)

print("\nMissing Position")
print(
    df[df["Position"].isna()][
        [
            "Year",
            "RaceName",
            "Abbreviation",
            "GridPosition",
            "Position"
        ]
    ]
)

