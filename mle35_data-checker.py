import pandas as pd

df = pd.read_csv(
    "f1_2020_2026_raw_v2.csv"
)

print("\nRows with missing Position:")
print(
    df[
        df["Position"].isna()
    ]
)

print("\nRows with missing GridPosition:")
print(
    df[
        df["GridPosition"].isna()
    ]
)

print("\nRows with missing DriverId:")
print(
    df[
        df["DriverId"].isna()
    ]
)

print("\nRows with missing TeamId:")
print(
    df[
        df["TeamId"].isna()
    ]
)

print("\nRows with missing bestqualitime:")
print(
    df[
        df["bestqualitime"].isna()
    ][
        [
            "Year",
            "RaceName",
            "Abbreviation",
            "TeamName"
        ]
    ]
)