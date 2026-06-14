import pandas as pd

df = pd.read_csv("backup_dataset.csv")

df = df[
    ~(
        (df["Year"] == 2022)
        &
        (df["RaceName"] == "Pre-Season Track Session")
    )
]

df.to_csv(
    "backup_dataset.csv",
    index=False
)

print("Removed Pre-Season Track Session")

