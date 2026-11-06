import pandas as pd

df = pd.read_csv("backup_dataset.csv")

print("Shape:")
print(df.shape)

print()

print("Negative gaps:")
print((df["gaptopole_bestquali"] < 0).sum())

print()

print(
    df[df["gaptopole_bestquali"] < 0][
        [
            "Year",
            "RaceName",
            "Abbreviation",
            "gaptopole_bestquali"
        ]
    ].head(20)
)