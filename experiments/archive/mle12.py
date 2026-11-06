# import pandas as pd

# # Load dataset
# df = pd.read_csv("f1_2020_2025_dataset.csv")

# # Remove rows with missing target
# df = df.dropna(subset=["Position"])

# # Create podium target
# df["Podium"] = (
#     df["Position"] <= 3
# ).astype(int)

# # Count rows with negative GapToPole
# negative_rows = len(
#     df[df["gaptopole_bestquali"] < 0]
# )

# print(
#     "Number of negative GapToPole rows:",
#     negative_rows
# )

# # Percentage of dataset affected
# print(
#     "Percentage of dataset affected:",
#     (negative_rows / len(df)) * 100
# )


import pandas as pd

df = pd.read_csv("f1_2020_2025_dataset.csv")

df = df.dropna(subset=["Position"])

race = df[
    (df["Year"] == 2022) &
    (df["RaceName"] == "Emilia Romagna Grand Prix")
]

print(
    race[
        [
            "Abbreviation",
            "gaptopole_bestquali",
            "GridPosition",
            "Position"
        ]
    ].sort_values("GridPosition")
)