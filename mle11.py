# import pandas as pd

# df = pd.read_csv("f1_2020_2025_dataset.csv")

# print(
#     df[df["GridPosition"] == 0][
#         ["Year","RaceName","Abbreviation","GridPosition","Position"]
#     ]
# )

# print(
#     "\nCount:",
#     len(df[df["GridPosition"] == 0])
# ) 
import pandas as pd

df = pd.read_csv("f1_2020_2025_dataset.csv")

for idx, row in df[df["GridPosition"] == 0].iterrows():

    race = df[
        (df["Year"] == row["Year"]) &
        (df["RaceName"] == row["RaceName"])
    ]

    print(
        row["Year"],
        row["RaceName"],
        "Max Grid:",
        race["GridPosition"].max()
    )