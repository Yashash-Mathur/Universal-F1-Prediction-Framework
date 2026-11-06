# import fastf1
# fastf1.Cache.enable_cache("cache")
# quali = fastf1.get_session(
#     2022,
#     "Emilia Romagna Grand Prix",
#     "Q"
# )
# quali.load()

# quali.results.loc[
#     quali.results["Abbreviation"] == "VER",
#     ["Q1","Q2","Q3"]
# ]

# print(
#     quali.results[
#         ["Abbreviation","Q1","Q2","Q3"]
#     ]
# )

import pandas as pd

df = pd.read_csv("f1_2020_2025_dataset.csv")

df = df.dropna(subset=["Position"])

df["Podium"] = (
    df["Position"] <= 3
).astype(int)

df["HasQualiTime"] = (
    df["gaptopole_bestquali"]
    .notna()
    .astype(int)
)

print(df["HasQualiTime"].value_counts())
