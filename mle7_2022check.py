import pandas as pd
import fastf1

df = pd.read_csv("f1_2020_2025_dataset.csv")

df = df.dropna(subset=["Position"])

df["Podium"] = (
    df["Position"] <= 3
).astype(int)

print(df["Podium"].value_counts())

print(
    df.sort_values(
        "gaptopole_bestquali"
    ).head(20)
)

print(
    df.sort_values(
        "gaptopole_bestquali",
        ascending=False
    ).head(20)
)


bad = df[df["gaptopole_bestquali"] < 0]

print(
    bad.groupby(
        ["Year", "RaceName"]
    ).size()
)


print("previous EDA thing below ")
print(df.shape)

print(df["Position"].describe())

print(df["GridPosition"].describe())

print(df["gaptopole_bestquali"].describe())

print(
    df[
        ["GridPosition",
         "gaptopole_bestquali",
         "Position"]
    ].corr()
)

# print(df.isnull().sum())

# missing_gap = df[df["gaptopole_bestquali"].isna()]

# print(
#     missing_gap.groupby("Year")
#     .size()
# )
# backup = pd.read_csv("backup_dataset.csv")
# races_2022 = (
#     backup[
#         backup["Year"] == 2022
#     ]["RaceName"]
#     .drop_duplicates()
#     .tolist()
# )

# # print("2022 races collected:")
# # for race in races_2022:
# #     print(race)

# # print("\nCount:", len(races_2022))


# schedule = fastf1.get_event_schedule(2022)

# print(
#     schedule[
#         ["RoundNumber", "EventName"]
#     ]
# )