# import pandas as pd

# df = pd.read_csv("f1_2020_2026_features_v6.csv")

# print("Shape:", df.shape)

# # 1. No duplicate driver-race rows
# print("\nDuplicates:",
#       df.duplicated(["Year", "RoundNumber", "Abbreviation"]).sum())

# # 2. 2026 races
# print("\n2026 races:")
# print(
#     df[df["Year"] == 2026]
#     .groupby(["RoundNumber", "RaceName"])
#     .size()
# )

# # 3. Target
# print("\nPodium:")
# print(df["Podium"].value_counts())

# # 4. Leakage-sensitive features
# features = [
#     "DriverChampionshipPoints",
#     "ConstructorChampionshipPoints",
#     "AverageFinishLast5",
#     "AverageFinishLast3",
#     "AverageGridLast3",
#     "ConstructorAverageFinishLast3"
# ]

# print("\nRound 1 feature values:")
# print(
#     df[df["RoundNumber"] == 1][features].describe()
# )

import pandas as pd

df = pd.read_csv("f1_2020_2026_features_v6.csv")

# Check 2026 Hungarian GP
hungary = df[
    (df["Year"] == 2026) &
    (df["RoundNumber"] == 11)
]

print("Hungarian GP rows:", len(hungary))

features = [
    "DriverChampionshipPoints",
    "ConstructorChampionshipPoints",
    "AverageFinishLast5",
    "AverageFinishLast3",
    "AverageGridLast3",
    "ConstructorAverageFinishLast3",
    "StreetCircuitPerformance",
    "PermanentCircuitPerformance",
    "HighSpeedCircuitPerformance",
    "HighDownforceCircuitPerformance"
]

print("\nHungarian GP feature summary:")
print(hungary[features].describe().T)

print("\nMissing values:")
print(hungary[features].isna().sum())