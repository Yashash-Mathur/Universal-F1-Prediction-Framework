# import fastf1

# fastf1.Cache.enable_cache("cache")

# session = fastf1.get_session(
#     2025,
#     "Monaco",
#     "Q"
# )

# session.load()

# print(session.results.columns)

# import fastf1

# fastf1.Cache.enable_cache("cache")

# schedule = fastf1.get_event_schedule(2026)

# print(
#     schedule[
#         ["RoundNumber", "EventName"]
#     ]
# )

# import fastf1

# schedule = fastf1.get_event_schedule(2026)

# print(schedule.columns)

# import fastf1

# fastf1.Cache.enable_cache("cache")

# session = fastf1.get_session(
#     2025,
#     "Monaco",
#     "R"
# )

# session.load()

# print(session.results[
#     [
#         "Abbreviation",
#         "TeamName",
#         "Points",
#         "Position"
#     ]
# ].head())

# import fastf1

# schedule = fastf1.get_event_schedule(2026)

# print(
#     schedule[
#         ["EventName", "EventDate"]
#     ].head(10)
# )

# import pandas as pd

# df = pd.read_csv(
#     "f1_2020_2026_features_v2.csv"
# )

# print(
#     df[
#         [
#             "Year",
#             "RoundNumber",
#             "FullName",
#             "TeamName",
#             "DriverChampionshipPoints",
#             "ConstructorChampionshipPoints",
#             "DriverChampionshipPosition",
#             "ConstructorChampionshipPosition"
#         ]
#     ]
#     .head(30)
# )

# import pandas as pd

# df = pd.read_csv(
#     "f1_2020_2026_features_v2.csv"
# )

# test = df[
#     (df["Year"] == 2026)
#     &
#     (df["RoundNumber"] == 6)
# ]

# print(
#     test[
#         [
#             "FullName",
#             "TeamName",
#             "DriverChampionshipPoints",
#             "ConstructorChampionshipPoints",
#             "DriverChampionshipPosition",
#             "ConstructorChampionshipPosition"
#         ]
#     ]
#     .sort_values(
#         "DriverChampionshipPosition"
#     )
# )

# import pandas as pd

# df = pd.read_csv(
#     "f1_2020_2026_features_v2.csv"
# )

# test = df[
#     (df["Year"] == 2026)
#     &
#     (df["RoundNumber"] == 6)
# ]

# print(
#     test[
#         [
#             "FullName",
#             "TeamName",
#             "DriverChampionshipPoints",
#             "ConstructorChampionshipPoints"
#         ]
#     ]
#     .sort_values(
#         "DriverChampionshipPoints",
#         ascending=False
#     )
# )

# import pandas as pd

# df = pd.read_csv(
#     "f1_2020_2026_features_v3.csv"
# )

# test = df[
#     df["FullName"] == "Max Verstappen"
# ]

# print(
#     test[
#         [
#             "Year",
#             "RoundNumber",
#             "Position",
#             "AverageFinishLast5"
#         ]
#     ].head(15)
# )
# import pandas as pd

# df = pd.read_csv(
#     "f1_2020_2026_features_v3.csv"
# )

# print(df["bestqualitime"].dtype)

# print(
#     df["bestqualitime"].head()
# )

# import pandas as pd

# df = pd.read_csv("f1_2020_2026_cleaned.csv")

# features = [
#     "GridPosition",
#     "gaptopole_bestquali",
#     "PitLaneStart",
#     "HasQualiTime",
#     "RoundNumber",
#     "ConstructorChampionshipPoints",
#     "DriverChampionshipPoints",
#     "ConstructorChampionshipPosition",
#     "DriverChampionshipPosition",
#     "AverageFinishLast5",
#     "TeammateQualifyingGap"
# ]

# print("=" * 50)
# print("MISSING VALUES")
# print("=" * 50)

# print(df[features].isnull().sum())

# print("\n")
# print("=" * 50)
# print("DATASET SHAPE")
# print("=" * 50)

# print(df.shape)

# import pandas as pd

# df = pd.read_csv("f1_2020_2026_features_v4.csv")

# for col in df.columns:
#     print(col)

# import pandas as pd

# df = pd.read_csv("f1_2020_2026_features_v4.csv")

# features = [
#     "GridPosition",
#     "gaptopole_bestquali",
#     "PitLaneStart",
#     "HasQualiTime",
#     "RoundNumber",
#     "ConstructorChampionshipPoints",
#     "DriverChampionshipPoints",
#     "ConstructorChampionshipPosition",
#     "DriverChampionshipPosition",
#     "AverageFinishLast5",
#     "TeammateQualifyingGap"
# ]

# print("=" * 50)
# print("MISSING VALUES")
# print("=" * 50)

# print(df[features].isnull().sum())

# print("\nDataset Shape:")
# print(df.shape)

# import pandas as pd

# df = pd.read_csv("f1_2020_2026_features_v4.csv")

# print(sorted(df["RaceName"].unique()))

# import pandas as pd

# df = pd.read_csv("f1_2020_2026_features_v4.csv")

# print(df[["Year", "RoundNumber", "FullName", "RaceName"]].head())

# import pandas as pd

# df = pd.read_csv("f1_2020_2026_features_v5.csv")

# print(df[
#     [
#         "Year",
#         "RaceName",
#         "StreetCircuitPerformance",
#         "HasStreetCircuitHistory"
#     ]
# ].head(50))

# import pandas as pd

# df = pd.read_csv("f1_2020_2026_features_v5.csv")

# print(df.groupby("Year")["StreetCircuitPerformance"]
#       .apply(lambda x: x.isna().sum()))

# import pandas as pd

# df = pd.read_csv("f1_2020_2026_features_v5.csv")

# driver = "Max Verstappen"

# sample = df[
#     df["FullName"] == driver
# ][[
#     "Year",
#     "RoundNumber",
#     "RaceName",
#     "Position",
#     "StreetCircuitPerformance"
# ]]

# print(sample.head(30).to_string(index=False))

# import pandas as pd

# df = pd.read_csv("f1_2020_2026_features_v5.csv")

# cols = [
#     "StreetCircuitPerformance",
#     "PermanentCircuitPerformance",
#     "HighSpeedCircuitPerformance",
#     "HighDownforceCircuitPerformance"
# ]

# print(df[cols].describe())

# import catboost
# print(catboost.__version__) 

# import pandas as pd

# df = pd.read_csv("f1_2020_2026_features_v5.csv")

# print(df["Position"].dtype)

# print("\nMin Position:")
# print(df["Position"].min())

# print("\nMax Position:")
# print(df["Position"].max())

# print("\nUnique sample:")
# print(sorted(df["Position"].unique())[:30])

# import pandas as pd

# df = pd.read_csv("f1_2020_2026_features_v5.csv")

# print(df.columns.tolist())

# import pandas as pd

# df = pd.read_csv("f1_2020_2026_features_v6.csv")

# for col in df.columns:
#     print(col)

# import pandas as pd

# df = pd.read_csv("f1_2020_2026_features_v6.csv")

# print(
#     sorted(
#         df[df["Year"] == 2026]["RaceName"].unique()
#     )
# )

# import pandas as pd

# df = pd.read_csv("f1_2020_2026_features_v6.csv")

# print(
#     sorted(
#         df[df["Year"] == 2026]["RaceName"].unique()
#     )
# )

# import pandas as pd

# df = pd.read_csv("barcelona_2026_prediction.csv")

# print(df.isna().sum())

# print()
# print(df[[
#     "FullName",
#     "GridPosition",
#     "gaptopole_bestquali",
#     "TeammateQualifyingGap"
# ]].head(22))

# import fastf1

# schedule = fastf1.get_event_schedule(2026)

# print(
#     schedule[
#         ["RoundNumber", "EventName", "EventDate"]
#     ]
# )

# import fastf1

# session = fastf1.get_session(
#     2026,
#     "Barcelona Grand Prix",
#     "Q"
# )

# session.load()

# print(session.results[
#     [
#         "Abbreviation",
#         "GridPosition",
#         "Q1",
#         "Q2",
#         "Q3"
#     ]
# ].head(25))

# import pandas as pd

# df = pd.read_csv(
#     "barcelona_2026_prediction_final.csv"
# )

# print(df.columns.tolist())

# import pandas as pd

# df = pd.read_csv(
#     "barcelona_2026_prediction_final.csv"
# )

# cols = [
#     "FullName",
#     "GridPosition",
#     "gaptopole_bestquali",
#     "DriverChampionshipPosition",
#     "ConstructorChampionshipPosition",
#     "AverageFinishLast5",
#     "AverageFinishLast3"
# ]

# print(
#     df.sort_values("GridPosition")[cols]
# ) 

import pandas as pd

df = pd.read_csv("barcelona_2026_predictions.csv")

features = pd.read_csv(
    "barcelona_2026_prediction_final.csv"
)

audit = df.merge(
    features,
    on=["FullName", "TeamName"],
    how="left"
)

print(audit.columns.tolist())