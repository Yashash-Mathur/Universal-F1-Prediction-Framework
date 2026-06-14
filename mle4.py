import fastf1
import pandas as pd

fastf1.Cache.enable_cache("cache")

def build_race_dataset(year, race_name):

    session = fastf1.get_session(year, race_name, "Q")
    session.load()

    race = fastf1.get_session(year, race_name, "R")
    race.load()

    bestqualitime = session.results[["Q3", "Q2", "Q1"]].min(axis=1)

    pole_time = session.results.iloc[0]["Q3"]

    session.results["bestqualitime"] = bestqualitime

    session.results["gaptopole_bestquali"] = (
        bestqualitime - pole_time
    )

    quali_df = session.results[
    ["Abbreviation", "gaptopole_bestquali"]
    ]

    race_df = race.results[
        ["Abbreviation","GridPosition", "Position"]
    ]

    mlds1 = pd.merge(
        quali_df,
        race_df,
        on="Abbreviation",
        how="inner"
    )
    print(session.results["GridPosition"].head())
    print(race.results["GridPosition"].head())
    return mlds1

monaco_df = build_race_dataset(2023, "Monaco")
print(monaco_df.head())
