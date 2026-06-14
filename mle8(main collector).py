import fastf1
import pandas as pd
import os

fastf1.Cache.enable_cache("cache")


# ==================================
# BUILD ONE RACE
# ==================================

def build_race_dataset(year, race_name):

    print(f"\nLoading {year} - {race_name}")

    # ----------------------------
    # QUALIFYING
    # ----------------------------

    quali = fastf1.get_session(
        year,
        race_name,
        "Q"
    )

    quali.load()

    # ----------------------------
    # RACE
    # ----------------------------

    race = fastf1.get_session(
        year,
        race_name,
        "R"
    )

    race.load()

    # ----------------------------
    # BEST QUALIFYING LAP
    # ----------------------------

    bestqualitime = quali.results[
        ["Q3", "Q2", "Q1"]
    ].min(axis=1)

    quali.results["bestqualitime"] = (
        bestqualitime
    )

    # ----------------------------
    # POLE TIME
    # ----------------------------

    pole_time = (
        bestqualitime.min()
    )

    quali.results["gaptopole_bestquali"] = (
        bestqualitime - pole_time
    )

    quali.results["gaptopole_bestquali"] = (
        quali.results["gaptopole_bestquali"]
        .dt.total_seconds()
    )

    # ----------------------------
    # DEBUG CHECK
    # ----------------------------

    if (
        quali.results[
            "gaptopole_bestquali"
        ].lt(0).any()
    ):

        print(
            f"WARNING NEGATIVE GAP: "
            f"{year} - {race_name}"
        )
    # ----------------------------
    # QUALIFYING FEATURES
    # ----------------------------

    quali_df = quali.results[
        [
            "Abbreviation",
            "gaptopole_bestquali"
        ]
    ]

    # ----------------------------
    # RACE FEATURES
    # ----------------------------

    race_df = race.results[
        [
            "Abbreviation",
            "GridPosition",
            "Position"
        ]
    ]

    # ----------------------------
    # MERGE
    # ----------------------------

    race_dataset = pd.merge(
        quali_df,
        race_df,
        on="Abbreviation",
        how="inner"
    )

# ----------------------------
# METADATA
# ----------------------------

    race_dataset["Year"] = year

    race_dataset["RoundNumber"] = (
        race.event["RoundNumber"]
    )

    race_dataset["RaceName"] = race_name

    return race_dataset


# ==================================
# LOAD EXISTING BACKUP
# ==================================

if os.path.exists("backup_dataset.csv"):

    backup_df = pd.read_csv(
        "backup_dataset.csv"
    )

    print(
        f"\nLoaded backup: {backup_df.shape}"
    )

    completed_races = set(
        zip(
            backup_df["Year"],
            backup_df["RaceName"]
        )
    )

    all_races = [
        group
        for _, group
        in backup_df.groupby(
            ["Year", "RaceName"]
        )
    ]

else:

    print(
        "\nNo backup found. Starting fresh."
    )

    completed_races = set()
    all_races = []


# ==================================
# YEARS
# ==================================

years = [
    2020,
    2021,
    2022,
    2023,
    2024,
    2025
]


# ==================================
# MAIN LOOP
# ==================================

for year in years:

    print("\n" + "=" * 50)
    print(f"PROCESSING YEAR {year}")
    print("=" * 50)

    schedule = fastf1.get_event_schedule(
        year
    )

    for _, event in schedule.iterrows():

        race_name = event["EventName"]

        # --------------------------------
        # SKIP TESTING EVENTS
        # --------------------------------

        if event["RoundNumber"] == 0:

            print(
                f"SKIPPING TEST EVENT: {race_name}"
            )

            continue

        # --------------------------------
        # ALREADY COLLECTED?
        # --------------------------------

        if (
            year,
            race_name
        ) in completed_races:

            print(
                f"SKIPPING: {year} - {race_name}"
            )

            continue

        try:

            race_df = build_race_dataset(
                year,
                race_name
            )

            all_races.append(
                race_df
            )

            completed_races.add(
                (
                    year,
                    race_name
                )
            )

            # ----------------------------
            # BACKUP
            # ----------------------------

            backup_save = pd.concat(
                all_races,
                ignore_index=True
            )

            backup_save.to_csv(
                "backup_dataset.csv",
                index=False
            )

            print(
                f"SUCCESS: {year} - {race_name}"
            )

        except Exception as e:

            print(
                f"FAILED: {year} - {race_name}"
            )

            print(e)

            continue


# ==================================
# FINAL DATASET
# ==================================

if len(all_races) > 0:

    final_df = pd.concat(
        all_races,
        ignore_index=True
    )

    final_df.to_csv(
        "f1_2020_2025_dataset.csv",
        index=False
    )

    print("\n====================")
    print("DATASET COMPLETE")
    print("====================")

    print(
        "Shape:",
        final_df.shape
    )

    print(
        "\nRaces per year:"
    )

    print(
        final_df.groupby(
            "Year"
        )["RaceName"]
        .nunique()
    )

    print(
        "\nMissing Values:"
    )

    print(
        final_df.isnull().sum()
    )

else:

    print(
        "No data collected."
    )