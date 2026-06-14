import fastf1
import pandas as pd
import os

from datetime import datetime

fastf1.Cache.enable_cache("cache")


# ==================================
# BUILD ONE RACE
# ==================================

def build_race_dataset(year, race_name, event_date):

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
    # BEST QUALIFYING TIME
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

    pole_time = bestqualitime.min()

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
    # QUALIFYING DATA
    # ----------------------------

    quali_df = quali.results[
        [
            "DriverNumber",
            "Abbreviation",
            "DriverId",
            "FullName",
            "TeamName",
            "TeamId",
            "Q1",
            "Q2",
            "Q3",
            "bestqualitime",
            "gaptopole_bestquali"
        ]
    ]

    # ----------------------------
    # RACE DATA
    # ----------------------------

    race_df = race.results[
        [
            "Abbreviation",
            "GridPosition",
            "Position",
            "Points"
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

    race_dataset["EventDate"] = (
        pd.to_datetime(event_date)
    )

    return race_dataset


# ==================================
# LOAD EXISTING BACKUP
# ==================================

BACKUP_FILE = "backup_dataset_v2.csv"
FINAL_FILE = "f1_2020_2026_raw_v2.csv"

if os.path.exists(BACKUP_FILE):

    backup_df = pd.read_csv(
        BACKUP_FILE
    )

    print(
        f"\nLoaded backup: "
        f"{backup_df.shape}"
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
        "\nNo backup found. "
        "Starting fresh."
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
    2025,
    2026
]


# ==================================
# TODAY
# ==================================

today = pd.Timestamp.today().normalize()


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

        event_date = pd.to_datetime(
            event["EventDate"]
        )

        # ----------------------------
        # SKIP TEST EVENTS
        # ----------------------------

        if event["RoundNumber"] == 0:

            print(
                f"SKIPPING TEST EVENT: "
                f"{race_name}"
            )

            continue

        # ----------------------------
        # SKIP FUTURE RACES
        # ----------------------------

        if event_date > today:

            print(
                f"SKIPPING FUTURE RACE: "
                f"{race_name}"
            )

            continue

        # ----------------------------
        # ALREADY COLLECTED?
        # ----------------------------

        if (
            year,
            race_name
        ) in completed_races:

            print(
                f"SKIPPING: "
                f"{year} - {race_name}"
            )

            continue

        try:

            race_df = build_race_dataset(
                year,
                race_name,
                event_date
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
                BACKUP_FILE,
                index=False
            )

            print(
                f"SUCCESS: "
                f"{year} - {race_name}"
            )

        except Exception as e:

            print(
                f"FAILED: "
                f"{year} - {race_name}"
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
        FINAL_FILE,
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