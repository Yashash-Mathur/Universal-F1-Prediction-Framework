import fastf1
import pandas as pd
import os

fastf1.Cache.enable_cache("cache")


def build_race_dataset(year, race_name):

    print(f"\nLoading {year} - {race_name}")

    # ----------------------------
    # QUALIFYING SESSION
    # ----------------------------

    session = fastf1.get_session(
        year,
        race_name,
        "Q"
    )

    session.load()

    # ----------------------------
    # RACE SESSION
    # ----------------------------

    race = fastf1.get_session(
        year,
        race_name,
        "R"
    )

    race.load()

    # ----------------------------
    # QUALIFYING FEATURES
    # ----------------------------

    bestqualitime = session.results[
        ["Q3", "Q2", "Q1"]
    ].min(axis=1)

    # Pole sitter's Q3 lap
    pole_time = session.results.iloc[0]["Q3"]

    session.results["bestqualitime"] = (
        bestqualitime
    )

    session.results["gaptopole_bestquali"] = (
        bestqualitime - pole_time
    )

    session.results["gaptopole_bestquali"] = (
        session.results["gaptopole_bestquali"]
        .dt.total_seconds()
    )

    # ----------------------------
    # QUALIFYING DATA
    # ----------------------------

    quali_df = session.results[
        [
            "Abbreviation",
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
            "Position"
        ]
    ]

    # ----------------------------
    # MERGE
    # ----------------------------

    mlds1 = pd.merge(
        quali_df,
        race_df,
        on="Abbreviation",
        how="inner"
    )

    # ----------------------------
    # METADATA
    # ----------------------------

    mlds1["Year"] = year
    mlds1["RaceName"] = race_name

    return mlds1


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
        for _, group in backup_df.groupby(
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
# YEARS TO PROCESS
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

    for race_name in schedule["EventName"]:

        # Skip testing events
        if "Test" in race_name:
            continue

        # Skip races already collected
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

            # ----------------------
            # SAVE BACKUP
            # ----------------------

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