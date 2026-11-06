import fastf1
import pandas as pd
import os

fastf1.Cache.enable_cache("cache")

RAW_DATASET = "f1_2020_2026_raw_v2.csv"


# ==========================================================
# BUILD ONE RACE
# ==========================================================

def build_race_dataset(year, race_name, event_date):

    print(f"\nLoading {year} - {race_name}")

    # ------------------------------------------------------
    # QUALIFYING
    # ------------------------------------------------------

    quali = fastf1.get_session(
        year,
        race_name,
        "Q"
    )

    quali.load()

    # ------------------------------------------------------
    # RACE
    # ------------------------------------------------------

    race = fastf1.get_session(
        year,
        race_name,
        "R"
    )

    race.load()

    # ------------------------------------------------------
    # BEST QUALIFYING TIME
    # ------------------------------------------------------

    bestqualitime = (
        quali.results[
            ["Q3", "Q2", "Q1"]
        ].min(axis=1)
    )

    quali.results["bestqualitime"] = bestqualitime

    pole_time = bestqualitime.min()

    quali.results["gaptopole_bestquali"] = (
        bestqualitime - pole_time
    ).dt.total_seconds()

    # ------------------------------------------------------
    # QUALIFYING DATA
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # RACE DATA
    # ------------------------------------------------------

    race_df = race.results[
        [
            "Abbreviation",
            "GridPosition",
            "Position",
            "Points"
        ]
    ]

    # ------------------------------------------------------
    # MERGE
    # ------------------------------------------------------

    race_dataset = pd.merge(
        quali_df,
        race_df,
        on="Abbreviation",
        how="inner"
    )

    race_dataset["Year"] = year
    race_dataset["RoundNumber"] = race.event["RoundNumber"]
    race_dataset["RaceName"] = race_name
    race_dataset["EventDate"] = pd.to_datetime(event_date)

    return race_dataset


# ==========================================================
# LOAD CURRENT DATASET
# ==========================================================

if not os.path.exists(RAW_DATASET):
    raise FileNotFoundError(
        f"{RAW_DATASET} not found."
    )

df = pd.read_csv(RAW_DATASET)

print("\n================================================")
print("CURRENT DATASET")
print("================================================")

print("Shape:", df.shape)

# ==========================================================
# IDENTIFY EXISTING RACES
# ==========================================================

completed_races = set(
    zip(
        df["Year"],
        df["RoundNumber"]
    )
)

# ==========================================================
# CHECK 2026 SCHEDULE
# ==========================================================

schedule = fastf1.get_event_schedule(2026)

today = pd.Timestamp.today().normalize()

new_races = []

print("\n================================================")
print("CHECKING 2026 RACES")
print("================================================")

for _, event in schedule.iterrows():

    round_number = event["RoundNumber"]

    if round_number == 0:
        continue

    event_date = pd.to_datetime(
        event["EventDate"]
    )

    # Future race → skip
    if event_date > today:
        continue

    race_name = event["EventName"]

    # Already present → skip
    if (2026, round_number) in completed_races:

        print(
            f"Already exists : "
            f"Round {round_number} - {race_name}"
        )

        continue

    # ------------------------------------------------------
    # DOWNLOAD NEW RACE
    # ------------------------------------------------------

    try:

        race_df = build_race_dataset(
            2026,
            race_name,
            event_date
        )

        new_races.append(race_df)

        print(
            f"Added : "
            f"Round {round_number} - {race_name} "
            f"({len(race_df)} rows)"
        )

    except Exception as e:

        print(
            f"FAILED : "
            f"Round {round_number} - {race_name}"
        )

        print(e)


# ==========================================================
# APPEND NEW RACES
# ==========================================================

if len(new_races) == 0:

    print("\nNo new races found.")

else:

    new_df = pd.concat(
        new_races,
        ignore_index=True
    )

    updated = pd.concat(
        [
            df,
            new_df
        ],
        ignore_index=True
    )

    # ------------------------------------------------------
    # REMOVE ACCIDENTAL DUPLICATES
    # ------------------------------------------------------

    updated = updated.drop_duplicates(
        subset=[
            "Year",
            "RoundNumber",
            "Abbreviation"
        ],
        keep="last"
    )

    # ------------------------------------------------------
    # SORT
    # ------------------------------------------------------

    updated = updated.sort_values(
        [
            "Year",
            "RoundNumber",
            "GridPosition"
        ]
    ).reset_index(drop=True)

    # ------------------------------------------------------
    # SAVE
    # ------------------------------------------------------

    updated.to_csv(
        RAW_DATASET,
        index=False
    )

    print("\n================================================")
    print("UPDATED DATASET")
    print("================================================")

    print("Shape:", updated.shape)

    print("\n2026 races:")

    print(
        updated[
            updated["Year"] == 2026
        ]
        .groupby(
            ["RoundNumber", "RaceName"]
        )
        .size()
        .reset_index(name="Rows")
        .to_string(index=False)
    )

    print("\nAdded races:")

    print(
        new_df[
            [
                "RoundNumber",
                "RaceName"
            ]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )