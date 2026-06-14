import fastf1
import pandas as pd

fastf1.Cache.enable_cache("cache")

affected_races = [
    (2020, "Portuguese Grand Prix"),
    (2020, "Styrian Grand Prix"),
    (2021, "Belgian Grand Prix"),
    (2021, "British Grand Prix"),
    (2021, "Portuguese Grand Prix"),
    (2022, "British Grand Prix"),
    (2022, "Emilia Romagna Grand Prix"),
    (2022, "São Paulo Grand Prix"),
    (2023, "Canadian Grand Prix"),
    (2023, "Miami Grand Prix"),
    (2023, "São Paulo Grand Prix"),
    (2024, "Bahrain Grand Prix"),
    (2024, "Canadian Grand Prix"),
    (2025, "Hungarian Grand Prix")
]

for year, race_name in affected_races:

    print("\n" + "=" * 60)
    print(year, "-", race_name)

    quali = fastf1.get_session(
        year,
        race_name,
        "Q"
    )

    quali.load()

    results = quali.results.copy()

    bestqualitime = (
        results[
            ["Q1", "Q2", "Q3"]
        ]
        .min(axis=1)
    )

    overall_fastest = (
        bestqualitime.min()
    )

    new_gap = (
        bestqualitime
        - overall_fastest
    )

    new_gap = (
        new_gap.dt.total_seconds()
    )

    negatives = (
        new_gap < 0
    ).sum()

    print(
        "Negative values:",
        negatives
    )

    print(
        "Missing values:",
        new_gap.isna().sum()
    )



# import pandas as pd

# df = pd.read_csv(
#     "f1_2020_2025_cleaned.csv"
# )

# print(
#     df[
#         df["gaptopole_bestquali"] < 0
#     ][
#         ["Year","RaceName"]
#     ]
#     .drop_duplicates()
#     .sort_values(
#         ["Year","RaceName"]
#     )
# )