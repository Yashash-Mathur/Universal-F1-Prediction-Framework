import fastf1
import pandas as pd

fastf1.Cache.enable_cache("cache")

quali = fastf1.get_session(
    2022,
    "Emilia Romagna Grand Prix",
    "Q"
)

quali.load()

results = quali.results.copy()

# Best lap for each driver
results["bestqualitime"] = (
    results[["Q1", "Q2", "Q3"]]
    .min(axis=1)
)

# Fastest lap of each session
q1_fastest = results["Q1"].min()
q2_fastest = results["Q2"].min()
q3_fastest = results["Q3"].dropna().min()

gaps = []

for _, row in results.iterrows():

    best = row["bestqualitime"]

    if pd.isna(best):
        gaps.append(None)
        continue

    if pd.notna(row["Q3"]) and best == row["Q3"]:
        gap = (
            row["Q3"] - q3_fastest
        ).total_seconds()

    elif pd.notna(row["Q2"]) and best == row["Q2"]:
        gap = (
            row["Q2"] - q2_fastest
        ).total_seconds()

    else:
        gap = (
            row["Q1"] - q1_fastest
        ).total_seconds()

    gaps.append(gap)

results["new_gap"] = gaps

print(
    results[
        [
            "Abbreviation",
            "Q1",
            "Q2",
            "Q3",
            "new_gap"
        ]
    ]
)