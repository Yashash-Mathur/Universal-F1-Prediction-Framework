import fastf1
import pandas as pd

fastf1.Cache.enable_cache("cache")

# Load session
quali = fastf1.get_session(
    2022,
    "Emilia Romagna Grand Prix",
    "Q"
)

quali.load()

results = quali.results.copy()

# Driver's best qualifying lap
results["bestqualitime"] = (
    results[["Q1", "Q2", "Q3"]]
    .min(axis=1)
)

# Fastest qualifying lap of the whole session
overall_fastest = (
    results["bestqualitime"]
    .min()
)

print("Overall fastest lap:")
print(overall_fastest)
print()

# New gap calculation
results["new_gap"] = (
    results["bestqualitime"]
    - overall_fastest
)

results["new_gap"] = (
    results["new_gap"]
    .dt.total_seconds()
)

print(
    results[
        [
            "Abbreviation",
            "Q1",
            "Q2",
            "Q3",
            "bestqualitime",
            "new_gap"
        ]
    ]
)

print()
print("Negative gaps:")

print(
    len(
        results[
            results["new_gap"] < 0
        ]
    )
)