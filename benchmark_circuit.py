import time
import statistics
import pandas as pd

from src.features.circuit_features import add_circuit_features as optimized


STREET_CIRCUITS = {
    "Australian Grand Prix",
    "Azerbaijan Grand Prix",
    "Canadian Grand Prix",
    "Miami Grand Prix",
    "Monaco Grand Prix",
    "Saudi Arabian Grand Prix",
    "Singapore Grand Prix",
    "Las Vegas Grand Prix"
}

HIGH_SPEED_CIRCUITS = {
    "Australian Grand Prix",
    "Austrian Grand Prix",
    "Azerbaijan Grand Prix",
    "Belgian Grand Prix",
    "British Grand Prix",
    "Canadian Grand Prix",
    "Italian Grand Prix",
    "Las Vegas Grand Prix",
    "Mexico City Grand Prix",
    "Miami Grand Prix",
    "Saudi Arabian Grand Prix",
    "Styrian Grand Prix",
    "70th Anniversary Grand Prix",
    "Sakhir Grand Prix"
}

HIGH_DOWNFORCE_CIRCUITS = {
    "Monaco Grand Prix",
    "Singapore Grand Prix",
    "Hungarian Grand Prix",
    "Dutch Grand Prix",
    "Japanese Grand Prix",
    "Spanish Grand Prix",
    "Qatar Grand Prix",
    "British Grand Prix"
}


def original(df):
    import numpy as np

    df = df.copy()

    permanent_circuits = (
        set(df["RaceName"].unique()) - STREET_CIRCUITS
    )

    columns = [
        "StreetCircuitPerformance",
        "PermanentCircuitPerformance",
        "HighSpeedCircuitPerformance",
        "HighDownforceCircuitPerformance",
        "HasStreetCircuitHistory",
        "HasPermanentCircuitHistory",
        "HasHighSpeedCircuitHistory",
        "HasHighDownforceCircuitHistory"
    ]

    for col in columns:
        df[col] = np.nan

    for idx in range(len(df)):
        row = df.iloc[idx]

        driver = row["FullName"]
        current_year = row["Year"]
        current_round = row["RoundNumber"]

        history = df[
            (df["FullName"] == driver)
            &
            (
                (
                    (df["Year"] >= 2023)
                    &
                    (df["Year"] < current_year)
                )
                |
                (
                    (df["Year"] == current_year)
                    &
                    (df["Year"] >= 2023)
                    &
                    (df["RoundNumber"] < current_round)
                )
            )
        ]

        categories = [
            (
                "StreetCircuitPerformance",
                "HasStreetCircuitHistory",
                STREET_CIRCUITS
            ),
            (
                "PermanentCircuitPerformance",
                "HasPermanentCircuitHistory",
                permanent_circuits
            ),
            (
                "HighSpeedCircuitPerformance",
                "HasHighSpeedCircuitHistory",
                HIGH_SPEED_CIRCUITS
            ),
            (
                "HighDownforceCircuitPerformance",
                "HasHighDownforceCircuitHistory",
                HIGH_DOWNFORCE_CIRCUITS
            )
        ]

        for performance, flag, circuits in categories:
            category_history = history[
                history["RaceName"].isin(circuits)
            ]

            if len(category_history) > 0:
                df.at[
                    idx, performance
                ] = category_history["Position"].mean()

                df.at[idx, flag] = 1
            else:
                df.at[idx, flag] = 0

    return df


# ------------------------------------------------------------
# Load + prep input (same as real pipeline)
# ------------------------------------------------------------

df = pd.read_csv("f1_2020_2026_raw_v2.csv")
df = df.dropna(subset=["Position"]).copy()
df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
df["GridPosition"] = pd.to_numeric(df["GridPosition"], errors="coerce")
df["Points"] = pd.to_numeric(df["Points"], errors="coerce")
df["bestqualitime"] = pd.to_timedelta(df["bestqualitime"], errors="coerce")
df = df.sort_values(["Year", "RoundNumber", "Abbreviation"]).reset_index(drop=True)

RUNS = 10

# ------------------------------------------------------------
# Benchmark original (1 run — it's slow, 1 is enough for median)
# shortcut: only 1 original run to avoid ~55s total wait; acceptable for reference
# ------------------------------------------------------------

print(f"Running ORIGINAL (1 run for reference)...")
t0 = time.perf_counter()
old_result = original(df)
old_times = [time.perf_counter() - t0]
print(f"  run 1: {old_times[0]:.4f}s")

# ------------------------------------------------------------
# Benchmark optimized (RUNS iterations)
# ------------------------------------------------------------

print(f"\nRunning OPTIMIZED ({RUNS} runs)...")
new_times = []
for i in range(RUNS):
    t0 = time.perf_counter()
    new_result = optimized(df)
    elapsed = time.perf_counter() - t0
    new_times.append(elapsed)
    print(f"  run {i+1}: {elapsed:.4f}s")

# ------------------------------------------------------------
# Correctness gate
# ------------------------------------------------------------

print("\nCORRECTNESS")
print("=" * 50)
identical = old_result.equals(new_result)
print("FULL DATA IDENTICAL:", identical)
if not identical:
    # Show which columns differ
    for col in old_result.columns:
        if col in new_result.columns:
            if not old_result[col].equals(new_result[col]):
                print(f"  DIFFERS: {col}")

# ------------------------------------------------------------
# Performance summary
# ------------------------------------------------------------

orig_median = statistics.median(old_times)
opt_median = statistics.median(new_times)
speedup = orig_median / opt_median

print("\nPERFORMANCE")
print("=" * 50)
print(f"Original  (1-run):       {orig_median:.4f} s")
print(f"Optimized (median/{RUNS}): {opt_median:.4f} s")
print(f"Speedup:                 {speedup:.2f}x")
print(f"Runtime reduction:       {(1 - opt_median / orig_median) * 100:.2f}%")
