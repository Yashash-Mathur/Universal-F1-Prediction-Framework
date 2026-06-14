import pandas as pd
import numpy as np

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("f1_2020_2026_features_v4.csv")

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)
print(df.shape)

# ==========================================================
# SORT DATA FOR ANTI-LEAKAGE
# ==========================================================

df = df.sort_values(
    by=["FullName", "Year", "RoundNumber"]
).reset_index(drop=True)

# ==========================================================
# CIRCUIT CLASSIFICATIONS
# ==========================================================

street_circuits = {
    "Australian Grand Prix",
    "Azerbaijan Grand Prix",
    "Canadian Grand Prix",
    "Miami Grand Prix",
    "Monaco Grand Prix",
    "Saudi Arabian Grand Prix",
    "Singapore Grand Prix",
    "Las Vegas Grand Prix"
}

high_speed_circuits = {
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

high_downforce_circuits = {
    "Monaco Grand Prix",
    "Singapore Grand Prix",
    "Hungarian Grand Prix",
    "Dutch Grand Prix",
    "Japanese Grand Prix",
    "Spanish Grand Prix",
    "Qatar Grand Prix",
    "British Grand Prix"
}

all_races = set(df["RaceName"].unique())

permanent_circuits = all_races - street_circuits

# ==========================================================
# CREATE EMPTY COLUMNS
# ==========================================================

new_columns = [
    "StreetCircuitPerformance",
    "PermanentCircuitPerformance",
    "HighSpeedCircuitPerformance",
    "HighDownforceCircuitPerformance",
    "HasStreetCircuitHistory",
    "HasPermanentCircuitHistory",
    "HasHighSpeedCircuitHistory",
    "HasHighDownforceCircuitHistory"
]

for col in new_columns:
    df[col] = np.nan

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

print("\nBuilding circuit performance features...")

for idx in range(len(df)):

    if idx % 500 == 0:
        print(f"Processed {idx}/{len(df)} rows")

    row = df.iloc[idx]

    driver = row["FullName"]
    current_year = row["Year"]
    current_round = row["RoundNumber"]

    # ======================================================
# HISTORY FILTER
# ROLLING 3-YEAR WINDOW
# NO DATA LEAKAGE
# ======================================================

    window_start = current_year - 3

    history = df[
        (df["FullName"] == driver)
        &
        (
            (
                (df["Year"] < current_year)
                &
                (df["Year"] >= window_start)
            )
            |
            (
                (df["Year"] == current_year)
                &
                (df["RoundNumber"] < current_round)
            )
        )
    ]

    # ======================================================
    # STREET
    # ======================================================

    street_history = history[
        history["RaceName"].isin(street_circuits)
    ]

    if len(street_history) > 0:
        df.at[idx, "StreetCircuitPerformance"] = (
            street_history["Position"].mean()
        )
        df.at[idx, "HasStreetCircuitHistory"] = 1
    else:
        df.at[idx, "HasStreetCircuitHistory"] = 0

    # ======================================================
    # PERMANENT
    # ======================================================

    permanent_history = history[
        history["RaceName"].isin(permanent_circuits)
    ]

    if len(permanent_history) > 0:
        df.at[idx, "PermanentCircuitPerformance"] = (
            permanent_history["Position"].mean()
        )
        df.at[idx, "HasPermanentCircuitHistory"] = 1
    else:
        df.at[idx, "HasPermanentCircuitHistory"] = 0

    # ======================================================
    # HIGH SPEED
    # ======================================================

    high_speed_history = history[
        history["RaceName"].isin(high_speed_circuits)
    ]

    if len(high_speed_history) > 0:
        df.at[idx, "HighSpeedCircuitPerformance"] = (
            high_speed_history["Position"].mean()
        )
        df.at[idx, "HasHighSpeedCircuitHistory"] = 1
    else:
        df.at[idx, "HasHighSpeedCircuitHistory"] = 0

    # ======================================================
    # HIGH DOWNFORCE
    # ======================================================

    high_downforce_history = history[
        history["RaceName"].isin(high_downforce_circuits)
    ]

    if len(high_downforce_history) > 0:
        df.at[idx, "HighDownforceCircuitPerformance"] = (
            high_downforce_history["Position"].mean()
        )
        df.at[idx, "HasHighDownforceCircuitHistory"] = 1
    else:
        df.at[idx, "HasHighDownforceCircuitHistory"] = 0

# ==========================================================
# SAVE
# ==========================================================

output_file = "f1_2020_2026_features_v5.csv"

df.to_csv(output_file, index=False)

print("\n")
print("=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)

print("Saved as:")
print(output_file)

print("\nMissing Values:")

check_cols = [
    "StreetCircuitPerformance",
    "PermanentCircuitPerformance",
    "HighSpeedCircuitPerformance",
    "HighDownforceCircuitPerformance"
]

print(df[check_cols].isna().sum())  