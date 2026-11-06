import pandas as pd
import fastf1

# ==========================================================
# LOAD PREDICTION FILE
# ==========================================================

df = pd.read_csv(
    "barcelona_2026_prediction.csv"
)

print("=" * 60)
print("LOADED PREDICTION FILE")
print("=" * 60)
print(df.shape)

# ==========================================================
# REMOVE EMPTY QUALI COLUMNS
# ==========================================================

df = df.drop(
    columns=[
        "GridPosition",
        "gaptopole_bestquali",
        "TeammateQualifyingGap"
    ],
    errors="ignore"
)

# ==========================================================
# LOAD BARCELONA 2026 QUALIFYING
# ==========================================================

fastf1.Cache.enable_cache("cache")

session = fastf1.get_session(
    2026,
    "Barcelona Grand Prix",
    "Q"
)

session.load()

results = session.results.copy()

# ==========================================================
# BEST QUALIFYING TIME
# ==========================================================

results["BestQualiTime"] = (
    results[["Q3", "Q2", "Q1"]]
    .bfill(axis=1)
    .iloc[:, 0]
)

# ==========================================================
# GRID POSITION
# ==========================================================

results = results.sort_values(
    by="BestQualiTime"
).reset_index(drop=True)

results["GridPosition"] = (
    results.index + 1
)

# ==========================================================
# GAP TO POLE
# ==========================================================

pole_time = results["BestQualiTime"].min()

results["gaptopole_bestquali"] = (
    results["BestQualiTime"] - pole_time
).dt.total_seconds()

# ==========================================================
# TEAMMATE QUALIFYING GAP
# ==========================================================

results["TeammateQualifyingGap"] = pd.NA

for team in results["TeamName"].unique():

    team_rows = results[
        results["TeamName"] == team
    ]

    if len(team_rows) != 2:
        continue

    idx1 = team_rows.index[0]
    idx2 = team_rows.index[1]

    t1 = team_rows.iloc[0]["BestQualiTime"]
    t2 = team_rows.iloc[1]["BestQualiTime"]

    gap = (
        t1 - t2
    ).total_seconds()

    results.loc[idx1,
                "TeammateQualifyingGap"] = gap

    results.loc[idx2,
                "TeammateQualifyingGap"] = -gap

# ==========================================================
# DRIVER NAME -> ABBREVIATION
# ==========================================================

mapping = {
    "Alexander Albon":"ALB",
    "Arvid Lindblad":"LIN",
    "Carlos Sainz":"SAI",
    "Charles Leclerc":"LEC",
    "Esteban Ocon":"OCO",
    "Fernando Alonso":"ALO",
    "Franco Colapinto":"COL",
    "Gabriel Bortoleto":"BOR",
    "George Russell":"RUS",
    "Isack Hadjar":"HAD",
    "Kimi Antonelli":"ANT",
    "Lance Stroll":"STR",
    "Lando Norris":"NOR",
    "Lewis Hamilton":"HAM",
    "Liam Lawson":"LAW",
    "Max Verstappen":"VER",
    "Nico Hulkenberg":"HUL",
    "Oliver Bearman":"BEA",
    "Oscar Piastri":"PIA",
    "Pierre Gasly":"GAS",
    "Sergio Perez":"PER",
    "Valtteri Bottas":"BOT"
}

df["Abbreviation"] = (
    df["FullName"].map(mapping)
)

# ==========================================================
# MERGE QUALIFYING DATA
# ==========================================================

merge_cols = [
    "Abbreviation",
    "GridPosition",
    "gaptopole_bestquali",
    "TeammateQualifyingGap"
]

df = df.merge(
    results[merge_cols],
    on="Abbreviation",
    how="left"
)

# ==========================================================
# SAVE
# ==========================================================

output_file = (
    "barcelona_2026_prediction_final.csv"
)

df.to_csv(
    output_file,
    index=False
)

# ==========================================================
# VERIFY
# ==========================================================

print()
print("=" * 60)
print("QUALIFYING DATA INJECTED")
print("=" * 60)

print(df[[
    "FullName",
    "GridPosition",
    "gaptopole_bestquali",
    "TeammateQualifyingGap"
]])

print()
print("Missing Values:")
print()

print(df[[
    "GridPosition",
    "gaptopole_bestquali",
    "TeammateQualifyingGap"
]].isna().sum())

print()
print("Saved:")
print(output_file)