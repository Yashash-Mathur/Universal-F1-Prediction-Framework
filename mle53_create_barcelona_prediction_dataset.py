import pandas as pd

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("f1_2020_2026_features_v6.csv")

# ==========================================================
# GET LATEST DRIVER STATE
# ==========================================================

latest = df[
    (df["Year"] == 2026)
    &
    (df["RaceName"] == "Monaco Grand Prix")
].copy()

# ==========================================================
# PREPARE BARCELONA DATASET
# ==========================================================

prediction_df = latest[[
    "FullName",
    "TeamName",

    "ConstructorChampionshipPoints",
    "DriverChampionshipPoints",

    "ConstructorChampionshipPosition",
    "DriverChampionshipPosition",

    "AverageFinishLast5",

    "StreetCircuitPerformance",
    "PermanentCircuitPerformance",
    "HighSpeedCircuitPerformance",
    "HighDownforceCircuitPerformance",

    "AverageFinishLast3",
    "AverageGridLast3",
    "ConstructorAverageFinishLast3"
]].copy()

# ==========================================================
# BARCELONA ROUND
# ==========================================================

prediction_df["RoundNumber"] = 7

# ==========================================================
# QUALIFYING FIELDS
# ==========================================================

prediction_df["GridPosition"] = None
prediction_df["gaptopole_bestquali"] = None
prediction_df["TeammateQualifyingGap"] = None

# ==========================================================
# SAVE
# ==========================================================

prediction_df.to_csv(
    "barcelona_2026_prediction.csv",
    index=False
)

print(prediction_df.head())
print()
print("Rows:", len(prediction_df))
print()
print("Saved:")
print("barcelona_2026_prediction.csv")