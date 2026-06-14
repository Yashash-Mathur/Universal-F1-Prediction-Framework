import pandas as pd

# ==========================================================
# LOAD FILE
# ==========================================================

audit = pd.read_csv(
    "barcelona_2026_predictions.csv"
)

# ==========================================================
# SORT BY MODEL RANKING
# ==========================================================

audit = audit.sort_values(
    by="PodiumProbability",
    ascending=False
)

# ==========================================================
# TOP 10 AUDIT
# ==========================================================

print("\n" + "="*80)
print("TOP 10 DRIVER AUDIT")
print("="*80)

print(
    audit[[
        "FullName",
        "TeamName",
        "PodiumProbability",
        "GridPosition",
        "gaptopole_bestquali",
        "ConstructorChampionshipPoints",
        "DriverChampionshipPoints",
        "DriverChampionshipPosition",
        "AverageFinishLast5",
        "AverageFinishLast3"
    ]]
    .head(10)
    .to_string(index=False)
)