import pandas as pd

from catboost import CatBoostClassifier

# ==========================================================
# LOAD TRAINING DATA
# ==========================================================

train_df = pd.read_csv(
    "f1_2020_2026_features_v6.csv"
)

print("=" * 60)
print("TRAINING DATA")
print("=" * 60)
print(train_df.shape)

# ==========================================================
# FEATURES
# ==========================================================

features = [

    "GridPosition",
    "gaptopole_bestquali",
    "AverageFinishLast5",

    "AverageFinishLast3",
    "AverageGridLast3",
    "ConstructorAverageFinishLast3",

    "DriverChampionshipPosition",
    "DriverChampionshipPoints",

    "ConstructorChampionshipPosition",
    "ConstructorChampionshipPoints",

    "TeammateQualifyingGap",

    "StreetCircuitPerformance",
    "PermanentCircuitPerformance",
    "HighSpeedCircuitPerformance",
    "HighDownforceCircuitPerformance",

    "RoundNumber"
]

# ==========================================================
# TRAINING SET
# ==========================================================

X_train = train_df[features]

y_train = train_df["Podium"]

# ==========================================================
# FINAL MODEL
# ==========================================================

print()
print("=" * 60)
print("TRAINING FINAL MODEL")
print("=" * 60)

model = CatBoostClassifier(

    iterations=300,
    depth=6,
    learning_rate=0.05,

    loss_function="Logloss",

    verbose=100,

    random_seed=42
)

model.fit(
    X_train,
    y_train
)

# ==========================================================
# LOAD BARCELONA FILE
# ==========================================================

pred_df = pd.read_csv(
    "barcelona_2026_prediction_final.csv"
)

print()
print("=" * 60)
print("BARCELONA DRIVERS")
print("=" * 60)
print(pred_df.shape)

# ==========================================================
# PREDICT
# ==========================================================

X_pred = pred_df[features]

pred_df["PodiumProbability"] = (
    model.predict_proba(X_pred)[:, 1]
)

# ==========================================================
# SORT
# ==========================================================

pred_df = pred_df.sort_values(
    by="PodiumProbability",
    ascending=False
).reset_index(drop=True)

# ==========================================================
# TOP 3
# ==========================================================

print()
print("=" * 60)
print("PREDICTED PODIUM")
print("=" * 60)

for i in range(3):

    row = pred_df.iloc[i]

    print(
        f"P{i+1} | "
        f"{row['FullName']} | "
        f"{row['TeamName']} | "
        f"{row['PodiumProbability']:.2%}"
    )

# ==========================================================
# FULL GRID
# ==========================================================

print()
print("=" * 60)
print("FULL RANKING")
print("=" * 60)

print(
    pred_df[
        [
            "FullName",
            "TeamName",
            "PodiumProbability"
        ]
    ]
)

# ==========================================================
# SAVE
# ==========================================================

pred_df.to_csv(
    "barcelona_2026_predictions.csv",
    index=False
)

print()
print("Saved:")
print("barcelona_2026_predictions.csv")