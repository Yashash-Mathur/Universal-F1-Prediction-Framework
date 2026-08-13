import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================================
# CONFIG
# ==========================================================

DATASET = "f1_2020_2026_features_v6.csv"

TARGET_YEAR = 2025
TARGET_RACE = "Dutch Grand Prix"


# ==========================================================
# FEATURES
# ==========================================================

features = [
    "GridPosition",
    "gaptopole_bestquali",
    "PitLaneStart",
    "HasQualiTime",
    "TeammateQualifyingGap",

    "ConstructorChampionshipPoints",
    "DriverChampionshipPoints",
    "ConstructorChampionshipPosition",
    "DriverChampionshipPosition",

    "RoundNumber",

    "AverageFinishLast5",
    "AverageFinishLast3",
    "AverageGridLast3",
    "ConstructorAverageFinishLast3",

    "StreetCircuitPerformance",
    "PermanentCircuitPerformance",
    "HighSpeedCircuitPerformance",
    "HighDownforceCircuitPerformance"
]


# ==========================================================
# LOAD
# ==========================================================

df = pd.read_csv(DATASET)

df = df.sort_values(
    ["Year", "RoundNumber"]
).reset_index(drop=True)

df["Podium"] = (
    df["Position"] <= 3
).astype(int)


# ==========================================================
# FIND TARGET RACE
# ==========================================================

target = df[
    (df["Year"] == TARGET_YEAR)
    &
    (df["RaceName"] == TARGET_RACE)
].copy()

if len(target) == 0:
    raise ValueError(
        "Target race not found."
    )


target_round = target["RoundNumber"].iloc[0]


# ==========================================================
# STRICT TEMPORAL SPLIT
# ==========================================================

train = df[
    (
        (df["Year"] < TARGET_YEAR)
        |
        (
            (df["Year"] == TARGET_YEAR)
            &
            (df["RoundNumber"] < target_round)
        )
    )
].copy()


print("=" * 70)
print("CATBOOST HISTORICAL BACKTEST")
print("=" * 70)

print(
    f"Target: {TARGET_YEAR} {TARGET_RACE}"
)

print(
    f"Target round: {target_round}"
)

print(
    "\nTraining rows:",
    len(train)
)

print(
    "Target rows:",
    len(target)
)


# ==========================================================
# VALIDATION
# ==========================================================

if len(target) < 20:
    raise ValueError(
        "Unexpected target race size."
    )


if train["RoundNumber"].isna().any():
    raise ValueError(
        "Missing RoundNumber detected."
    )


# ==========================================================
# TRAIN
# ==========================================================

X_train = train[features]
y_train = train["Podium"]

X_target = target[features]
y_target = target["Podium"]


model = CatBoostClassifier(

    iterations=300,
    depth=6,
    learning_rate=0.05,

    loss_function="Logloss",

    random_seed=42,

    verbose=False
)


model.fit(
    X_train,
    y_train
)


# ==========================================================
# PREDICTIONS
# ==========================================================

probabilities = (
    model.predict_proba(X_target)[:, 1]
)

predictions = (
    probabilities >= 0.5
).astype(int)


# ==========================================================
# METRICS
# ==========================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION PERFORMANCE")
print("=" * 70)

print(
    f"Accuracy : "
    f"{accuracy_score(y_target, predictions):.4f}"
)

print(
    f"Precision: "
    f"{precision_score(y_target, predictions, zero_division=0):.4f}"
)

print(
    f"Recall   : "
    f"{recall_score(y_target, predictions, zero_division=0):.4f}"
)

print(
    f"F1 Score : "
    f"{f1_score(y_target, predictions, zero_division=0):.4f}"
)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_target,
        predictions
    )
)


# ==========================================================
# PODIUM RANKING
# ==========================================================

target["PodiumProbability"] = probabilities

target["PredictedPodium"] = predictions

ranking = (
    target[
        [
            "Abbreviation",
            "FullName",
            "TeamName",
            "Position",
            "Podium",
            "PodiumProbability",
            "PredictedPodium"
        ]
    ]
    .sort_values(
        "PodiumProbability",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\n")
print("=" * 70)
print("PREDICTED PODIUM RANKING")
print("=" * 70)

print(
    ranking.to_string(
        index=False
    )
)


# ==========================================================
# TOP 3
# ==========================================================

top3 = ranking.head(3)

actual_podium = set(
    target.loc[
        target["Podium"] == 1,
        "Abbreviation"
    ]
)

predicted_top3 = set(
    top3["Abbreviation"]
)

correct = (
    predicted_top3
    .intersection(actual_podium)
)


print("\n")
print("=" * 70)
print("TOP 3 RESULT")
print("=" * 70)

print(
    "Predicted:",
    list(top3["Abbreviation"])
)

print(
    "Actual:",
    list(
        target[
            target["Podium"] == 1
        ]["Abbreviation"]
    )
)

print(
    "Correct podium drivers:",
    len(correct),
    "/ 3"
)


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance = model.get_feature_importance()

importance_df = (
    pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })
    .sort_values(
        "Importance",
        ascending=False
    )
)

print("\n")
print("=" * 70)
print("TOP FEATURES")
print("=" * 70)

print(
    importance_df.head(10)
    .to_string(index=False)
)