import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("f1_2020_2026_features_v5.csv")

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(df.shape)

# ==========================================================
# FEATURES
# ==========================================================

features = [
    "GridPosition",
    "gaptopole_bestquali",
    "PitLaneStart",
    "HasQualiTime",
    "RoundNumber",

    "ConstructorChampionshipPoints",
    "DriverChampionshipPoints",
    "ConstructorChampionshipPosition",
    "DriverChampionshipPosition",

    "AverageFinishLast5",

    "TeammateQualifyingGap",
    "HasTeammateGap",

    "StreetCircuitPerformance",
    "PermanentCircuitPerformance",
    "HighSpeedCircuitPerformance",
    "HighDownforceCircuitPerformance",

    "HasStreetCircuitHistory",
    "HasPermanentCircuitHistory",
    "HasHighSpeedCircuitHistory",
    "HasHighDownforceCircuitHistory"
]

target = "Podium"

# ==========================================================
# MISSING VALUE HANDLING
# ==========================================================

for col in [
    "gaptopole_bestquali",
    "TeammateQualifyingGap",
    "StreetCircuitPerformance",
    "PermanentCircuitPerformance",
    "HighSpeedCircuitPerformance",
    "HighDownforceCircuitPerformance"
]:
    df[col] = df[col].fillna(df[col].median())

# ==========================================================
# STABILITY SPLITS
# ==========================================================

splits = [

    {
        "name": "Train 2020-2023 | Test 2024",
        "train_years": [2020, 2021, 2022, 2023],
        "test_years": [2024]
    },

    {
        "name": "Train 2020-2024 | Test 2025",
        "train_years": [2020, 2021, 2022, 2023, 2024],
        "test_years": [2025]
    },

    {
        "name": "Train 2020-2025 | Test 2026",
        "train_years": [2020, 2021, 2022, 2023, 2024, 2025],
        "test_years": [2026]
    }
]

results = []

# ==========================================================
# LOOP THROUGH SPLITS
# ==========================================================

for split in splits:

    print("\n")
    print("=" * 60)
    print(split["name"])
    print("=" * 60)

    train_df = df[df["Year"].isin(split["train_years"])]

    test_df = df[df["Year"].isin(split["test_years"])]

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    print("Train Rows:", len(train_df))
    print("Test Rows :", len(test_df))

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="F1",
        verbose=False,
        random_seed=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    results.append([
        split["name"],
        accuracy,
        precision,
        recall,
        f1
    ])

# ==========================================================
# RESULTS TABLE
# ==========================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Split",
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]
)

print("\n")
print("=" * 80)
print("STABILITY TEST RESULTS")
print("=" * 80)

print(results_df.to_string(index=False))

print("\n")
print("=" * 80)
print("AVERAGE PERFORMANCE")
print("=" * 80)

print(
    results_df[
        ["Accuracy", "Precision", "Recall", "F1"]
    ].mean()
)