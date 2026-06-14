import pandas as pd
import numpy as np

from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("f1_2020_2026_features_v6.csv")

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(df.shape)

# ============================================================
# TARGET
# ============================================================

df["Podium"] = (df["Position"] <= 3).astype(int)

# ============================================================
# MISSING FLAGS
# ============================================================

df["HasGapToPole"] = (
    df["gaptopole_bestquali"].notna()
).astype(int)

df["HasTeammateGap"] = (
    df["TeammateQualifyingGap"].notna()
).astype(int)

# ============================================================
# FEATURES
# ============================================================

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
    "HighDownforceCircuitPerformance",

    "HasGapToPole",
    "HasTeammateGap",

    "HasStreetCircuitHistory",
    "HasPermanentCircuitHistory",
    "HasHighSpeedCircuitHistory",
    "HasHighDownforceCircuitHistory"
]

# ============================================================
# TIME SPLIT
# ============================================================

train_df = df[df["Year"] <= 2024].copy()
test_df = df[df["Year"] >= 2025].copy()

print("\n")
print("=" * 60)
print("TIME SPLIT")
print("=" * 60)

print("Train Rows:", len(train_df))
print("Test Rows :", len(test_df))

X_train = train_df[features]
X_test = test_df[features]

y_train = train_df["Podium"]
y_test = test_df["Podium"]

# ============================================================
# CONFIGURATIONS
# ============================================================

configs = [

    {
        "name": "Model_A",
        "depth": 4,
        "learning_rate": 0.03,
        "iterations": 500
    },

    {
        "name": "Model_B",
        "depth": 5,
        "learning_rate": 0.03,
        "iterations": 500
    },

    {
        "name": "Model_C",
        "depth": 6,
        "learning_rate": 0.02,
        "iterations": 800
    },

    {
        "name": "Model_D",
        "depth": 4,
        "learning_rate": 0.05,
        "iterations": 300
    },

    {
        "name": "Model_E",
        "depth": 5,
        "learning_rate": 0.05,
        "iterations": 300
    },

    {
        "name": "Model_F",
        "depth": 6,
        "learning_rate": 0.05,
        "iterations": 300
    }
]

results = []

# ============================================================
# TRAIN ALL MODELS
# ============================================================

for config in configs:

    print("\n")
    print("=" * 60)
    print(f"TRAINING {config['name']}")
    print("=" * 60)

    model = CatBoostClassifier(

        depth=config["depth"],
        learning_rate=config["learning_rate"],
        iterations=config["iterations"],

        loss_function="Logloss",
        eval_metric="F1",

        random_seed=42,
        verbose=False
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test)
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    results.append({

        "Model": config["name"],
        "Depth": config["depth"],
        "LearningRate": config["learning_rate"],
        "Iterations": config["iterations"],

        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1",
    ascending=False
)

print("\n")
print("=" * 80)
print("CATBOOST TUNING RESULTS")
print("=" * 80)

print(results_df.to_string(index=False))

# ============================================================
# BEST MODEL
# ============================================================

best_model = results_df.iloc[0]

print("\n")
print("=" * 80)
print("BEST CONFIGURATION")
print("=" * 80)

print(best_model)