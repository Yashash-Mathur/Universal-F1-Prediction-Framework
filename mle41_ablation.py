import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("f1_2020_2026_features_v4.csv")

# =====================================================
# CREATE MISSING VALUE FLAGS
# =====================================================

df["HasGapToPole"] = (
    df["gaptopole_bestquali"]
    .notna()
    .astype(int)
)

df["HasTeammateGap"] = (
    df["TeammateQualifyingGap"]
    .notna()
    .astype(int)
)

# =====================================================
# MEDIAN IMPUTATION
# =====================================================

df["gaptopole_bestquali"] = (
    df["gaptopole_bestquali"]
    .fillna(df["gaptopole_bestquali"].median())
)

df["TeammateQualifyingGap"] = (
    df["TeammateQualifyingGap"]
    .fillna(df["TeammateQualifyingGap"].median())
)

# =====================================================
# TARGET
# =====================================================

y = df["Podium"]

# =====================================================
# FULL FEATURE LIST
# =====================================================

all_features = [
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
    "HasGapToPole",
    "HasTeammateGap"
]

# =====================================================
# FUNCTION TO TRAIN MODEL
# =====================================================

def evaluate_model(features_to_use):

    X = df[features_to_use]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=5000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    }

# =====================================================
# BASELINE (ALL FEATURES)
# =====================================================

results = []

baseline_metrics = evaluate_model(all_features)

results.append({
    "Removed Feature": "NONE (BASELINE)",
    **baseline_metrics
})

# =====================================================
# ABLATION TESTING
# =====================================================

for feature in all_features:

    remaining_features = [
        f for f in all_features
        if f != feature
    ]

    metrics = evaluate_model(remaining_features)

    results.append({
        "Removed Feature": feature,
        **metrics
    })

# =====================================================
# RESULTS TABLE
# =====================================================

results_df = pd.DataFrame(results)

results_df["F1 Drop"] = (
    baseline_metrics["F1"]
    - results_df["F1"]
)

results_df = results_df.sort_values(
    by="F1",
    ascending=False
)

print("\n")
print("=" * 90)
print("ABLATION TEST RESULTS")
print("=" * 90)

print(results_df.to_string(index=False))

print("\n")
print("=" * 90)
print("MOST IMPORTANT FEATURES")
print("=" * 90)

importance_df = results_df[
    results_df["Removed Feature"] != "NONE (BASELINE)"
].copy()

importance_df = importance_df.sort_values(
    by="F1 Drop",
    ascending=False
)

print(
    importance_df[
        ["Removed Feature", "F1 Drop"]
    ].to_string(index=False)
)