import pandas as pd
import numpy as np

from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
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
# MISSING VALUE FLAGS
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

    # Qualifying
    "GridPosition",
    "gaptopole_bestquali",
    "PitLaneStart",
    "HasQualiTime",
    "TeammateQualifyingGap",

    # Championship
    "ConstructorChampionshipPoints",
    "DriverChampionshipPoints",
    "ConstructorChampionshipPosition",
    "DriverChampionshipPosition",

    # Season
    "RoundNumber",

    # Form
    "AverageFinishLast5",
    "AverageFinishLast3",
    "AverageGridLast3",
    "ConstructorAverageFinishLast3",

    # Circuit History
    "StreetCircuitPerformance",
    "PermanentCircuitPerformance",
    "HighSpeedCircuitPerformance",
    "HighDownforceCircuitPerformance",

    # Missing Flags
    "HasGapToPole",
    "HasTeammateGap",

    "HasStreetCircuitHistory",
    "HasPermanentCircuitHistory",
    "HasHighSpeedCircuitHistory",
    "HasHighDownforceCircuitHistory"
]

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

train_df = df[df["Year"] <= 2024].copy()
test_df = df[df["Year"] >= 2025].copy()

print("\n")
print("=" * 60)
print("TIME SPLIT")
print("=" * 60)

print("Train Rows:", len(train_df))
print("Test Rows :", len(test_df))

# ============================================================
# X / Y
# ============================================================

X_train = train_df[features]
X_test = test_df[features]

y_train = train_df["Podium"]
y_test = test_df["Podium"]

# ============================================================
# CATBOOST
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING CATBOOST V2")
print("=" * 60)

model = CatBoostClassifier(

    iterations=300,
    depth=6,
    learning_rate=0.05,

    loss_function="Logloss",
    eval_metric="F1",

    random_seed=42,
    verbose=50
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test)
)

# ============================================================
# PREDICTIONS
# ============================================================

predictions = model.predict(X_test)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\n")
print("=" * 60)
print("CATBOOST V2 PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, predictions)

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = model.get_feature_importance()

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n")
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(importance_df.to_string(index=False))

# ============================================================
# TOP 10
# ============================================================

print("\n")
print("=" * 60)
print("TOP 10 FEATURES")
print("=" * 60)

print(
    importance_df.head(10).to_string(index=False)
)