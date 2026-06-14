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

    "StreetCircuitPerformance",
    "PermanentCircuitPerformance",
    "HighSpeedCircuitPerformance",
    "HighDownforceCircuitPerformance",

    "HasStreetCircuitHistory",
    "HasPermanentCircuitHistory",
    "HasHighSpeedCircuitHistory",
    "HasHighDownforceCircuitHistory"
]

# ==========================================================
# TARGET
# ==========================================================

X = df[features]
y = df["Podium"]

# ==========================================================
# TIME SPLIT
#
# TRAIN = 2020-2024
# TEST  = 2025-2026
# ==========================================================

train_mask = df["Year"] <= 2024
test_mask = df["Year"] >= 2025

X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print("\n")
print("=" * 60)
print("TIME SPLIT")
print("=" * 60)

print(f"Train Rows: {len(X_train)}")
print(f"Test Rows : {len(X_test)}")

print("\nTrain Years:")
print(sorted(df.loc[train_mask, "Year"].unique()))

print("\nTest Years:")
print(sorted(df.loc[test_mask, "Year"].unique()))

# ==========================================================
# MODEL
# ==========================================================

model = CatBoostClassifier(

    iterations=300,

    depth=4,

    learning_rate=0.02,

    loss_function="Logloss",

    eval_metric="F1",

    random_seed=42,

    verbose=50
)

# ==========================================================
# TRAIN
# ==========================================================

print("\n")
print("=" * 60)
print("TRAINING CATBOOST")
print("=" * 60)

model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test),
    use_best_model=True
)

# ==========================================================
# PREDICTIONS
# ==========================================================

predictions = model.predict(X_test)

# ==========================================================
# METRICS
# ==========================================================

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\n")
print("=" * 60)
print("TIME SPLIT CATBOOST PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(confusion_matrix(y_test, predictions))

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.get_feature_importance()
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