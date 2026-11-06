import pandas as pd

from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
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

X = df[features]

y = df["Podium"]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================================
# CATBOOST MODEL
# ==========================================================

model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.03,

    loss_function="Logloss",
    eval_metric="F1",

    random_seed=42,

    verbose=100
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
# PREDICT
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
print("CATBOOST PERFORMANCE")
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

# ==========================================================
# TOP 10 FEATURES
# ==========================================================

print("\n")
print("=" * 60)
print("TOP 10 FEATURES")
print("=" * 60)

print(
    importance_df.head(10).to_string(index=False)
)