import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("f1_2020_2026_features_v4.csv")

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(df.shape)

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
# FEATURES
# =====================================================

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
    "HasGapToPole",
    "HasTeammateGap"
]

X = df[features]
y = df["Podium"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================================
# DECISION TREE MODEL
# =====================================================

model = DecisionTreeClassifier(
    random_state=42,
    max_depth=3
)

model.fit(X_train, y_train)

# =====================================================
# PREDICTIONS
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# METRICS
# =====================================================

print("\n")
print("=" * 60)
print("DECISION TREE PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")

# =====================================================
# CONFUSION MATRIX
# =====================================================

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(confusion_matrix(y_test, y_pred))

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
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