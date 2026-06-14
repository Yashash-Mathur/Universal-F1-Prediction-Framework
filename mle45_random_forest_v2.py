import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
# MISSING VALUE FLAGS
# ==========================================================

df["HasGapToPole"] = df["gaptopole_bestquali"].notna().astype(int)

df["HasTeammateGap"] = (
    df["TeammateQualifyingGap"].notna().astype(int)
)

# ==========================================================
# IMPUTATION
# ==========================================================

df["gaptopole_bestquali"] = (
    df.groupby("FullName")["gaptopole_bestquali"]
      .transform(lambda x: x.fillna(x.median()))
)

df["TeammateQualifyingGap"] = (
    df.groupby("FullName")["TeammateQualifyingGap"]
      .transform(lambda x: x.fillna(x.median()))
)

# ==========================================================
# NEW CIRCUIT FEATURE IMPUTATION
# ==========================================================

circuit_features = [
    "StreetCircuitPerformance",
    "PermanentCircuitPerformance",
    "HighSpeedCircuitPerformance",
    "HighDownforceCircuitPerformance"
]

for feature in circuit_features:

    df[feature] = (
        df.groupby("FullName")[feature]
        .transform(lambda x: x.fillna(x.median()))
    )

# ==========================================================
# REMAINING NaNs
# (rookies / no history)
# ==========================================================

for feature in circuit_features:
    df[feature] = df[feature].fillna(df[feature].median())

df["gaptopole_bestquali"] = (
    df["gaptopole_bestquali"]
    .fillna(df["gaptopole_bestquali"].median())
)

df["TeammateQualifyingGap"] = (
    df["TeammateQualifyingGap"]
    .fillna(df["TeammateQualifyingGap"].median())
)

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

    "HasGapToPole",
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
# MODEL
# ==========================================================

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

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
print("RANDOM FOREST V2 PERFORMANCE")
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
    "Importance": model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)

print("\n")
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(importance_df.to_string(index=False))