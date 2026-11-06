import pandas as pd
import numpy as np

from catboost import CatBoostRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
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

y = df["Position"]

# ==========================================================
# TIME SPLIT
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

print("Train Rows:", len(X_train))
print("Test Rows :", len(X_test))

# ==========================================================
# MODEL
# ==========================================================

model = CatBoostRegressor(

    iterations=500,
    depth=4,
    learning_rate=0.03,

    loss_function="RMSE",

    random_seed=42,

    verbose=50
)

# ==========================================================
# TRAIN
# ==========================================================

print("\n")
print("=" * 60)
print("TRAINING POSITION MODEL")
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

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

print("\n")
print("=" * 60)
print("POSITION MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")

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
# SAMPLE PREDICTIONS
# ==========================================================

results = pd.DataFrame({

    "ActualPosition": y_test.values,
    "PredictedPosition": predictions

})

print("\n")
print("=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

print(results.head(20))