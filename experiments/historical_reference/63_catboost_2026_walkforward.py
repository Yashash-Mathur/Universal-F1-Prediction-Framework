import pandas as pd
import numpy as np


from src.models.catboost_model import create_catboost_model

from src.features.feature_config import MODEL_FEATURES

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================================
# CONFIG
# ==========================================================

DATASET = "f1_2020_2026_features_v6.csv"

TEST_YEAR = 2026

MIN_RACE_ROWS = 20


# ==========================================================
# FEATURES
# ==========================================================

features = MODEL_FEATURES

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(DATASET)

df = df.sort_values(
    ["Year", "RoundNumber"]
).reset_index(drop=True)


# ==========================================================
# TARGET
# ==========================================================

df["Podium"] = (
    df["Position"] <= 3
).astype(int)


# ==========================================================
# FIND 2026 RACES
# ==========================================================

races = (
    df[df["Year"] == TEST_YEAR]
    [
        [
            "RoundNumber",
            "RaceName"
        ]
    ]
    .drop_duplicates()
    .sort_values("RoundNumber")
)


print("=" * 70)
print("CATBOOST 2026 WALK-FORWARD BACKTEST")
print("=" * 70)

print(
    f"\nCandidate 2026 races: {len(races)}"
)


# ==========================================================
# RESULTS STORAGE
# ==========================================================

results = []


# ==========================================================
# WALK-FORWARD BACKTEST
# ==========================================================

for _, race_info in races.iterrows():

    round_number = race_info["RoundNumber"]
    race_name = race_info["RaceName"]

    target = df[
        (df["Year"] == TEST_YEAR)
        &
        (df["RoundNumber"] == round_number)
        &
        (df["RaceName"] == race_name)
    ].copy()


    # ------------------------------------------------------
    # Skip incomplete races
    # ------------------------------------------------------

    if len(target) < MIN_RACE_ROWS:

        print(
            f"Skipping {TEST_YEAR} "
            f"R{int(round_number):02d} "
            f"{race_name} "
            f"(only {len(target)} rows)"
        )

        continue


    # ------------------------------------------------------
    # TRAIN ONLY BEFORE TARGET RACE
    # ------------------------------------------------------

    train = df[
        (
            (df["Year"] < TEST_YEAR)
            |
            (
                (df["Year"] == TEST_YEAR)
                &
                (df["RoundNumber"] < round_number)
            )
        )
    ].copy()


    if len(train) == 0:

        print(
            f"Skipping {race_name} "
            f"(no training data)"
        )

        continue


    # ------------------------------------------------------
    # DATA
    # ------------------------------------------------------

    X_train = train[features]
    y_train = train["Podium"]

    X_target = target[features]
    y_target = target["Podium"]


    # ------------------------------------------------------
    # MODEL
    # ------------------------------------------------------
    model = create_catboost_model()

    model.fit(
        X_train,
        y_train
    )


    # ------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------

    probabilities = (
        model
        .predict_proba(X_target)[:, 1]
    )


    predictions = (
        probabilities >= 0.5
    ).astype(int)


    # ------------------------------------------------------
    # CLASSIFICATION METRICS
    # ------------------------------------------------------

    accuracy = accuracy_score(
        y_target,
        predictions
    )

    precision = precision_score(
        y_target,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_target,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_target,
        predictions,
        zero_division=0
    )


    # ------------------------------------------------------
    # PODIUM RANKING
    # ------------------------------------------------------

    target["PodiumProbability"] = probabilities

    ranking = (
        target
        .sort_values(
            "PodiumProbability",
            ascending=False
        )
        .reset_index(drop=True)
    )


    predicted_top3 = list(
        ranking
        .head(3)["Abbreviation"]
    )


    actual_top3 = list(
        target[
            target["Podium"] == 1
        ]
        .sort_values("Position")
        ["Abbreviation"]
    )


    correct_top3 = len(
        set(predicted_top3)
        &
        set(actual_top3)
    )


    # ------------------------------------------------------
    # PRINT
    # ------------------------------------------------------

    print(
        f"{TEST_YEAR} "
        f"R{int(round_number):02d} "
        f"{race_name:<30}"
        f"Top3: {correct_top3}/3 "
        f"F1: {f1:.3f}"
    )


    # ------------------------------------------------------
    # STORE RESULT
    # ------------------------------------------------------

    results.append({

        "Year": TEST_YEAR,

        "RoundNumber": round_number,

        "RaceName": race_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "CorrectTop3": correct_top3,

        "PredictedTop3":
            ",".join(predicted_top3),

        "ActualTop3":
            ",".join(actual_top3)
    })


# ==========================================================
# RESULTS DATAFRAME
# ==========================================================

results_df = pd.DataFrame(results)


if len(results_df) == 0:

    raise ValueError(
        "No valid 2026 races were available."
    )


# ==========================================================
# OVERALL PERFORMANCE
# ==========================================================

print("\n")
print("=" * 70)
print("2026 BACKTEST PERFORMANCE")
print("=" * 70)

print(
    "Races tested:",
    len(results_df)
)

print(
    f"Average Accuracy : "
    f"{results_df['Accuracy'].mean():.4f}"
)

print(
    f"Average Precision: "
    f"{results_df['Precision'].mean():.4f}"
)

print(
    f"Average Recall   : "
    f"{results_df['Recall'].mean():.4f}"
)

print(
    f"Average F1       : "
    f"{results_df['F1'].mean():.4f}"
)


# ==========================================================
# TOP-3 PERFORMANCE
# ==========================================================

average_top3 = (
    results_df["CorrectTop3"]
    .mean()
)

perfect_top3 = (
    results_df["CorrectTop3"] == 3
).sum()

two_or_more = (
    results_df["CorrectTop3"] >= 2
).sum()


print("\n")
print("=" * 70)
print("2026 TOP-3 PERFORMANCE")
print("=" * 70)

print(
    f"Average correct podium drivers: "
    f"{average_top3:.2f} / 3"
)

print(
    f"Perfect Top-3 predictions: "
    f"{perfect_top3} / {len(results_df)}"
)

print(
    f"2+ correct podium drivers: "
    f"{two_or_more} / {len(results_df)}"
)


# ==========================================================
# SAVE
# ==========================================================

OUTPUT = (
    "catboost_2026_walkforward_results.csv"
)

results_df.to_csv(
    OUTPUT,
    index=False
)


print("\n")
print("Saved:")
print(OUTPUT)