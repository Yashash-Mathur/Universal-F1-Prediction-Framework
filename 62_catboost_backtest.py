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
# CONFIG
# ==========================================================

DATASET = "f1_2020_2026_features_v6.csv"

# Historical seasons to evaluate
TEST_YEARS = [2023, 2024, 2025]

# Number of races from each season to test
# None = every race after the minimum training history
RACES_PER_YEAR = None

MIN_TRAIN_RACES = 5


# ==========================================================
# FEATURES
# ==========================================================

features = [
    "GridPosition",
    "gaptopole_bestquali",
    "PitLaneStart",
    "HasQualiTime",
    "HasGapToPole",
    "TeammateQualifyingGap",
    "HasTeammateGap",

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
    "HighDownforceCircuitPerformance"
]


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(DATASET)

df = df.sort_values(
    ["Year", "RoundNumber", "Abbreviation"]
).reset_index(drop=True)

df["Podium"] = (
    df["Position"] <= 3
).astype(int)


# ==========================================================
# VALIDATE FEATURES
# ==========================================================

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        f"Missing features: {missing_features}"
    )


# ==========================================================
# CATBOOST PARAMETERS
# ==========================================================

MODEL_PARAMS = {

    "iterations": 300,

    "depth": 6,

    "learning_rate": 0.05,

    "loss_function": "Logloss",

    "random_seed": 42,

    "verbose": False
}


# ==========================================================
# FIND TEST RACES
# ==========================================================

all_races = (
    df[
        df["Year"].isin(TEST_YEARS)
    ]
    [
        [
            "Year",
            "RoundNumber",
            "RaceName"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "Year",
            "RoundNumber"
        ]
    )
)


# ==========================================================
# BACKTEST
# ==========================================================

results = []

race_counter = 0

print("=" * 70)
print("CATBOOST MULTI-RACE HISTORICAL BACKTEST")
print("=" * 70)

print(
    "Test years:",
    TEST_YEARS
)

print(
    "Total candidate races:",
    len(all_races)
)

print()


for _, race_info in all_races.iterrows():

    year = race_info["Year"]

    round_number = race_info["RoundNumber"]

    race_name = race_info["RaceName"]


    # ------------------------------------------------------
    # TRAINING DATA
    # STRICTLY BEFORE TARGET RACE
    # ------------------------------------------------------

    train = df[
        (
            (df["Year"] < year)
            |
            (
                (df["Year"] == year)
                &
                (
                    df["RoundNumber"]
                    < round_number
                )
            )
        )
    ].copy()


    # ------------------------------------------------------
    # SKIP EARLY RACES
    # ------------------------------------------------------

    train_races = (
        train[
            [
                "Year",
                "RoundNumber"
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    if train_races < MIN_TRAIN_RACES:

        continue


    # ------------------------------------------------------
    # TARGET
    # ------------------------------------------------------

    target = df[
        (df["Year"] == year)
        &
        (df["RoundNumber"] == round_number)
    ].copy()


    if len(target) < 20:

        print(
            f"Skipping {year} {race_name} "
            f"(only {len(target)} rows)"
        )

        continue


    # ------------------------------------------------------
    # OPTIONAL RACE LIMIT
    # ------------------------------------------------------

    if (
        RACES_PER_YEAR is not None
        and year in TEST_YEARS
    ):

        already_tested = sum(
            1
            for r in results
            if r["Year"] == year
        )

        if already_tested >= RACES_PER_YEAR:

            continue


    # ------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------

    X_train = train[features]

    y_train = train["Podium"]

    X_target = target[features]

    y_target = target["Podium"]


    model = CatBoostClassifier(
        **MODEL_PARAMS
    )


    model.fit(
        X_train,
        y_train
    )


    # ------------------------------------------------------
    # PREDICTIONS
    # ------------------------------------------------------

    probabilities = (
        model
        .predict_proba(X_target)[:, 1]
    )


    predictions = (
        probabilities >= 0.5
    ).astype(int)


    # ------------------------------------------------------
    # TOP 3 RANKING
    # ------------------------------------------------------

    target["PodiumProbability"] = (
        probabilities
    )

    ranking = (
        target
        .sort_values(
            "PodiumProbability",
            ascending=False
        )
        .reset_index(drop=True)
    )


    predicted_top3 = set(
        ranking
        .head(3)["Abbreviation"]
    )


    actual_top3 = set(
        target[
            target["Podium"] == 1
        ]["Abbreviation"]
    )


    correct_top3 = len(
        predicted_top3
        .intersection(actual_top3)
    )


    # ------------------------------------------------------
    # METRICS
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
    # STORE
    # ------------------------------------------------------

    results.append({

        "Year": year,

        "RoundNumber": round_number,

        "RaceName": race_name,

        "TrainingRows": len(train),

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "CorrectTop3": correct_top3,

        "PredictedTop3":
            ",".join(
                ranking
                .head(3)["Abbreviation"]
            ),

        "ActualTop3":
            ",".join(
                target[
                    target["Podium"] == 1
                ]
                .sort_values("Position")
                ["Abbreviation"]
            )
    })


    race_counter += 1


    print(
        f"{year} R{round_number:02d} "
        f"{race_name:<30} "
        f"Top3: {correct_top3}/3 "
        f"F1: {f1:.3f}"
    )


# ==========================================================
# RESULTS DATAFRAME
# ==========================================================

results_df = pd.DataFrame(results)


if len(results_df) == 0:

    raise ValueError(
        "No races were successfully backtested."
    )


# ==========================================================
# OVERALL PERFORMANCE
# ==========================================================

print("\n")
print("=" * 70)
print("OVERALL BACKTEST PERFORMANCE")
print("=" * 70)

print(
    f"Races tested: "
    f"{len(results_df)}"
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
# TOP 3 PERFORMANCE
# ==========================================================

print("\n")
print("=" * 70)
print("TOP 3 PREDICTION PERFORMANCE")
print("=" * 70)

print(
    f"Average correct podium drivers: "
    f"{results_df['CorrectTop3'].mean():.2f} / 3"
)

print(
    f"Perfect Top-3 predictions: "
    f"{(results_df['CorrectTop3'] == 3).sum()}"
    f" / {len(results_df)}"
)

print(
    f"2+ correct podium drivers: "
    f"{(results_df['CorrectTop3'] >= 2).sum()}"
    f" / {len(results_df)}"
)


# ==========================================================
# YEAR-WISE PERFORMANCE
# ==========================================================

print("\n")
print("=" * 70)
print("YEAR-WISE PERFORMANCE")
print("=" * 70)

year_summary = (
    results_df
    .groupby("Year")
    .agg(
        Races=("RaceName", "count"),
        Accuracy=("Accuracy", "mean"),
        Precision=("Precision", "mean"),
        Recall=("Recall", "mean"),
        F1=("F1", "mean"),
        AvgCorrectTop3=("CorrectTop3", "mean")
    )
    .round(4)
)

print(
    year_summary.to_string()
)


# ==========================================================
# RACE RESULTS
# ==========================================================

print("\n")
print("=" * 70)
print("RACE-BY-RACE RESULTS")
print("=" * 70)

print(
    results_df[
        [
            "Year",
            "RoundNumber",
            "RaceName",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "CorrectTop3",
            "PredictedTop3",
            "ActualTop3"
        ]
    ]
    .to_string(index=False)
)


# ==========================================================
# SAVE RESULTS
# ==========================================================

OUTPUT = "catboost_multirace_backtest_results.csv"

results_df.to_csv(
    OUTPUT,
    index=False
)

print("\n")
print("Saved:")
print(OUTPUT)