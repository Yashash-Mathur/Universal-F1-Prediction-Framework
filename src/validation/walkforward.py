"""
Reusable chronological walk-forward evaluation.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.models.catboost_model import create_catboost_model
from src.features.feature_config import MODEL_FEATURES


def run_walkforward(df: pd.DataFrame, test_year: int = 2026, min_race_rows: int = 20) -> tuple:
    """
    Run chronological walk-forward validation for the given season.

    Returns:
        (results_df, summary_metrics_dict)
    """
    df = df.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)
    if "Podium" not in df.columns:
        df["Podium"] = (df["Position"] <= 3).astype(int)

    races = (
        df[df["Year"] == test_year][["RoundNumber", "RaceName"]]
        .drop_duplicates()
        .sort_values("RoundNumber")
    )

    results = []

    for _, race_info in races.iterrows():
        round_number = race_info["RoundNumber"]
        race_name = race_info["RaceName"]

        target = df[
            (df["Year"] == test_year)
            & (df["RoundNumber"] == round_number)
            & (df["RaceName"] == race_name)
        ].copy()

        if len(target) < min_race_rows:
            continue

        train = df[
            (df["Year"] < test_year)
            | ((df["Year"] == test_year) & (df["RoundNumber"] < round_number))
        ].copy()

        if len(train) == 0:
            continue

        X_train = train[MODEL_FEATURES]
        y_train = train["Podium"]
        X_target = target[MODEL_FEATURES]
        y_target = target["Podium"]

        model = create_catboost_model()
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_target)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        acc = accuracy_score(y_target, predictions)
        prec = precision_score(y_target, predictions, zero_division=0)
        rec = recall_score(y_target, predictions, zero_division=0)
        f1 = f1_score(y_target, predictions, zero_division=0)

        target["PodiumProbability"] = probabilities
        ranking = target.sort_values("PodiumProbability", ascending=False).reset_index(drop=True)
        predicted_top3 = list(ranking.head(3)["Abbreviation"])
        actual_top3 = list(target[target["Podium"] == 1]["Abbreviation"])
        correct_top3 = len(set(predicted_top3) & set(actual_top3))

        results.append({
            "Round": round_number,
            "RaceName": race_name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "CorrectPodium": correct_top3,
            "PredictedTop3": ", ".join(predicted_top3),
            "ActualTop3": ", ".join(actual_top3)
        })

    results_df = pd.DataFrame(results)
    summary = {
        "Races": len(results_df),
        "MeanAccuracy": results_df["Accuracy"].mean(),
        "MeanPrecision": results_df["Precision"].mean(),
        "MeanRecall": results_df["Recall"].mean(),
        "MeanF1": results_df["F1"].mean(),
        "MeanCorrectPodium": results_df["CorrectPodium"].mean(),
        "PerfectPodiums": (results_df["CorrectPodium"] == 3).sum(),
    }
    return results_df, summary
