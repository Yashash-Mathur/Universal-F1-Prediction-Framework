"""
Generalized race prediction.

Usage:
    python -m src.prediction.predict \\
        --features f1_2020_2026_features_v6.csv \\
        --target dutch_gp_catboost_prediction_fixed.csv \\
        --train-through-round 11 \\
        --train-year 2026 \\
        --output prediction_output.csv

The --target file must already contain qualifying columns
(GridPosition, gaptopole_bestquali, TeammateQualifyingGap, HasQualiTime)
as produced by the race-specific data-fix or qualifying-injector script.
"""

import argparse

import pandas as pd

from src.models.catboost_model import create_catboost_model
from src.features.feature_config import MODEL_FEATURES as FEATURES


def _load_and_validate(features_path: str, target_path: str,
                       train_year: int, train_through_round: int) -> tuple:
    df = pd.read_csv(features_path)
    target = pd.read_csv(target_path)

    train = df[
        (df["Year"] < train_year)
        | ((df["Year"] == train_year) & (df["RoundNumber"] <= train_through_round))
    ].copy()
    train["Podium"] = (train["Position"] <= 3).astype(int)

    # Availability flags (may already be present in a fixed target file)
    if "HasGapToPole" not in target.columns:
        target["HasGapToPole"] = target["gaptopole_bestquali"].notna().astype(int)
    if "HasTeammateGap" not in target.columns:
        target["HasTeammateGap"] = target["TeammateQualifyingGap"].notna().astype(int)

    missing_train = [f for f in FEATURES if f not in train.columns]
    missing_target = [f for f in FEATURES if f not in target.columns]
    if missing_train:
        raise ValueError(f"Missing training features: {missing_train}")
    if missing_target:
        raise ValueError(f"Missing target features: {missing_target}")

    return train, target


def predict(features_path: str, target_path: str,
            train_year: int, train_through_round: int,
            output_path: str) -> pd.DataFrame:
    """Train on historical data and predict podium probabilities for the target race."""
    train, target = _load_and_validate(
        features_path, target_path, train_year, train_through_round
    )

    model = create_catboost_model()
    model.fit(train[FEATURES], train["Podium"])

    target["PodiumProbability"] = model.predict_proba(target[FEATURES])[:, 1]

    ranking = (
        target[["Abbreviation", "FullName", "TeamName", "GridPosition",
                "PodiumProbability"]]
        .sort_values("PodiumProbability", ascending=False)
        .reset_index(drop=True)
    )
    ranking["PredictedRank"] = ranking.index + 1
    ranking["PredictedPodium"] = (ranking["PredictedRank"] <= 3).astype(int)

    ranking.to_csv(output_path, index=False)

    print(f"\nPredicted Top 3:")
    print(ranking.head(3)[["PredictedRank", "Abbreviation", "FullName",
                            "TeamName", "PodiumProbability"]].to_string(index=False))
    print(f"\nSaved: {output_path}")

    return ranking


def main():
    parser = argparse.ArgumentParser(description="Predict F1 race podium.")
    parser.add_argument("--features", required=True,
                        help="Path to historical features CSV (e.g. f1_2020_2026_features_v6.csv)")
    parser.add_argument("--target", required=True,
                        help="Path to target race features CSV (post-qualifying)")
    parser.add_argument("--train-year", type=int, required=True,
                        help="Season year for training cutoff")
    parser.add_argument("--train-through-round", type=int, required=True,
                        help="Last round number to include in training data")
    parser.add_argument("--output", required=True,
                        help="Path for the output predictions CSV")
    args = parser.parse_args()

    predict(
        features_path=args.features,
        target_path=args.target,
        train_year=args.train_year,
        train_through_round=args.train_through_round,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
