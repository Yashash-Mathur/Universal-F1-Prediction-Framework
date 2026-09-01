from catboost import CatBoostClassifier


def create_catboost_model():
    """
    Create the standard CatBoost classifier
    used by the F1 prediction framework.
    """

    return CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        random_seed=42,
        verbose=False
    )