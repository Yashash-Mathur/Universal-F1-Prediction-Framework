import pandas as pd
from catboost import CatBoostClassifier


# ==========================================================
# CONFIG
# ==========================================================

HISTORICAL_DATASET = "f1_2020_2026_features_v6.csv"

# IMPORTANT:
# This is the output produced by 65_dutchgp_datafix.py
DUTCH_DATASET = "f1_2020_2026_dutch_gp_prediction_fixed.csv"

OUTPUT = "dutch_gp_catboost_prediction_final.csv"

TRAIN_YEAR = 2026
TRAIN_UP_TO_ROUND = 11

TARGET_YEAR = 2026
TARGET_ROUND = 12
TARGET_RACE = "Dutch Grand Prix"


# ==========================================================
# FEATURES
# EXACT SAME FEATURES AS 63 BACKTEST
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
    "HighDownforceCircuitPerformance",

    "HasStreetCircuitHistory",
    "HasPermanentCircuitHistory",
    "HasHighSpeedCircuitHistory",
    "HasHighDownforceCircuitHistory"
]


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(HISTORICAL_DATASET)
target = pd.read_csv(DUTCH_DATASET)


print("=" * 70)
print("DUTCH GP FINAL CATBOOST PREDICTION")
print("=" * 70)

print("\nHistorical dataset:", HISTORICAL_DATASET)
print("Dutch GP dataset:", DUTCH_DATASET)

print("\nHistorical shape:", df.shape)
print("Dutch GP shape:", target.shape)


# ==========================================================
# VALIDATE TARGET RACE
# ==========================================================

if len(target) != 22:
    raise ValueError(
        f"Expected 22 Dutch GP drivers, found {len(target)}."
    )


if "RaceName" in target.columns:

    target_races = target["RaceName"].dropna().unique()

    if len(target_races) != 1 or target_races[0] != TARGET_RACE:

        raise ValueError(
            f"Unexpected target race: {target_races}"
        )


if "RoundNumber" in target.columns:

    target_rounds = target["RoundNumber"].dropna().unique()

    if len(target_rounds) != 1 or target_rounds[0] != TARGET_ROUND:

        raise ValueError(
            f"Expected Dutch GP Round {TARGET_ROUND}, "
            f"found {target_rounds}"
        )


# ==========================================================
# TRAINING DATA
# STRICTLY THROUGH HUNGARIAN GP
# ==========================================================

train = df[
    (
        (df["Year"] < TRAIN_YEAR)
        |
        (
            (df["Year"] == TRAIN_YEAR)
            &
            (df["RoundNumber"] <= TRAIN_UP_TO_ROUND)
        )
    )
].copy()


# ==========================================================
# TARGET
# ==========================================================

train["Podium"] = (
    train["Position"] <= 3
).astype(int)


print("\n")
print("=" * 70)
print("TRAINING DATA")
print("=" * 70)

print(
    f"Training through: "
    f"{TRAIN_YEAR} Round {TRAIN_UP_TO_ROUND}"
)

print("Training rows:", len(train))
print("Dutch GP rows:", len(target))


# ==========================================================
# QUALIFYING FEATURES
# THESE ARE NOW ACTUAL DUTCH GP QUALIFYING VALUES
# FROM SCRIPT 65
# ==========================================================

required_quali_columns = [
    "GridPosition",
    "gaptopole_bestquali",
    "TeammateQualifyingGap",
    "HasQualiTime"
]

missing_quali = [
    col
    for col in required_quali_columns
    if col not in target.columns
]

if missing_quali:

    raise ValueError(
        "Missing qualifying columns from fixed Dutch dataset:\n"
        + "\n".join(missing_quali)
    )


# ==========================================================
# CREATE QUALIFYING AVAILABILITY FLAGS
# ==========================================================

target["HasGapToPole"] = (
    target["gaptopole_bestquali"]
    .notna()
    .astype(int)
)

target["HasTeammateGap"] = (
    target["TeammateQualifyingGap"]
    .notna()
    .astype(int)
)


# ==========================================================
# HISTORICAL DATA FOR CIRCUIT HISTORY FLAGS
# THROUGH HUNGARIAN GP ONLY
# ==========================================================

history = df[
    (
        (df["Year"] > 2023)
        |
        (
            (df["Year"] == 2023)
            &
            (df["RoundNumber"] >= 1)
        )
    )
    &
    (
        (df["Year"] < TARGET_YEAR)
        |
        (
            (df["Year"] == TARGET_YEAR)
            &
            (df["RoundNumber"] <= TRAIN_UP_TO_ROUND)
        )
    )
].copy()


# ==========================================================
# CIRCUIT DEFINITIONS
# MATCH BACKTEST
# ==========================================================

street_circuits = {
    "Australian Grand Prix",
    "Azerbaijan Grand Prix",
    "Canadian Grand Prix",
    "Miami Grand Prix",
    "Monaco Grand Prix",
    "Saudi Arabian Grand Prix",
    "Singapore Grand Prix",
    "Las Vegas Grand Prix"
}


high_speed_circuits = {
    "Australian Grand Prix",
    "Austrian Grand Prix",
    "Azerbaijan Grand Prix",
    "Belgian Grand Prix",
    "British Grand Prix",
    "Canadian Grand Prix",
    "Italian Grand Prix",
    "Las Vegas Grand Prix",
    "Mexico City Grand Prix",
    "Miami Grand Prix",
    "Saudi Arabian Grand Prix",
    "Styrian Grand Prix",
    "70th Anniversary Grand Prix",
    "Sakhir Grand Prix"
}


high_downforce_circuits = {
    "Monaco Grand Prix",
    "Singapore Grand Prix",
    "Hungarian Grand Prix",
    "Dutch Grand Prix",
    "Japanese Grand Prix",
    "Spanish Grand Prix",
    "Qatar Grand Prix",
    "British Grand Prix"
}


# ==========================================================
# CIRCUIT HISTORY FLAGS
# ==========================================================

target["HasStreetCircuitHistory"] = 0
target["HasPermanentCircuitHistory"] = 0
target["HasHighSpeedCircuitHistory"] = 0
target["HasHighDownforceCircuitHistory"] = 0


for idx, row in target.iterrows():

    driver = row["FullName"]

    driver_history = history[
        history["FullName"] == driver
    ]


    # ------------------------------------------------------
    # STREET
    # ------------------------------------------------------

    if driver_history["RaceName"].isin(
        street_circuits
    ).any():

        target.at[
            idx,
            "HasStreetCircuitHistory"
        ] = 1


    # ------------------------------------------------------
    # PERMANENT
    # ------------------------------------------------------

    if driver_history[
        ~driver_history["RaceName"].isin(
            street_circuits
        )
    ].shape[0] > 0:

        target.at[
            idx,
            "HasPermanentCircuitHistory"
        ] = 1


    # ------------------------------------------------------
    # HIGH SPEED
    # ------------------------------------------------------

    if driver_history["RaceName"].isin(
        high_speed_circuits
    ).any():

        target.at[
            idx,
            "HasHighSpeedCircuitHistory"
        ] = 1


    # ------------------------------------------------------
    # HIGH DOWNFORCE
    # ------------------------------------------------------

    if driver_history["RaceName"].isin(
        high_downforce_circuits
    ).any():

        target.at[
            idx,
            "HasHighDownforceCircuitHistory"
        ] = 1


# ==========================================================
# FEATURE VALIDATION
# ==========================================================

missing_train = [
    feature
    for feature in features
    if feature not in train.columns
]

missing_target = [
    feature
    for feature in features
    if feature not in target.columns
]


if missing_train:

    raise ValueError(
        "Missing training features:\n"
        + "\n".join(missing_train)
    )


if missing_target:

    raise ValueError(
        "Missing Dutch GP features:\n"
        + "\n".join(missing_target)
    )


# ==========================================================
# FINAL FEATURE CHECK
# ==========================================================

print("\n")
print("=" * 70)
print("FINAL FEATURE VALIDATION")
print("=" * 70)

feature_status = pd.DataFrame({
    "Feature": features,
    "MissingValues": [
        target[f].isna().sum()
        for f in features
    ]
})

print(
    feature_status.to_string(index=False)
)


# ==========================================================
# CRITICAL QUALIFYING VALIDATION
# ==========================================================

print("\n")
print("=" * 70)
print("QUALIFYING DATA VALIDATION")
print("=" * 70)

print(
    target[
        [
            "GridPosition",
            "gaptopole_bestquali",
            "TeammateQualifyingGap",
            "HasQualiTime",
            "HasGapToPole",
            "HasTeammateGap"
        ]
    ].isna().sum()
)


# ==========================================================
# DRIVER / TEAM VALIDATION
# ==========================================================

print("\n")
print("=" * 70)
print("DUTCH GP ROSTER")
print("=" * 70)

print(
    target[
        [
            "GridPosition",
            "Abbreviation",
            "FullName",
            "TeamName"
        ]
    ]
    .sort_values("GridPosition")
    .to_string(index=False)
)


# ==========================================================
# SPECIAL DUTCH GP ROSTER CHECK
# ==========================================================

lawson_team = (
    target.loc[
        target["Abbreviation"] == "LAW",
        "TeamName"
    ]
)

tsu_team = (
    target.loc[
        target["Abbreviation"] == "TSU",
        "TeamName"
    ]
)

if len(lawson_team) == 1:

    print(
        "\nLawson team:",
        lawson_team.iloc[0]
    )

if len(tsu_team) == 1:

    print(
        "Tsunoda team:",
        tsu_team.iloc[0]
    )


# ==========================================================
# TRAIN CATBOOST
# SAME MODEL SETTINGS AS 63
# ==========================================================

X_train = train[features]
y_train = train["Podium"]

X_target = target[features]


print("\n")
print("=" * 70)
print("TRAINING CATBOOST")
print("=" * 70)

model = CatBoostClassifier(

    iterations=300,
    depth=6,
    learning_rate=0.05,

    loss_function="Logloss",

    random_seed=42,

    verbose=False
)


model.fit(
    X_train,
    y_train
)


# ==========================================================
# PREDICT
# ==========================================================

probabilities = (
    model.predict_proba(X_target)[:, 1]
)


# ==========================================================
# RANK
# ==========================================================

target["PodiumProbability"] = probabilities


ranking = (
    target[
        [
            "Abbreviation",
            "FullName",
            "TeamName",
            "GridPosition",
            "gaptopole_bestquali",
            "TeammateQualifyingGap",
            "DriverChampionshipPoints",
            "DriverChampionshipPosition",
            "PodiumProbability"
        ]
    ]
    .sort_values(
        "PodiumProbability",
        ascending=False
    )
    .reset_index(drop=True)
)


ranking["PredictedRank"] = (
    ranking.index + 1
)


ranking["PredictedPodium"] = (
    ranking["PredictedRank"] <= 3
).astype(int)


# ==========================================================
# FINAL RANKING
# ==========================================================

print("\n")
print("=" * 70)
print("DUTCH GP PODIUM PREDICTION")
print("=" * 70)

print(
    ranking[
        [
            "PredictedRank",
            "Abbreviation",
            "FullName",
            "TeamName",
            "GridPosition",
            "gaptopole_bestquali",
            "TeammateQualifyingGap",
            "DriverChampionshipPosition",
            "PodiumProbability",
            "PredictedPodium"
        ]
    ].to_string(index=False)
)


# ==========================================================
# TOP 3
# ==========================================================

top3 = ranking.head(3)


print("\n")
print("=" * 70)
print("PREDICTED TOP 3")
print("=" * 70)

print(
    top3[
        [
            "PredictedRank",
            "Abbreviation",
            "FullName",
            "TeamName",
            "GridPosition",
            "PodiumProbability"
        ]
    ].to_string(index=False)
)


# ==========================================================
# SAVE
# ==========================================================

ranking.to_csv(
    OUTPUT,
    index=False
)


print("\n")
print("=" * 70)
print("PREDICTION COMPLETE")
print("=" * 70)

print("Saved:")
print(OUTPUT)