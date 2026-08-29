"""
Central feature contract for the Universal F1 Prediction Framework.

This file defines the features used by the final CatBoost podium model.
It contains NO feature calculations and NO data loading.

The purpose is to give every pipeline component one authoritative
definition of the model schema.
"""


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = "Podium"


# ============================================================
# IDENTIFIERS / METADATA
# ============================================================

IDENTIFIER_COLUMNS = [
    "Year",
    "RoundNumber",
    "RaceName",
    "Abbreviation",
    "FullName",
    "TeamName",
]


# ============================================================
# FINAL MODEL FEATURES
# ============================================================

MODEL_FEATURES = [

    # Qualifying / starting position
    "GridPosition",
    "gaptopole_bestquali",
    "PitLaneStart",
    "HasQualiTime",
    "HasGapToPole",

    # Teammate comparison
    "TeammateQualifyingGap",
    "HasTeammateGap",

    # Championship context
    "ConstructorChampionshipPoints",
    "DriverChampionshipPoints",
    "ConstructorChampionshipPosition",
    "DriverChampionshipPosition",

    # Race context
    "RoundNumber",

    # Driver recent form
    "AverageFinishLast5",
    "AverageFinishLast3",
    "AverageGridLast3",

    # Constructor recent form
    "ConstructorAverageFinishLast3",

    # Circuit-type performance
    "StreetCircuitPerformance",
    "PermanentCircuitPerformance",
    "HighSpeedCircuitPerformance",
    "HighDownforceCircuitPerformance",

    # Circuit-history indicators
    "HasStreetCircuitHistory",
    "HasPermanentCircuitHistory",
    "HasHighSpeedCircuitHistory",
    "HasHighDownforceCircuitHistory",
]


# ============================================================
# FEATURE GROUPS
# ============================================================

QUALIFYING_FEATURES = [
    "GridPosition",
    "gaptopole_bestquali",
    "PitLaneStart",
    "HasQualiTime",
    "HasGapToPole",
]

TEAMMATE_FEATURES = [
    "TeammateQualifyingGap",
    "HasTeammateGap",
]

CHAMPIONSHIP_FEATURES = [
    "ConstructorChampionshipPoints",
    "DriverChampionshipPoints",
    "ConstructorChampionshipPosition",
    "DriverChampionshipPosition",
]

RECENT_FORM_FEATURES = [
    "AverageFinishLast5",
    "AverageFinishLast3",
    "AverageGridLast3",
    "ConstructorAverageFinishLast3",
]

CIRCUIT_PERFORMANCE_FEATURES = [
    "StreetCircuitPerformance",
    "PermanentCircuitPerformance",
    "HighSpeedCircuitPerformance",
    "HighDownforceCircuitPerformance",
]

CIRCUIT_HISTORY_FEATURES = [
    "HasStreetCircuitHistory",
    "HasPermanentCircuitHistory",
    "HasHighSpeedCircuitHistory",
    "HasHighDownforceCircuitHistory",
]


# ============================================================
# VALIDATION
# ============================================================

ALL_FEATURE_GROUPS = (
    QUALIFYING_FEATURES
    + TEAMMATE_FEATURES
    + CHAMPIONSHIP_FEATURES
    + ["RoundNumber"]
    + RECENT_FORM_FEATURES
    + CIRCUIT_PERFORMANCE_FEATURES
    + CIRCUIT_HISTORY_FEATURES
)


if set(ALL_FEATURE_GROUPS) != set(MODEL_FEATURES):
    raise ValueError(
        "Feature group definitions do not match MODEL_FEATURES."
    )

if len(MODEL_FEATURES) != len(set(MODEL_FEATURES)):
    raise ValueError(
        "Duplicate feature detected in MODEL_FEATURES."
    )