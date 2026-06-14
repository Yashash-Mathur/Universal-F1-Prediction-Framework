import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("f1_2020_2025_cleaned.csv")

# Remove rows without GapToPole

df = df.dropna(
    subset=["gaptopole_bestquali"]
)

# ==========================
# FEATURES
# ==========================

X = df[
    [
        "GridPosition",
        "gaptopole_bestquali",
        "PitLaneStart",
        "HasQualiTime"
    ]
]

# ==========================
# TARGET
# ==========================

y = df["Podium"]

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# MODEL
# ==========================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("Model trained successfully!")

print()

print("Training rows:")
print(len(X_train))

print()

print("Testing rows:")
print(len(X_test))

# ==========================
# PREDICTIONS
# ==========================

y_pred = model.predict(X_test)

# ==========================
# ACCURACY
# ==========================

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    y_pred
)

print()

print("Accuracy:")
print(accuracy)

# ==========================
# CONFUSION MATRIX
# ==========================

from sklearn.metrics import confusion_matrix

print()

print("Confusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

# ==========================
# CLASSIFICATION REPORT
# ==========================

from sklearn.metrics import classification_report

print()

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ==========================
# TRAIN / TEST ACCURACY
# ==========================

train_accuracy = model.score(
    X_train,
    y_train
)

test_accuracy = model.score(
    X_test,
    y_test
)

print()

print("Train Accuracy:")
print(train_accuracy)

print()

print("Test Accuracy:")
print(test_accuracy)

# ==========================
# FEATURE IMPORTANCE
# ==========================

print()

print("Feature Importances:")

for feature, importance in zip(
    X.columns,
    model.feature_importances_
):
    print(
        feature,
        ":",
        round(importance, 4)
    )

print("NOW PRINTING THE COLUMNS WE HAVE IN THE DATASET JUST TO KNOW WHAT EXISTS AND WHAT DOES NOT")
print(df.columns)