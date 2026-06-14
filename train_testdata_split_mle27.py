import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.read_csv("f1_2020_2025_cleaned.csv")

# Features
X = df[
    [
        "GridPosition",
        "gaptopole_bestquali",
        "PitLaneStart",
        "HasQualiTime"
    ]
]

# Target
y = df["Podium"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("X_train shape:")
print(X_train.shape)

print()

print("X_test shape:")
print(X_test.shape)

print()

print("y_train distribution:")
print(y_train.value_counts())

print()

print("y_test distribution:")
print(y_test.value_counts())