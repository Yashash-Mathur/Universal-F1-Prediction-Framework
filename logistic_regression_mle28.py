import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

model = LogisticRegression(
    max_iter=1000
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

#accuracy_score check from sklearn 
from sklearn.metrics import accuracy_score

# ==========================
# PREDICTIONS
# ==========================

y_pred = model.predict(X_test)

# ==========================
# ACCURACY
# ==========================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print()

print("Accuracy:")
print(accuracy)

#CONFUSION MATRIX 
#TRUE NEGATIVES, FALSE POSITIVES, TRUE POSITIVES, FALSE NEGATIVES

from sklearn.metrics import confusion_matrix

print()

print("Confusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

#GETTING A CLASSIFICATION REPORT
#THIS WILL GET US THE PRECISION, RECALL AND F1 SCORE. 

from sklearn.metrics import classification_report

print()

print(
    classification_report(
        y_test,
        y_pred
    )
)


#Logistic Regression coefficients 
#Logistic Regression coefficients.
print("Features:")
print(X.columns)

print()

print("Coefficients:")
print(model.coef_)

print()

print("Intercept:")
print(model.intercept_)