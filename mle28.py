import pandas as pd

df = pd.read_csv("f1_2020_2025_cleaned.csv")

print("Before:")
print(df.shape)

df = df.dropna(
    subset=["gaptopole_bestquali"]
)

print()

print("After:")
print(df.shape)

print()

print("Rows removed:")
print(2615 - len(df))
