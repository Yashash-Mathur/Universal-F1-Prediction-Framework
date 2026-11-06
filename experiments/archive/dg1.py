import pandas as pd

df = pd.read_csv("f1_2020_2026_raw_v2.csv")

print("Dataset shape:", df.shape)

print("\n2026 races:")
print(
    df[df["Year"] == 2026]
    .groupby(["RoundNumber", "RaceName"])
    .size()
    .reset_index(name="Rows")
    .to_string(index=False)
)

print("\n2026 total rows:")
print(df[df["Year"] == 2026].shape[0])

print("\n2026 race count:")
print(df[df["Year"] == 2026]["RaceName"].nunique())