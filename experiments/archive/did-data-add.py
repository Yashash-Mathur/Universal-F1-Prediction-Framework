import pandas as pd

df = pd.read_csv("f1_2020_2026_raw_v2.csv")

print(df.shape)

print(df[df["RaceName"] == "Belgian Grand Prix"].shape)

print(df[df["RaceName"] == "British Grand Prix"].shape)

print(df[df["RaceName"] == "Austrian Grand Prix"].shape)

print(df[df["RaceName"] == "Barcelona Grand Prix"].shape)