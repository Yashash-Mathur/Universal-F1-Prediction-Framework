# import pandas as pd

# df = pd.read_csv("f1_2020_2025_cleaned.csv")

# print(df.shape)

# print()

# print(df["Position"].describe())

# print()

# print(df["GridPosition"].describe())

# print()

# print(df["gaptopole_bestquali"].describe())

# print()

# print(
#     df[
#         [
#             "GridPosition",
#             "gaptopole_bestquali",
#             "Position"
#         ]
#     ].corr()
# )

import pandas as pd

df = pd.read_csv("f1_2020_2025_cleaned.csv")

print(
    df.groupby("Podium")[
        ["GridPosition", "gaptopole_bestquali"]
    ].mean()
)