import pandas as pd

df = pd.read_csv("f1_2020_2025_dataset.csv")

df = df.dropna(subset=["Position"])

print(df.shape)

print(df.isnull().sum())


# import pandas as pd

# df = pd.read_csv("f1_2020_2025_dataset.csv")

# print("Shape:")
# print(df.shape)

# print()

# print("Negative gaps:")
# print(
#     (df["gaptopole_bestquali"] < 0).sum()
# )

# print()

# print("Missing values:")
# print(
#     df.isnull().sum()
# )