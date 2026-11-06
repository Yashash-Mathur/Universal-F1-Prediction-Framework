import pandas as pd

df = pd.read_csv("backup_dataset.csv")

print("Rows:", len(df))
print(df.groupby("Year")["RaceName"].nunique())

# import pandas as pd

# backup = pd.read_csv("backup_dataset.csv")

# print("Rows:", len(backup))

# print("\nRaces collected per year:")
# print(
#     backup.groupby("Year")["RaceName"]
#     .nunique()
# )

# print("\nLast races collected:")
# print(
#     backup[
#         ["Year", "RaceName"]
#     ]
#     .drop_duplicates()
#     .tail(20)
# )