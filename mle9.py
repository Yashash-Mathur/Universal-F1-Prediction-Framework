import pandas as pd

backup = pd.read_csv("backup_dataset.csv")

print("Rows:", len(backup))

print("\nRa ces per year:")
print(
    backup.groupby("Year")["RaceName"]
    .nunique()
)

print("\n2022 races:")
for race in sorted(
    backup[
        backup["Year"] == 2022
    ]["RaceName"].unique()
):
    print(race)