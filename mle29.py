# import pandas as pd

# df = pd.read_csv("f1_2020_2025_cleaned.csv")

# print(df.head())

import pandas as pd

df = pd.read_csv("f1_2020_2025_cleaned.csv")

print(
    df[
        ["Year", "RaceName"]
    ]
    .drop_duplicates()
    .head(30)
)