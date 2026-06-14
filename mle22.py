import pandas as pd

df = pd.read_csv("backup_dataset.csv")

print(
    df.loc[
        (df["Year"] == 2023)
        &
        (df["RaceName"] == "Canadian Grand Prix")
    ]
    [["Abbreviation","gaptopole_bestquali"]]
    .sort_values("gaptopole_bestquali")
)


# import fastf1
# import pandas as pd

# fastf1.Cache.enable_cache("cache")

# quali = fastf1.get_session(
#     2023,
#     "Canadian Grand Prix",
#     "Q"
# )

# quali.load()

# results = quali.results.copy()

# # Driver best lap
# results["bestqualitime"] = (
#     results[["Q1", "Q2", "Q3"]]
#     .min(axis=1)
# )

# # Overall fastest lap in session
# overall_fastest = (
#     results["bestqualitime"]
#     .min()
# )

# print("Overall fastest lap:")
# print(overall_fastest)

# results["new_gap"] = (
#     results["bestqualitime"]
#     - overall_fastest
# ).dt.total_seconds()

# print(
#     results[
#         [
#             "Abbreviation",
#             "Q1",
#             "Q2",
#             "Q3",
#             "bestqualitime",
#             "new_gap"
#         ]
#     ]
# )

# print()

# print(
#     "Negative gaps:",
#     (results["new_gap"] < 0).sum()
# )

