import fastf1
import pandas as pd

df = pd.read_csv("f1_2020_2026_raw_v2.csv")

schedule = fastf1.get_event_schedule(2026)

print("\n==============================")
print("FASTF1 2026 SCHEDULE")
print("==============================")

for _, event in schedule.iterrows():

    if event["RoundNumber"] == 0:
        continue

    print(
        event["RoundNumber"],
        "|",
        event["EventName"],
        "|",
        pd.to_datetime(event["EventDate"]).date()
    )

print("\n==============================")
print("RACES ALREADY IN DATASET")
print("==============================")

existing = (
    df[df["Year"] == 2026]
    .groupby(["RoundNumber", "RaceName"])
    .size()
    .reset_index(name="Rows")
)

print(existing.to_string(index=False))