import fastf1
# Enables local caching so data downloads faster after the first run
fastf1.Cache.enable_cache("cache")
session = fastf1.get_session(2023,"Monaco","R")
#download the sesion data from the below line
session.load()
print(session.results[["Abbreviation","TeamName","GridPosition","Position","Points"]].head(10))
