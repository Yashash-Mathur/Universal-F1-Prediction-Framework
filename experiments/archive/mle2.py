import fastf1
fastf1.Cache.enable_cache("cache")
session = fastf1.get_session(2023,"Monaco","Q")
session.load()
pole_time = session.results.iloc[0]["Q3"]
#print("the pole position timing is",pole_time)
#print(type(pole_time))
gaptopole = session.results["Q3"] - pole_time
print(session.results[["Abbreviation","Q3","gaptopole"]])