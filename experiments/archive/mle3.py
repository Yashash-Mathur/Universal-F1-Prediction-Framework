import fastf1
fastf1.Cache.enable_cache("cache")
session = fastf1.get_session(2023,"Monaco","Q")
session.load()
bestqualitime = session.results[["Q3","Q2","Q1"]].min(axis = 1)
pole_time = session.results.iloc[0]["Q3"]
#print("the pole position timing is",pole_time)
#print(type(pole_time))
gaptopole = session.results["Q3"] - pole_time
#print(session.results[["Abbreviation","Q3","gaptopole"]]) 
race = fastf1.get_session(2023, "Monaco", "R")
race.load()
print(race.results.columns)
session.results["bestqualitime"] = bestqualitime 
gaptopole_bestquali = bestqualitime - pole_time 
session.results["gaptopole_bestquali"] = gaptopole_bestquali  
#print(session.results[["Abbreviation","Q1", "Q2","Q3","bestqualitime","gaptopole_bestquali"]])
