from fastf1.ergast import Ergast

ergast = Ergast()
res = ergast.get_race_results(season=2024, round=8)  # Monaco GP
df = res.content[0]

print("Available columns:")
print(df.columns.tolist())

print("\nRace Results & Statuses:")
print(df[["driverCode", "grid", "position", "status"]])

