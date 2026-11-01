import time
import pandas as pd
from fastf1.ergast import Ergast

ergast = Ergast()
all_results = []

# Fetching the modern era seasons (2021-2024)
seasons = [2021, 2022, 2023, 2024]

print("Starting data collection...")

for season in seasons:
    print(f"--> Fetching Season {season}...")
    # Get the total number of rounds in this season
    schedule = ergast.get_race_schedule(season)
    rounds = len(schedule)
    
    for rnd in range(1, rounds + 1):
        try:
            res = ergast.get_race_results(season=season, round=rnd)
            if res.content:
                df = res.content[0]
                # Tag metadata columns
                df["season"] = season
                df["round"] = rnd
                all_results.append(df)
                print(f"    Round {rnd}/{rounds} fetched")
        except Exception as e:
            print(f"    Failed Round {rnd}: {e}")
        
        # Respect Jolpica API rate limits
        time.sleep(0.5)

# Concatenate everything into one DataFrame
dataset = pd.concat(all_results, ignore_index=True)

# Keep the relevant columns for predicting DNF
columns_to_keep = [
    "season", "round", "raceName", "circuitId", 
    "driverCode", "constructorName", "grid", "position", "status"
]
available_cols = [c for c in columns_to_keep if c in dataset.columns]
dataset = dataset[available_cols]

# Save locally to avoid re-downloading
output_file = "f1_races_dataset.csv"
dataset.to_csv(output_file, index=False)
print(f"\nDone! Saved {len(dataset)} driver race records to '{output_file}'.")

