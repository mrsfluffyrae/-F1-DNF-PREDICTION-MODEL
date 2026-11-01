"""
STEP 1: Download F1 race results (including finishing status) from Jolpica-F1.

Why this version is better than a one-request-per-race loop:
  - It asks for a WHOLE SEASON at once (about 5 requests per season instead of ~24).
  - It waits between requests and retries politely if the server says "slow down".
  - It saves each finished season to disk, so if something fails you just re-run
    the script and it skips seasons it already has.
"""
import json
import os
import time

import pandas as pd
import requests

CURRENT_SEASON = 2026            # the season still in progress
SEASONS = range(2014, CURRENT_SEASON + 1)   # 2014 (start of the hybrid era) to now
BASE_URL = "https://api.jolpi.ca/ergast/f1"
PAGE_SIZE = 100                  # the API returns at most 100 rows per request
PAUSE_SECONDS = 1.5              # wait between requests so we don't get blocked
CACHE_DIR = "raw_cache"
OUTPUT_CSV = "f1_races_dataset.csv"

os.makedirs(CACHE_DIR, exist_ok=True)


def get_json(url, params, max_tries=8):
    """Download one page. If the server says 429 (Too Many Requests), wait and retry."""
    for attempt in range(1, max_tries + 1):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            wait = 10 * attempt   # wait 10s, then 20s, then 30s ...
            print(f"   Server said slow down. Waiting {wait}s (try {attempt}/{max_tries})...")
            time.sleep(wait)
            continue
        resp.raise_for_status()   # any other error: stop and show it
    raise RuntimeError(f"Gave up on {url} after {max_tries} tries")


def fetch_season(season):
    """Return a list of rows (one per driver per race) for one season."""
    cache_file = os.path.join(CACHE_DIR, f"{season}.json")
    # Finished seasons are saved to disk. The CURRENT season is always re-downloaded
    # because new races get added every two weeks.
    if season < CURRENT_SEASON and os.path.exists(cache_file):
        print(f"--> Season {season}: already downloaded, using saved copy")
        with open(cache_file) as f:
            return json.load(f)

    print(f"--> Season {season}: downloading...")
    rows = []
    offset = 0
    while True:
        data = get_json(f"{BASE_URL}/{season}/results.json",
                        {"limit": PAGE_SIZE, "offset": offset})
        total = int(data["MRData"]["total"])   # total result rows in the season

        for race in data["MRData"]["RaceTable"]["Races"]:
            for r in race["Results"]:
                rows.append({
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "race_name": race["raceName"],
                    "circuit_id": race["Circuit"]["circuitId"],
                    "date": race["date"],
                    "driver_id": r["Driver"]["driverId"],
                    "driver_code": r["Driver"].get("code"),
                    "constructor_id": r["Constructor"]["constructorId"],
                    "grid": int(r["grid"]),
                    "position": int(r["position"]),
                    "laps": int(r["laps"]),
                    "status": r["status"],
                })

        offset += PAGE_SIZE
        time.sleep(PAUSE_SECONDS)
        if offset >= total:
            break

    # Only save finished seasons, and only once the WHOLE season downloaded
    if season < CURRENT_SEASON:
        with open(cache_file, "w") as f:
            json.dump(rows, f)
    return rows


all_rows = []
for season in SEASONS:
    all_rows.extend(fetch_season(season))

df = pd.DataFrame(all_rows)
df = df.drop_duplicates(subset=["season", "round", "driver_id"])  # safety net
df = df.sort_values(["season", "round", "position"]).reset_index(drop=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nDone! Saved {len(df)} driver-race records to '{OUTPUT_CSV}'.")
print("\nRaces per season (sanity check, should be ~16-24 each):")
print(df.groupby("season")["round"].nunique().to_string())
print("\nMost common finishing statuses:")
print(df["status"].value_counts().head(15).to_string())
