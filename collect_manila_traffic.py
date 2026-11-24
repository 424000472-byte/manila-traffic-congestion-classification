import csv
import os
import time
from datetime import datetime
import requests
from config import API_KEY, MANILA_ROADS, estimate_vehicle_volume

OUTPUT_FILE = "manila_traffic_raw.csv"
INTERVAL_SECONDS = 120  # Poll every 2 minutes (300 rows/hour)

if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "Timestamp",
            "Road_Name",
            "Vehicle_Volume",
            "Travel_Time_Sec",
            "FreeFlow_Time_Sec",
            "Current_Speed_Kph",
            "Congestion",  # 1 = Yes, 0 = No
        ])


def count_rows():
    with open(OUTPUT_FILE, mode="r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


print("Collecting live Manila traffic data... Press Ctrl+C to stop.")

while True:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    batch_added = 0

    for road_name, lat, lon, lanes, cap_per_lane in MANILA_ROADS:
        url = (
            f"https://api.tomtom.com/traffic/services/4/flowSegmentData/"
            f"absolute/10/json?point={lat},{lon}&key={API_KEY}"
        )
        try:
            resp = requests.get(url, timeout=10).json()
            data = resp.get("flowSegmentData", {})

            curr_tt = data.get("currentTravelTime")
            free_tt = data.get("freeFlowTravelTime")
            curr_spd = data.get("currentSpeed")
            free_spd = data.get("freeFlowSpeed")

            if curr_tt and free_tt and curr_spd and free_spd:
                volume = estimate_vehicle_volume(curr_spd, free_spd, lanes, cap_per_lane)
                # Label: 1 (Yes) if travel time is >= 35% longer than free-flow, else 0 (No)
                congestion = 1 if (curr_tt / free_tt) >= 1.35 else 0

                with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        now, road_name, volume, curr_tt, free_tt, curr_spd, congestion
                    ])
                batch_added += 1
        except Exception as e:
            print(f"[{now}] Error on {road_name}: {e}")

    total_rows = count_rows()
    print(f"[{now}] +{batch_added} rows | Total Dataset Rows: {total_rows} (Target: 2,500–10,000)")
    time.sleep(INTERVAL_SECONDS)