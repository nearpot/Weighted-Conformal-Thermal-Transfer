import fastf1
import pandas as pd
import os

cache_dir = "./f1_cache"
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# You can change year/race/driver — Monza 2023 kept for continuity with the
# original paper, but consider ALSO pulling a second circuit (e.g. Silverstone,
# a very different track profile: fewer straight-line-only bursts, more
# sustained high-speed corners) to test generalization across circuits.
RACES = [
    (2023, "Monza", "R", "VER"),
    (2023, "Silverstone", "R", "VER"),  # second circuit for generalization check
]

all_frames = []
for year, gp, session_type, driver in RACES:
    print(f"Loading {year} {gp} {session_type} — driver {driver} ...")
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, weather=False, messages=False)
    laps = session.laps.pick_driver(driver)
    tel = laps.get_telemetry()
    tel = tel[["Time", "Speed", "Throttle", "Brake", "DRS", "nGear", "RPM", "Distance"]].copy()
    tel["year"] = year
    tel["gp"] = gp
    tel["driver"] = driver
    all_frames.append(tel)
    print(f"  -> {len(tel)} telemetry frames")

out = pd.concat(all_frames, ignore_index=True)
out.to_csv("f1_telemetry_clean.csv", index=False)
print(f"\nWrote {len(out)} total rows to f1_telemetry_clean.csv")
print(out.groupby(["year", "gp"]).size())