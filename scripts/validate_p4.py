import json, os
import pandas as pd

states = [s.strip() for s in open("configs/states.txt") if s.strip()]
ok = True


def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg)
    ok = ok and bool(cond)


def have(p):
    if not os.path.exists(p):
        print("SKIP", p, "not found")
    return os.path.exists(p)


if have("data/processed/india_states.geojson"):
    g = json.load(open("data/processed/india_states.geojson"))
    ids = [f["properties"]["state_id"] for f in g["features"]]
    check(len(ids) == 36 and set(ids) == set(states), "GeoJSON has exactly the 36 state_ids")
    check(os.path.getsize("data/processed/india_states.geojson") < 10e6, "GeoJSON under 10 MB")
if have("data/processed/weather_area_monthly.csv"):
    w = pd.read_csv("data/processed/weather_area_monthly.csv", parse_dates=["start_date"])
    ny = w.start_date.dt.year.nunique()
    check(len(w) == 36 * 12 * ny and set(w.state_id) == set(states), f"weather: 36 states x {12 * ny} months")
    check(not w[["temp_mean_c", "rainfall_mm", "rh_mean_pct", "wind_ms"]].isna().any().any(), "weather: no NaN")
    check((w.rainfall_mm >= 0).all() and w.temp_mean_c.between(-50, 60).all(), "weather: plausible ranges")
    check(w.rh_mean_pct.between(0, 100).all() and (w.wind_ms >= 0).all(), "humidity 0 to 100, wind not negative")
if have("data/processed/static_state_features.csv"):
    s = pd.read_csv("data/processed/static_state_features.csv")
    check(set(s.state_id) == set(states) and len(s) == 36, "static: 36 states")
    check(not s[["area_km2", "population", "population_density_per_km2", "urban_fraction"]].isna().any().any(),
          "static: no NaN in area, population, density, urban fraction")
if have("data/processed/annual_environment.csv"):
    e = pd.read_csv("data/processed/annual_environment.csv")
    check(len(e) == 36 * e.year.nunique(), "annual environment: 36 states x all years")
print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")