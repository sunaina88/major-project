"""
apply_area_weather.py: puts REAL NASA POWER weather into the model tensor.

Replaces build_weather_tensor.py (synthetic climatology, kept in archive/ for reference).
Merges the monthly, area-averaged temperature and rainfall in data/processed/weather_area_monthly.csv
(made by download_area_weather.py: NASA POWER sample points inside each state boundary, averaged)
into the disease-only tensor, adds the weather lag features, and writes the enriched tensor.
build_lag_features.py is no longer needed (it would overwrite these lags).

    python src/ingestion/weather/apply_area_weather.py [--out PATH]
"""
import argparse

import pandas as pd

DISEASE = "data/processed/pan_india_state_spatiotemporal_tensor.csv"
WEATHER = "data/processed/weather_area_monthly.csv"
DEFAULT_OUT = "data/processed/pan_india_state_enriched_tensor.csv"

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=DEFAULT_OUT)
a = ap.parse_args()

d = pd.read_csv(DISEASE, parse_dates=["start_date"])
w = pd.read_csv(WEATHER, parse_dates=["start_date"])[["state_id", "start_date", "temp_mean_c", "rainfall_mm"]]
w = w.sort_values(["state_id", "start_date"]).copy()
g = w.groupby("state_id")   # lags come from the full monthly weather series, so they are correct in every state's first months
w["temp_lag_1"], w["temp_lag_2"] = g.temp_mean_c.shift(1), g.temp_mean_c.shift(2)
w["rainfall_lag_1"], w["rainfall_lag_2"] = g.rainfall_mm.shift(1), g.rainfall_mm.shift(2)
w["rainfall_rolling_3m"] = g.rainfall_mm.transform(lambda x: x.rolling(3, min_periods=1).sum())
out = d.merge(w, on=["state_id", "start_date"], how="left")
missing = int(out[["temp_mean_c", "rainfall_mm"]].isna().sum().sum())   # lags are legitimately empty only before 1999
if missing or len(out) != len(d):
    raise SystemExit(f"Weather merge incomplete: {missing} missing values, {len(out)} rows vs {len(d)}.")
out["start_date"] = out["start_date"].dt.strftime("%Y-%m-%d")
out.to_csv(a.out, index=False)
print(f"wrote {a.out} {out.shape}: real NASA POWER area-averaged temperature and rainfall, no missing values")