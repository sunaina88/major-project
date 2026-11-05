import pandas as pd

w = pd.read_csv("data/processed/weather_area_monthly.csv", parse_dates=["start_date"])
w["year"], w["month"] = w.start_date.dt.year, w.start_date.dt.month
out = w.groupby(["state_id", "year"]).agg(
    mean_temp_c=("temp_mean_c", "mean"), total_rain_mm=("rainfall_mm", "sum"),
    hottest_month_temp_c=("temp_mean_c", "max"), mean_rh_pct=("rh_mean_pct", "mean"),
    mean_wind_ms=("wind_ms", "mean")).reset_index()
mon = (w[w.month.between(6, 9)].groupby(["state_id", "year"]).rainfall_mm.sum()
       .rename("monsoon_rain_mm").reset_index())
out = out.merge(mon, on=["state_id", "year"]).sort_values(["state_id", "year"])
cols = [c for c in out.columns if c not in ("state_id", "year")]
for c in cols:
    out[c + "_lag1"] = out.groupby("state_id")[c].shift(1)
out.round(3).to_csv("data/processed/annual_environment.csv", index=False)
print(out.shape, f"({36 * w.year.nunique()} rows expected)")