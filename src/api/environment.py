"""Environmental context served to the dashboard (Person 4's data).

Area-averaged monthly weather, yearly climate summaries and fixed state geography. Yearly
temperature and rainfall feed the forecast model; the other variables were tested and are not used
(see docs/feature_ablation_results.txt).
"""
from functools import lru_cache

import pandas as pd

from src.api.config import REPO_ROOT

PROCESSED = REPO_ROOT / "data" / "processed"
STATE_IDS = [s.strip() for s in open(REPO_ROOT / "configs" / "states.txt", encoding="utf-8") if s.strip()]

MONTHLY_COLS = ["temp_mean_c", "rainfall_mm", "rh_mean_pct", "wind_ms", "n_points"]
ANNUAL_COLS = ["mean_temp_c", "total_rain_mm", "monsoon_rain_mm", "hottest_month_temp_c", "mean_rh_pct", "mean_wind_ms"]
STATIC_COLS = ["area_km2", "population", "population_year", "population_density_per_km2",
               "urban_fraction", "mean_elevation_m"]


def _records(df):
    return df.astype(object).where(df.notna(), None).to_dict("records")


@lru_cache(maxsize=1)
def _monthly():
    df = pd.read_csv(PROCESSED / "weather_area_monthly.csv", parse_dates=["start_date"])
    df["year"] = df.start_date.dt.year
    df["month"] = df.start_date.dt.month
    df[["temp_mean_c", "rainfall_mm", "rh_mean_pct", "wind_ms"]] = df[["temp_mean_c", "rainfall_mm", "rh_mean_pct", "wind_ms"]].round(2)
    return df[["state_id", "year", "month"] + MONTHLY_COLS]


@lru_cache(maxsize=1)
def _annual():
    df = pd.read_csv(PROCESSED / "annual_environment.csv")
    return df[["state_id", "year"] + ANNUAL_COLS].round(2)


@lru_cache(maxsize=1)
def _static():
    df = pd.read_csv(PROCESSED / "static_state_features.csv").set_index("state_id").reindex(STATE_IDS).reset_index()
    return df[["state_id"] + STATIC_COLS]


def monthly(state_id=None, year=None, month=None):
    df = _monthly()
    if state_id is not None:
        df = df[df.state_id == state_id]
    if year is not None:
        df = df[df.year == year]
    if month is not None:
        df = df[df.month == month]
    return _records(df)


def annual(state_id=None):
    df = _annual()
    return _records(df if state_id is None else df[df.state_id == state_id])


def profile(state_id=None):
    df = _static()
    return _records(df if state_id is None else df[df.state_id == state_id])