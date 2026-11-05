import argparse, calendar, json, os, time
from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely

ap = argparse.ArgumentParser()
ap.add_argument("--spacing", type=float, default=0.75)
ap.add_argument("--maxp", type=int, default=30)
ap.add_argument("--workers", type=int, default=3)
a = ap.parse_args()

URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
VARS = ("T2M", "PRECTOTCORR", "RH2M", "WS2M")
RAW = "data/raw/nasa_power_area"
os.makedirs(RAW, exist_ok=True)
states = [s.strip() for s in open("configs/states.txt") if s.strip()]
geo = gpd.read_file("data/processed/india_states.geojson").set_index("state_id")


def sample_points(poly):
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(np.floor(minx / a.spacing) * a.spacing, maxx + a.spacing, a.spacing)
    ys = np.arange(np.floor(miny / a.spacing) * a.spacing, maxy + a.spacing, a.spacing)
    X, Y = np.meshgrid(xs, ys)
    X, Y = X.ravel(), Y.ravel()
    inside = shapely.contains_xy(poly, X, Y)
    pts = list(zip(Y[inside], X[inside]))
    if not pts:
        rp = poly.representative_point()
        pts = [(rp.y, rp.x)]
    if len(pts) > a.maxp:
        idx = np.linspace(0, len(pts) - 1, a.maxp).round().astype(int)
        pts = [pts[i] for i in idx]
    return [(round(float(la), 3), round(float(lo), 3)) for la, lo in pts]


jobs = [(s, la, lo) for s in states for la, lo in sample_points(geo.loc[s].geometry)]
print(f"{len(jobs)} points for {len(states)} states (about {len(jobs) * 3 / a.workers / 60:.0f} min if each takes 3 s)")


def fetch(job):
    s, lat, lon = job
    f = f"{RAW}/{s}__{lat:.3f}_{lon:.3f}.json"
    if os.path.exists(f):
        return job, json.load(open(f))
    q = dict(parameters=",".join(VARS), community="AG", latitude=lat, longitude=lon,
             start=1999, end=2022, format="JSON")
    err = None
    for attempt in range(5):
        try:
            r = requests.get(URL, params=q, timeout=120)
            r.raise_for_status()
            js = r.json()
            json.dump(js, open(f, "w"))
            return job, js
        except Exception as e:
            err = getattr(getattr(e, "response", None), "text", None) or e
            time.sleep(3 * (attempt + 1))
    print("FAILED", job, str(err)[:300])
    return job, None


j0, js0 = fetch(jobs[0])
if js0 is None:
    raise SystemExit("First request failed: check your network and the parameter names printed above.")
with ThreadPoolExecutor(max_workers=a.workers) as ex:
    results = list(ex.map(fetch, jobs))

rows, elev, failed = [], [], 0
for (s, lat, lon), js in results:
    if js is None:
        failed += 1
        continue
    par = js["properties"]["parameter"]
    co = js.get("geometry", {}).get("coordinates", [])
    if len(co) >= 3 and co[2] is not None:
        elev.append((s, float(co[2])))
    for yr in range(1999, 2023):
        for mo in range(1, 13):
            k = f"{yr}{mo:02d}"
            v = {}
            for p in VARS:
                x = par[p].get(k, np.nan)
                v[p] = np.nan if x is None or x <= -990 else x
            v["PRECTOTCORR"] = v["PRECTOTCORR"] * calendar.monthrange(yr, mo)[1]   # mm/day -> mm/month
            rows.append((s, pd.Timestamp(yr, mo, 1), v["T2M"], v["PRECTOTCORR"], v["RH2M"], v["WS2M"]))
print("failed downloads:", failed, "(rerun the script to retry; finished downloads are cached)")

df = pd.DataFrame(rows, columns=["state_id", "start_date", "temp_mean_c", "rainfall_mm", "rh_mean_pct", "wind_ms"])
w = df.groupby(["state_id", "start_date"], as_index=False).mean()
npts = pd.Series([j[0] for j in jobs]).value_counts().rename("n_points")
w = w.merge(npts, left_on="state_id", right_index=True)
print("rows:", len(w), "(10368 expected) | states:", w.state_id.nunique())
print("NaN counts:", w[["temp_mean_c", "rainfall_mm", "rh_mean_pct", "wind_ms"]].isna().sum().to_dict())
w.to_csv("data/processed/weather_area_monthly.csv", index=False)

if elev:
    e = (pd.DataFrame(elev, columns=["state_id", "elev_m"]).groupby("state_id").elev_m.mean().round(1)
         .rename("mean_elevation_m").reset_index())
    e.to_csv("data/processed/state_elevation_m.csv", index=False)
    print("elevation saved for", len(e), "states")
else:
    print("NOTE: no elevation in the NASA responses; elevation needs another source")

# ---- copy of the model tensor with the new weather and recomputed lags ----
t = pd.read_csv("data/processed/pan_india_state_enriched_tensor_real.csv", parse_dates=["start_date"])
old = ["temp_mean_c", "rainfall_mm", "temp_lag_1", "temp_lag_2", "rainfall_lag_1", "rainfall_lag_2", "rainfall_rolling_3m"]
t = t.drop(columns=[c for c in old if c in t.columns])
full = w.drop(columns=["n_points"]).sort_values(["state_id", "start_date"]).copy()
gb = full.groupby("state_id")
full["temp_lag_1"], full["temp_lag_2"] = gb.temp_mean_c.shift(1), gb.temp_mean_c.shift(2)
full["rainfall_lag_1"], full["rainfall_lag_2"] = gb.rainfall_mm.shift(1), gb.rainfall_mm.shift(2)
full["rainfall_rolling_3m"] = gb.rainfall_mm.transform(lambda x: x.rolling(3, min_periods=1).sum())
out = t.merge(full, on=["state_id", "start_date"], how="left")
out.to_csv("data/processed/pan_india_state_enriched_tensor_area.csv", index=False)
print("saved data/processed/pan_india_state_enriched_tensor_area.csv", out.shape)