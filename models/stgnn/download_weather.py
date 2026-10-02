import os, json, time, calendar, requests, numpy as np, pandas as pd

states = open("configs/states.txt").read().split("\n")
PTS = {  # approximate centroids (lat, lon)
 "andaman_and_nicobar_islands": (11.74, 92.66), "andhra_pradesh": (15.91, 79.74),
 "arunachal_pradesh": (28.22, 94.73), "assam": (26.20, 92.94), "bihar": (25.10, 85.31),
 "chandigarh": (30.73, 76.78), "chhattisgarh": (21.28, 81.87),
 "dadra_and_nagar_haveli_and_daman_and_diu": (20.27, 73.02), "delhi": (28.66, 77.23),
 "goa": (15.30, 74.12), "gujarat": (22.26, 71.19), "haryana": (29.06, 76.09),
 "himachal_pradesh": (31.10, 77.17), "jammu_and_kashmir": (33.78, 76.58),
 "jharkhand": (23.61, 85.28), "karnataka": (15.32, 75.71), "kerala": (10.85, 76.27),
 "ladakh": (34.15, 77.58), "lakshadweep": (10.57, 72.64), "madhya_pradesh": (22.97, 78.66),
 "maharashtra": (19.75, 75.71), "manipur": (24.66, 93.91), "meghalaya": (25.47, 91.37),
 "mizoram": (23.16, 92.94), "nagaland": (26.16, 94.56), "odisha": (20.95, 85.10),
 "puducherry": (11.94, 79.81), "punjab": (31.15, 75.34), "rajasthan": (27.02, 74.22),
 "sikkim": (27.53, 88.51), "tamil_nadu": (11.13, 78.66), "telangana": (18.11, 79.02),
 "tripura": (23.94, 91.99), "uttar_pradesh": (26.85, 80.91), "uttarakhand": (30.07, 79.02),
 "west_bengal": (22.99, 87.85),
}
assert set(PTS) == set(states)

os.makedirs("data/raw/nasa_power", exist_ok=True)
URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"

def fetch(s):
    f = f"data/raw/nasa_power/{s}.json"
    if os.path.exists(f):
        return json.load(open(f))
    lat, lon = PTS[s]
    q = dict(parameters="T2M,PRECTOTCORR", community="AG", latitude=lat, longitude=lon,
             start=1999, end=2022, format="JSON")
    for attempt in range(4):
        try:
            r = requests.get(URL, params=q, timeout=120)
            r.raise_for_status()
            js = r.json()
            json.dump(js, open(f, "w"))
            return js
        except Exception as e:
            print(f"  retry {attempt+1} for {s}: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed: {s}")

rows = []
for s in states:
    print("downloading", s, flush=True)
    par = fetch(s)["properties"]["parameter"]
    t2m, pr = par["T2M"], par["PRECTOTCORR"]
    for yr in range(1999, 2023):
        for mo in range(1, 13):
            k = f"{yr}{mo:02d}"
            t = t2m.get(k, np.nan); p = pr.get(k, np.nan)
            t = np.nan if t is None or t <= -990 else t
            p = np.nan if p is None or p <= -990 else p * calendar.monthrange(yr, mo)[1]  # mm/day -> mm/month
            rows.append((s, pd.Timestamp(yr, mo, 1), t, p))
    time.sleep(1)

w = pd.DataFrame(rows, columns=["state_id", "start_date", "temp_mean_c", "rainfall_mm"])
print("NaN in downloaded weather:", int(w.temp_mean_c.isna().sum()), int(w.rainfall_mm.isna().sum()))
w.to_csv("data/processed/weather_real_monthly.csv", index=False)

# ---- build new enriched tensor with real weather + recomputed lags ----
df = pd.read_csv("data/processed/pan_india_state_enriched_tensor.csv", parse_dates=["start_date"])
df = df.drop(columns=["temp_mean_c", "rainfall_mm", "temp_lag_1", "temp_lag_2",
                      "rainfall_lag_1", "rainfall_lag_2", "rainfall_rolling_3m"])
full = w.sort_values(["state_id", "start_date"]).copy()          # full 288-month grid per state
g = full.groupby("state_id")
full["temp_lag_1"] = g.temp_mean_c.shift(1);  full["temp_lag_2"] = g.temp_mean_c.shift(2)
full["rainfall_lag_1"] = g.rainfall_mm.shift(1); full["rainfall_lag_2"] = g.rainfall_mm.shift(2)
full["rainfall_rolling_3m"] = g.rainfall_mm.transform(lambda x: x.rolling(3, min_periods=1).sum())
out = df.merge(full, on=["state_id", "start_date"], how="left")
print("rows:", len(out), "| NaN temp/rain:", int(out.temp_mean_c.isna().sum()), int(out.rainfall_mm.isna().sum()))
out.to_csv("data/processed/pan_india_state_enriched_tensor_real.csv", index=False)
print("saved data/processed/pan_india_state_enriched_tensor_real.csv")