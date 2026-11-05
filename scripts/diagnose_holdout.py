import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "models/stgnn")
from annual_data import load_annual

DIS = ["dengue", "malaria", "chikungunya"]
states, years, Y, _ = load_annual("data/processed/pan_india_state_enriched_tensor_real.csv")
yl = [int(y) for y in years]
i22 = yl.index(2022)

fc = pd.read_csv("data/processed/stgnn_forecast_final_v1.csv")
fc = fc[fc.year.isin(yl)].copy()
fc["true"] = [Y[yl.index(int(r.year)), states.index(r.state_id), DIS.index(r.disease)] for r in fc.itertuples()]
fc["pers_new"] = [Y[i22, states.index(r.state_id), DIS.index(r.disease)] for r in fc.itertuples()]
m = fc[fc.true.notna() & fc.persistence.notna() & fc.pers_new.notna()].copy()
m["lm"] = np.log1p(m.forecast) - np.log1p(m.true)            # signed: > 0 means the model forecast too high
m["lp"] = np.log1p(m.persistence) - np.log1p(m.true)         # old 2022 value
m["lpn"] = np.log1p(m.pers_new) - np.log1p(m.true)           # new 2022 value
print("1) holdout by disease (errors are mean absolute log error; bias is signed)")
g = m.groupby(["horizon", "disease"]).agg(
    n=("lm", "size"), model=("lm", lambda x: x.abs().mean()), pers_old2022=("lp", lambda x: x.abs().mean()),
    pers_new2022=("lpn", lambda x: x.abs().mean()), bias_model=("lm", "mean"), bias_pers_new=("lpn", "mean"))
print(g.round(3).to_string())
print("\n   how much did the 2022 values change between the old and the new data?")
d22 = np.abs(np.log1p(m.pers_new) - np.log1p(m.persistence))
print(f"   mean abs log difference {d22.mean():.3f} | share of pairs with identical 2022 value {(d22 < 1e-9).mean():.0%}")

print("\n2) typical size of the year-on-year change (mean abs log change across states); a seam shows as a jump")
rows = {}
for k, d in enumerate(DIS):
    r = {}
    for i in range(1, len(yl)):
        a, b = Y[i - 1, :, k], Y[i, :, k]
        ok = ~np.isnan(a) & ~np.isnan(b) & (a > 0) & (b > 0)
        if ok.sum() >= 8:
            r[yl[i]] = np.mean(np.abs(np.log(b[ok] / a[ok])))
    rows[d] = r
print(pd.DataFrame(rows).loc[2015:].round(2).to_string())

print("\n3) are the old forecasts for 2023 the same as the new model's 2023 predictions?")
p = pd.read_csv("data/processed/stgnn_annual_geo_w_h1_pairs.csv")
p = p[p.year == 2023][["state_id", "disease", "model", "persistence", "true"]]
v = fc[fc.horizon == 1][["state_id", "disease", "forecast", "persistence"]]
j = p.merge(v, on=["state_id", "disease"], suffixes=("_new", "_v1"))
print(f"   {len(j)} pairs | same forecast (within 1 case): {((j.model - j.forecast).abs() < 1.5).mean():.0%}"
      f" | same persistence: {((j.persistence_new - j.persistence_v1).abs() < 0.5).mean():.0%}")

print("\n4) new rolling results for the last folds")
f = pd.read_csv("data/processed/stgnn_annual_geo_w_h1.csv")
print(f[f.year >= 2021][["year", "n", "log_model", "log_last"]].round(3).to_string(index=False))
