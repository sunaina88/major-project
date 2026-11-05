import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "models/stgnn")
from annual_data import load_annual

DIS = ["dengue", "malaria", "chikungunya"]
fc = pd.read_csv("data/processed/stgnn_forecast_final_v1.csv")
states, years, Y, _ = load_annual("data/processed/pan_india_state_enriched_tensor_real.csv")
yl = [int(y) for y in years]
fc = fc[fc.year.isin(yl)].copy()
fc["true"] = [Y[yl.index(int(r.year)), states.index(r.state_id), DIS.index(r.disease)] for r in fc.itertuples()]
m = fc[fc.true.notna() & fc.persistence.notna()].copy()
m["e_model"] = np.abs(np.log1p(m.forecast) - np.log1p(m.true))
m["e_pers"] = np.abs(np.log1p(m.persistence) - np.log1p(m.true))
m["d"] = m.e_model - m.e_pers
rng = np.random.default_rng(0)


def boot(df, n=5000):
    s = df.groupby("state_id").d.agg(["sum", "count"])
    idx = rng.integers(0, len(s), (n, len(s)))
    means = s["sum"].values[idx].sum(1) / s["count"].values[idx].sum(1)
    return np.percentile(means, [2.5, 97.5])


print("Forecasts made from data through 2022; persistence = the 2022 value. Diff < 0 = model better.\n")
for h, g in m.groupby("horizon"):
    for label, sub in (("all rows", g), ("confidence normal", g[g.confidence == "normal"])):
        if len(sub) < 10:
            continue
        lo, hi = boot(sub)
        print(f"horizon {h} (year {int(sub.year.iloc[0])}) {label:18s} n={len(sub):3d} | model {sub.e_model.mean():.3f} "
              f"| persistence {sub.e_pers.mean():.3f} | diff {sub.d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
