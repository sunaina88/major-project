"""Build frontend/static/data_story.json (small static file for the Data story page).

Run from the repo root:  python scripts/make_data_story.py

It holds two things the API does not serve:
  1. The fixed monthly split of an annual total (identical for every state, year and disease).
  2. Annual values for 2021 and 2022 before (graph/dataset.npz, v1 data) and after (current tensor) the source revision.
"""
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DIS = ["dengue", "malaria", "chikungunya"]
STATES = [s.strip() for s in open(ROOT / "configs/states.txt") if s.strip()]

new = pd.read_csv(ROOT / "data/processed/pan_india_state_enriched_tensor.csv", parse_dates=["start_date"])
new["year"] = new.start_date.dt.year

# 1. Monthly shares: the mean share of the annual total that falls in each month, and how much it varies.
shares, spread = [], 0.0
for dis in DIS:
    col = dis + "_cases"
    d = new.dropna(subset=[col]).copy()
    d["month"] = d.start_date.dt.month
    total = d.groupby(["state_id", "year"])[col].transform("sum")
    d = d[total > 0].copy()
    d["share"] = d[col] / total[total > 0]
    g = d.groupby("month").share.agg(["mean", "std"])
    shares.append(g["mean"].round(4).tolist())
    spread = max(spread, float(g["std"].max()))
monthly = {"months": list(range(1, 13)), "share": shares[0], "max_std_across_states_years_diseases": round(spread, 6),
           "same_for_all_diseases": all(s == shares[0] for s in shares)}

# 2. Old (v1) against new values for 2021 and 2022.
ds = np.load(ROOT / "data/processed/graph/dataset.npz", allow_pickle=True)
Y, M = ds["Yraw"], ds["M"]
years = np.array([pd.Timestamp(x).year for x in ds["dates"]])
rows = []
for k, dis in enumerate(DIS):
    annual = new.groupby(["state_id", "year"])[dis + "_cases"].sum(min_count=1)
    for yr in (2021, 2022):
        sel = years == yr
        for i, s in enumerate(STATES):
            seen = M[sel, i, k] > 0
            nv = annual.get((s, yr), np.nan)
            if not seen.any() or np.isnan(nv):
                continue
            ov = float(Y[sel, i, k][seen].sum())
            rows.append({"year": yr, "disease": dis, "state_id": s, "old": round(ov, 1), "new": round(float(nv), 1),
                         "changed": bool(not np.isclose(ov, nv, rtol=2e-3, atol=1.01))})
out = {"monthly": monthly, "revisions": rows}
json.dump(out, open(ROOT / "frontend/static/data_story.json", "w"), indent=1)
n_changed = sum(r["changed"] for r in rows)
print(f"wrote frontend/static/data_story.json | revised pairs: {n_changed} of {len(rows)}")
