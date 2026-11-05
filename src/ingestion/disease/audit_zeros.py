from pathlib import Path

import numpy as np
import pandas as pd

AUDIT_PATH = Path("data/processed/zero_audit.csv")
KEYS = ["state_id", "disease", "year"]

df = pd.read_csv("data/processed/pan_india_state_enriched_tensor.csv", parse_dates=["start_date"])
df["year"] = df.start_date.dt.year
rows = []
for d in ["dengue", "malaria", "chikungunya"]:
    a = df.groupby(["state_id", "year"])[f"{d}_cases"].sum(min_count=1).unstack("year")
    for s in a.index:
        x = a.loc[s]
        for yr in x.index[x == 0]:
            later = x.loc[yr + 1:]
            rows.append(dict(
                state_id=s, disease=d, year=int(yr), series_max=x.max(),
                prev_year=x.get(yr - 1, np.nan), next_year=x.get(yr + 1, np.nan),
                years_observed=int(x.notna().sum()), zeros_in_series=int((x == 0).sum()),
                trailing=bool(((later == 0) | later.isna()).all()),
                heuristic="likely_gap" if x.max() >= 100 else "small_series_maybe_real",
                verdict="", evidence=""))
out = pd.DataFrame(rows).sort_values(["heuristic", "state_id", "disease", "year"])
if AUDIT_PATH.exists():
    previous = pd.read_csv(AUDIT_PATH)
    if {"verdict", "evidence"}.issubset(previous.columns):
        previous = previous[KEYS + ["verdict", "evidence"]].drop_duplicates(KEYS)
        out = out.merge(previous, on=KEYS, how="left", suffixes=("", "_previous"))
        for column in ("verdict", "evidence"):
            out[column] = out[f"{column}_previous"].fillna(out[column])
            out.drop(columns=f"{column}_previous", inplace=True)
out.to_csv(AUDIT_PATH, index=False)
print(len(out), "zero state-years |", out.heuristic.value_counts().to_dict())
print(out[out.heuristic == "likely_gap"].groupby(["state_id", "disease"]).year.apply(list).to_string())