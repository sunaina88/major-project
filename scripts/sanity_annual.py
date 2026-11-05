import io, subprocess, sys
import numpy as np
import pandas as pd

NEW = "data/processed/pan_india_state_spatiotemporal_tensor.csv"
DIS = ["dengue", "malaria", "chikungunya"]


def annual(df):
    df = df.assign(year=df.start_date.dt.year)
    return pd.DataFrame({d: df.groupby(["state_id", "year"])[f"{d}_cases"].sum(min_count=1) for d in DIS})


new = pd.read_csv(NEW, parse_dates=["start_date"])
a_new = annual(new)

print("NATIONAL TOTAL (sum over reporting states) and number of states, per year")
tot = pd.concat({d: a_new[d].groupby("year").agg(["sum", "count"]) for d in DIS}, axis=1)
print(tot.loc[2008:].round(0).to_string())
for d in DIS:
    s = a_new[d].groupby("year").sum(min_count=1)
    drop = s[(s / s.shift(1) < 0.5)]
    if len(drop):
        print(f"WARNING {d}: national total falls by more than half in {list(drop.index)}"
              " (partial year, missing states, or an unreported-zero problem?)")

try:
    txt = subprocess.run(["git", "show", "main:" + NEW], capture_output=True, text=True, check=True).stdout
    old = pd.read_csv(io.StringIO(txt), parse_dates=["start_date"])
except Exception as e:
    print("\ncould not read the old tensor from main:", e)
    sys.exit()
a_old = annual(old)
print("\nOLD vs NEW for years up to 2022 (state-years):")
rows, changed = [], []
for d in DIS:
    j = pd.concat([a_old[d].rename("old"), a_new[d].rename("new")], axis=1)
    j = j[j.index.get_level_values("year") <= 2022]
    both = j.dropna()
    diff = (both.old - both.new).abs() >= 0.5
    rows.append(dict(disease=d, unchanged=int((~diff).sum()), changed=int(diff.sum()),
                     added=int((j.old.isna() & j.new.notna()).sum()),
                     removed=int((j.old.notna() & j.new.isna()).sum())))
    changed.append(both[diff].assign(disease=d).reset_index())
print(pd.DataFrame(rows).to_string(index=False))
ch = pd.concat(changed)
if len(ch):
    ch["rel"] = (ch.new - ch.old).abs() / ch.old.clip(lower=1)
    print("\nlargest changes:")
    print(ch.sort_values("rel", ascending=False).head(20)[["disease", "state_id", "year", "old", "new", "rel"]].round(2).to_string(index=False))