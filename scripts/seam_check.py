import io, subprocess
import numpy as np
import pandas as pd

DIS = ["dengue", "malaria", "chikungunya"]
P = "data/processed/pan_india_state_spatiotemporal_tensor.csv"


def annual(df):
    df = df.assign(year=df.start_date.dt.year)
    return pd.DataFrame({d: df.groupby(["state_id", "year"])[f"{d}_cases"].sum(min_count=1) for d in DIS})


new = annual(pd.read_csv(P, parse_dates=["start_date"]))
old_txt = subprocess.run(["git", "show", "main:" + P], capture_output=True, text=True, check=True).stdout
old = annual(pd.read_csv(io.StringIO(old_txt), parse_dates=["start_date"]))

print("1) OLD vs NEW source, same state-year (up to 2022). Rows not shown are identical.")
rows = []
for d in DIS:
    j = pd.concat([old[d].rename("old"), new[d].rename("new")], axis=1)
    j = j[j.index.get_level_values("year") <= 2022]
    for yr, g in j.groupby(level="year"):
        b = g.dropna()
        pos = b[(b.old > 0) & (b.new > 0)]
        lr = np.log(pos.new / pos.old) if len(pos) else pd.Series(dtype=float)
        rows.append(dict(disease=d, year=yr, both=len(b),
                         identical=round(float(((b.old - b.new).abs() < 0.5).mean()), 2) if len(b) else np.nan,
                         median_log_ratio=round(float(lr.median()), 3) if len(lr) else np.nan,
                         mean_abs_log=round(float(lr.abs().mean()), 3) if len(lr) else np.nan,
                         added=int((g.old.isna() & g.new.notna()).sum()),
                         removed=int((g.old.notna() & g.new.isna()).sum())))
t = pd.DataFrame(rows)
print(t[(t.identical < 1) | (t.added > 0) | (t.removed > 0)].to_string(index=False))

print("\n2) NEW series: median |log change| from the previous year across states (a seam shows as a jump)")
out = {}
for d in DIS:
    s = new[d].unstack("year")
    ch = {}
    for y in s.columns:
        if y - 1 in s.columns:
            a, b = s[y - 1], s[y]
            ok = a.notna() & b.notna() & (a > 0) & (b > 0)
            if ok.sum() >= 8:
                ch[y] = round(float(np.median(np.abs(np.log(b[ok] / a[ok])))), 2)
    out[d] = pd.Series(ch)
print(pd.DataFrame(out).loc[2005:].to_string())

f = pd.read_csv("data/processed/stgnn_annual_geo_w_h1.csv")
f["diff"] = f.log_model - f.log_last
print("\n3) per-fold difference, model minus persistence (negative = model better)")
print(f[["year", "n", "log_model", "log_last", "diff"]].round(3).to_string(index=False))
print("mean", round(f["diff"].mean(), 3), "| without 2021:", round(f.loc[f.year != 2021, "diff"].mean(), 3))
