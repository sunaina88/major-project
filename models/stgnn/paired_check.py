import numpy as np, pandas as pd

P = "data/processed/stgnn_rolling_jan_{}.csv"
tags = ["full", "noweather", "nograph", "gat"]
R = {t: pd.read_csv(P.format(t)) for t in tags}
rng = np.random.default_rng(0)

def boot(diff, n=10000):
    m = [rng.choice(diff, len(diff)).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])

print("difference = model minus comparator, per fold; negative means the model is better\n")
for t in tags:
    for col, ref in [("mae_model", "mae_last"), ("log_model", "log_last")]:
        dlt = (R[t][col] - R[t][ref]).values
        lo, hi = boot(dlt)
        print(f"{t:10s} vs last-year {col:10s} mean diff {dlt.mean():7.3f}  95% CI [{lo:7.3f}, {hi:7.3f}]  folds better {(dlt<0).sum()}/13")
    print()

print("graph effect: nograph minus full")
for col in ["mae_model", "log_model"]:
    dlt = (R["nograph"][col] - R["full"][col]).values
    lo, hi = boot(dlt)
    print(f"  {col:10s} mean diff {dlt.mean():7.3f}  95% CI [{lo:7.3f}, {hi:7.3f}]  nograph better in {(dlt<0).sum()}/13")
print("weather effect: noweather minus full")
for col in ["mae_model", "log_model"]:
    dlt = (R["noweather"][col] - R["full"][col]).values
    lo, hi = boot(dlt)
    print(f"  {col:10s} mean diff {dlt.mean():7.3f}  95% CI [{lo:7.3f}, {hi:7.3f}]  noweather better in {(dlt<0).sum()}/13")