import numpy as np, pandas as pd

name = "gcn_lag1_cap10"
d = np.load("data/processed/graph/dataset.npz")
states = open("configs/states.txt").read().split("\n")
Y = d["Yraw"]; M = d["M"]; dates = pd.DatetimeIndex(d["dates"])
te = d["test_idx"]; t = d["ts"][te]
pred = np.load(f"data/processed/stgnn_test_pred_{name}.npy")   # [24,N,3]
true = Y[t]; m = M[t] > 0
tm = dates[t].month.values
tr_t = np.array(dates < pd.Timestamp("2021-01-01"))

# 1. is the within-year ratio constant? (month m / month m-1, raw counts)
print("ratio y[m]/y[m-1] within a year, train data (median, IQR):")
ratios = {}
for mo in range(2, 13):
    r = []
    for j in range(36):
        for k in range(3):
            for yr in range(1999, 2021):
                a = Y[(dates.year == yr) & (dates.month == mo - 1), j, k]
                b = Y[(dates.year == yr) & (dates.month == mo), j, k]
                if a.size and b.size and a[0] > 0 and not np.isnan(a[0]) and not np.isnan(b[0]):
                    r.append(b[0] / a[0])
    r = np.array(r)
    ratios[mo] = np.median(r)
    print(f"  month {mo:2d}: median {np.median(r):.3f}  IQR {np.percentile(r,25):.3f}-{np.percentile(r,75):.3f}")

# 2. January vs other months: model error vs a pure ratio rule
lag1 = Y[t - 1]
rule = lag1.copy()
for i, mo in enumerate(tm):
    rule[i] = lag1[i] * (ratios[mo] if mo >= 2 else 1.0)

def mae(p, sel):
    e = np.abs(p - true)
    mk = m & sel[:, None, None]
    return e[mk].mean() if mk.any() else np.nan

jan = tm == 1
print("\nMAE by target month group (test 2021-22):")
print(f"  model        Jan: {mae(pred, jan):8.1f} | Feb-Dec: {mae(pred, ~jan):8.1f}")
print(f"  ratio rule   Jan: {mae(rule, jan):8.1f} | Feb-Dec: {mae(rule, ~jan):8.1f}")
print(f"  last-month   Jan: {mae(lag1, jan):8.1f} | Feb-Dec: {mae(lag1, ~jan):8.1f}")