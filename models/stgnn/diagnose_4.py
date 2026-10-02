import numpy as np, pandas as pd

name = "gcn_lag1_cap10"
d = np.load("data/processed/graph/dataset.npz")
states = open("configs/states.txt").read().split("\n")
Y = d["Yraw"]; M = d["M"]; dates = pd.DatetimeIndex(d["dates"])
te = d["test_idx"]; t = d["ts"][te]
pred = np.load(f"data/processed/stgnn_test_pred_{name}.npy")
tm = dates[t].month.values
jan = tm == 1

true = Y[t][jan]; m = (M[t][jan] > 0)
cands = {
    "model (gcn_lag1_cap10)": pred[jan],
    "last-month (Dec)": Y[t[jan] - 1],
    "scaled persistence (Dec x 2/3)": Y[t[jan] - 1] * (2 / 3),
    "seasonal naive (Jan last year)": Y[t[jan] - 12],
}
print("January targets in test:", int(jan.sum()), "months |", int(m.sum()), "observed pairs")
for k, p in cands.items():
    ok = m & ~np.isnan(p)
    e = np.abs(p - true)[ok]
    print(f"{k:34s} MAE {e.mean():8.1f} | median {np.median(e):7.1f} | n {ok.sum()}")