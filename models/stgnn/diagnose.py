import numpy as np, pandas as pd

G = "data/processed/graph/"
name = "gcn_all_norm_s0"
d = np.load(G + "dataset.npz")
states = open("configs/states.txt").read().split("\n")
DIS = ["dengue", "malaria", "chikungunya"]
pred = np.load(f"data/processed/stgnn_test_pred_{name}.npy")   # [24,N,3]
te = d["test_idx"]; t = d["ts"][te]
true = d["Yraw"][t]; m = d["M"][t] > 0
mu, sd = d["mu"], d["sd"]
Yraw = d["Yraw"]; tr_t = d["dates"] < np.datetime64("2021-01-01")

rows = []
for j, s in enumerate(states):
    for k, dz in enumerate(DIS):
        mk = m[:, j, k]
        if mk.sum() == 0:
            continue
        tr_vals = Yraw[tr_t][:, j, k]
        tr_vals = tr_vals[~np.isnan(tr_vals)]
        rows.append(dict(state=s, dis=dz,
            mae=np.abs(pred[:, j, k][mk] - true[:, j, k][mk]).mean(),
            pred_mean=pred[:, j, k][mk].mean(), true_mean=true[:, j, k][mk].mean(),
            train_max=tr_vals.max(), train_mean=tr_vals.mean(),
            sd=sd[j, k], n_train=len(tr_vals)))
r = pd.DataFrame(rows).sort_values("mae", ascending=False)
pd.set_option("display.width", 200)
print(r.head(15).round(2).to_string(index=False))
print("\nnumber of pairs with MAE > 5000:", int((r.mae > 5000).sum()), "of", len(r))
print("mean MAE excluding those:", round(r[r.mae <= 5000].mae.mean(), 2))