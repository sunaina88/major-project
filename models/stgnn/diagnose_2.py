import numpy as np, pandas as pd

d = np.load("data/processed/graph/dataset.npz")
states = open("configs/states.txt").read().split("\n")
dates = pd.DatetimeIndex(d["dates"]); Y = d["Yraw"]   # [T,N,3]
DIS = ["dengue", "malaria", "chikungunya"]
pred = np.load("data/processed/stgnn_test_pred_gcn_all_norm_s0.npy")
t = d["ts"][d["test_idx"]]
pd.set_option("display.width", 220)

for s, k in [("telangana", 2), ("madhya_pradesh", 2), ("maharashtra", 2),
             ("odisha", 2), ("kerala", 0), ("west_bengal", 0)]:
    j = states.index(s)
    ser = pd.Series(Y[:, j, k], index=dates)
    ann = ser.groupby(ser.index.year).sum(min_count=1).loc[2014:]
    print(f"\n{s} / {DIS[k]} annual totals 2014-2022:")
    print(ann.round(0).astype("Int64").to_string())
    p = pd.Series(pred[:, j, k], index=dates[t])
    print("predicted annual (2021, 2022):", p.groupby(p.index.year).sum().round(0).to_dict())

# how many test pairs have all-zero truth?
te_true = Y[t]
zero_pairs = [(states[j], DIS[k]) for j in range(36) for k in range(3)
              if (te_true[:, j, k] == 0).all()]
print("\npairs with all-zero test truth:", len(zero_pairs))
print(zero_pairs)