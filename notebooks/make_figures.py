import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.evaluation.data import get_data, DISEASES
D, F = "data/processed/", "docs/figures/"
states, years, Y, _ = get_data()

f = pd.read_csv(D + "stgnn_annual_geo_w_h1.csv")
plt.figure(figsize=(7, 4)); plt.plot(f.year, f.log_model, "o-", label="ST-GNN"); plt.plot(f.year, f.log_last, "s--", label="Persistence")
plt.xlabel("Test year"); plt.ylabel("log-MAE"); plt.legend(); plt.title("Forecast vs persistence per fold"); plt.tight_layout(); plt.savefig(F + "forecast_vs_persistence.png", dpi=150); plt.close()

p = pd.read_csv(D + "stgnn_annual_geo_w_h1_pairs.csv")
p["e_model"] = (np.log1p(p.model) - np.log1p(p.true)).abs(); p["e_pers"] = (np.log1p(p.persistence) - np.log1p(p.true)).abs()
e = p.groupby("state_id")[["e_model", "e_pers"]].mean().sort_values("e_pers")
e.plot.barh(figsize=(7, 9)); plt.xlabel("log-MAE"); plt.title("Error by state"); plt.tight_layout(); plt.savefig(F + "error_by_state.png", dpi=150); plt.close()

a = pd.read_csv(D + "anomaly_annual.csv")
top = pd.Series(np.nansum(Y, axis=(0, 2)), index=states).nlargest(5).index
fig, ax = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
for i, s in enumerate(top):
    sub = a[(a.state_id == s) & (a.disease == "dengue")]
    ax[i].bar(sub.year, sub.zscore); ax[i].axhline(2, color="r", ls="--"); ax[i].set_ylabel(str(s))
ax[0].set_title("Dengue anomaly z-score, 5 highest-burden states"); plt.tight_layout(); plt.savefig(F + "anomaly_timelines.png", dpi=150); plt.close()

from src.evaluation.early_warning import threshold, auroc, DEFS, MIN_PRIOR
years_l = list(years); sidx = {s: i for i, s in enumerate(states)}
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for k, dfn in enumerate(DEFS):
    recs = []
    for r in p.itertuples():
        t = years_l.index(r.year); x = Y[:t, sidx[r.state_id], DISEASES.index(r.disease)]; x = x[~np.isnan(x)]
        if len(x) < MIN_PRIOR: continue
        thr, is_log = threshold(x, dfn); lt = thr if is_log else np.log1p(thr)
        lab = int((np.log1p(r.true) if is_log else r.true) >= thr)
        recs.append((lab, np.log1p(r.model) - lt, np.log1p(r.persistence) - lt))
    R = np.array(recs)
    for j, nm in [(1, "model"), (2, "persistence")]:
        o = np.argsort(-R[:, j]); y = R[o, 0]
        tpr = np.cumsum(y) / max(y.sum(), 1); fpr = np.cumsum(1 - y) / max((1 - y).sum(), 1)
        ax[k].plot(fpr, tpr, label=f"{nm} AUC={auroc(R[:, j], R[:, 0]):.2f}")
    ax[k].plot([0, 1], [0, 1], "k:"); ax[k].set_title(f"{dfn} (n_outbreak={int(R[:, 0].sum())})"); ax[k].legend()
plt.tight_layout(); plt.savefig(F + "early_warning_roc.png", dpi=150)
