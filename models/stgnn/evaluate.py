import sys, re, numpy as np, pandas as pd, torch
from model import STModel

name = sys.argv[1] if len(sys.argv) > 1 else "gcn_all_norm_s0"
G = "data/processed/graph/"
states = open("configs/states.txt").read().split("\n")
DIS = ["dengue", "malaria", "chikungunya"]

d = np.load(G + "dataset.npz")
ck = torch.load(f"models/stgnn/{name}.pt", weights_only=False)
a = ck["args"]
if a["graph"] == "norm":
    A_norm = torch.tensor(np.load(G + "adjacency_norm.npy"))
    A_raw = torch.tensor(np.load(G + "adjacency_raw.npy"))
else:
    A_norm = torch.eye(36); A_raw = torch.zeros(36, 36)
model = STModel(A_norm, A_raw, feat_idx=ck["feat_idx"], hidden=a["hidden"], kind=a["kind"], cap=a.get("cap", 1.5), base=a.get("base", "lag12"))
model.load_state_dict(ck["state"]); model.eval()

te = d["test_idx"]
with torch.no_grad():
    pred_n = model(torch.tensor(d["xs"][te])).numpy()          # [24,N,3] normalized
mu, sd = d["mu"], d["sd"]
tr_t = d["dates"] < np.datetime64("2021-01-01")
tmax = np.nan_to_num(np.nanmax(d["Yraw"][tr_t], axis=0), nan=0.0)   # [N,3] train max
logp = np.clip(pred_n * sd + mu, 0, np.log1p(tmax))                 # cap at historical max
pred = np.expm1(logp)                                               # real cases          # real cases
t = d["ts"][te]
true = d["Yraw"][t]                                            # [24,N,3] raw (NaN if unobserved)
m = d["M"][t] > 0

rows = []
for j, s in enumerate(states):
    for k, dz in enumerate(DIS):
        mk = m[:, j, k]
        if mk.sum() == 0:
            continue
        e = pred[:, j, k][mk] - true[:, j, k][mk]
        rows.append((s, dz, float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))))
mine = pd.DataFrame(rows, columns=["state_id", "disease", "mae", "rmse"])
mine["model"] = name
mine.to_csv(f"data/processed/stgnn_metrics_{name}.csv", index=False)
np.save(f"data/processed/stgnn_test_pred_{name}.npy", pred)

# ---- join with baselines ----
b = pd.read_csv("data/processed/baseline_model_metrics.csv")
print("baseline models:", b.model.unique().tolist())
print("baseline disease values:", b.disease.unique().tolist())

def slug(x): return re.sub(r"[^a-z0-9]+", "_", str(x).lower().replace("&", "and")).strip("_")
def dnorm(x):
    x = str(x).lower()
    return "dengue" if "den" in x else "malaria" if "mal" in x else "chikungunya"
b["state_id"] = b.state_name.map(slug)
b["disease"] = b.disease.map(dnorm)

unmatched = set(b.state_id) - set(states)
print("unmatched baseline states:", unmatched)

pairs = mine[["state_id", "disease"]].merge(b[["state_id", "disease"]].drop_duplicates())
bb = b.merge(pairs)
mm = mine.merge(pairs)
print(f"\ncommon state x disease pairs: {len(pairs)}")

summary = bb.groupby("model")[["mae", "rmse"]].mean()
summary.loc[name] = [mm.mae.mean(), mm.rmse.mean()]
print("\n=== MEAN over common pairs ===")
print(summary.round(2).sort_values("mae").to_string())

print("\n=== by disease (MAE) ===")
t1 = bb.groupby(["disease", "model"]).mae.mean().unstack()
t1[name] = mm.groupby("disease").mae.mean()
print(t1.round(2).to_string())

# win rate vs best baseline per pair
bl = bb.pivot_table(index=["state_id", "disease"], columns="model", values="mae")
best_bl = bl.min(axis=1)
cmp = mm.set_index(["state_id", "disease"]).mae
win_all = (cmp < best_bl.reindex(cmp.index)).mean()
gb = [c for c in bl.columns if "Weather" not in c and "Gradient" in c]
if gb:
    win_gb = (cmp < bl[gb[0]].reindex(cmp.index)).mean()
    print(f"\nwin rate vs {gb[0]}: {win_gb:.2%}")
print(f"win rate vs best baseline per pair: {win_all:.2%}")

print("\n=== robust view ===")
print(f"median MAE  mine {mm.mae.median():.1f} | GB {bb[bb.model=='Gradient_Boosting'].mae.median():.1f} | SeasNaive {bb[bb.model=='Seasonal_Naive'].mae.median():.1f}")
w = mm.sort_values("mae", ascending=False).head(8)
print("\nworst 8 pairs (mine):")
print(w[["state_id", "disease", "mae"]].round(1).to_string(index=False))
print("\nmean MAE of my 10 worst pairs vs rest:",
      round(mm.mae.nlargest(10).mean(), 1), "|", round(mm.mae.nsmallest(98).mean(), 1))


print("\n=== floor: pure same-month-last-year (cap=0) ===")
base_n = d["xs"][te][:, 0 if a.get("base", "lag12") == "lag12" else -1, :, 0:3]
bl_pred = np.expm1(np.clip(base_n * sd + mu, 0, np.log1p(tmax)))
fl = []
for j, s in enumerate(states):
    for k, dz in enumerate(DIS):
        mk = m[:, j, k]
        if mk.sum():
            fl.append((s, dz, float(np.abs(bl_pred[:, j, k][mk] - true[:, j, k][mk]).mean())))
fl = pd.DataFrame(fl, columns=["state_id", "disease", "floor_mae"])
cmpd = mm.merge(fl, on=["state_id", "disease"])
print(f"floor mean MAE {cmpd.floor_mae.mean():.1f} | model mean MAE {cmpd.mae.mean():.1f}")
print(f"model beats its own floor on {(cmpd.mae < cmpd.floor_mae).mean():.1%} of pairs")
print(f"median: floor {cmpd.floor_mae.median():.1f} | model {cmpd.mae.median():.1f}")