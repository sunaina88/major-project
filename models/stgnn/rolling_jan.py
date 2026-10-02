import warnings, numpy as np, pandas as pd, torch
from model import STModel, ALL, masked_huber
warnings.filterwarnings("ignore")

import argparse
from model import NO_WEATHER
p = argparse.ArgumentParser()
p.add_argument("--kind", default="gcn")
p.add_argument("--features", default="all")
p.add_argument("--graph", default="norm")
p.add_argument("--tag", default="full")
args = p.parse_args()
CAP, EPOCHS, SEEDS, HIDDEN = 0.5, 150, [0, 1, 2], 32
FEAT = ALL if args.features == "all" else NO_WEATHER
G = "data/processed/graph/"
d = np.load(G + "dataset.npz")
if args.graph == "norm":
    A_norm = torch.tensor(np.load(G + "adjacency_norm.npy"))
    A_raw = torch.tensor(np.load(G + "adjacency_raw.npy"))
else:
    A_norm = torch.eye(36); A_raw = torch.zeros(36, 36)
xs, ts = d["xs"], d["ts"]
Yraw = d["Yraw"].astype(np.float64); M = d["M"]
dates = pd.DatetimeIndex(d["dates"]); T = len(dates)
Ylog = np.nan_to_num(np.log1p(Yraw), nan=0.0)

# rebuild the full feature tensor X[T,N,15] from the saved windows
X = np.zeros((T, 36, 15), dtype=np.float32)
X[:12] = xs[0]
for i, t in enumerate(ts):
    X[t - 1] = xs[i][-1]

jan_t = [t for t in ts if dates[t].month == 1]

def run_fold(Y, seed):
    cut = pd.Timestamp(Y, 1, 1)
    trm = np.array(dates < cut)
    cnt = M[trm].sum(0).clip(min=1)
    mu = (Ylog[trm] * M[trm]).sum(0) / cnt
    sd = np.sqrt((((Ylog[trm] - mu) ** 2) * M[trm]).sum(0) / cnt) + 1e-3
    Yn = ((Ylog - mu) / sd) * M
    Xf = X.copy(); Xf[:, :, 0:3] = Yn
    tmax = np.nan_to_num(np.nanmax(Yraw[trm], axis=0), nan=0.0)

    tr = [t for t in jan_t if dates[t] < cut]
    te = [t for t in jan_t if dates[t].year == Y][0]
    xtr = torch.tensor(np.stack([Xf[t - 12:t] for t in tr]))
    ytr = torch.tensor(np.stack([Yn[t] for t in tr]))
    mtr = torch.tensor(np.stack([M[t] for t in tr]))
    xte = torch.tensor(Xf[te - 12:te][None])

    torch.manual_seed(seed); np.random.seed(seed)
    model = STModel(A_norm, A_raw, feat_idx=FEAT, hidden=HIDDEN, kind=args.kind, cap=CAP, base="lag12")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(len(tr))
        for i in range(0, len(perm), 8):
            b = perm[i:i + 8]
            opt.zero_grad()
            masked_huber(model(xtr[b]), ytr[b], mtr[b]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        pn = model(xte)[0].numpy()
    return np.expm1(np.clip(pn * sd + mu, 0, np.log1p(tmax))), te

rows = []
for Y in range(2010, 2023):
    preds = []
    for s in SEEDS:
        p, te = run_fold(Y, s); preds.append(p)
    pred = np.mean(preds, 0)                      # average over seeds
    true = Yraw[te]
    ly = Yraw[te - 12]
    avg2 = np.nanmean(np.stack([Yraw[te - 12], Yraw[te - 24]]), 0)
    ok = (M[te] > 0) & ~np.isnan(ly) & ~np.isnan(avg2)
    def mae(p): return np.abs(p - true)[ok].mean()
    def lmae(p): return np.abs(np.log1p(p) - np.log1p(true))[ok].mean()
    rows.append(dict(year=Y, n=int(ok.sum()),
        mae_model=mae(pred), mae_last=mae(ly), mae_avg2=mae(avg2),
        log_model=lmae(pred), log_last=lmae(ly), log_avg2=lmae(avg2)))
    print(f"{Y} done", flush=True)

r = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\n", r.round(2).to_string(index=False))
print("\nMEAN over folds:")
print(r.drop(columns=["year", "n"]).mean().round(3).to_string())
print("\nfolds where model beats last-year  (MAE):", int((r.mae_model < r.mae_last).sum()), "of", len(r))
print("folds where model beats last-year  (log-MAE):", int((r.log_model < r.log_last).sum()), "of", len(r))
r.to_csv(f"data/processed/stgnn_rolling_jan_{args.tag}.csv", index=False)