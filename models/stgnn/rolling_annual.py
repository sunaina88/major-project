import argparse, warnings, numpy as np, pandas as pd, torch
from model import STModel, masked_huber
warnings.filterwarnings("ignore")

p = argparse.ArgumentParser()
p.add_argument("--kind", default="gcn")
p.add_argument("--graph", default="norm")
p.add_argument("--weather", type=int, default=0)
p.add_argument("--zero_missing", type=int, default=1)
p.add_argument("--K", type=int, default=3)
p.add_argument("--tag", default="full")
p.add_argument("--forecast", type=int, default=0)
a = p.parse_args()
CAP, EPOCHS, SEEDS = 0.5, 150, [0, 1, 2]

G = "data/processed/graph/"
states = open("configs/states.txt").read().split("\n")
DIS = ["dengue_cases", "malaria_cases", "chikungunya_cases"]
if a.graph == "norm":
    A_norm = torch.tensor(np.load(G + "adjacency_norm.npy"))
    A_raw = torch.tensor(np.load(G + "adjacency_raw.npy"))
else:
    A_norm = torch.eye(36); A_raw = torch.zeros(36, 36)

df = pd.read_csv("data/processed/pan_india_state_enriched_tensor_real.csv", parse_dates=["start_date"])
df["year"] = df.start_date.dt.year
years = np.arange(1999, 2023); T = len(years)

def annual(col, how):
    g = df.groupby(["state_id", "year"])[col]
    s = g.sum(min_count=1) if how == "sum" else g.mean()
    return s.unstack("year").reindex(index=states, columns=years).values.T   # [T,N]

Y = np.stack([annual(c, "sum") for c in DIS], -1)            # [T,N,3]
nz = 0
if a.zero_missing:
    for j in range(36):
        for k in range(3):
            s = Y[:, j, k]
            if np.nanmax(s) >= 100 if (~np.isnan(s)).any() else False:
                z = s == 0
                nz += int(z.sum()); s[z] = np.nan
print(f"zeros treated as missing: {nz} | weather features: {bool(a.weather)} | graph: {a.graph} | {a.kind}")
M = (~np.isnan(Y)).astype(np.float32)
Ylog = np.nan_to_num(np.log1p(Y), nan=0.0)
Wraw = np.stack([annual("temp_mean_c", "mean"), annual("rainfall_mm", "sum")], -1)

def fold(Yr, seed):
    cut = Yr - 1999
    trm = np.arange(T) < cut
    cnt = M[trm].sum(0).clip(min=1)
    mu = (Ylog[trm] * M[trm]).sum(0) / cnt
    sd = np.sqrt((((Ylog[trm] - mu) ** 2) * M[trm]).sum(0) / cnt) + 0.05
    Yn = ((Ylog - mu) / sd) * M
    W = (Wraw - np.nanmean(Wraw[trm], axis=(0, 1))) / (np.nanstd(Wraw[trm], axis=(0, 1)) + 1e-6)
    W = np.nan_to_num(W, nan=0.0)
    X = np.concatenate([Yn, M, W], -1).astype(np.float32)    # [T,N,8]
    K = a.K
    tt = list(range(K, cut))
    xtr = torch.tensor(np.stack([X[t - K:t] for t in tt]))
    ytr = torch.tensor(np.stack([Yn[t] for t in tt]).astype(np.float32))
    mtr = torch.tensor(np.stack([M[t] for t in tt]))
    xte = torch.tensor(X[cut - K:cut][None])
    tmax = np.nan_to_num(np.nanmax(Y[trm], axis=0), nan=0.0)
    feat = list(range(8)) if a.weather else list(range(6))
    torch.manual_seed(seed); np.random.seed(seed)
    model = STModel(A_norm, A_raw, feat_idx=feat, hidden=16, kind=a.kind, cap=CAP, base="lag1")
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(len(tt))
        for i in range(0, len(perm), 8):
            b = perm[i:i + 8]
            opt.zero_grad()
            masked_huber(model(xtr[b]), ytr[b], mtr[b]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        pn = model(xte)[0].numpy()
    return np.expm1(np.clip(pn * sd + mu, 0, np.log1p(tmax))), cut


if a.forecast:
    pred = np.mean([fold(2023, s)[0] for s in SEEDS], 0)          # [N,3] annual 2023
    out = []
    for j, s in enumerate(states):
        for k, dz in enumerate(DIS):
            out.append(dict(state_id=s, disease=dz.replace("_cases", ""),
                            forecast_2023=round(float(pred[j, k]), 1),
                            last_year_2022=None if np.isnan(Y[-1, j, k]) else float(Y[-1, j, k])))
    pd.DataFrame(out).to_csv(f"data/processed/stgnn_forecast_2023_{a.tag}.csv", index=False)
    print("saved forecast for 2023 -> data/processed/stgnn_forecast_2023_%s.csv" % a.tag)
    raise SystemExit

rows = []
for Yr in range(2010, 2023):
    pred = np.mean([fold(Yr, s)[0] for s in SEEDS], 0)
    cut = Yr - 1999
    true = Y[cut]; ly = Y[cut - 1]
    avg2 = np.nanmean(np.stack([Y[cut - 1], Y[cut - 2]]), 0)
    ok = (M[cut] > 0) & ~np.isnan(ly) & ~np.isnan(avg2)
    mae = lambda q: np.abs(q - true)[ok].mean()
    lmae = lambda q: np.abs(np.log1p(q) - np.log1p(true))[ok].mean()
    rows.append(dict(year=Yr, n=int(ok.sum()), mae_model=mae(pred), mae_last=mae(ly), mae_avg2=mae(avg2),
                     log_model=lmae(pred), log_last=lmae(ly), log_avg2=lmae(avg2)))
    print(Yr, "done", flush=True)

r = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\n", r.round(2).to_string(index=False))
print("\nMEAN over folds:"); print(r.drop(columns=["year", "n"]).mean().round(3).to_string())
rng = np.random.default_rng(0)
for col, ref in [("mae_model", "mae_last"), ("log_model", "log_last")]:
    dl = (r[col] - r[ref]).values
    bs = [rng.choice(dl, len(dl)).mean() for _ in range(10000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"{col} minus last-year: mean {dl.mean():.3f} CI [{lo:.3f}, {hi:.3f}] folds better {(dl<0).sum()}/{len(dl)}")
r.to_csv(f"data/processed/stgnn_rolling_annual_{a.tag}.csv", index=False)