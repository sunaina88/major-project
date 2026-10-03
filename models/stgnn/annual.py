import argparse, warnings, numpy as np, pandas as pd, torch
from model import STModel, masked_huber
from annual_data import load_annual, load_graph, to_cases, forecast_table, DIS
warnings.filterwarnings("ignore")

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/processed/pan_india_state_enriched_tensor_real.csv")
p.add_argument("--kind", default="gcn", choices=["gcn", "gat"])
p.add_argument("--graph", default="geo", choices=["geo", "identity", "knn", "dist", "hybrid"])
p.add_argument("--weather", type=int, default=1)
p.add_argument("--single", type=int, default=0)       # 1 = separate model per disease
p.add_argument("--zero_missing", type=int, default=1)
p.add_argument("--K", type=int, default=3)            # input window in years
p.add_argument("--horizons", default="1")             # e.g. 1,2,3 (years ahead)
p.add_argument("--cap", type=float, default=0.5)
p.add_argument("--hidden", type=int, default=16)
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--tag", default="run")
p.add_argument("--forecast", type=int, default=0)
a = p.parse_args()
HS = [int(x) for x in a.horizons.split(",")]

states, years, Y, Wraw = load_annual(a.data, a.zero_missing)
T, N = Y.shape[0], Y.shape[1]
M = (~np.isnan(Y)).astype(np.float32)
Ylog = np.nan_to_num(np.log1p(Y), nan=0.0)
A_norm, A_raw = load_graph(a.graph)
groups = [[0], [1], [2]] if a.single else [[0, 1, 2]]
print(f"[cfg] {a.kind} graph={a.graph} weather={a.weather} single={a.single} K={a.K} horizons={HS} tag={a.tag}")


def prepare(o):
    """normalisation statistics use only years before index o"""
    trm = np.arange(T) < o
    cnt = M[trm].sum(0).clip(min=1)
    mu = (Ylog[trm] * M[trm]).sum(0) / cnt
    sd = np.sqrt((((Ylog[trm] - mu) ** 2) * M[trm]).sum(0) / cnt) + 0.05
    Yn = ((Ylog - mu) / sd) * M
    W = (Wraw - np.nanmean(Wraw[trm], axis=(0, 1))) / (np.nanstd(Wraw[trm], axis=(0, 1)) + 1e-6)
    W = np.nan_to_num(W, nan=0.0)
    X = np.concatenate([Yn, M, W], -1).astype(np.float32)          # [T,N,8]
    tmax = np.nan_to_num(np.nanmax(Y[trm], axis=0), nan=0.0)
    return X, Yn, mu, sd, tmax


def fit(o, h, seed, X, Yn, g):
    K = a.K
    tt = list(range(K, o - h + 1))                  # targets t+h-1 must be known (< o)
    if len(tt) == 0:
        return None
    feat = list(g) + [3 + d for d in g] + ([6, 7] if a.weather else [])
    xtr = torch.tensor(np.stack([X[t - K:t] for t in tt]))
    ytr = torch.tensor(np.stack([Yn[t + h - 1] for t in tt]).astype(np.float32))
    oh = np.zeros(3, dtype=np.float32); oh[g] = 1.0
    mtr = torch.tensor((np.stack([M[t + h - 1] for t in tt]) * oh).astype(np.float32))
    xte = torch.tensor(X[o - K:o][None])
    torch.manual_seed(seed); np.random.seed(seed)
    model = STModel(A_norm, A_raw, feat_idx=feat, hidden=a.hidden, kind=a.kind, cap=a.cap, base="lag1")
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for ep in range(a.epochs):
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
    return pn, {k: v.clone() for k, v in model.state_dict().items()}, feat


def run(o, h):
    X, Yn, mu, sd, tmax = prepare(o)
    out = np.zeros((N, 3)); capped = np.zeros((N, 3), bool); store = []
    for g in groups:
        res = [fit(o, h, s, X, Yn, g) for s in range(a.seeds)]
        if res[0] is None:
            return None
        cs = [to_cases(r[0], mu, sd, tmax) for r in res]
        out[:, g] = np.mean([c[0] for c in cs], 0)[:, g]
        capped[:, g] = (np.mean([c[1] for c in cs], 0) > 0.5)[:, g]
        store.append({"group": g, "feat": res[0][2], "states": [r[1] for r in res]})
    return out, capped, store, (X[o - a.K:o].copy(), mu, sd, tmax)


# ---------------- forecast + checkpoint mode ----------------
if a.forecast:
    models, frames = {}, []
    last, nobs = Y[-1], M.sum(0)
    for h in HS:
        pred, capped, store, (Xl, mu, sd, tmax) = run(T, h)
        models[h] = store
        frames.append(forecast_table(states, int(years[-1]) + h, h, pred, capped, last, nobs))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(f"data/processed/stgnn_forecast_{a.tag}.csv", index=False)
    torch.save({"args": vars(a), "states": states, "years": [int(y) for y in years],
                "X_last": Xl, "mu": mu, "sd": sd, "tmax": tmax, "last": last, "nobs": nobs,
                "models": models}, f"models/stgnn/annual_{a.tag}.pt")
    print(f"saved data/processed/stgnn_forecast_{a.tag}.csv ({len(df)} rows) and models/stgnn/annual_{a.tag}.pt")
    print(df.confidence.value_counts().to_string())
    raise SystemExit

# ---------------- rolling-origin evaluation ----------------
rng = np.random.default_rng(0)
for h in HS:
    rows = []
    pair_rows = []
    for Yr in range(2010, int(years[-1]) + 1):
        tgt = Yr - int(years[0]); o = tgt - (h - 1)
        r = run(o, h)
        if r is None:
            continue
        pred = r[0]
        true, ly = Y[tgt], Y[o - 1]
        avg2 = np.nanmean(np.stack([Y[o - 1], Y[o - 2]]), 0)
        ok = (M[tgt] > 0) & ~np.isnan(ly) & ~np.isnan(avg2)
        for j in range(N):
            for k in range(3):
                if ok[j, k]:
                    pair_rows.append(dict(year=Yr, state_id=states[j],
                                          disease=DIS[k].replace("_cases", ""),
                                          true=float(true[j, k]), model=float(pred[j, k]),
                                          persistence=float(ly[j, k])))
        mae = lambda q: np.abs(q - true)[ok].mean()
        lmae = lambda q: np.abs(np.log1p(q) - np.log1p(true))[ok].mean()
        rows.append(dict(year=Yr, n=int(ok.sum()), mae_model=mae(pred), mae_last=mae(ly), mae_avg2=mae(avg2),
                         log_model=lmae(pred), log_last=lmae(ly), log_avg2=lmae(avg2)))
        print(f"h={h} {Yr} done", flush=True)
    r = pd.DataFrame(rows)
    print(f"\n=== horizon {h} year(s) ahead | MEAN over {len(r)} folds ===")
    print(r.drop(columns=["year", "n"]).mean().round(3).to_string())
    for col, ref in [("mae_model", "mae_last"), ("log_model", "log_last")]:
        dl = (r[col] - r[ref]).values
        bs = [rng.choice(dl, len(dl)).mean() for _ in range(10000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{col} minus persistence: mean {dl.mean():.3f} CI [{lo:.3f}, {hi:.3f}] folds better {(dl < 0).sum()}/{len(dl)}")
    r.to_csv(f"data/processed/stgnn_annual_{a.tag}_h{h}.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(f"data/processed/stgnn_annual_{a.tag}_h{h}_pairs.csv", index=False)