import numpy as np, pandas as pd

PATH = "data/processed/pan_india_state_enriched_tensor_real.csv"
states = open("configs/states.txt").read().split("\n")
N = len(states)
dis = ["dengue_cases", "malaria_cases", "chikungunya_cases"]
wx = ["temp_mean_c", "rainfall_mm", "temp_lag_1", "temp_lag_2",
      "rainfall_lag_1", "rainfall_lag_2", "rainfall_rolling_3m"]
L = 12

df = pd.read_csv(PATH, parse_dates=["start_date"])
dates = pd.date_range("1999-01-01", "2022-12-01", freq="MS")
T = len(dates)  # 288

def grid(col):
    p = df.pivot(index="start_date", columns="state_id", values=col)
    return p.reindex(index=dates, columns=states).values.astype(np.float64)  # [T,N]

# targets, mask
Yraw = np.stack([grid(c) for c in dis], -1)          # [T,N,3]
M = (~np.isnan(Yraw)).astype(np.float32)
Ylog = np.log1p(Yraw)
Ylog = np.nan_to_num(Ylog, nan=0.0)

# weather: fill NaN lags (first months) with column value from same state forward/back fill
W = np.stack([grid(c) for c in wx], -1)              # [T,N,7]
for k in range(W.shape[-1]):
    w = pd.DataFrame(W[:, :, k]).bfill().ffill().values
    W[:, :, k] = np.nan_to_num(w, nan=0.0)

# month encoding
mon = dates.month.values
sc = np.stack([np.sin(2*np.pi*mon/12), np.cos(2*np.pi*mon/12)], -1)
sc = np.repeat(sc[:, None, :], N, 1)                 # [T,N,2]

# train-only statistics (dates before 2021)
tr_t = dates < pd.Timestamp("2021-01-01")
cnt = M[tr_t].sum(0).clip(min=1)                     # [N,3]
mu = (Ylog[tr_t] * M[tr_t]).sum(0) / cnt             # [N,3]
var = (((Ylog[tr_t] - mu) ** 2) * M[tr_t]).sum(0) / cnt
sd = np.sqrt(var) + 1e-3                             # [N,3]

Yn = ((Ylog - mu) / sd) * M                          # unobserved -> 0
wmu = W[tr_t].mean((0, 1)); wsd = W[tr_t].std((0, 1)) + 1e-6
Wn = (W - wmu) / wsd

X = np.concatenate([Yn, M, Wn, sc], -1).astype(np.float32)   # [T,N,15]
print("X:", X.shape, "| NaNs:", int(np.isnan(X).sum()))

# windows: input [t-L, t), target at t
xs, ys, ms, ts = [], [], [], []
for t in range(L, T):
    xs.append(X[t-L:t]); ys.append(Yn[t]); ms.append(M[t]); ts.append(t)
xs = np.array(xs, dtype=np.float32)
ys = np.array(ys, dtype=np.float32)
ms = np.array(ms, dtype=np.float32)
ts = np.array(ts)

tdate = dates[ts]
train_idx = np.where(tdate < pd.Timestamp("2019-01-01"))[0]
val_idx = np.where((tdate >= pd.Timestamp("2019-01-01")) & (tdate < pd.Timestamp("2021-01-01")))[0]
test_idx = np.where(tdate >= pd.Timestamp("2021-01-01"))[0]
print("windows:", xs.shape, "| train/val/test:", len(train_idx), len(val_idx), len(test_idx))
print("observed targets  train/val/test:",
      int(ms[train_idx].sum()), int(ms[val_idx].sum()), int(ms[test_idx].sum()))

np.savez_compressed("data/processed/graph/dataset.npz",
    xs=xs, ys=ys, ms=ms, ts=ts,
    train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    mu=mu, sd=sd, wmu=wmu, wsd=wsd,
    dates=dates.values.astype("datetime64[D]"),
    Yraw=Yraw.astype(np.float32), M=M)
print("saved data/processed/graph/dataset.npz")