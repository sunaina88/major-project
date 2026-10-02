import numpy as np, pandas as pd, torch

DIS = ["dengue_cases", "malaria_cases", "chikungunya_cases"]


def load_annual(path, zero_missing=1):
    states = open("configs/states.txt").read().split("\n")
    df = pd.read_csv(path, parse_dates=["start_date"])
    df["year"] = df.start_date.dt.year
    years = np.arange(1999, int(df.year.max()) + 1)

    def annual(col, how):
        g = df.groupby(["state_id", "year"])[col]
        s = g.sum(min_count=1) if how == "sum" else g.mean()
        return s.unstack("year").reindex(index=states, columns=years).values.T   # [T,N]

    Y = np.stack([annual(c, "sum") for c in DIS], -1).astype(np.float64)       # [T,N,3]
    nz = 0
    if zero_missing:
        for j in range(Y.shape[1]):
            for k in range(3):
                s = Y[:, j, k]
                if (~np.isnan(s)).any() and np.nanmax(s) >= 100:
                    z = s == 0
                    nz += int(z.sum()); s[z] = np.nan
    print(f"[data] years {years[0]}-{years[-1]} | zeros treated as missing: {nz}")
    W = np.stack([annual("temp_mean_c", "mean"), annual("rainfall_mm", "sum")], -1)
    return states, years, Y, W


def load_graph(name):
    G = "data/processed/graph/"
    if name == "identity":
        return torch.eye(36), torch.zeros(36, 36)
    if name in ("geo", "norm"):
        return (torch.tensor(np.load(G + "adjacency_norm.npy")),
                torch.tensor(np.load(G + "adjacency_raw.npy")))
    return (torch.tensor(np.load(G + f"adjacency_{name}_norm.npy")),
            torch.tensor(np.load(G + f"adjacency_{name}_raw.npy")))


def to_cases(pn, mu, sd, tmax):
    """normalized log prediction -> cases, capped at the historical max. Also returns the cap flag."""
    z = pn * sd + mu
    cap = np.log1p(tmax)
    return np.expm1(np.clip(z, 0, cap)), z > cap


def forecast_table(states, year, h, pred, capped, last, nobs):
    rows = []
    for j, s in enumerate(states):
        for k, d in enumerate(DIS):
            notes = []
            if np.isnan(last[j, k]): notes.append("no_last_year_value")
            if nobs[j, k] < 8: notes.append("short_history")
            if capped[j, k]: notes.append("capped_at_historical_max")
            rows.append(dict(
                state_id=s, disease=d.replace("_cases", ""), year=int(year), horizon=int(h),
                forecast=int(round(float(pred[j, k]))),
                persistence=None if np.isnan(last[j, k]) else int(round(float(last[j, k]))),
                confidence="low" if notes else "normal", notes=";".join(notes),
                n_years_observed=int(nobs[j, k])))
    return pd.DataFrame(rows)