import numpy as np, pandas as pd
from src.evaluation.data import get_data, DISEASES

def compute_anomalies(Y, years, states, min_prior=5, window=5, floor=0.1):
    rows = []
    for s in range(Y.shape[1]):
        for d in range(Y.shape[2]):
            x = Y[:, s, d]
            for t, yr in enumerate(years):
                if np.isnan(x[t]):
                    continue
                prior = x[:t]; prior = prior[~np.isnan(prior)]
                if len(prior) < min_prior:
                    continue
                w = prior[-window:]
                expected = float(np.median(w))
                lw = np.log1p(w)
                scale = max(1.4826 * np.median(np.abs(lw - np.median(lw))), floor)
                z = (np.log1p(x[t]) - np.log1p(expected)) / scale
                pct = float(np.mean(prior < x[t]) + 0.5 * np.mean(prior == x[t]))
                rows.append(dict(state_id=states[s], disease=DISEASES[d], year=int(yr),
                                 observed=x[t], expected=expected, zscore=float(z), pctile=pct))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    states, years, Y, _ = get_data()
    df = compute_anomalies(Y, years, states)
    df.to_csv("data/processed/anomaly_annual.csv", index=False)
    print(len(df), "rows ->", "data/processed/anomaly_annual.csv")
