"""Rolling-origin backtest. predict_fn only ever sees Y[:t], so leakage is impossible by construction."""
import numpy as np, pandas as pd
from .data import DISEASES
from .metrics import mae, log_mae

COLS = ["year", "n", "mae_model", "mae_last", "mae_avg2", "log_model", "log_last", "log_avg2"]

def scored_mask(Y, t):
    # Matches Person 2's code (models/stgnn/annual.py): target observed and last year observed.
    # (The handoff text says "last two years", but the code only requires the last year; avg2 uses nanmean.)
    return ~np.isnan(Y[t]) & ~np.isnan(Y[t - 1])

def run_backtest(Y, years, states, predict_fn, test_years=range(2010, 2023)):
    """predict_fn(Y_train[t,S,D], years_train, target_year) -> array [S,D] in cases.
    Returns (per_fold_df, pairs_df) with the same columns as Person 2's files."""
    years = list(years)
    rows, pairs = [], []
    for yr in test_years:
        t = years.index(yr)
        m = scored_mask(Y, t)
        if m.sum() == 0:
            continue
        pred = np.asarray(predict_fn(Y[:t].copy(), np.array(years[:t]), yr), float)
        true, last, avg2 = Y[t][m], Y[t - 1][m], ((Y[t - 1] + Y[t - 2]) / 2)[m]
        p = pred[m]
        rows.append(dict(year=yr, n=int(m.sum()),
                         mae_model=mae(p, true), mae_last=mae(last, true), mae_avg2=mae(avg2, true),
                         log_model=log_mae(p, true), log_last=log_mae(last, true), log_avg2=log_mae(avg2, true)))
        for s, d in np.argwhere(m):
            pairs.append(dict(year=yr, state_id=states[s], disease=DISEASES[d],
                              true=Y[t, s, d], model=pred[s, d], persistence=Y[t - 1, s, d]))
    return pd.DataFrame(rows, columns=COLS), pd.DataFrame(pairs)

def persistence_fn(Ytr, ytr, yr):
    return Ytr[-1]

def median5_fn(Ytr, ytr, yr):
    return np.nanmedian(Ytr[-5:], axis=0)
