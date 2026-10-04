import numpy as np, pandas as pd
from src.evaluation.backtest import run_backtest, persistence_fn
from src.anomaly.annual import compute_anomalies

def _toy(seed=0, S=4):
    rng = np.random.default_rng(seed)
    years = np.arange(1999, 2023)
    Y = np.exp(rng.normal(6, 1, (24, S, 3)))
    return years, Y, [f"s{i}" for i in range(S)]

def test_avg2_uses_nanmean():
    years, Y, states = _toy()
    t = list(years).index(2015)
    Y[t - 2, 0, 0] = np.nan                      # year before last is missing for one series
    f, p = run_backtest(Y, years, states, persistence_fn, test_years=[2015])
    m = ~np.isnan(Y[t]) & ~np.isnan(Y[t - 1])
    avg2 = np.nanmean(np.stack([Y[t - 1], Y[t - 2]]), 0)[m]
    expected = np.mean(np.abs(np.log1p(avg2) - np.log1p(Y[t][m])))
    assert abs(f.log_avg2.iloc[0] - expected) < 1e-12
    assert f.n.iloc[0] == m.sum()                # series with one missing earlier year is still scored

def test_backtest_never_sees_future():
    years, Y, states = _toy()
    seen = []
    def spy(Ytr, ytr, yr):
        seen.append((len(Ytr), ytr.max(), yr)); return Ytr[-1]
    run_backtest(Y, years, states, spy)
    assert all(ymax < yr and n == yr - 1999 for n, ymax, yr in seen)

def test_anomaly_has_no_leakage():
    years, Y, states = _toy()
    a = compute_anomalies(Y, years, states)
    Y2 = Y.copy(); Y2[list(years).index(2012) + 1:] *= 50      # change everything after 2012
    b = compute_anomalies(Y2, years, states)
    cols = ["state_id", "disease", "year"]
    A = a[a.year <= 2012].sort_values(cols).reset_index(drop=True)
    B = b[b.year <= 2012].sort_values(cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(A, B)
