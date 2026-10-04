import numpy as np, pandas as pd
from src.evaluation.metrics import mae, log_mae, paired_diff, bootstrap_ci

def test_basic():
    assert mae([1, 3], [2, 5]) == 1.5
    assert abs(log_mae([0], [np.e - 1])) - 1 < 1e-9
    assert np.allclose(paired_diff([1, 2], [2, 2]), [-1, 0])
    m, lo, hi = bootstrap_ci([1, 1, 1, 1])
    assert m == lo == hi == 1

def test_reproduces_person2():
    df = pd.read_csv("data/processed/stgnn_annual_geo_w_h1.csv")
    assert round(df["log_model"].mean(), 3) == 0.828
    assert round(df["log_last"].mean(), 3) == 0.846
