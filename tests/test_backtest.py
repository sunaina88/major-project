import numpy as np, pandas as pd
from src.evaluation.data import get_data
from src.evaluation.backtest import run_backtest, persistence_fn

def test_persistence_matches_person2():
    states, years, Y, _ = get_data()
    f, _ = run_backtest(Y, years, states, persistence_fn)
    g = pd.read_csv("data/processed/stgnn_annual_geo_w_h1.csv")
    assert (f.n.values == g.n.values).all()
    assert round(f.log_last.mean(), 3) == round(g.log_last.mean(), 3) == 0.846
