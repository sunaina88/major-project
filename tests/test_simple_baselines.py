import numpy as np
import pandas as pd

import scripts.evaluate_simple_baselines as baseline_script
from scripts.evaluate_simple_baselines import (
    _damped_holt_forecast,
    _ses_forecast,
    _shrunken_trend_forecast,
    make_predictor,
)


def test_ses_constant_history_stays_constant():
    assert _ses_forecast(np.array([2.0, 2.0, 2.0]), 0.5) == 2.0


def test_shrunken_trend_uses_only_training_observations():
    forecast = _shrunken_trend_forecast(
        np.array([2000, 2001, 2002]),
        np.array([1.0, 2.0, 3.0]),
        shrinkage=0.5,
        target_year=2003,
    )
    assert forecast == 3.5


def test_damped_holt_forecast_is_below_undamped_linear_extrapolation():
    forecast = _damped_holt_forecast(
        np.array([2000, 2001, 2002]),
        np.array([1.0, 2.0, 3.0]),
        target_year=2003,
    )
    assert 3.0 < forecast < 4.0


def test_persistence_predictor_returns_last_observed_value():
    history = np.array([[[1.0]], [[np.nan]], [[4.0]]])
    prediction = make_predictor("persistence")(
        history, np.array([2000, 2001, 2002]), 2003
    )
    assert prediction[0, 0] == 4.0


def test_baseline_forecast_is_capped_at_training_max():
    history = np.array([[[1.0]], [[10.0]], [[100.0]]])
    prediction = make_predictor("shrink_trend_0.75")(
        history, np.array([2000, 2001, 2002]), 2003
    )
    assert prediction[0, 0] <= 100.0


def test_evaluate_writes_only_prefixed_outputs(monkeypatch, tmp_path):
    years = np.arange(1999, 2005)
    targets = np.array([10.0, 12.0, 15.0, 13.0, 16.0, 19.0])[:, None, None]
    monkeypatch.setattr(
        baseline_script,
        "get_data",
        lambda _path: (["test_state"], years, targets, None),
    )
    prefix = tmp_path / "simple_baselines"

    baseline_script.evaluate("unused.csv", 2001, str(prefix))

    folds = pd.read_csv(tmp_path / "simple_baselines_folds.csv")
    pairs = pd.read_csv(tmp_path / "simple_baselines_pairs.csv")
    summary = pd.read_csv(tmp_path / "simple_baselines_summary.csv")
    assert folds.columns.is_unique
    assert folds.year.tolist() == [2001, 2002, 2003, 2004]
    assert len(pairs) == 4
    assert "persistence" in summary.method.values
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "simple_baselines_folds.csv",
        "simple_baselines_pairs.csv",
        "simple_baselines_summary.csv",
    ]
