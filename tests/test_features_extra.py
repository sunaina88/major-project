import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "models/stgnn")
from features_extra import static_channels, env_channels


def test_static_shape_and_log(tmp_path):
    p = tmp_path / "s.csv"
    pd.DataFrame({"state_id": ["a", "b"], "population": [100.0, 1000.0], "urban_fraction": [0.2, 0.4]}).to_csv(p, index=False)
    x = static_channels(["a", "b"], [2000, 2001, 2002], ["population", "urban_fraction"], path=str(p))
    assert x.shape == (3, 2, 2)
    assert np.allclose(x[0, :, 0], np.log1p([100.0, 1000.0]))
    assert np.allclose(x[:, 1, 1], 0.4)


def test_env_aligns_by_year_and_state(tmp_path):
    p = tmp_path / "e.csv"
    pd.DataFrame({"state_id": ["a", "a", "b", "b"], "year": [2000, 2001, 2000, 2001],
                  "mean_rh_pct": [1.0, 2.0, 3.0, 4.0]}).to_csv(p, index=False)
    x = env_channels(["b", "a"], [2000, 2001], ["mean_rh_pct"], path=str(p))
    assert x[0, 0, 0] == 3.0 and x[1, 1, 0] == 2.0


def test_missing_column_stops(tmp_path):
    p = tmp_path / "s.csv"
    pd.DataFrame({"state_id": ["a"], "population": [1.0]}).to_csv(p, index=False)
    with pytest.raises(SystemExit):
        static_channels(["a"], [2000], ["nope"], path=str(p))
