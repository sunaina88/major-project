import numpy as np
import pandas as pd

from src.risk import calibrate, uncertainty


def test_interval_formula_and_direction():
    # model always 2x too high in log terms -> true is below the forecast
    e = np.full(50, np.log(2.0))
    q_lo, q_hi = uncertainty.fit_quantiles(e)
    lo, hi = uncertainty.apply_interval(200, q_lo, q_hi)
    assert lo == hi == 100 or abs(lo - 100) <= 1 and abs(hi - 100) <= 1


def test_zero_error_gives_point_interval():
    lo, hi = uncertainty.apply_interval(123, 0.0, 0.0)
    assert lo == hi == 123


def test_interval_never_negative():
    lo, hi = uncertainty.apply_interval(3, -1.0, 5.0)
    assert lo >= 0 and hi >= lo


def _synthetic_pairs(rng, years, scale_by_year):
    rows = []
    for y in years:
        for i in range(60):
            true = float(rng.integers(10, 5000))
            model = float(np.expm1(np.log1p(true) + rng.normal(0, scale_by_year[y])))
            rows.append({"year": y, "state_id": f"s{i}", "disease": "dengue", "true": true, "model": model})
    return pd.DataFrame(rows)


def test_walk_forward_uses_only_earlier_years():
    rng = np.random.default_rng(1)
    years = list(range(2010, 2016))
    scale = {y: 0.2 for y in years}
    scale[2015] = 3.0                           # last year is much noisier than the past
    pairs = _synthetic_pairs(rng, years, scale)
    table, covered, total = calibrate.walk_forward(pairs, 0.8, False, min_pairs=100)
    last = table[table["year"] == 2015].iloc[0]
    # intervals built from the calm past cannot cover the noisy last year; a leak would
    row_2012 = table[table["year"] == 2012].iloc[0]
    assert last["coverage"] < 0.5
    assert 0.6 <= row_2012["coverage"] <= 1.0


def test_real_pairs_coverage_is_reported_not_assumed():
    pairs = pd.read_csv("data/processed/stgnn_annual_geo_w_h1_pairs.csv")
    table, covered, total = calibrate.walk_forward(pairs, 0.8, False, 100)
    cov = covered / total
    assert 0.6 < cov < 0.95         # roughly near nominal; exact value is in docs/calibration_report.md
