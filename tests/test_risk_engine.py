import numpy as np

from src.risk import engine

CFG = engine.load_config()
PRIOR = list(range(1, 21))   # 20 previous years: 1..20


def test_percentile_is_strict_share_below():
    assert engine.percentile_rank(10.5, PRIOR) == 50.0     # 10 of 20 values are below 10.5
    assert engine.percentile_rank(0, [0, 0, 0, 0, 0, 0]) == 0.0   # flat zero series is not 'high'
    assert engine.percentile_rank(100, PRIOR) == 100.0


def test_level_cutoffs_come_from_config():
    c = CFG["risk"]["cutoffs"]
    assert engine.level_from_percentile(49.9, c) == "Low"
    assert engine.level_from_percentile(50, c) == "Moderate"
    assert engine.level_from_percentile(74.9, c) == "Moderate"
    assert engine.level_from_percentile(75, c) == "High"
    assert engine.level_from_percentile(90, c) == "Very high"


def test_low_confidence_is_never_a_level():
    r = engine.assess(5, 4, "low", ["short_history"], PRIOR, CFG)
    assert r["risk_level"] == "Insufficient evidence"
    assert "short_history" in r["reason"] and "low_confidence" in r["reason"]
    assert r["percentile"] is None            # no number that could be turned into a colour


def test_too_few_prior_years_is_insufficient():
    r = engine.assess(5, 4, "normal", [], [1, 2, 3], CFG)
    assert r["risk_level"] == "Insufficient evidence"
    assert "fewer_than_min_prior_years" in r["reason"]


def test_growth_is_separate_from_level():
    a = engine.assess(15, 5, "normal", [], PRIOR, CFG)     # large growth
    b = engine.assess(15, 15, "normal", [], PRIOR, CFG)    # no growth
    assert a["risk_level"] == b["risk_level"]
    assert a["growth_vs_persistence_pct"] == 200.0
    assert b["growth_vs_persistence_pct"] == 0.0


def test_growth_missing_when_no_persistence():
    assert engine.growth_vs_persistence(10, None) is None
    assert engine.growth_vs_persistence(10, 0) is None
    assert engine.growth_vs_persistence(10, np.nan) is None
