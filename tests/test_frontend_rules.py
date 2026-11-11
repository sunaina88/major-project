import sys, json, pathlib
sys.path.insert(0, "frontend")
from logic import build_display, COLOURS, INSUFF

ROWS = json.load(open(pathlib.Path("frontend/mocks/forecast.json")))

def test_every_forecast_has_persistence_text():
    for h in (1, 2, 3):
        assert (build_display(ROWS, h)["persistence_text"].str.len() > 0).all()

def test_low_confidence_has_no_risk_colour():
    df = build_display(ROWS, 1); low = df[df.confidence == "low"]
    assert (low.risk == INSUFF).all() and (low.colour == COLOURS[INSUFF]).all()

def test_normal_rows_get_a_risk_level():
    df = build_display(ROWS, 1); assert (df[df.confidence == "normal"].risk != INSUFF).all()


import re
from logic import closer_years, is_tie, reported_grid, verdict


def test_tie_and_verdict():
    assert is_tie([-0.02, 0.05]) and verdict([-0.02, 0.05]) == "tie"
    assert verdict([-0.09, -0.01]) == "better than persistence"
    assert verdict([0.01, 0.09]) == "worse than persistence"
    assert is_tie(None)


def test_closer_years_counts():
    d = {"true": [100, 100, 100], "model": [100, 50, 100], "persistence": [50, 100, 100]}
    assert closer_years(d) == (1, 1, 1)


def test_reported_grid_marks_gaps():
    import pandas as pd
    h = pd.DataFrame([{"state_id": "kerala", "disease": "dengue", "year": 2013, "cases": 5.0}])
    assert reported_grid(h, "dengue", ["kerala"], [2013, 2014]) == [[1, 0]]


def test_extra_pages_avoid_overstated_wording():
    text = open("frontend/extra_pages.py").read().lower()
    assert not re.search(r"\blive\b|real-time|predicts? outbreaks?", text)
