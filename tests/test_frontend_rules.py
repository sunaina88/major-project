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
