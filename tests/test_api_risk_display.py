import sys, json, pathlib
sys.path.insert(0, "frontend")
from logic import build_display, COLOURS, INSUFF

def test_api_risk_is_used_and_low_confidence_is_grey():
    rows = [
        {"state_id": "a", "disease": "dengue", "horizon": 1, "forecast": 10, "persistence": None,
         "confidence": "low", "notes": "capped_at_historical_max", "risk_level": INSUFF},
        {"state_id": "b", "disease": "dengue", "horizon": 1, "forecast": 20, "persistence": 15.0,
         "confidence": "normal", "notes": None, "risk_level": "Very high"},
    ]
    df = build_display(rows, 1).set_index("state_id")
    assert df.loc["a", "colour"] == COLOURS[INSUFF]
    assert "not available" in df.loc["a", "persistence_text"]
    assert df.loc["b", "risk"] == "Very high"