import json
import pathlib

import pytest
from streamlit.testing.v1 import AppTest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGES = ["Home", "State", "Disease explorer", "Model comparison", "Annual replay", "System status", "Limitations"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_with_mocks(page, monkeypatch):
    monkeypatch.setenv("USE_MOCKS", "true")
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "frontend" / "app.py"), default_timeout=60).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception


def test_geojson_ids_match_states_both_ways():
    ids = {f["properties"]["state_id"] for f in json.load(open(ROOT / "data/processed/india_states.geojson"))["features"]}
    states = {s.strip() for s in open(ROOT / "configs/states.txt") if s.strip()}
    assert ids == states and len(ids) == 36
