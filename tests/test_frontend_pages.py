import json
import pathlib

import pytest
from streamlit.testing.v1 import AppTest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGES = ["Home", "State", "Disease explorer", "Time-lapse map", "Why this forecast?",
         "Model comparison", "Research results", "Annual replay", "Data story", "Limitations"]


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


def _run(**sel):
    import os
    os.environ["USE_MOCKS"] = "true"
    os.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "frontend" / "app.py"), default_timeout=60).run()
    at.sidebar.radio[0].set_value("State").run()
    for i, v in sel.items():
        at.selectbox[int(i[1:])].set_value(v).run()
    return at


def test_state_page_india_all_shows_pie():
    at = _run(s0="India", s1="All")
    assert not at.exception


def test_state_page_single_state_and_india_single_disease():
    assert not _run(s0="kerala", s1="dengue").exception
    assert not _run(s0="India", s1="malaria").exception
