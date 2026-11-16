import pytest
from fastapi.testclient import TestClient

from src.api import environment
from src.api.main import app

client = TestClient(app)


def test_state_profile_has_all_states_in_order():
    rows = client.get("/state-profile").json()
    assert [r["state_id"] for r in rows] == environment.STATE_IDS and len(rows) == 36
    assert all(r["population_year"] == 2011 for r in rows)


def test_climate_monthly_filters():
    r = client.get("/climate/monthly", params={"state_id": "kerala", "year": 2020, "month": 7}).json()
    assert len(r) == 1 and r[0]["state_id"] == "kerala" and r[0]["n_points"] >= 1
    assert len(client.get("/climate/monthly", params={"year": 2020, "month": 7}).json()) == 36


def test_climate_annual_one_state():
    r = client.get("/climate/annual", params={"state_id": "assam"}).json()
    assert [x["year"] for x in r] == sorted(x["year"] for x in r) and r[0]["state_id"] == "assam"
    assert {"mean_temp_c", "total_rain_mm", "monsoon_rain_mm", "hottest_month_temp_c"} <= set(r[0])


@pytest.mark.parametrize("url", ["/climate/monthly", "/climate/annual", "/state-profile"])
def test_unknown_state_is_404(url):
    assert client.get(url, params={"state_id": "atlantis"}).status_code == 404


def test_bad_month_is_422_and_empty_selection_404():
    assert client.get("/climate/monthly", params={"month": 13}).status_code == 422
    assert client.get("/climate/monthly", params={"year": 1800}).status_code == 404