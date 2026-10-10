import json
import os

import pytest

pytest.importorskip("torch")     # ingest uses Person 2's load_annual, which imports torch
from fastapi.testclient import TestClient  # noqa: E402

from src.api import ingest  # noqa: E402
from src.api.config import DISEASES, REPO_ROOT  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("db") / "test.db")
    mp = pytest.MonkeyPatch()
    mp.setenv("VW_DB_PATH", path)
    ingest.main()
    from src.api.main import app
    yield TestClient(app)
    mp.undo()


STATES = open(REPO_ROOT / "configs" / "states.txt").read().split("\n")


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["latest_observed_year"] == 2022
    assert r["data_granularity"] == "annual" and "live" not in r["data_note"].lower()


def test_states_in_fixed_order(client):
    ids = [s["state_id"] for s in client.get("/states").json()]
    assert ids == STATES and len(ids) == 36


def test_historical(client):
    r = client.get("/historical/kerala").json()
    assert set(r["history"]) == set(DISEASES)
    assert client.get("/historical/atlantis").status_code == 404


def test_forecast_default_is_horizon_1_with_persistence_confidence_notes(client):
    for d in DISEASES:
        r = client.get(f"/forecast/kerala/{d}").json()
        assert [f["horizon"] for f in r["forecasts"]] == [1]
        for f in r["forecasts"]:
            assert "persistence" in f and "confidence" in f and isinstance(f["notes"], list)
            assert f["experimental"] is False
            lo, hi = f["interval_80"]
            assert lo <= hi
        assert "tied" in r["model"]["summary"]


def test_every_forecast_has_persistence_unless_last_year_missing(client):
    for s in STATES:
        for d in DISEASES:
            r = client.get(f"/forecast/{s}/{d}?include_experimental=true").json()
            assert len(r["forecasts"]) == 3
            for f in r["forecasts"]:
                if f["persistence"] is None:
                    assert "no_last_year_value" in f["notes"]


def test_experimental_horizons_flagged_and_have_no_interval(client):
    r = client.get("/forecast/kerala/malaria?include_experimental=true").json()
    for f in r["forecasts"]:
        assert f["experimental"] == (f["horizon"] > 1)
        if f["horizon"] > 1:
            assert f["interval_80"] is None


def test_low_confidence_is_insufficient_evidence_everywhere(client):
    n_low = 0
    for d in DISEASES:
        rm = client.get(f"/risk-map?disease={d}").json()
        for s in rm["states"]:
            if s["confidence"] == "low":
                n_low += 1
                assert s["risk_level"] == "Insufficient evidence" and s["notes"]
                assert s["percentile"] is None
            else:
                assert s["risk_level"] in {"Low", "Moderate", "High", "Very high"}
    assert n_low > 0


def test_risk_map_shape_and_years(client):
    rm = client.get("/risk-map?disease=dengue").json()
    assert rm["year"] == 2023 and rm["horizon"] == 1 and rm["experimental"] is False
    assert [s["state_id"] for s in rm["states"]] == STATES
    assert "outbreak" in rm["disclaimer"]
    assert client.get("/risk-map?disease=dengue&year=2024").json()["experimental"] is True
    assert client.get("/risk-map?disease=dengue&year=2030").status_code == 404
    assert client.get("/risk-map?disease=zika").status_code == 422


def test_model_performance_matches_handoff_table(client):
    r = client.get("/model-performance").json()
    h = r["headline"]
    assert h["log_mae_model"] == 0.828 and h["log_mae_persistence"] == 0.846
    assert h["tie"] is True and h["ci_95"][0] < 0 < h["ci_95"][1]
    assert r["variants_run"] >= 10 and "chance" in r["multiple_comparisons_note"]
    assert len(r["folds"]) == 13
    gat = [c for c in r["model_comparisons"] if c["a"] == "GAT"][0]
    assert gat["diff_a_minus_b"] > 0 and gat["tie"] is False      # GAT worse than GCN (README)


def test_openapi_page_exists(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ["/health", "/states", "/historical/{state_id}", "/forecast/{state_id}/{disease}",
              "/risk-map", "/model-performance", "/explanation/{state_id}/{disease}"]:
        assert p in paths


def test_explanation(client):
    r = client.get("/explanation/kerala/dengue").json()
    assert "not a causal effect" in r["label"].lower()
    assert [g["group"] for g in r["groups"]] == ["own_history", "other_diseases", "neighbours", "weather"]
    f = client.get("/forecast/kerala/dengue").json()["forecasts"][0]
    assert round(r["base_forecast"]) == f["forecast"]


def test_ingest_is_idempotent(tmp_path):
    import sqlite3
    path = str(tmp_path / "twice.db")
    os.environ["VW_DB_PATH"] = path
    try:
        ingest.main()
        a = sqlite3.connect(path).execute("select count(*) from forecasts").fetchone()
        ingest.main()
        b = sqlite3.connect(path).execute("select count(*) from forecasts").fetchone()
        tables = ["states", "disease_observations", "forecasts", "risk_scores", "pair_predictions"]
        counts = [sqlite3.connect(path).execute(f"select count(*) from {t}").fetchone()[0] for t in tables]
    finally:
        os.environ.pop("VW_DB_PATH", None)
    assert a == b == (324,) and counts == [36, 1716, 324, 324, 1046]


def test_registry_has_only_annual_final():
    reg = json.load(open(REPO_ROOT / "models" / "registry.json"))
    assert [m["file"] for m in reg["models"]] == ["models/stgnn/annual_final.pt"]
    assert "archive_monthly" not in json.dumps(reg["models"][0]["file"])
