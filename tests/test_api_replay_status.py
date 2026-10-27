import json
import pytest
from fastapi.testclient import TestClient
from src.api import db
from src.api.main import app, get_db


@pytest.fixture()
def client(tmp_path):
    conn = db.connect(str(tmp_path / "t.db"))
    db.recreate(conn)
    conn.execute("INSERT INTO states VALUES ('kerala', 'Kerala')")
    conn.executemany("INSERT INTO pair_predictions VALUES (?,?,?,?,?,?,?,?)", [
        ("geo_w", 1, 2010, "kerala", "dengue", 100.0, 120.0, 90.0),
        ("geo_w", 1, 2011, "kerala", "dengue", 110.0, 105.0, 100.0),
        ("geo_w", 2, 2011, "kerala", "dengue", 110.0, 95.0, 80.0)])
    conn.commit()
    app.dependency_overrides[get_db] = lambda: conn
    yield TestClient(app)
    app.dependency_overrides.clear()
    conn.close()


def test_replay_default_is_headline_horizon_1(client):
    r = client.get("/replay")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 2
    assert set(rows[0]) == {"year", "state_id", "disease", "true", "model", "persistence"}
    assert rows[0]["true"] == 100.0 and rows[0]["persistence"] == 90.0


def test_replay_filters(client):
    assert len(client.get("/replay?year=2011").json()) == 1
    assert len(client.get("/replay?horizon=2").json()) == 1
    assert len(client.get("/replay?disease=dengue&state_id=kerala").json()) == 2


def test_replay_404s(client):
    assert client.get("/replay?tag=nope").status_code == 404
    assert client.get("/replay?state_id=atlantis").status_code == 404


def test_status_503_when_report_missing(client, tmp_path, monkeypatch):
    monkeypatch.setenv("VW_DRIFT_REPORT", str(tmp_path / "missing.json"))
    r = client.get("/status")
    assert r.status_code == 503 and "drift.py" in r.json()["detail"]


def test_status_returns_report(client, tmp_path, monkeypatch):
    p = tmp_path / "drift.json"
    p.write_text(json.dumps({"newest_year": 2022, "flagged": False}))
    monkeypatch.setenv("VW_DRIFT_REPORT", str(p))
    assert client.get("/status").json() == {"newest_year": 2022, "flagged": False}
