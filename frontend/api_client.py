"""The ONLY place the frontend gets data. USE_MOCKS=true -> mocks, else Person 5's API (docs/api.md)."""
import os, json, pathlib, requests
from concurrent.futures import ThreadPoolExecutor

MOCKS = pathlib.Path(__file__).parent / "mocks"
USE_MOCKS = os.getenv("USE_MOCKS", "true").lower() == "true"
API = os.getenv("API_URL", "http://localhost:8000")
ROOT = pathlib.Path(__file__).parent.parent
STATES = [s.strip() for s in open(ROOT / "configs/states.txt") if s.strip()]
DISEASES = ["dengue", "malaria", "chikungunya"]


class NotAvailable(Exception):
    """The API does not serve this yet."""


def _req(path, **params):
    r = requests.get(API + path, params=params, timeout=30)
    if r.status_code == 404:
        raise NotAvailable(f"The API has no {path} endpoint yet.")
    r.raise_for_status()
    return r.json()


_cache = {}


def _fetch_all():
    """One call per state and disease returns history AND forecasts for all three horizons."""
    if "all" in _cache:
        return _cache["all"]
    pairs = [(s, d) for s in STATES for d in DISEASES]

    def one(p):
        return p, _req(f"/forecast/{p[0]}/{p[1]}", include_experimental="true")

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(one, pairs))
    forecast, history = [], []
    for (s, d), j in results:
        n_obs = sum(1 for h in j["history"] if h["cases"] is not None)
        for h in j["history"]:
            history.append({"state_id": s, "disease": d, "year": h["year"], "cases": h["cases"]})
        for f in j["forecasts"]:
            forecast.append({
                "state_id": s, "disease": d, "year": f["year"], "horizon": f["horizon"],
                "forecast": f["forecast"], "persistence": f["persistence"],
                "confidence": f["confidence"], "notes": ";".join(f["notes"]) or None,
                "n_years_observed": n_obs,
                "risk_level": f["risk_level"], "percentile": f["percentile"],
                "interval_80": f["interval_80"],
                "growth_vs_persistence_pct": f["growth_vs_persistence_pct"],
                "risk_reason": f["risk_reason"], "experimental": f["experimental"]})
    _cache["all"] = {"forecast": forecast, "history": history}
    return _cache["all"]


def get(name):
    if USE_MOCKS:
        return json.load(open(MOCKS / f"{name}.json"))
    if name in ("forecast", "history"):
        return _fetch_all()[name]
    if name == "replay":
        return _req("/replay")                  
    if name == "drift_report":
        return _req("/status")  
    if name == "model_performance":
        j = _req("/model-performance")
        rows = [{"variant": "Persistence", "log_mae": j["headline"]["log_mae_persistence"],
                 "diff": None, "ci": None, "tag": None, "horizon": 1}]
        seen = set()
        for s in j["summary"]:
            v = s["variant"] if s["horizon"] == 1 else f'{s["variant"]} ({s["horizon"]} years ahead, vs its own persistence)'
            if v in seen:
                continue
            seen.add(v)
            rows.append({"variant": v, "log_mae": s["log_mae_model"],
                         "diff": s["diff_vs_persistence"], "ci": s["ci_95"],
                         "tag": s["tag"], "horizon": s["horizon"]})
        return rows                
    raise KeyError(name)


def get_folds(tag, horizon=1):
    """Per-year errors of one variant: rows with year, log_model (model error), log_last (persistence error)."""
    if USE_MOCKS:
        rows = json.load(open(MOCKS / "model_folds.json"))
        return [r for r in rows if r["tag"] == tag and r["horizon"] == horizon]
    return _req("/model-performance", tag=tag, horizon=horizon)["folds"]


def get_explanation(state_id, disease, horizon=1):
    """Occlusion explanation for one state and disease. Raises NotAvailable with a readable reason."""
    if USE_MOCKS:
        path = MOCKS / f"explanation_{state_id}_{disease}.json"
        if not path.exists():
            raise NotAvailable("Demo mode has an explanation for Kerala dengue only. Start the API to see the others.")
        return json.load(open(path))
    r = requests.get(f"{API}/explanation/{state_id}/{disease}", params={"horizon": horizon}, timeout=120)
    if r.status_code in (404, 503):
        try:
            detail = r.json().get("detail", "")
        except ValueError:
            detail = ""
        raise NotAvailable(detail or "The API cannot explain this forecast.")
    r.raise_for_status()
    return r.json()
