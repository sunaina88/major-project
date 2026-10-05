"""The ONLY place the frontend gets data. USE_MOCKS=true -> mocks, else Person 5's API.
Endpoint paths below are guesses: align them with person5.md."""
import os, json, pathlib, requests
MOCKS = pathlib.Path(__file__).parent / "mocks"
USE_MOCKS = os.getenv("USE_MOCKS", "true").lower() == "true"
API = os.getenv("API_URL", "http://localhost:8000")
ENDPOINTS = {"forecast": "/forecast", "history": "/history", "replay": "/replay",
             "model_performance": "/model-performance", "drift_report": "/status"}

def get(name):
    if USE_MOCKS:
        return json.load(open(MOCKS / f"{name}.json"))
    r = requests.get(API + ENDPOINTS[name], timeout=10)
    r.raise_for_status()
    return r.json()
