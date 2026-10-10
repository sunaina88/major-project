# VectorWatch India API

Owner: Person 5. Code: `src/api/`. Interactive OpenAPI page: `http://localhost:8000/docs`
(a static copy is in `docs/openapi.json`, example responses are in `docs/api_examples/`).

## Run it (two commands, from the repository root)

```bash
pip install -r requirements-stgnn.txt
python -m src.api.ingest                      # builds data/vectorwatch.db, safe to run again
uvicorn src.api.main:app --port 8000          # start the API
```

`VW_DB_PATH` sets the database file (default `data/vectorwatch.db`). The database is SQLite for development. `/explanation` also needs `models/stgnn/annual_final.pt` and `data/processed/graph/`.

Other commands: `python -m src.risk.calibrate` (rewrites `docs/calibration_report.md`), `python scripts/make_registry.py` (rewrites `models/registry.json`), `python -m src.api.export_examples` (rewrites the examples), `pytest tests`.

## Rules every response follows

- Annual data only. No monthly values, no "live" or "real-time".
- A forecast always comes with `persistence`, `confidence` and `notes`. `persistence` is `null` only when `notes` contains `no_last_year_value`.
- Horizon 1 is the default. Horizons 2 and 3 appear only with `include_experimental=true` (or an explicit later `year` on `/risk-map`), are flagged `experimental`, and have `interval_80: null`.
- `risk_level` is one of `Low`, `Moderate`, `High`, `Very high`, `Insufficient evidence`. For `Insufficient evidence` do not draw a risk colour; show `risk_reason`. `percentile` is `null` then.
- Join on `state_id`; order is `configs/states.txt`.

## Endpoints

| Endpoint | Returns |
|---|---|
| `GET /health` | `status`, `data_version`, `latest_observed_year`, `data_note` (503 if the database is missing) |
| `GET /states` | 36 states: `state_id`, `state_name` |
| `GET /historical/{state_id}` | Annual cases per disease. 404 for an unknown state |
| `GET /forecast/{state_id}/{disease}` | `history`, `forecasts[]` (forecast, persistence, `interval_80`, confidence, notes, risk fields), `model`, `disclaimer`. `disease` is `dengue`, `malaria` or `chikungunya` (else 422) |
| `GET /risk-map?disease=&year=` | One row per state: `risk_level`, `percentile`, `growth_vs_persistence_pct`, forecast, persistence, interval, confidence, notes. `year` defaults to 2023; 404 outside 2023 to 2025 |
| `GET /model-performance?tag=&horizon=` | Summary of every variant against persistence with 95% intervals, model-vs-model comparisons, per-fold rows of `tag` (default `geo_w`, horizon 1), number of variants run |
| `GET /explanation/{state_id}/{disease}?horizon=` | Occlusion attribution per input group. 503 if torch is not installed |

Example (shortened):

```json
{
  "state_id": "andhra_pradesh", "disease": "dengue",
  "forecasts": [{
    "year": 2023, "horizon": 1, "forecast": 5626, "persistence": 6391.0,
    "interval_80": [1484, 20194], "confidence": "low", "notes": ["capped_at_historical_max"],
    "risk_level": "Insufficient evidence", "percentile": null,
    "growth_vs_persistence_pct": -11.97, "risk_reason": ["low_confidence", "capped_at_historical_max"],
    "experimental": false
  }],
  "model": {"name": "annual_final", "summary": "Annual forecasts are statistically tied with persistence ..."}
}
```

## Explanations

Occlusion, not SHAP. For one state and disease, one input group at a time is set to the training average (and its "observed" flag to 0), and the forecast is recomputed:

| Group | What is changed |
|---|---|
| `own_history` | The state's own history of this disease |
| `other_diseases` | The state's history of the other two diseases |
| `neighbours` | Disease history of the states linked in `adjacency_raw.npy` (this graph also links island territories, for example Lakshadweep to Kerala) |
| `weather` | The state's own temperature and rainfall (neighbours' weather is not changed) |

`change_cases` and `change_log` are forecast-without-group minus the full forecast. `share_of_total_change` is each group's absolute log change divided by the sum. Read it as **model attribution, not a causal effect**. Limits: "average" is not the same as "absent", the groups interact, and when a forecast sits at its historical-maximum cap, increases are hidden (`notes` says so). The full-input forecast reproduces `stgnn_forecast_final.csv` exactly (tested for horizon 1).

## Database

SQLite file with the tables `meta`, `states`, `disease_observations`, `forecasts`, `fold_results`, `pair_predictions`, `anomaly_annual` (optional input from Person 3), `risk_scores` (risk fields and intervals per forecast row) and `model_registry`. Observations come from Person 2's `load_annual`, so the zeros-as-missing rule applies. `ingest.py` drops and recreates every table in one transaction, so running it twice gives the same database. To move to PostgreSQL later, the SQL in `src/api/db.py` and `ingest.py` is plain SQL; only the connection code would change.

## Docker note for Person 6

The container needs these files in addition to what `docker/Dockerfile.api` copies now: `data/processed/*.csv` (ingest reads them), `models/registry.json`, and `models/stgnn/annual_final.pt`. Run `python -m src.api.ingest` before `uvicorn` (for example `CMD python -m src.api.ingest && uvicorn ...`). The compose `db` service is not used yet because the API runs on SQLite.
