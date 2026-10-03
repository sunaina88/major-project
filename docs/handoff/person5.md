# Person 5: Backend, Risk Engine, Uncertainty and Explainability

**Role:** you turn model outputs into a service: database, API, risk levels, intervals and explanations.
**Branch:** `person5/backend-risk`
**Read first:** `handoff/README.md`.

---

## 1. What you build on

- `data/processed/stgnn_forecast_final.csv` holds forecasts for 2023 to 2025 (324 rows). Use it as your data source from day one.
- `models/stgnn/annual_final.pt` and `models/stgnn/predict.py` regenerate the same forecast from a checkpoint. `predict.py` prints `matches the CSV written at training time: True`.
- Forecasts are **annual**. The model is tied with persistence, so the API must always return persistence next to the forecast.
- Person 3 will deliver `data/processed/anomaly_annual.csv`. Person 4 will deliver environmental files. Treat both as optional inputs until they exist.

---

## 2. Tasks in priority order

### Task 1: Database and loader

1. Use PostgreSQL (with PostGIS only if Person 6 needs it) or SQLite for development. Keep the connection string in an environment variable.
2. Tables: `states(state_id, state_name)`, `disease_observations(state_id, disease, year, cases)`, `forecasts(state_id, disease, year, horizon, forecast, persistence, confidence, notes, n_years_observed)`, `fold_results(tag, horizon, year, n, mae_model, mae_last, log_model, log_last)`, `pair_predictions(tag, horizon, year, state_id, disease, true, model, persistence)`, `risk_scores(...)`, `model_registry(...)`.
3. Write `src/api/ingest.py`, idempotent (running it twice gives the same database). Load observations through `load_annual` (see `handoff/README.md` section 5) and the other tables from the CSVs.

### Task 2: API

Build FastAPI under `src/api/`. Minimum endpoints:

| Endpoint | Returns |
|---|---|
| `GET /health` | Status, data version, latest observed year |
| `GET /states` | The 36 states with `state_id` |
| `GET /historical/{state_id}` | Annual cases per disease |
| `GET /forecast/{state_id}/{disease}` | History, forecasts (horizon 1 to 3), persistence, interval, confidence, notes |
| `GET /risk-map?disease=&year=` | Risk level per state |
| `GET /model-performance` | Per-fold and summary results with intervals |
| `GET /explanation/{state_id}/{disease}` | Input contributions (Task 5) |

Example of the shape (values illustrative):

```json
{
  "state_id": "kerala", "disease": "dengue",
  "history": [{"year": 2021, "cases": 0}, {"year": 2022, "cases": 0}],
  "forecasts": [{
    "year": 2023, "horizon": 1, "forecast": 0, "persistence": 0,
    "interval_80": [0, 0], "confidence": "normal", "notes": []
  }],
  "model": {"name": "annual_final", "summary": "statistically tied with persistence"}
}
```

Always include `persistence`, `confidence` and `notes`. Document the API with the automatic OpenAPI page and write tests in `tests/`.

### Task 3: Risk engine

Write `src/risk/engine.py`. Risk is a documented rule, not an arbitrary colour.

1. For each state, disease and year, compute the **percentile of the forecast among that state's previous observed years**.
2. Map it to `Low` (< 50), `Moderate` (50 to 75), `High` (75 to 90), `Very high` (≥ 90). These cut-offs are a configuration in `configs/risk.yaml`, not a finding.
3. Add the growth relative to persistence as a separate field, not blended into the level.
4. **If `confidence` is `low`, return `Insufficient evidence` as the level** and show the notes. Do not hide the reason.
5. Put the full method in `docs/risk_method.md`: inputs, thresholds, and what the level does not mean.
6. Treat any weighted composite (trend, anomaly, environment) as an experiment. Do not add it unless Person 3 and 4's inputs exist and you can say how you validated the weights.

### Task 4: Uncertainty

Use `data/processed/stgnn_annual_geo_w_h1_pairs.csv` (columns `year, state_id, disease, true, model, persistence`).

1. Compute the error `e = log1p(model) - log1p(true)` for each pair.
2. Take quantiles of `e` (for example 10% and 90%, optionally per disease). The interval for a forecast `f` is `expm1(log1p(f) - q90)` to `expm1(log1p(f) - q10)`.
3. **Check calibration on held-out folds**: build the quantiles from folds before year Y and measure how often the true value falls inside the interval in year Y. Report the observed coverage next to the nominal 80%.
4. Expect wide intervals (typical log error is about 0.8). Show them honestly.

The per-fold results files alone are not enough for this, because they hold fold averages, not individual errors. The `_pairs.csv` file is the one to use.

### Task 5: Explanations

Gradient-based SHAP does not fit the dense GCN or GRU well, so use occlusion. Load the checkpoint like `predict.py` does and, for one state and disease, measure how the forecast changes when you zero out one group of inputs at a time:

| Group | What to zero |
|---|---|
| Own history | The state's own disease features |
| Other diseases | The other two diseases' features for that state |
| Neighbours | Features of the neighbouring states (use `adjacency_raw.npy`) |
| Weather | The two weather features |

Return the change in the forecast per group. Label it clearly as "model attribution, not a causal effect". With a model that ties persistence, the honest summary is often "mostly last year's value".

### Task 6: Model registry

Create `models/registry.json` with one entry for `annual_final.pt`: version, training years, features, metrics (summary of the table in `handoff/README.md` section 4), git commit and timestamp. **Register only `annual_final.pt`.** The files in `models/stgnn/archive_monthly/` are invalid.

---

## 3. Deliverables

- `src/api/`, `src/risk/`, `src/explainability/`, `configs/risk.yaml`.
- Database schema and `ingest.py`.
- `docs/risk_method.md` and `docs/api.md`.
- `models/registry.json`.
- Tests and a calibration report for the intervals.

## 4. Definition of done

- A fresh clone can run `ingest.py` and start the API with two commands, documented in `docs/api.md`.
- Every forecast response includes persistence, confidence and notes.
- The interval calibration report is committed, even if coverage is poor.
- Person 6 has the OpenAPI page and mock data.

## 5. Pitfalls

- Do not turn a `low` confidence row into a colour. Show it as insufficient evidence.
- Do not expose horizons 2 and 3 by default. They are no better than persistence.
- Do not present a risk level as a prediction of an outbreak.
- Do not query CSV files from the frontend. All access goes through the API.
