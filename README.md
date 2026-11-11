# VectorWatch India

**Annual forecasting and risk-mapping of dengue, malaria and chikungunya across India's 36 states and union territories, with a spatio-temporal graph neural network, an API and a dashboard.**

A six-person project. Data: annual state-wise case totals, 1999 to 2022 (NCVBDC / NDAP exports), plus NASA POWER weather.

---

## Status 

The whole pipeline works end to end: data, graph, model, evaluation, API, risk levels, dashboard. The central research result is **negative and well measured**: on annual totals, the graph neural network is **statistically tied with the simplest possible forecast, "next year equals last year"** (persistence). The limit is the data, not the code. The source reports only annual totals, so the monthly values supplied with the dataset are an exact fixed split of those totals. About 1,600 real numbers exist in total. This repository is built so that the same scripts can be rerun if real monthly or newer data arrives.

---

## Key findings

1. **The monthly data was computed.** Every annual total was split into 12 months with one fixed curve (weights 2, 2, 3, 4, 5, 8, 18, 22, 18, 10, 5, 3 percent, January to December). Month-to-month ratios are identical for every state and year, so a one-line rule scores zero error on February to December. Early models that appeared to beat the baselines by a wide margin were learning this curve, not disease.
2. **On annual totals, no variant beats persistence.** 13 rolling-origin test years (2010 to 2022), log-error, 95% bootstrap intervals over years (below). Every interval against persistence includes zero.
3. **Slight evidence that the graph helps.** Land-border, distance, kNN and hybrid graphs and an identity graph (no neighbours) all score within about 0.01 of each other.
4. **Weather is directionally favourable but not significant** (log-MAE 0.828 with NASA POWER weather against 0.852 without; the interval includes zero).
5. **Independent confirmation:** pooled GRU and LSTM baselines, written separately, are also tied with persistence.

### Results (log-MAE, lower is better)

Log-MAE is the mean of `|log1p(forecast) - log1p(actual)|`. Persistence is last year's value.

| Variant | log-MAE | Difference from persistence [95% CI] |
|---|---|---|
| Persistence (last year) | 0.846 | n/a |
| GCN, land-border graph, real weather (headline) | 0.828 | -0.018 [-0.061, +0.026] |
| GCN, no weather | 0.852 | +0.006 [-0.014, +0.029] |
| GCN, no graph (identity) | 0.836 | -0.009 [-0.036, +0.014] |
| GCN, kNN / distance / hybrid graph | 0.827 / 0.824 / 0.828 | all intervals include 0 |
| GAT | 0.859 | +0.013 [-0.027, +0.050] |
| Separate model per disease | 0.891 | +0.045 [-0.009, +0.106] |
| Input window 2 / 5 years | 0.831 / 0.838 | all intervals include 0 |
| 2 / 3 years ahead | 1.127 / 1.259 | persistence 1.093 / 1.240 |

Eleven variants were compared, so one can look better by chance. See [`docs/person2_handoff.md`](docs/person2_handoff.md) and `GET /model-performance`.

---

## Quick start

Requires Python 3.10 or newer. Commands are for Git Bash, macOS or Linux; on Windows PowerShell set environment variables on a separate line first.

```bash
git clone https://github.com/sunaina88/major-project.git
cd major-project

python -m venv .venv
source .venv/Scripts/activate          # macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt -r requirements-stgnn.txt -r requirements-frontend.txt
pip install pytest
python -m pytest tests -q              # run the test suite
```

### Run the API and the dashboard

```bash
python -m src.api.ingest               # builds data/vectorwatch.db (safe to rerun)
python src/monitoring/drift.py --out data/processed/drift_report.json   # only if the file is missing
uvicorn src.api.main:app --port 8000   # API; interactive docs at http://localhost:8000/docs
```

In a second terminal (venv active):

```bash
USE_MOCKS=false API_URL=http://localhost:8000 streamlit run frontend/app.py
```

To run the dashboard on mock data (no API needed): `python scripts/make_mocks.py`, then `streamlit run frontend/app.py`. See [`docs/frontend.md`](docs/frontend.md).

Docker files are in `docker/` (see `docs/frontend.md` and `docs/api.md`).

### Reproduce the experiments

```bash
python models/stgnn/step2_build_graph.py        # land-border graph (hand-built neighbour list)
python models/stgnn/step2b_more_graphs.py       # distance, kNN, hybrid graphs
python models/stgnn/download_weather.py         # NASA POWER weather (needs internet)
bash models/stgnn/run_all.sh                    # every experiment, comparison table, final checkpoint
python models/stgnn/predict.py                  # forecasts from models/stgnn/annual_final.pt
```

All scripts run from the repository root.

---

## How it works

```
NCVBDC / NDAP annual totals        NASA POWER weather
          \                                /
           v                              v
   data tensor (36 states, 1999-2022)  ---->  annual series  [years x states x diseases]
                                                     |
          land-border graph (71 edges) ----->  ST-GNN (graph conv -> GRU over 3 years -> graph conv)
                                                     |
                           shared head, 3 outputs, bounded correction to last year's value
                                                     |
                       forecasts 2023-2025 + prediction intervals + risk levels + attributions
                                                     |
                                    SQLite  ->  FastAPI  ->  Streamlit dashboard
```

- **Nodes:** 36 states and union territories, fixed alphabetical order in `configs/states.txt`. Join key is `state_id`.
- **Graph:** states are linked by land borders; island territories are attached to the nearest mainland state (a modelling choice). Distance, kNN and hybrid graphs are also built.
- **Model:** dense graph convolution (or graph attention) at each time step, a GRU over a 3-year window, a second graph layer, and one shared head for the three diseases. The head predicts a bounded correction to last year's value; forecasts are capped at each series' historical maximum, so the model cannot forecast a record year. Plain PyTorch, runs on CPU.
- **Evaluation:** rolling origin. For each test year from 2010 to 2022, train on earlier years only and compute normalisation from them only. A state and disease is scored when the target year and last year are both observed.
- **Missing data:** missing values stay missing and a masked loss ignores them. An annual total of exactly 0 is treated as missing when that series has ever reached 100 or more (a heuristic, not yet confirmed against the source).
- **Risk level:** the percentile of the forecast among that state's own previous observed years (`Low` under 50, `Moderate` 50 to 75, `High` 75 to 90, `Very high` 90 and above). Rows with low confidence show `Insufficient evidence`. A level compares a state with its own past, not with other states, and can be driven by trend. It is not a prediction of an outbreak.
- **Intervals:** 80% intervals from out-of-sample errors. Observed coverage is 76.9% pooled against the 80% target, and the intervals are very wide (see `docs/calibration_report.md`).
- **Explanations:** occlusion attributions (inputs replaced by averages one group at a time). They show how the forecast changes.

---

## API

Annual data only. Every forecast comes with `persistence`, `confidence` and `notes`. Horizon 1 is the default; horizons 2 and 3 are experimental and carry no interval.

| Endpoint | Returns |
|---|---|
| `GET /health` | Status, data version, latest observed year |
| `GET /states` | The 36 states |
| `GET /historical/{state_id}` | Annual cases per disease |
| `GET /forecast/{state_id}/{disease}` | History, forecasts, persistence, interval, confidence, notes, risk level |
| `GET /risk-map?disease=&year=` | Risk level per state |
| `GET /model-performance` | Variants against persistence with 95% intervals, per-fold rows |
| `GET /explanation/{state_id}/{disease}` | Occlusion attribution per input group |
| `GET /replay` | Backtest: model, actual and persistence per state, disease and year |
| `GET /status` | Data-drift report |

Details: [`docs/api.md`](docs/api.md).

## Dashboard

Streamlit pages: Home (risk map and table), State (history, forecast beside persistence, interval, confidence and reason), Disease explorer, Model comparison, Annual replay (backtest), System status, Limitations. The home map is currently a placeholder tile grid until state boundaries are delivered.

---

## Repository layout

```
configs/            node order (states.txt) and settings
config/             drift-monitor thresholds
data/
  raw/              source exports and downloaded weather
  processed/        tensors, graph files, experiment results, forecasts, anomaly scores
models/
  stgnn/            graph, model, experiments, annual_final.pt, predict.py
  registry.json     registered model (annual_final only)
src/
  ingestion/        disease data and weather ingestion
  models/           baselines and sequence models
  evaluation/       metrics, backtest, early-warning evaluation
  anomaly/          annual anomaly detector
  risk/             risk engine and interval calibration
  explainability/   occlusion attributions
  api/              FastAPI app, database, schemas
  monitoring/       drift monitor
frontend/           Streamlit dashboard (and mocks)
docker/             container files
scripts/            helper scripts
docs/               documentation (see below)
tests/              test suite
notebooks/          figure scripts
logs/               experiment logs
.github/workflows/  continuous integration
```

---

## Documentation

| File | Contents |
|---|---|
| [`docs/data_pipeline.md`](docs/data_pipeline.md) | Original data and baseline pipeline (Person 1) |
| [`docs/person2_handoff.md`](docs/person2_handoff.md) | Graph and model work, results, caveats |
| [`docs/person3_findings.md`](docs/person3_findings.md) | Metrics, anomaly detection, early-warning evaluation, sequence baselines |
| [`docs/api.md`](docs/api.md) | API reference and rules |
| [`docs/risk_method.md`](docs/risk_method.md) | How risk levels are built and what they mean |
| [`docs/calibration_report.md`](docs/calibration_report.md) | Interval calibration |
| [`docs/frontend.md`](docs/frontend.md) | Dashboard structure and how to run it |
| [`docs/handoff/`](docs/handoff/) | Role-by-role instructions and shared rules |

---

## Team and status

| Role | Scope | Status |
|---|---|---|
| Person 1: disease data and baselines | Ingestion, monthly tensor, baselines | Delivered; follow-ups open (newer annual data, real monthly data, which zeros are unreported, Andhra Pradesh rescaling) |
| Person 2: spatio-temporal GNN | Graph, model, experiments, checkpoint, forecasts; also `/replay`, `/status` and the dashboard's API adapter | Delivered |
| Person 3: time series, anomaly, evaluation | Metrics, backtest, anomaly detector, early-warning evaluation, GRU/LSTM baselines | Delivered |
| Person 4: geography and environment | State boundaries (GeoJSON), area-averaged weather, static features | Not started |
| Person 5: backend and risk | Database, API, risk engine, intervals, explanations, registry | Delivered |
| Person 6: frontend and MLOps | Dashboard, containers, CI, drift monitor | Delivered |

---

## Known limitations

1. **Monthly values are modelled.** No monthly forecasting, seasonality, monthly anomaly detection or lead time in months.
2. **Moderate real data:** about 1,600 annual numbers (10 to 21 observed years per state and disease).
3. **Some zeros are probably unreported years** (for example chikungunya in Maharashtra and Odisha in 2021 to 2022, and all of Ladakh).
4. **Andhra Pradesh history has a rescaling break:** its values before a cutoff are whole numbers multiplied by 0.5861509524794264 (dengue through 2009, malaria through at least 2010 in the data seen). The cause is unconfirmed.
5. **Weather is one NASA POWER point per state**, a rough stand-in for large or varied states.
6. **Forecasts are capped at the historical maximum**, which limits any use for early warning.
7. **Risk levels reflect trend** (reported dengue has risen, reported malaria has fallen), so almost all malaria rows show Low and many dengue rows High. Levels do not compare states.
8. **Thirteen test years** give wide uncertainty. Small true effects cannot be ruled out.
9. **Baseline numbers from the original pipeline** are monthly and not comparable with the annual evaluation.

## Future work

- Real monthly surveillance data (for example NCVBDC monthly malaria reports, IDSP weekly reports), then rerun the same scripts.
- Annual data for 2023 onward, giving a true out-of-sample test.
- Confirm which zeros are real; replace the rest with missing values at the source.
- State boundaries for a clickable map; area-averaged weather; humidity, population, elevation.
- District-level data where it exists, for a larger graph.

## How to describe this project

Use "statistically tied with persistence" and "no evidence of", not "does not work". Always show persistence next to a forecast. Do not report monthly results or call the system live or early-warning capable.
