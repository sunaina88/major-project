# VectorWatch India

**Annual forecasting and risk-mapping of dengue, malaria and chikungunya across India's 36 states and union territories, using a spatio-temporal graph neural network (ST-GNN), a REST API and an interactive dashboard.**

A six-person project. Data: annual state-wise case totals, 1999 to 2025 (NCVBDC / NDAP exports), and real monthly weather from NASA POWER averaged over each state (see [How it works](#how-it-works)).

---

## Status

The whole pipeline works end to end: data, graph, model, evaluation, API, risk levels and dashboard.

The central research result is **negative and carefully measured**. In this rolling-origin evaluation, we did not detect a statistically significant difference from persistence ("next year equals last year"): 0.840 vs 0.855 log-MAE, with a 95% interval for the difference of [-0.061, +0.026]. This does not establish that the methods are equivalent.

The disease source reports only annual totals. The monthly disease values supplied with the original dataset are an exact fixed split of the annual totals, so this repository works at annual resolution; the model uses yearly values aggregated from monthly NASA POWER weather. The processed tensor holds about 2,300 reported state-disease-year entries before the model's zero-as-missing heuristic. The scripts can be rerun if real monthly disease data arrives or source revisions are resolved.

---

## Key findings

1. **The monthly disease data was computed.** Every annual total was split into 12 months with one fixed curve (weights 2, 2, 3, 4, 5, 8, 18, 22, 18, 10, 5, 3 percent, January to December). Month-to-month ratios are identical for every state and year, so a one-line rule scores zero error on February to December. Early models that appeared to beat the baselines by a wide margin were learning this curve, not disease.
2. **No detectable difference from persistence for the headline model.** It scores 0.840 vs 0.855 on 16 rolling-origin test years (2010 to 2025); the 95% bootstrap interval for the difference is [-0.061, +0.026].
3. **Graph variants are close.** The no-graph, land-border (headline), hybrid, kNN and distance graphs score 0.841, 0.840, 0.834, 0.827 and 0.818. The distance graph's interval against persistence just excludes zero ([-0.080, -0.001]), but it is one of eleven variants compared and its difference from the headline model ([-0.056, +0.004]) includes zero, so we treat it as exploratory. All other graph variants have intervals that include zero.
4. **Real weather gave a borderline benefit over no weather.** Without weather the score is 0.865; the paired difference from the headline model is +0.025 [+0.001, +0.057]. This is weak evidence, and one of many comparisons.
5. **Extra environmental and demographic features did not produce a detectable improvement** over the base model (see [Feature ablation](#feature-ablation-supplemental)).
6. **Earlier GRU/LSTM runs (2010-2022) were also tied with persistence.** They were not rerun on the current data snapshot.
7. **Simple annual baselines show the same pattern.** SES with alpha 0.8 scores 0.848 vs persistence at 0.855, but the paired difference interval [-0.023, +0.009] includes zero. Other tested smoothing, trend and recent-average variants score worse.
8. **Raw-case MAE is a secondary metric.** On it, several variants fall below persistence with intervals that exclude zero. Raw MAE is dominated by a few high-burden states, so we treat it as exploratory and base conclusions on log-MAE.

### Results (log-MAE, lower is better)

Log-MAE is the mean of `|log1p(forecast) - log1p(actual)|`. Persistence is last year's value.

| Variant | log-MAE | Difference from persistence [95% CI] |
|---|---|---|
| Persistence (last year) | 0.855 | n/a |
| GCN, land-border graph, real weather (headline) | 0.840 | -0.015 [-0.061, +0.026] |
| GCN, no weather | 0.865 | +0.010 [-0.019, +0.044] |
| GCN, no graph (identity) | 0.841 | -0.014 [-0.043, +0.016] |
| GCN, kNN graph | 0.827 | -0.028 [-0.080, +0.025] |
| GCN, distance graph | 0.818 | -0.037 [-0.080, -0.001] |
| GCN, hybrid graph | 0.834 | -0.020 [-0.069, +0.021] |
| GAT | 0.868 | +0.013 [-0.037, +0.056] |
| Separate model per disease | 0.879 | +0.024 [-0.034, +0.084] |
| Input window 2 years | 0.853 | -0.002 [-0.051, +0.049] |
| Input window 5 years | 0.854 | -0.000 [-0.026, +0.028] |
| 2 years ahead | 1.098 | persistence 1.110; -0.012 [-0.037, +0.014] |
| 3 years ahead | 1.276 | persistence 1.282; -0.006 [-0.036, +0.025] |

Eleven main experiment variants were compared, so one can look better by chance. The separate simple-baseline comparison is in `data/processed/simple_baselines_summary.csv`. See [`models/registry.json`](models/registry.json) and `GET /model-performance`.

### Feature ablation (supplemental)

The registered model consumes only disease histories, observed flags and yearly temperature and rainfall. To check whether other available variables help, each group was added to the headline model and compared on the same folds (`docs/feature_ablation_results.txt`).

| Added to the headline model | Difference from persistence [95% CI] | Difference from headline [95% CI] |
|---|---|---|
| Monsoon rainfall | -0.032 [-0.069, -0.002] | -0.018 [-0.039, +0.000] |
| Humidity and wind | -0.020 [-0.058, +0.011] | -0.005 [-0.024, +0.010] |
| Population density and urban share | -0.004 [-0.051, +0.044] | +0.011 [-0.004, +0.027] |
| Elevation | -0.027 [-0.067, +0.008] | -0.012 [-0.025, +0.000] |
| All of the above | -0.012 [-0.063, +0.028] | +0.003 [-0.018, +0.022] |

No group was detectably better than the headline model: every interval against it includes zero or reaches it. Monsoon rainfall and elevation had the most favourable estimates, but adding all groups together was no better than the base, consistent with overfitting on a small dataset. With about twenty comparisons in total, borderline intervals can occur by chance. None of these groups is in the registered model.

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

To run the dashboard on mock data (no API needed): `python -m scripts.make_mocks` and `python -m scripts.make_climate_mocks`, then `streamlit run frontend/app.py`. See [`docs/frontend.md`](docs/frontend.md).

Docker: `docker compose -f docker/docker-compose.yml up --build`, then open http://localhost:8501.

### Reproduce the experiments

```bash
python src/ingestion/weather/download_area_weather.py   # real NASA POWER weather (cached per sample point)
python src/ingestion/weather/apply_area_weather.py      # puts it into the model tensor
python models/stgnn/build_graph.py              # land-border graph (hand-built neighbour list)
python models/stgnn/graphs.py                   # distance, kNN and hybrid graphs
bash models/stgnn/run_all.sh                    # every experiment, comparison table, final checkpoint
python models/stgnn/predict.py                  # forecasts from models/stgnn/annual_final.pt
```

The full data pipeline (disease tensor, weather, zero audit, baselines and experiments) is in `scripts/rerun_disease_pipeline.ps1`. Extra static or environmental columns can be tried with the `--static` and `--env` options of `models/stgnn/annual.py` (see `models/stgnn/features_extra.py`). They are off in the registered run. The earlier synthetic weather generator is kept in `src/ingestion/weather/archive/` for reference only.

All scripts run from the repository root.

---

## How it works

```
NCVBDC / NDAP annual totals        NASA POWER weather (area-averaged)
          \                                /
           v                              v
   data tensor (36 states, 1999-2025)  ---->  annual series  [years x states x diseases]
                                                     |
          land-border graph (71 edges) ----->  ST-GNN (graph conv -> GRU over 3 years -> graph conv)
                                                     |
                           shared head, 3 outputs, bounded correction to last year's value
                                                     |
                       forecasts 2026-2028 + prediction intervals + risk levels + attributions
                                                     |
                                    SQLite  ->  FastAPI  ->  Streamlit dashboard
```

- **Nodes:** 36 states and union territories, fixed alphabetical order in `configs/states.txt`. Join key is `state_id`.
- **Inputs the model consumes (8 per state and year):** log1p annual dengue, malaria and chikungunya cases (normalised), three observed flags, and annual mean temperature and total rainfall from NASA POWER (normalised). Humidity, wind, monsoon rainfall, population and elevation exist in the repository but are not used by the registered model.
- **Weather:** NASA POWER monthly temperature, precipitation, humidity and wind, sampled on a 0.75 degree grid inside each state boundary (up to 30 points) and averaged (`src/ingestion/weather/download_area_weather.py`). `apply_area_weather.py` merges temperature and rainfall into the model tensor, and the model aggregates them to yearly values. NASA POWER values are satellite- and reanalysis-based estimates, not station measurements.
- **Graph:** states are linked by land borders; island territories are attached to the nearest mainland state (a modelling choice). Distance, kNN and hybrid graphs are also built.
- **Model:** dense graph convolution (or graph attention) at each time step, a GRU over a 3-year window, a second graph layer, and one shared head for the three diseases. The head predicts a bounded correction to last year's value; forecasts are capped at each series' historical maximum, so the model cannot forecast a record year. Plain PyTorch, runs on CPU.
- **Evaluation:** rolling origin. For each test year from 2010 to 2025, train on earlier years only and compute normalisation from them only. A state and disease is scored when the target year and last year are both observed.
- **Missing data:** missing values stay missing and a masked loss ignores them. An annual total of exactly 0 is treated as missing when that series has ever reached 100 or more (a heuristic, not yet confirmed against the source).
- **Risk level:** the percentile of the forecast among that state's own previous observed years (`Low` under 50, `Moderate` 50 to 75, `High` 75 to 90, `Very high` 90 and above). Rows with low confidence show `Insufficient evidence`. A level compares a state with its own past, not with other states, and can be driven by trend. It is not a prediction of an outbreak.
- **Intervals:** 80% intervals from out-of-sample errors. Observed pooled coverage is 79.0% against the 80% target, and the intervals are wide (see `docs/calibration_report.md`).
- **Explanations:** occlusion attributions (inputs replaced by averages one group at a time). They show how the forecast changes.

---

## API

Disease data is annual only. Every forecast comes with `persistence`, `confidence` and `notes`. Horizon 1 is the default; horizons 2 and 3 are experimental and carry no interval.

| Endpoint | Returns |
|---|---|
| `GET /health` | Status, data version, latest observed year |
| `GET /states` | The 36 states |
| `GET /historical/{state_id}` | Annual cases per disease |
| `GET /forecast/{state_id}/{disease}` | History, forecast, persistence, interval, confidence, notes, risk level |
| `GET /risk-map?disease=&year=` | Risk level per state |
| `GET /model-performance` | Variants against persistence with 95% intervals, per-fold rows |
| `GET /explanation/{state_id}/{disease}` | Occlusion attribution per input group |
| `GET /replay` | Backtest: model, actual and persistence per state, disease and year |
| `GET /climate/monthly` | Monthly area-averaged weather per state |
| `GET /climate/annual` | Yearly climate summaries per state |
| `GET /state-profile` | Area, population (Census 2011), density, urban fraction, elevation |
| `GET /status` | Data-drift report |

Details: [`docs/api.md`](docs/api.md).

## Dashboard

Streamlit, dark theme, state-boundary maps drawn from `data/processed/india_states.geojson`.

| Page | What it shows |
|---|---|
| Home | Risk map by disease, summary figures, colour key and a searchable table |
| How it works | The pipeline in five steps: data, graph, model, forecast, risk level |
| State | History, forecast beside persistence with its 80% interval, state comparison, CSV download |
| Disease explorer | Reported cases by state and year, top states, national trend |
| Time-lapse map | Reported cases by year, 1999 to latest year |
| Climate & Geography | Monthly weather map, state climate trends, state profile |
| Why this forecast? | Occlusion attributions |
| Network | The state graph over the map, with a build-the-graph slider and neighbour details |
| Research results | Forest plot against persistence, per-year errors, all variants |
| Annual replay | Backtest overview, by year, one state over time |
| Data story | Monthly split, coverage and source revisions |
| Limitations | Plain-language caveats |

Rules enforced by tests: persistence beside every forecast, no risk colour on low-confidence rows, GeoJSON ids equal the 36 ids in `configs/states.txt`, every page renders.

---

## Repository layout

```
configs/            node order (states.txt) and settings
config/             drift-monitor thresholds
data/
  raw/              source exports, state boundaries, cached NASA POWER downloads, static features
  processed/        tensors, graph files, GeoJSON, experiment results, forecasts, anomaly scores
models/
  stgnn/            graph, model, experiments, annual_final.pt, predict.py
  registry.json     registered model (annual_final only)
src/
  ingestion/        disease, weather and geography ingestion
  features/         annual environment and static state features
  models/           baselines and sequence models
  evaluation/       metrics, backtest, early-warning evaluation
  anomaly/          annual anomaly detector
  risk/             risk engine and interval calibration
  explainability/   occlusion attributions
  api/              FastAPI app, database, schemas, environment data
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
| [`docs/data_sources.md`](docs/data_sources.md) | Source exports, 2025 extension, revisions and data caveats |
| [`docs/person2_handoff.md`](docs/person2_handoff.md) | Graph and model work, results, caveats |
| [`docs/person3_findings.md`](docs/person3_findings.md) | Metrics, anomaly detection, early-warning evaluation, sequence baselines |
| [`docs/api.md`](docs/api.md) | API reference and rules |
| [`docs/risk_method.md`](docs/risk_method.md) | How risk levels are built and what they mean |
| [`docs/calibration_report.md`](docs/calibration_report.md) | Interval calibration |
| [`docs/feature_ablation_results.txt`](docs/feature_ablation_results.txt) | Feature-ablation results |
| [`docs/frontend.md`](docs/frontend.md) | Dashboard structure and how to run it |
| [`docs/figures/`](docs/figures/) | Anomaly timelines, early-warning ROC, error by state, forecast vs persistence |
| [`docs/handoff/`](docs/handoff/) | Role-by-role instructions and shared rules |

---

## Team and status

| Role | Scope | Status |
|---|---|---|
| Person 1: disease data and baselines | Ingestion, monthly tensor, baselines | Delivered; follow-ups open (real monthly disease data, which zeros are unreported, Andhra Pradesh rescaling) |
| Person 2: spatio-temporal GNN | Graph, model, experiments, checkpoint, forecasts; also `/replay`, `/status` and the dashboard's API adapter | Delivered |
| Person 3: time series, anomaly, evaluation | Metrics, backtest, anomaly detector, early-warning evaluation, GRU/LSTM baselines | Delivered |
| Person 4: geography and environment | State boundaries (GeoJSON), area-averaged NASA POWER weather (now the model's weather input), static features, feature ablation | Delivered |
| Person 5: backend and risk | Database, API, risk engine, intervals, explanations, registry | Delivered |
| Person 6: frontend and MLOps | Dashboard, containers, CI, drift monitor | Delivered |

---

## Known limitations

1. **Monthly disease values are modelled.** No monthly forecasting, seasonality, monthly anomaly detection or lead time in months.
2. **Limited real data:** about 2,300 reported state-disease-year entries before the zero-as-missing heuristic, with uneven coverage. Dengue values for 2014 to 2016 were added from a parliamentary question table; two states have no value in each of those years.
3. **Some zeros are probably unreported years** (for example chikungunya in Maharashtra and Odisha in 2021 to 2022, and all of Ladakh).
4. **Andhra Pradesh history has a rescaling break:** its values before a cutoff are whole numbers multiplied by 0.5861509524794264 (dengue through 2009, malaria through at least 2010 in the data seen). The cause is unconfirmed.
5. **Weather is estimated, not measured.** NASA POWER values are satellite- and reanalysis-based, averaged over a 0.75 degree sample grid inside each state; very small states have only one or a few points. Yearly aggregation hides within-year timing.
6. **The model uses only temperature and rainfall.** Humidity, wind, monsoon rainfall, population, density and elevation were tested separately and gave no detectable improvement.
7. **Forecasts are capped at the historical maximum**, which limits any use for early warning.
8. **Risk levels reflect trend** (reported dengue has risen, reported malaria has fallen), so almost all malaria rows show Low and many dengue rows High. Levels do not compare states.
9. **Sixteen test years and about twenty comparisons.** Small true effects cannot be ruled out, and borderline intervals (the distance graph, monsoon rainfall, weather versus no weather) may be chance.
10. **2022 source values were revised**, so a clean 2023-2025 holdout is not possible.
11. **Baseline numbers from the original pipeline** are monthly and not comparable with the annual evaluation.

## Future work

- Real monthly disease surveillance data (for example NCVBDC monthly malaria reports, IDSP weekly reports), then rerun the same scripts with monthly weather.
- Station-based or higher-resolution weather, plus vegetation indices (NDVI).
- Confirm which zeros are real; replace the rest with missing values at the source.
- District-level data where it exists, for a larger graph.
- A pre-registered comparison of the distance graph and monsoon rainfall on new data.
