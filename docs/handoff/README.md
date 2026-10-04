# VectorWatch India: Handoff from Person 2 (ST-GNN) to Persons 1, 3, 4, 5, 6

**Phase owner:** Person 2, Spatio-Temporal GNN Engineer
**Phase status:** Complete. Graph, models, ablations, annual checkpoint and forecast file are delivered.
**Downstream owners:** Persons 1, 3, 4, 5, 6. Each has their own file in this folder.

---

## 1. Where the project stands

The pipeline works end to end, but the data cannot support the original claims.

- **Monthly disease values are not real.** Person 1's tensor splits every annual total with one fixed curve (weights 2, 2, 3, 4, 5, 8, 18, 22, 18, 10, 5, 3 %). Month-to-month ratios are identical for every state and year. Any model that sees last month's value can reproduce Feb to Dec exactly, so monthly accuracy numbers are artefacts.
- **The real information is the annual total.** About 1,600 real numbers exist in total (10 to 21 observed years per state and disease).
- **On annual totals the ST-GNN is statistically tied with "same as last year" (persistence).** This holds across graph definitions, GCN vs GAT, window length, weather on/off and horizons. 13 rolling-origin folds, 2010 to 2022, bootstrap intervals on log-error.
- **Weather is now real** (NASA POWER, one centroid per state). It points the right way (log-MAE 0.828 vs 0.852 without weather) but the interval includes zero.
- **The limit is the data, not the code.** Rerunning the same scripts on real monthly or newer annual data is how the project can still get a positive result.

---

## 2. Ground rules for everyone

1. **Never report monthly forecast accuracy, monthly anomaly detection, lead time in months or monthly early-warning performance** from the current tensor.
2. **Always show persistence (last year's value) next to any model forecast.** Never present the model as better than persistence.
3. **Respect the `confidence` and `notes` columns** in the forecast file. 64 of 324 rows are `low`.
4. **Join on `state_id`**, in the order of `configs/states.txt`. Do not edit that file.
5. **No leakage.** Anything computed for test year Y may only use years before Y.
6. **Say "no evidence of" or "statistically tied", not "does not work".** The experiment hasn't been run on real monthly data.

---

## 3. Repo layout and branches

Run every command from the repository root. Create your branch from `main`:

```bash
git checkout main && git pull
git checkout -b person3/anomaly-eval     # use your own name
```

| Person | Branch name | Writes under |
|---|---|---|
| 1 | `person1/data-followups` | `src/ingestion/`, `data/raw/`, `data/processed/` |
| 3 | `person3/anomaly-eval` | `src/anomaly/`, `src/evaluation/`, `src/models/sequence/`, `notebooks/` |
| 4 | `person4/environment` | `src/ingestion/weather/`, `src/features/`, `data/raw/geography/` |
| 5 | `person5/backend-risk` | `src/api/`, `src/risk/`, `src/explainability/`, `models/registry.json` |
| 6 | `person6/frontend-mlops` | `frontend/`, `docker/`, `.github/workflows/` |

Person 2's code lives in `models/stgnn/`. **Do not edit it.** If you need a change, ask Person 2 (section 9). Open a pull request into `main` for every merge, and have one other person look at it.

---

## 4. What Person 2 delivered

| Path | What it is |
|---|---|
| `configs/states.txt` | Fixed node order: 36 states/UTs, alphabetical |
| `data/processed/graph/adjacency_{raw,norm,identity}.npy` | Land-border graph (71 edges), normalized and identity versions |
| `data/processed/graph/adjacency_{knn,dist,hybrid}_{raw,norm}.npy` | Alternative graphs |
| `models/stgnn/model.py` | Dense GCN and GAT layers, GRU, residual head |
| `models/stgnn/annual_data.py` | `load_annual`, `load_graph`, `forecast_table` helpers |
| `models/stgnn/annual.py` | Experiments, rolling evaluation, forecast and checkpoint mode |
| `models/stgnn/predict.py` | Loads the checkpoint and writes a forecast CSV |
| `models/stgnn/compare_runs.py`, `run_all.sh` | Reproduce the comparison table |
| `models/stgnn/download_weather.py` | NASA POWER download (one centroid per state) |
| `models/stgnn/annual_final.pt` | **The only checkpoint to register** |
| `models/stgnn/archive_monthly/` | Monthly-model checkpoints. **Invalid, do not use.** |
| `data/processed/stgnn_forecast_final.csv` | Forecasts for 2023, 2024, 2025 |
| `data/processed/stgnn_annual_<tag>_h<h>.csv` | Per-fold results of each experiment |
| `data/processed/stgnn_annual_<tag>_h<h>_pairs.csv` | Per state/disease predictions per fold |
| `data/processed/pan_india_state_enriched_tensor_real.csv` | Person 1's tensor with real weather |
| `docs/person2_handoff.md` | Person 2's own handoff note |

### Results (log-MAE, lower is better; persistence = last year's value)

| Variant | log-MAE | Difference vs persistence [95% CI] |
|---|---|---|
| Persistence | 0.846 | n/a |
| GCN, land-border graph, real weather | 0.828 | -0.018 [-0.060, +0.026] |
| Without weather | 0.852 | +0.006 [-0.014, +0.029] |
| No graph | 0.836 | -0.009 [-0.036, +0.014] |
| kNN / distance / hybrid graph | 0.827 / 0.824 / 0.828 | all intervals include 0 |
| GAT | 0.859 | worse than GCN by 0.031 [+0.006, +0.058] |
| Separate model per disease | 0.891 | worse than shared by 0.063 [0.000, +0.131] |
| 1 / 2 / 3 years ahead | 0.828 / 1.127 / 1.259 | persistence 0.846 / 1.093 / 1.240 |

---

## 5. Shared data contract

### Annual tensor (use this in all code)

```python
import sys; sys.path.insert(0, "models/stgnn")
from annual_data import load_annual
states, years, Y, W = load_annual("data/processed/pan_india_state_enriched_tensor_real.csv")
# years: 1999..2022 (24 values)
# Y: [24 years, 36 states, 3 diseases (dengue, malaria, chikungunya)] in cases, NaN = not reported
# W: [24, 36, 2] annual mean temperature and annual total rainfall
```

`load_annual` treats an annual total of exactly 0 as missing when that series has ever reached 100 or more. This is a heuristic that Person 1 is asked to confirm. `zero_missing=0` turns it off. `annual_data.py` imports `torch`, so install `requirements-stgnn.txt` first.

### Forecast file `data/processed/stgnn_forecast_final.csv`

| Column | Meaning |
|---|---|
| `state_id`, `disease` | Node and `dengue`, `malaria` or `chikungunya` |
| `year`, `horizon` | Target year and years ahead (1 to 3) |
| `forecast` | Whole cases |
| `persistence` | Last observed year's value (empty if missing) |
| `confidence` | `normal` or `low` |
| `notes` | `;`-separated: `no_last_year_value`, `short_history`, `capped_at_historical_max` |
| `n_years_observed` | Real annual values available |

Horizon 1 is the only one to show prominently. Horizons 2 and 3 are no better than persistence.

### Per-fold results `stgnn_annual_<tag>_h<h>.csv`

Columns: `year, n, mae_model, mae_last, mae_avg2, log_model, log_last, log_avg2`. One row per test year (2010 to 2022).

### Per-pair predictions `stgnn_annual_<tag>_h<h>_pairs.csv`

Columns: `year, state_id, disease, true, model, persistence`. One row per state and disease that was scored in that fold.

---

## 6. Metric definitions (use exactly these)

- **Test folds:** test year Y from 2010 to 2022, trained on years before Y only.
- **Scored pairs:** a state and disease are scored in a fold if the the target year and last year are both observed (the mean-of-last-2-years comparator averages whichever of the two previous years is available).
- **MAE:** mean absolute error in cases over the scored pairs.
- **log-MAE:** mean of `|log1p(pred) - log1p(true)|`. Use this as the main metric. Raw MAE is dominated by a few high-burden states.
- **Comparison:** report the per-fold difference (model minus persistence) with a 95% bootstrap interval over folds (10,000 resamples). If the interval includes 0, the result is a tie.
- **Many comparisons:** if you run a dozen variants, one will look significant by chance. Say how many you ran.

---

## 7. Claims you may and may not make

| You may say | You may not say |
|---|---|
| Annual forecasts are statistically tied with persistence | The model predicts outbreaks / early warning works |
| No evidence that the graph helps | Spatial spillover improves forecasts |
| Weather direction is favourable but not significant | Weather drives disease burden |
| Monthly values are a fixed split, so monthly accuracy is not meaningful | Monthly accuracy of X |
| Annual data to 2022 | "Live" or "real-time" |

---

## 8. Dependencies between roles

1. **Person 1** answers: which zeros are real, is 2023+ data available, is real monthly data available (MMIS, IDSP). Everything downstream improves if the answer is yes.
2. **Person 4** delivers state boundaries (needed by Person 6) and better environmental data (tested by Person 2).
3. **Person 3** can start now on annual totals.
4. **Person 5** can start now with the forecast CSV as stub data.
5. **Person 6** can start now with the API contract in `person5.md`, using Person 5's mock responses.

---

## 9. How to ask Person 2 for something

Send a message with: the task, the file you want me to run, and the exact column names. I will rerun `annual.py` and send back the comparison table. Typical requests:

- "Here is a new weather file, please test it."
- "Please add column X as a model input."
- "Real monthly data arrived, please rebuild the monthly pipeline."

---

## 10. Role files

| File | Role |
|---|---|
| `person1.md` | Disease data follow-ups and annual baselines |
| `person3.md` | Anomaly detection, evaluation, sequence baselines |
| `person4.md` | Geography and environment |
| `person5.md` | Backend, risk engine, uncertainty, explanations |
| `person6.md` | Frontend, deployment, MLOps |
