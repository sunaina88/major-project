# Person 2 handoff: ST-GNN (pan-India, 36 nodes)

## Bottom line
On annual state-level totals, no spatio-temporal variant is statistically distinguishable from "same as last year"
(13 rolling-origin folds, 2010-2022, bootstrap CIs on log-error). We found no evidence that the graph helps,
and weather is directionally positive but not significant. The limit is the data, not the code.

## Files
- configs/states.txt: fixed node order (36 states/UTs, alphabetical). Join key is state_id. Do not edit.
- data/processed/graph/: adjacency_{raw,norm,identity}.npy (land borders, 71 edges) and adjacency_{knn,dist,hybrid}_{raw,norm}.npy
- models/stgnn/model.py: dense GCN and GAT layers, GRU, residual head with bounded correction
- models/stgnn/annual_data.py: loaders and forecast-table builder
- models/stgnn/annual.py: experiments (graph, GCN/GAT, weather, single-disease, window, horizon) and forecast/checkpoint mode
- models/stgnn/predict.py: loads models/stgnn/annual_final.pt and writes a forecast CSV
- models/stgnn/compare_runs.py, run_all.sh: reproduces the comparison table
- models/stgnn/download_weather.py: NASA POWER monthly data (one centroid per state)
- models/stgnn/archive_monthly/: monthly-model checkpoints. INVALID, do not use (see caveat 1).
- data/processed/stgnn_forecast_final.csv: forecasts for 2023, 2024, 2025 (horizons 1-3)
  columns: state_id, disease, year, horizon, forecast, persistence, confidence, notes, n_years_observed

## Results (log-MAE, lower is better; persistence = last year's value)
| Variant | log-MAE | diff vs persistence [95% CI] |
|---|---|---|
| Persistence | 0.846 | n/a |
| GCN, geographic graph, real weather | 0.828 | -0.018 [-0.060, +0.026] |
| No weather | 0.852 | +0.006 [-0.014, +0.029] |
| No graph (identity) | 0.836 | -0.009 [-0.036, +0.014] |
| kNN / distance / hybrid graph | 0.827 / 0.824 / 0.828 | -0.018 / -0.022 / -0.018, all CIs include 0 |
| GAT | 0.859 | +0.013 (worse than GCN by 0.031 [+0.006, +0.058]) |
| Separate model per disease | 0.891 | +0.045 (worse than shared by 0.063 [0.000, +0.131]) |
| Window 2 / 5 years | 0.831 / 0.838 | no difference |
| Horizon 2 / 3 years | 1.127 / 1.259 | persistence 1.093 / 1.240: no gain |

## Data caveats (read before building on this)
1. Monthly disease values are an exact fixed split of annual totals (weights 2,2,3,4,5,8,18,22,18,10,5,3 %).
   Any monthly accuracy is an artefact; only annual totals are real information. Do not report monthly forecast
   accuracy, anomaly lead time or monthly early-warning performance from this tensor.
2. About 1,600 real annual numbers in total (10-21 observed years per state and disease).
3. Many zeros (e.g. Maharashtra and Odisha chikungunya 2021-22, all of Ladakh) are probably unreported years stored as 0.
   annual.py treats zeros as missing for series that ever exceeded 100 (heuristic; --zero_missing 0 turns it off).
   Awaiting Person 1 confirmation.
4. Weather is NASA POWER at one centroid per state: a rough stand-in for a whole state.
   Person 1's baselines were trained on the synthetic weather and should be rerun on the _real file.
5. Forecasts are capped at each series' historical maximum, so the model cannot forecast a record year.
   Rows with confidence = low (64 of 324) have a missing last-year value, short history, or hit the cap.
6. Person 1's baseline metrics are monthly 2021-22 numbers; our comparison is annual rolling-origin against persistence.
   They are not directly comparable.

## Reproduce
pip install -r requirements-stgnn.txt
python models/stgnn/step2_build_graph.py
python models/stgnn/step2b_more_graphs.py
bash models/stgnn/run_all.sh
python models/stgnn/predict.py

## Instructions for downstream owners
- Person 3: work on annual totals only. Use the same rolling folds (2010-2022) and the same state x disease pairs, and compare with
  data/processed/stgnn_annual_*_h1.csv. Report bootstrap intervals.
- Person 4: improve the weather (grid averages, humidity, population, elevation, NDVI) in the same schema
  (state_id, start_date). Deliver a new CSV; rerun annual.py --data <file> to test it.
- Person 5: serve stgnn_forecast_final.csv. Always return persistence beside the forecast, surface confidence and notes,
  and build uncertainty from the per-fold errors in stgnn_annual_*_h1.csv (wide, and honest). Register only models/stgnn/annual_final.pt.
- Person 6: label data as annual, latest 2022. No monthly replay or lead-time screens until real monthly data exists.
  Show persistence next to the model and add a limitations panel quoting the caveats above. Treat horizons 2-3 as experimental.
- Person 1: newer annual data (2023+), real monthly data (MMIS, IDSP), and confirmation of which zeros are unreported.