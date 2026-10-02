# Person 2 handoff: ST-GNN (pan-India, 36 nodes)

## What exists
- configs/states.txt: fixed node order (alphabetical, 36 states/UTs)
- data/processed/graph/adjacency_{raw,norm,identity}.npy: land-border graph (71 edges); islands attached to nearest mainland state (Lakshadweep-Kerala, Andaman-West Bengal/Odisha)
- models/stgnn/model.py: dense GCN and GAT layers, GRU, residual head with bounded correction
- models/stgnn/rolling_annual.py: annual forecasting, rolling-origin 2010-2022, ablations, 2023 forecast (--forecast 1)
- models/stgnn/rolling_jan.py: monthly-data January test
- data/processed/stgnn_forecast_2023_handoff.csv: model forecast vs last year

## Results (all on annual totals, 13 rolling folds)
The model is statistically tied with "same as last year". No evidence that the graph (GCN, GAT) or weather helps.

## Data caveats (read before building on this)
1. Monthly disease values are an exact fixed split of annual totals (weights 2,2,3,4,5,8,18,22,18,10,5,3 %). Any monthly result is an artefact; only the annual total is real information.
2. Annual coverage is about 10-21 observed years per state and disease (~1,600 real numbers total).
3. Many zeros (e.g. Maharashtra and Odisha chikungunya 2021-22, all of Ladakh) are probably unreported years stored as 0. models/stgnn/rolling_annual.py treats zeros as missing for series that ever exceeded 100 (heuristic, --zero_missing 0 turns it off). Awaiting Person 1 confirmation.
4. Weather: NASA POWER monthly at one centroid per state (data/processed/pan_india_state_enriched_tensor_real.csv). A single point is a rough stand-in for a whole state.

## For Persons 3-6
- Do not report monthly forecast accuracy or lead time from this tensor.
- Risk and early-warning layers should use annual or seasonal-level signals only until real monthly data exists.
- Always show "persistence (last year)" next to the model forecast.