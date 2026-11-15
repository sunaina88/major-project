# Frontend

Streamlit. The frontend calls only the API (`frontend/api_client.py`). The map outline is the static file `data/processed/india_states.geojson`, and the Network and How it works pages also read the graph files in `data/processed/graph/` (copied into the frontend image).

## Run
- Whole stack: `docker compose -f docker/docker-compose.yml up --build`, then open http://localhost:8501
- Local: `python -m src.api.ingest`, `uvicorn src.api.main:app --port 8000`, then `USE_MOCKS=false streamlit run frontend/app.py`
- Offline demo: `streamlit run frontend/app.py` (mocks in `frontend/mocks/`, rebuild with `python scripts/make_mocks.py`)

## Pages and data
| Page | Data |
|---|---|
| Home | `/forecast/{state_id}/{disease}`: risk map, summary figures and table |
| How it works | Forecast and history from the API; graph metrics from `adjacency_raw.npy` |
| State | `/forecast/...` (history, forecast, persistence, 80% interval, risk); compare states, CSV download |
| Disease explorer | History from `/forecast/...`: map by year, top states, national trend |
| Time-lapse map | History 1999 to latest year, fixed log colour scale |
| Why this forecast? | `/explanation/{state_id}/{disease}` (needs torch on the API side) |
| Network | `adjacency_*.npy` graphs with state centroids from the GeoJSON; forecasts and risk from the API |
| Research results | `/model-performance` (forest plot, per-year errors, all variants; `?tag=&horizon=` for folds) |
| Annual replay | `/replay` (all years, by year, one state over time) |
| Data story | static `frontend/static/data_story.json` (rebuild: `python scripts/make_data_story.py`) |
| Limitations | static text |

Rules enforced by tests: persistence beside every forecast, no risk colour on low-confidence rows, GeoJSON ids equal the 36 ids in `configs/states.txt`, every page renders.