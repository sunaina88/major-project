# Frontend (Person 6)

Streamlit. The frontend calls only the API (`frontend/api_client.py`); the map outline is the static file `data/processed/india_states.geojson`.

## Run
- Whole stack: `docker compose -f docker/docker-compose.yml up --build`, then open http://localhost:8501
- Local: `python -m src.api.ingest`, `uvicorn src.api.main:app --port 8000`, then `USE_MOCKS=false streamlit run frontend/app.py`
- Offline demo: `streamlit run frontend/app.py` (mocks in `frontend/mocks/`, rebuild with `python scripts/make_mocks.py`)

## Pages and endpoints
| Page | Endpoint |
|---|---|
| Home, State, Disease explorer | `/forecast/{state_id}/{disease}?include_experimental=true` (history, forecast, persistence, interval, risk) |
| Model comparison | `/model-performance` |
| Research results (forest plot, per-year errors) | `/model-performance` (`?tag=&horizon=` for per-year folds) |
| Why this forecast? | `/explanation/{state_id}/{disease}` (needs torch on the API side) |
| Time-lapse map | `/forecast/...` history (1999 to latest year) |
| Annual replay (by year, one state over time) | `/replay` |
| Data story | static `frontend/static/data_story.json` (rebuild: `python scripts/make_data_story.py`) |
| Start here, Limitations | static text |

Rules enforced by tests: persistence beside every forecast, no risk colour on low-confidence rows, geojson ids equal the 36 ids in `configs/states.txt`, every page renders.
