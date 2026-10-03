# Person 6: Frontend, Deployment and MLOps

**Role:** you build the product people see, and the machinery that builds, tests and ships it.
**Branch:** `person6/frontend-mlops`
**Read first:** `handoff/README.md`, then `person5.md` for the API you will call.

---

## 1. What changed from the original plan

The original plan had monthly replay, lead-time screens and a "live" data status. **None of these can be shown honestly with the current data.** The data is annual, up to 2022, and the model is statistically tied with last year's value. Your job is to make that clear in the interface, not to hide it.

| Original idea | Now |
|---|---|
| Monthly playback slider | Annual playback of **historical** cases (1999 to 2022) |
| Forecast replay for a month | Annual replay for years 2010 to 2022, from the per-pair predictions file |
| Lead-time and early-warning screens | Remove until real monthly data exists |
| "Live data" | "Latest available data: annual, 2022" |
| Forecast as the headline | Forecast **next to persistence**, always |

---

## 2. Tasks in priority order

### Task 1: Skeleton and mock API

1. Choose Next.js (with TypeScript) or Streamlit. Streamlit is faster if the team is short on time. Record the decision in `docs/frontend.md`.
2. Until Person 5's API exists, build against mock JSON that follows the shape in `person5.md`. Keep the mocks in `frontend/mocks/`.
3. The frontend calls **only the API**. It must not read CSV files or model checkpoints.

### Task 2: Pages

| Page | Contents |
|---|---|
| **Home** | Disease tabs, India map coloured by risk level, data status ("Annual data, latest year 2022"), a short banner stating the model is statistically tied with last year's value |
| **State page** | Annual history chart per disease, next-year forecast **beside persistence**, interval, confidence badge, and the `notes` text for low-confidence rows |
| **Disease explorer** | Trend, distribution across states, top states |
| **Model comparison** | The results table from `handoff/README.md` section 4 with the intervals. Loaded from `/model-performance` |
| **Annual replay** | Choose a year from 2010 to 2022; show what the model predicted vs what happened vs persistence, from the per-pair predictions file. Label it "backtest" |
| **Limitations** | The caveats from `handoff/README.md` section 1 in plain language |

Rules for every page: show persistence next to the forecast; show `Insufficient evidence` instead of a risk colour when confidence is `low`; never write "predicts outbreaks"; show horizon 1 by default and mark horizons 2 and 3 as experimental.

### Task 3: Map

1. Use `data/processed/india_states.geojson` from Person 4. It has a `state_id` property on each feature. Until it exists, use a placeholder and keep the join code ready.
2. Join to API data by `state_id`. Test that all 36 ids match in both directions.
3. Include the island territories and small UTs (Lakshadweep, Andaman and Nicobar, Chandigarh, Delhi, Dadra & Nagar Haveli and Daman & Diu) in a side list, since they are hard to click on a map.
4. Provide a text alternative for the colour scale for accessibility.

### Task 4: Containers and CI

1. `docker/Dockerfile.api` and `docker/Dockerfile.frontend`, plus `docker/docker-compose.yml` that starts database, API and frontend.
2. `.github/workflows/ci.yml` that on every pull request: installs `requirements-stgnn.txt`, runs the tests in `tests/`, runs `python models/stgnn/predict.py`, and checks it prints `matches the CSV written at training time: True`. Add a lint step and a frontend build.
3. Keep secrets out of the repo. Use environment variables.
4. A health check on `/health`.

### Task 4b: Model serving

The API (Person 5) calls `predict.py`'s logic. Your part is making sure the container contains `models/stgnn/annual_final.pt`, the adjacency files in `data/processed/graph/`, and the CPU build of PyTorch. The model is small and runs on CPU.

### Task 5: Data-drift monitor

Write a script `src/monitoring/drift.py` that compares the newest data to the training data and writes a small JSON report: share of missing values per disease, the distribution of annual log counts, and the distribution of temperature and rainfall. Flag the report if missingness rises or a distribution shifts clearly. Show the report on a "System status" page. Keep thresholds in a config file.

### Task 6: Tests

- A test per page that renders with mock data.
- A test that fails if any forecast is displayed without a persistence value.
- A test that fails if a `low` confidence row is displayed with a risk colour.

---

## 3. Deliverables

- `frontend/` with the pages above and mock data.
- `docker/` files and `.github/workflows/ci.yml`.
- `src/monitoring/drift.py` and the status page.
- `docs/frontend.md` with how to run it and how the pages map to API endpoints.

## 4. Definition of done

- A fresh clone starts the full system with one command (`docker compose up`) and opens on the home page.
- Every forecast on screen has persistence beside it.
- Every low-confidence row shows the reason.
- The limitations page is one click away from every page.
- CI is green on `main`.

## 5. Pitfalls

- Do not label anything "live" or "real-time".
- Do not use colour alone for risk levels.
- Do not show monthly charts. The monthly data is a modelled split and any monthly curve would look like real seasonality.
- Do not hide the tie. Evaluators who see a forecast without the comparison will assume it is better than it is.
