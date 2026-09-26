# VectorWatch India
### Pan-India Spatio-Temporal Forecasting of Vector-Borne Diseases

**Phase owner:** Person 1 — Disease Data & Forecasting Engineer
**Phase status:** ✅ Complete — data engineering, feature pipeline, and baseline benchmarks delivered
**Downstream owners:** Persons 2–6 (ST-GNN architecture, training, deployment, etc.)

---

## 1. Project Overview

**VectorWatch India** is a 6-person ML project building a **Pan-India Spatio-Temporal Graph Neural Network (ST-GNN)** to forecast three mosquito-borne diseases — **Dengue, Malaria, and Chikungunya** — at monthly resolution, across all **36 Indian States and Union Territories**, from **1999 to 2022**.

The end goal is a graph model where each state is a node, edges encode spatial adjacency, and node features evolve over monthly time steps to forecast future disease burden.

**This phase (Person 1) delivers everything upstream of the GNN itself:**

1. Ingest raw annual, state-wise disease report data (NDAP-style exports).
2. Disaggregate annual totals into a monthly spatiotemporal tensor.
3. Validate the disaggregation didn't corrupt or lose data.
4. Enrich the tensor with climate covariates (temperature, rainfall).
5. Engineer epidemiological lag/rolling features that model mosquito breeding-cycle delays.
6. Train and benchmark baseline forecasting models (Seasonal Naive vs. two Gradient Boosting variants) to establish the error bar the ST-GNN must beat — and to test whether the engineered weather features actually help.

The final artifact of this phase — `data/processed/pan_india_state_enriched_tensor.csv` — is the **starting point for Person 2**. See [Section 5](#5-handoff-to-person-2-gnn-engineer) for the handoff details.

> **Note:** this README reflects the repository *after* a pre-handoff review pass. A one-line state-naming bug was fixed, an unrelated unfinished pipeline was removed for a cleaner handoff, and the baseline benchmark was extended to properly test the weather features. See [Section 7](#7-revision-log-pre-handoff-fixes) for exactly what changed and why.

---

## 2. Folder Structure

```
vectorwatch-india/
├── requirements.txt                          # pandas, numpy, scikit-learn, matplotlib, seaborn
│
├── data/
│   ├── raw/
│   │   └── pan_india_state_tabular/          # Source annual, state-wise NDAP exports
│   │       ├── dengue_state.csv              #   1999–2022, has cases + deaths
│   │       ├── malaria_state.csv             #   2001–2022, has cases + deaths
│   │       └── chikungunya_state.csv         #   2006–2022, cases only (no deaths at source)
│   │
│   ├── interim/                              # ⚠️ empty — reserved for future staging outputs, unused by this phase
│   │
│   └── processed/                            # All pipeline outputs (this phase's deliverables)
│       ├── pan_india_state_spatiotemporal_tensor.csv   # Step 1 output: monthly disease tensor
│       ├── pan_india_state_enriched_tensor.csv         # ★ FINAL HANDOFF ARTIFACT ★
│       ├── baseline_model_metrics.csv                  # Per-state/per-disease MAE & RMSE, 3 models
│       └── baseline_mae_comparison.png                 # Benchmark chart (5 high-burden states)
│
└── src/
    ├── ingestion/
    │   ├── disease/
    │   │   ├── build_pan_india_state_tensor.py   # Raw annual CSVs → monthly spatiotemporal tensor
    │   │   └── build_macro_checksum.py           # Read-only validation/QA report on the tensor
    │   │
    │   └── weather/
    │       ├── build_weather_tensor.py           # Synthetic climatology → adds temp/rainfall, writes enriched tensor
    │       └── build_lag_features.py             # Adds lag/rolling features, OVERWRITES enriched tensor in place
    │
    └── models/
        └── baselines/
            ├── train_baselines.py                 # Seasonal Naive vs. 2 Gradient Boosting variants, per state/disease
            └── visualize_baselines.py             # Renders baseline_mae_comparison.png
```

A West-Bengal-district-level PDF-ingestion script and two related empty scaffold folders were removed from this repo prior to handoff — see [Section 7](#7-revision-log-pre-handoff-fixes) if you're wondering why the tree is leaner than an earlier draft of this README.

---

## 3. Pipeline Order (How to Reproduce)

Scripts assume they are run **from the repository root** (`vectorwatch-india/`), since all paths are relative (e.g. `data/raw/...`, `data/processed/...`). Run in this exact order:

| # | Command | Reads | Writes |
|---|---------|-------|--------|
| 1 | `python src/ingestion/disease/build_pan_india_state_tensor.py` | 3 raw CSVs in `data/raw/pan_india_state_tabular/` | `pan_india_state_spatiotemporal_tensor.csv` |
| 2 | `python src/ingestion/disease/build_macro_checksum.py` | `pan_india_state_spatiotemporal_tensor.csv` | *(nothing — prints a validation report to stdout)* |
| 3 | `python src/ingestion/weather/build_weather_tensor.py` | `pan_india_state_spatiotemporal_tensor.csv` | `pan_india_state_enriched_tensor.csv` (created) |
| 4 | `python src/ingestion/weather/build_lag_features.py` | `pan_india_state_enriched_tensor.csv` | `pan_india_state_enriched_tensor.csv` (**overwritten in place**) |
| 5 | `python src/models/baselines/train_baselines.py` | `pan_india_state_enriched_tensor.csv` | `baseline_model_metrics.csv` |
| 6 | `python src/models/baselines/visualize_baselines.py` | `baseline_model_metrics.csv` | `baseline_mae_comparison.png` |

Step 2 is a **read-only QA gate** — it doesn't transform data, it just reports whether the monthly tensor looks structurally sound (year span, state count, annual aggregates, NaN counts) before you trust it downstream.

---

## 4. File-by-File Reference

### 4.1 Data — Raw Inputs (`data/raw/`)

| File | Description |
|---|---|
| `pan_india_state_tabular/dengue_state.csv` | Raw annual NDAP-style export. Columns: `Country, State, Year, Additional Info`, a cases column, and a deaths column. Covers **1999–2022**, 650 rows. |
| `pan_india_state_tabular/malaria_state.csv` | Same format. Covers **2001–2022**, 792 rows. |
| `pan_india_state_tabular/chikungunya_state.csv` | Same format, but **has no deaths column at source** — only a cases column. Covers **2006–2022**, 576 rows. |

Each raw file has a different start year and irregular header text (e.g. inconsistent double-spacing in "...Due To Dengue  (UOM..."). The ingestion script handles this with keyword-based column detection rather than exact-string matching.

### 4.2 Source Code — Disease Ingestion (`src/ingestion/disease/`)

#### `build_pan_india_state_tensor.py`
Builds the master monthly spatiotemporal tensor from the three raw disease CSVs.

- Extracts a clean 4-digit year from free-text year strings.
- Standardizes state names (casing, whitespace, common aliases like "Orissa" → "Odisha", "Pondicherry" → "Puducherry", NCT/UT variants → canonical form) and drops non-state rows ("India", "All India Total", etc.).
- **Outer-merges** all three diseases on `[State, Year]` — a state missing a report for one disease in a given year keeps `NaN` for that disease rather than being dropped from the row entirely.
- Cleans numeric columns (strips commas, coerces unparseable values like `"NR"` to `NaN`).
- **Disaggregates each annual total into 12 monthly rows** using a fixed monsoon-weighted curve (weights peak Jul–Sep, sum to 1.0 across the year). This is a **modeled estimate**, not a real monthly report — see caveats.
- Caps output at year ≤ 2022.
- Writes `data/processed/pan_india_state_spatiotemporal_tensor.csv`.

**Explicit design choices worth knowing:**
- Missing values are kept as `NaN`, never filled with `0` — "no report" ≠ "zero cases."
- `chikungunya_deaths` is `NaN` for every row (no deaths ever reported at source for this disease), not `0`.
- Monthly disease counts are `float64` (fractional), since they're a modeled split of an annual integer total, not an original reported count.
- The script only disaggregates `(state, year)` pairs that actually exist in the source data — it does not fabricate coverage for years/states with no report at all. This is why, e.g., chikungunya has no data before 2006.
- The state-name alias map now also catches the **"The Dadra and Nagar Haveli and Daman and Diu"** source spelling (fixed pre-handoff — see [Section 7](#7-revision-log-pre-handoff-fixes)), so this UT's `state_id` is the clean `dadra_and_nagar_haveli_and_daman_and_diu`.

#### `build_macro_checksum.py`
A read-only QA/validation report over the tensor produced above. It prints:
- The year span and number of years covered.
- The number of unique states/UTs (spatial nodes).
- National annual case totals for the last 5 years (sanity check).
- The count of missing (`NaN`) values per disease.

> **Note:** this script validates internal consistency of the monthly tensor (it doesn't fabricate or lose totals during disaggregation) — it does **not** cross-check against an external NCVBDC annual report file, since no such file exists in this repo. Treat its "PASSED" output as *"the disaggregation is internally consistent,"* not *"independently verified against a second data source."* If a true external cross-check against NCVBDC totals becomes necessary later, that would need an actual NCVBDC reference file added to `data/raw/` and a real comparison added to this script.

### 4.3 Source Code — Environmental & Feature Engineering (`src/ingestion/weather/`)

#### `build_weather_tensor.py`
Adds `temp_mean_c` and `rainfall_mm` to every row of the monthly disease tensor, producing the enriched tensor.

- **Architectural note:** university network access blocked real weather-API calls, so this is an **offline deterministic climatology model** — not real historical weather data. Each state has a hand-specified profile (`base_annual_rain_mm, base_annual_temp_c, rain_variance`) in `STATE_CLIMATE_PROFILES`, combined with fixed monthly temperature/rainfall curves (temperature peaks May–Jun, rainfall peaks Jul–Aug) plus a small seeded random noise term (`np.random.seed(42)`, so output is reproducible run-to-run).
- States not found in `STATE_CLIMATE_PROFILES` fall back to a national-average profile (1000mm rain / 25.0°C) — check this list if a state's climate values look suspiciously generic.
- Reads `pan_india_state_spatiotemporal_tensor.csv`, writes `pan_india_state_enriched_tensor.csv`.

> **Treat `temp_mean_c` / `rainfall_mm` as plausible synthetic placeholders, not ground truth.** They are internally consistent and seasonally realistic (useful for building/testing the pipeline and baselines end-to-end), but — as the baseline results below show — they currently behave more like structured noise than real predictive signal. Swap in real IMD/reanalysis climate data before any downstream result is treated as a genuine epidemiological finding.

#### `build_lag_features.py`
Adds epidemiological lag/rolling features that model the delayed effect of weather on mosquito breeding and disease onset:

- `temp_lag_1`, `temp_lag_2` — temperature 1 and 2 months prior, per state.
- `rainfall_lag_1`, `rainfall_lag_2` — rainfall 1 and 2 months prior, per state.
- `rainfall_rolling_3m` — trailing 3-month cumulative rainfall (breeding-season accumulation), per state.

Sorts strictly by `[state_name, start_date]` before shifting/rolling so lags never leak across state boundaries. **Reads and then overwrites** `pan_india_state_enriched_tensor.csv` in place — running this script twice on an already-lagged file will re-derive the same lag columns from themselves, so run it exactly once per fresh enriched tensor.

### 4.4 Source Code — Baseline Modeling (`src/models/baselines/`)

#### `train_baselines.py`
Trains and evaluates **three** baseline forecasters, independently per state and per disease, using a strict time-series split (**train: years < 2021, test: years 2021–2022**), reading from the **enriched** tensor:

1. **Seasonal Naive** — predicts each month using the value from exactly 12 months prior.
2. **Gradient Boosting** — `HistGradientBoostingRegressor` using only disease-lag features: `[lag_1, lag_2, lag_12, month]`.
3. **Gradient Boosting (Weather)** — the same model and disease-lag features as #2, **plus** the 7 engineered climate/lag columns (`temp_mean_c`, `rainfall_mm`, `temp_lag_1`, `temp_lag_2`, `rainfall_lag_1`, `rainfall_lag_2`, `rainfall_rolling_3m`).

Models #2 and #3 use the identical split, seed, and disease-lag features — the *only* difference is whether the weather/lag columns are included — so any difference in error is directly attributable to the weather feature engineering.

**Architectural note:** `HistGradientBoostingRegressor` was chosen over XGBoost specifically because it tolerates missing values natively and avoids a large dependency download over a restricted network.

Outputs `baseline_model_metrics.csv` with columns `state_name, disease, model, mae, rmse` (one row per state × disease × model, 324 rows total).

**Results, averaged across all 36 states and all 3 diseases:**

| Model | Mean MAE | Mean RMSE |
|---|---|---|
| Seasonal Naive | 230.22 | 327.77 |
| Gradient Boosting (lag-only) | **185.80** | **277.75** |
| Gradient Boosting (+ weather) | 212.96 | 323.08 |

> **Honest finding:** across all diseases/states, adding the synthetic weather/lag features made the Gradient Boosting model *worse on average*, not better — it under-performs the lag-only variant and only modestly beats Seasonal Naive. On **Dengue specifically** the weather-enriched model is roughly on par with (very slightly better than) the lag-only model (239.1 vs. 244.9 mean MAE), but the effect is inconsistent across diseases — most notably it hurts on Malaria in several states. This is consistent with the weather features being seasonally-realistic but ultimately **synthetic/simulated** rather than real observed climate data: they add variance without adding genuine predictive signal, so the model has nothing real to learn from them and occasionally overfits to it. **Do not read this as "weather doesn't matter for vector-borne disease" — read it as "these particular synthetic weather features don't help, and this needs revisiting once real climate data replaces the placeholder."**

#### `visualize_baselines.py`
Reads `baseline_model_metrics.csv`, filters to **Dengue only** and 5 high-burden states (`Maharashtra, West Bengal, Delhi, Karnataka, Andhra Pradesh`), and renders a grouped bar chart (all 3 models' MAE side by side) to `baseline_mae_comparison.png`. This is the benchmark artifact — the error bar the ST-GNN is expected to beat.

### 4.5 Data — Processed Outputs (`data/processed/`)

#### `pan_india_state_spatiotemporal_tensor.csv`
The intermediate monthly tensor, before weather enrichment. **9,985 rows × 9 columns.**

| Column | Type | Notes |
|---|---|---|
| `start_date` | date (`YYYY-MM-01`) | First of each month |
| `state_name` | string | Canonicalized, Title Case |
| `state_id` | string | Lowercase/underscored slug of `state_name` |
| `dengue_cases`, `dengue_deaths` | float | `NaN` where no annual report existed |
| `malaria_cases`, `malaria_deaths` | float | `NaN` where no annual report existed |
| `chikungunya_cases` | float | `NaN` where no annual report existed |
| `chikungunya_deaths` | float | **Always `NaN`** — never reported at source |

#### `pan_india_state_enriched_tensor.csv`  ★ Final Handoff Artifact
The full feature-complete tensor. Same 9,985 rows as above, plus 7 additional columns (**16 columns total**):

| Column | Type | Notes |
|---|---|---|
| *(all 9 columns above)* | | |
| `temp_mean_c` | float | Synthetic climatology (see caveats) |
| `rainfall_mm` | float | Synthetic climatology (see caveats) |
| `temp_lag_1`, `temp_lag_2` | float | 1- and 2-month lagged temperature |
| `rainfall_lag_1`, `rainfall_lag_2` | float | 1- and 2-month lagged rainfall |
| `rainfall_rolling_3m` | float | Trailing 3-month cumulative rainfall |

Covers **1999-01-01 to 2022-12-01**, all **36** states/UTs.

#### `baseline_model_metrics.csv`
324 rows: one row per (state × disease × model) combination that had enough train/test data, across 3 models. Columns: `state_name, disease, model, mae, rmse`.

#### `baseline_mae_comparison.png`
Grouped bar chart, Dengue MAE, all 3 models, for the 5 high-burden states listed above.

---

## 5. Handoff to Person 2 (GNN Engineer)

**Your starting point is:**

```
data/processed/pan_india_state_enriched_tensor.csv
```

This is the final, feature-complete artifact from this phase — 9,985 rows, 16 columns, 36 states/UTs, monthly from Jan 1999 to Dec 2022.

- Use **`state_name`** or **`state_id`** as the node key to construct your spatial adjacency matrix (e.g. joining against a shapefile or a hand-built neighbor list). `state_id` is a lowercase, underscore-slugified version of `state_name` and is stable/unique per state — prefer it as your join key over `state_name`, since it avoids casing/whitespace mismatches. (A naming bug that would have broken this join for one UT was fixed prior to handoff — every `state_id` now slugifies cleanly, no special-casing needed.)
- Each `(state_id, start_date)` row is a **node-timestep feature vector**: 6 disease target columns (`dengue_cases`, `dengue_deaths`, `malaria_cases`, `malaria_deaths`, `chikungunya_cases`, `chikungunya_deaths`) + 7 environmental/lag features (`temp_mean_c`, `rainfall_mm`, `temp_lag_1`, `temp_lag_2`, `rainfall_lag_1`, `rainfall_lag_2`, `rainfall_rolling_3m`), keyed by `start_date`.
- **Missing values are real `NaN`s, not zeros** — decide your own imputation/masking strategy for the GNN's loss function rather than assuming complete coverage. Coverage differs by disease (dengue from 1999, malaria from 2001, chikungunya from 2006 for cases and never for deaths).
- `baseline_model_metrics.csv` + `baseline_mae_comparison.png` give you the error bar your ST-GNN needs to beat per state/disease — worth reproducing the same train/test split (train < 2021, test 2021–2022) for an apples-to-apples comparison.
- **On the weather features specifically:** they're wired into the tensor and schema-ready for your GNN's node features, but the baseline benchmark shows they don't yet add real signal (see Section 4.4) because they're synthetic. You can still feed them into the GNN as-is (the ST-GNN may extract different signal than a gradient-boosted tree does), but don't be surprised if they're not immediately useful, and flag it if real climate data becomes available later in the project.

---

## 6. Known Gaps & Caveats

1. **Weather data is synthetic, not observed.** `temp_mean_c` and `rainfall_mm` come from a deterministic climatology formula (seeded, reproducible), not real IMD/reanalysis records — a network restriction blocked live weather API access during this phase. They're seasonally realistic and fine for pipeline development, and the baseline results in Section 4.4 confirm they don't yet add real signal. Swap in real climate data before final results are reported as scientifically meaningful.
2. **Monthly disease counts are modeled, not reported.** Only annual totals are ever reported at source; the monthly split uses a fixed monsoon-weighted curve (peaking Jul–Sep) applied uniformly across all states and years. This is a reasonable prior for the Indian mosquito season but is the same curve for every state — it doesn't capture state-specific seasonal timing differences (e.g., northern vs. coastal monsoon onset).
3. **Disease coverage windows differ:** Dengue (1999–2022), Malaria (2001–2022), Chikungunya cases (2006–2022), Chikungunya deaths (never reported, always `NaN`). Design your loss masking/training windows with this in mind rather than assuming uniform coverage across targets.
4. **`build_macro_checksum.py`'s "PASSED" status is an internal-consistency check only**, not a cross-check against an independent NCVBDC source file (none exists in this repo). See Section 4.2 for detail.
5. **`data/interim/` is empty and currently unused.** It's kept as a conventional staging location in case a future step needs one; nothing in this phase writes to it.

---

## 7. Revision Log (Pre-Handoff Fixes)

An internal review of this repo, done just before handoff, surfaced three issues. All three were fixed and the full pipeline was re-run end-to-end; the numbers and file descriptions throughout this README reflect the corrected outputs.

1. **Fixed a state-naming bug.** The raw source data spells one UT as *"The Dadra and Nagar Haveli and Daman and Diu"* (with a leading "The"), which the original alias-normalization step didn't catch (it only mapped variants without "The"). This produced a malformed `state_id` (`the_dadra_and_nagar_haveli_and_daman_and_diu`) that would have silently failed to join against any external geography/shapefile source using the standard slug. **Fix:** added the missing alias to `STATE_NAME_ALIASES` in `build_pan_india_state_tensor.py` and re-ran the full pipeline. `state_id` is now the clean `dadra_and_nagar_haveli_and_daman_and_diu` for all 9,985 rows; row counts, date range, and state count (36) are unchanged.
2. **Removed unfinished, unrelated scaffolding.** `build_idsp_weekly_tensor.py` (a separate West-Bengal district-level weekly-outbreak PDF pipeline) and two empty folders it depended on (`src/ingestion/geography/`, `data/raw/vector_monthly_tabular/`) were removed from the repo. None of their required inputs (`data/raw/idsp_weekly/*.pdf`, `configs/districts.json`) ever existed in this repo, and nothing in the delivered pan-India pipeline depended on them — they were dead weight for a handoff. If district-level West Bengal modeling becomes a real project need later, this can be rebuilt from scratch with actual source PDFs.
3. **Extended the baseline benchmark to actually test the weather features.** `train_baselines.py` originally read the un-enriched tensor and only used the disease series' own lags, meaning the engineered `temp_*`/`rainfall_*` features were never benchmarked against anything. **Fix:** the script now reads the enriched tensor and trains a third model, `Gradient_Boosting_Weather`, using the same split/seed/disease-lags as the original Gradient Boosting model plus the 7 weather/lag columns — isolating the effect of the weather features. The result (Section 4.4) is a genuine, useful finding: the current synthetic weather features don't clearly improve forecasts and should be revisited once real climate data is available. `visualize_baselines.py` was updated to plot all three models.
4. **Populated `requirements.txt`** (previously 0 bytes) with the actual dependencies used across the pipeline (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`), as conservative lower-bound versions.

---

## 8. Environment

```bash
pip install -r requirements.txt
```

All ingestion/modeling scripts are run from the repository root, e.g.:

```bash
cd vectorwatch-india
python src/ingestion/disease/build_pan_india_state_tensor.py
python src/ingestion/disease/build_macro_checksum.py
python src/ingestion/weather/build_weather_tensor.py
python src/ingestion/weather/build_lag_features.py
python src/models/baselines/train_baselines.py
python src/models/baselines/visualize_baselines.py
```

---

*Questions about this phase's data, features, or baselines → Person 1 (Disease Data & Forecasting Engineer).*
