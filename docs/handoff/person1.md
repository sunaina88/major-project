# Person 1: Disease Data and Forecasting Engineer (follow-up phase)

**Role:** you own the disease data. Everything the team can honestly claim depends on what you find in this phase.
**Branch:** `person1/data-followups`
**Read first:** `handoff/README.md` (ground rules and data contract).

---

## 1. Why you are needed again

Person 2's experiments showed that your monthly tensor is an exact fixed split of the annual totals, so only the annual totals carry real information. On annual totals the ST-GNN ties persistence. A positive result is possible only with better data: real monthly values, more years, or cleaner zeros.

---

## 2. Tasks in priority order

### Task 1: Check for newer annual data (2023, 2024, 2025)

1. Look on the NCVBDC dengue, malaria and chikungunya situation pages and on NDAP for state-wise totals after 2022.
2. For each disease and year, record in `docs/data_sources.md`: source URL, date accessed, years and states covered, whether deaths are included.
3. If data exists, add it to `data/raw/pan_india_state_tabular/` in the same format as the existing CSVs.
4. Extend the year cap in `build_pan_india_state_tensor.py` (currently `year <= 2022`).
5. Rerun the pipeline in README order, then tell Person 2. Person 2's scripts read the last year from the data, so 2023+ becomes a genuine test year with no code change.

Do not assume the data exists. If it does not, write that down in `docs/data_sources.md`.

### Task 2: Look for real monthly data

Candidates named in the project plan are the NCVBDC Monthly Malaria Information System (MMIS) and IDSP weekly outbreak reports. For each, find out and write down:

- Which years and which states are covered.
- Format (PDF, Excel, HTML table) and whether it can be parsed automatically.
- Whether the values are state-level counts for dengue, malaria and chikungunya.
- Terms of use.

**Report this coverage table to the team before building any parser.** Even monthly malaria alone for all 36 states would let Person 2 test a genuine monthly model.

### Task 3: Audit the zeros

Several series look like unreported years stored as 0 (for example Maharashtra and Odisha chikungunya in 2021 to 2022, and all of Ladakh).

1. Write `src/ingestion/disease/audit_zeros.py` that lists every state, disease and year where the annual total is exactly 0, together with that series' maximum and its neighbouring years' values.
2. Save it as `data/processed/zero_audit.csv`.
3. Check each flagged series against the raw source and mark it `real_zero` or `not_reported`.
4. Replace `not_reported` values with NaN in the raw ingestion step, not afterwards. Keep real zeros as 0.
5. Tell Person 2 when done, so the heuristic in `annual_data.py` can be switched off (`zero_missing=0`).

### Task 4: Reporting-quality columns

Add the columns promised in the original plan to the tensor: `dengue_reporting_status`, `malaria_reporting_status`, `chikungunya_reporting_status`, with values `reported`, `reported_zero`, `not_reported`, `not_applicable`. Do this after Task 3.

### Task 5: Rerun the baselines on real weather

`data/processed/pan_india_state_enriched_tensor_real.csv` holds NASA POWER weather in place of the synthetic columns (same schema). Rerun `train_baselines.py` on it and save as `baseline_model_metrics_real_weather.csv`. Update your README caveat 1.

### Task 6: Annual baselines (so the team has a proper benchmark)

Your monthly baselines are not comparable to Person 2's annual comparison. Add `src/models/baselines/annual_baselines.py` that evaluates, on the same rolling folds (test years 2010 to 2022, training on earlier years only) and the same scored pairs (see `handoff/README.md` section 6):

| Baseline | Definition |
|---|---|
| Persistence | Last year's value |
| Mean of last 2 years | Average of the last two years |
| Mean of last 3 years | Average of the last three |
| Gradient Boosting | Pooled over states, features: last 3 years (log1p), disease, state |

Report MAE and log-MAE per fold and the bootstrap interval of each baseline against persistence. Save as `data/processed/annual_baselines.csv` using the same column names as `stgnn_annual_<tag>_h1.csv`.

### Task 7: Small README fixes

- Row count: the tensor has 9,984 rows, not 9,985.
- Caveat 1 refers to synthetic weather, which now exists as the `_real` file.
- Note that the first 11 `lag` rows per state have NaN lag columns.

---

## 3. Deliverables

- `docs/data_sources.md` with coverage tables for newer annual data and for monthly sources.
- `data/processed/zero_audit.csv` and a cleaned tensor.
- Reporting-status columns.
- `baseline_model_metrics_real_weather.csv` and `annual_baselines.csv`.

## 4. Definition of done

- Every claim in `docs/data_sources.md` has a URL and an access date.
- `annual_baselines.csv` loads with the same code as `stgnn_annual_<tag>_h1.csv`.
- Person 2 has been told which zeros are real.

## 5. Pitfalls

- Do not fill missing values with 0.
- Do not disaggregate annual totals into months again. If you obtain real monthly data, use it as is.
- Keep `state_id` slugs identical to `configs/states.txt`.
