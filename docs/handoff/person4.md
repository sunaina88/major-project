# Person 4: Geospatial and Environmental Engineer

**Role:** you own the physical environment: state boundaries, weather, static geography, and any environmental model.
**Branch:** `person4/environment`
**Read first:** `handoff/README.md`.

---

## 1. Where things stand

- Person 2 already replaced the synthetic weather with NASA POWER monthly temperature and rainfall at **one centroid point per state** (`models/stgnn/download_weather.py`, output `data/processed/pan_india_state_enriched_tensor_real.csv`). A single point is a rough stand-in for a whole state, especially for large or varied states.
- With real weather the annual model gets log-MAE 0.828 vs 0.852 without. The direction is favourable but not significant. Better environmental data is the most likely way to make that signal clearer.
- Person 6 needs a state-boundary file for the map, and nobody has one yet. **You are the owner.**

---

## 2. Tasks in priority order

### Task 1: State boundaries (needed by Person 6)

1. Find a state-boundary GeoJSON or shapefile for India that you are allowed to use. Record source, licence and date in `docs/data_sources.md`.
2. It must match the 36 nodes in `configs/states.txt`: Telangana separate from Andhra Pradesh, Ladakh separate from Jammu & Kashmir, and Dadra & Nagar Haveli and Daman & Diu merged.
3. Map every polygon to a `state_id` from `configs/states.txt`. Write the mapping explicitly in a small CSV, `data/raw/geography/state_name_map.csv`, and check that all 36 ids appear exactly once.
4. Simplify the geometry for web display and save `data/processed/india_states.geojson` with a `state_id` property on each feature. Keep file size small (a few MB at most).

Test: loading the file and joining to `configs/states.txt` leaves no unmatched ids in either direction.

### Task 2: Area-aware weather

Replace the single-point values with something representative of the whole state.

1. Pick a gridded source (for example NASA POWER, ERA5 or IMD gridded data) and document your choice and licence.
2. For each state, average over grid cells or sample points inside its polygon (Task 1).
3. Produce monthly `temp_mean_c` and `rainfall_mm` for 1999-01 to 2022-12 for all 36 states, with no missing values.
4. **Keep the exact column names `temp_mean_c` and `rainfall_mm`.** Person 2's scripts read those two columns.
5. Write the script in `src/ingestion/weather/`, cache downloads under `data/raw/`, and make it resumable (the existing `download_weather.py` is a working example).

### Task 3: Test your weather file

Make a copy of `data/processed/pan_india_state_enriched_tensor_real.csv`, replace only the `temp_mean_c` and `rainfall_mm` columns with yours, and run:

```bash
python models/stgnn/annual.py --data data/processed/<your_file>.csv --weather 1 --tag p4_areaweather
python models/stgnn/compare_runs.py geo_w p4_areaweather
```

`compare_runs.py` prints the log-MAE, the difference to persistence and the difference to the current weather, each with a bootstrap interval. If the interval includes 0, report it as a tie.

### Task 4: More variables and static features

Deliver two files with `state_id` as the key:

| File | Columns |
|---|---|
| `data/processed/static_state_features.csv` | `state_id`, area_km2, population (state the census year), population_density, mean_elevation_m, urban_fraction |
| `data/processed/monthly_environment_extra.csv` | `state_id`, `start_date`, relative humidity, wind speed, plus any others you add |

Document each column's source, unit and year. **Currently `annual.py` only reads `temp_mean_c` and `rainfall_mm`.** To test other variables, send Person 2 the file and the column names and ask for them to be wired in (see `handoff/README.md` section 9).

### Task 5: Annual environmental summaries

The disease signal is annual, so deliver `data/processed/annual_environment.csv` with `state_id, year` and: mean temperature, total rainfall, monsoon-season (June to September) rainfall, hottest-month temperature, and each of these lagged by one year.

### Task 6 (optional): Environmental suitability model

- Train a model that predicts annual disease burden (log1p) from annual environmental and static features, pooled over states.
- **Train on years before Y and test on year Y, for Y from 2010 to 2022**, the same folds as everyone else. Compare against persistence with a bootstrap interval.
- Report permutation importance or SHAP values. These are model attributions, **not causal effects**.
- If it does not beat persistence, report that. It is still a valid result.

---

## 3. Deliverables

- `data/processed/india_states.geojson` and `data/raw/geography/state_name_map.csv`.
- Area-aware monthly weather file with the two column names above.
- `static_state_features.csv`, `monthly_environment_extra.csv`, `annual_environment.csv`.
- `docs/data_sources.md` with source, licence, units and years for every column.
- The comparison output from Task 3.

## 4. Definition of done

- No NaN in any delivered file. 36 states, 1999 to 2022.
- Every `state_id` matches `configs/states.txt` exactly.
- Person 6 can draw the map from the GeoJSON without further changes.
- Person 2 has run the test and sent back the comparison table.

## 5. Pitfalls

- **Dadra & Nagar Haveli and Daman & Diu**, **Ladakh** and **Telangana** are the usual boundary-file mismatches.
- NASA POWER rainfall arrives in mm per day. Convert to mm per month (the existing script already does).
- Do not use 2023+ weather for training rows before 2023, and do not use a state's future climate to explain its past.
- Island territories (Lakshadweep, Andaman and Nicobar) may have very few grid cells. Check them separately.
