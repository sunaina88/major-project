# Person 3: Time-Series, Anomaly Detection and Evaluation Engineer

**Role:** you own "is something unusual happening?" and the evaluation framework the whole team reports with.
**Branch:** `person3/anomaly-eval`
**Read first:** `handoff/README.md`, especially the ground rules and metric definitions.

---

## 1. What changed from the original plan

The original plan asked for monthly seasonality, anomaly detection on monthly counts, and "lead time in months". **That is not possible with the current data**, because the monthly values are a fixed split of annual totals. Your work is done at **annual resolution**. Seasonality and monthly lead time are blocked until Person 1 delivers real monthly data.

---

## 2. Setup

```bash
git checkout main && git pull && git checkout -b person3/anomaly-eval
pip install -r requirements-stgnn.txt
```

```python
import sys; sys.path.insert(0, "models/stgnn")
from annual_data import load_annual
states, years, Y, W = load_annual("data/processed/pan_india_state_enriched_tensor_real.csv")
# Y: [24 years, 36 states, 3 diseases], NaN = not reported
```

Result files you can compare against: `data/processed/stgnn_annual_geo_w_h1.csv` (per fold) and `stgnn_annual_geo_w_h1_pairs.csv` (per state, disease and year).

---

## 3. Tasks

### Task 1: Metrics module (do this first)

Create `src/evaluation/metrics.py` with: `mae`, `log_mae`, `paired_diff(model_errors, reference_errors)`, `bootstrap_ci(values, n=10000, seed=0)`. Use the definitions in `handoff/README.md` section 6.

**Test:** `tests/test_metrics.py` must reproduce the mean `log_model` (0.828) and `log_last` (0.846) from `stgnn_annual_geo_w_h1.csv` to three decimals. All other work depends on this being right.

### Task 2: Backtesting harness

Create `src/evaluation/backtest.py`:

- Folds: test year Y from 2010 to 2022, training on years before Y only. Any normalization is computed from those training years only.
- Scored pairs: target year and last year both observed (so your numbers are comparable to Person 2's).
- Output: a DataFrame with the same columns as `stgnn_annual_<tag>_h1.csv`.

### Task 3: Annual anomaly detector

Create `src/anomaly/annual.py`. For each state, disease and year Y, using years before Y only (expanding window, no leakage):

| Quantity | Definition |
|---|---|
| `expected` | Median of the last 5 observed years |
| `scale` | Median absolute deviation of log1p values over the same window (with a floor, to avoid division by zero) |
| `zscore` | `(log1p(observed) - log1p(expected)) / scale` |
| `pctile` | Rank of the observed value among the state's previous years |

Save `data/processed/anomaly_annual.csv` with columns `state_id, disease, year, observed, expected, zscore, pctile`. Require at least 5 prior observed years; otherwise leave the row out.

### Task 4: Outbreak-year definitions and early-warning evaluation

Define an "outbreak year" for state, disease and year Y in three ways, each using only earlier years, and compare them:

1. Observed value at or above the 90th percentile of the state's previous years.
2. Same with the 95th percentile.
3. Log value above previous mean + 2 standard deviations.

For each definition, evaluate how well a year-ahead flag predicts it:

| Predictor | How |
|---|---|
| Model forecast | Flag if the Person 2 model's forecast for Y (from `_pairs.csv`, column `model`) exceeds the threshold |
| Persistence | Flag if last year's value exceeds the threshold |

Report precision, recall, F1, AUROC and the false-alarm rate for both. **Lead time at annual resolution is one year by construction. Do not report lead time in months.**

If few outbreak years exist per definition, say so, and report the counts alongside the metrics.

### Task 5: Sequence baselines

Create `src/models/sequence/`:

- A pooled GRU and an LSTM. Input: last 3 years of log1p counts (plus missing-value mask), output: next year's log1p count. One model shared by all states, with a masked loss.
- Optionally a small temporal CNN.
- Same folds, same scored pairs, same metrics as Task 1.

With about 15 real points per series, expect a tie with persistence. Report it with the bootstrap interval, as Person 2 did.

### Task 6: Research plots

In `notebooks/`, produce: forecast vs persistence per fold, error by state, anomaly timelines for 5 high-burden states, and the early-warning ROC curves. Save PNGs to `docs/figures/`.

### Task 7 (blocked): Seasonality engine

Only start this if Person 1 confirms real monthly data. Then: seasonal profile per state and disease, a seasonal-baseline anomaly score, and lead-time analysis.

---

## 4. Deliverables

- `src/evaluation/metrics.py`, `src/evaluation/backtest.py`, tests.
- `src/anomaly/annual.py` and `data/processed/anomaly_annual.csv`.
- `src/models/sequence/` and its results CSV.
- Early-warning evaluation table and figures.
- A short `docs/person3_findings.md`.

## 5. Definition of done

- Metrics tests pass and reproduce Person 2's numbers.
- Every comparison reports a difference with a bootstrap interval, and states how many comparisons were run.
- Person 5 can read `anomaly_annual.csv` without asking you what the columns mean.

## 6. Pitfalls

- **Leakage.** Thresholds, percentiles and normalization must use earlier years only.
- **Zeros.** A zero in a series that once reached 100 or more is probably an unreported year, and `load_annual` already blanks those. Do not turn them back into zeros.
- **Few outbreak events.** A high AUROC from a handful of events is not evidence. Show the counts.
- **Multiple comparisons.** Three definitions × two predictors × three diseases is 18 results. Say so.
