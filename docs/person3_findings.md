# Person 3 findings

Scored pairs follow Person 2's code (`models/stgnn/annual.py`): target year and last year observed; the "average of two years" baseline uses `nanmean`. Comparisons run: ST-GNN vs persistence (1); GRU and LSTM vs persistence (2); early warning = 3 definitions x 2 predictors x 3 diseases = 18 disease-level results (plus 6 pooled rows), each reported under 3 AUROC variants (pooled relative, pooled raw-count, within-series).

## Headline: early warning is not shown to work within a series

Year-ahead flag, lead time = 1 year by construction. Score = log1p(forecast) minus log1p(that state-disease's own threshold from earlier years). "Within-series" = AUROC computed separately inside each state-disease series, then averaged over series that contain both outbreak and non-outbreak years (usable series / total in last column); the interval is a bootstrap over series (5,000 resamples, a rough guide only). M = ST-GNN forecast, P = persistence. Events = outbreak years among the n scored pairs.

| Definition | Disease | Events / n | Within-series AUROC M | Within-series AUROC P | Series usable |
|---|---|---|---|---|---|
| p90 | dengue | 117 / 238 | 0.47 [0.39, 0.56] | 0.53 [0.45, 0.62] | 32/36 |
| p90 | malaria | 28 / 463 | 0.78 [0.70, 0.86] | 0.79 [0.70, 0.85] | 15/36 |
| p90 | chikungunya | 57 / 235 | 0.67 [0.55, 0.80] | 0.67 [0.53, 0.80] | 18/29 |
| p95 | dengue | 99 / 238 | 0.49 [0.41, 0.57] | 0.52 [0.43, 0.60] | 32/36 |
| p95 | malaria | 22 / 463 | 0.71 [0.55, 0.84] | 0.70 [0.54, 0.83] | 13/36 |
| p95 | chikungunya | 35 / 235 | 0.57 [0.38, 0.75] | 0.57 [0.37, 0.75] | 11/29 |
| mean+2sd | dengue | 20 / 238 | 0.49 [0.32, 0.65] | 0.56 [0.37, 0.74] | 12/36 |
| mean+2sd | malaria | 12 / 463 | 0.70 [0.48, 0.88] | 0.70 [0.48, 0.88] | 9/36 |
| mean+2sd | chikungunya | 26 / 235 | 0.42 [0.18, 0.69] | 0.46 [0.23, 0.69] | 6/29 |

- Dengue is at chance level within series for every definition (intervals include 0.5). Malaria shows some signal and chikungunya is mixed, but the model is never distinguishable from persistence.
- Per-series AUROCs rest on at most 13 test years each, so they are noisy; mean+2sd rows have 12 to 26 events in total and should not be read as evidence.
- Per-disease pooled AUROC (all states together) is higher: p90 dengue 0.58 (M) / 0.60 (P), malaria 0.85 / 0.86, chikungunya 0.82 / 0.82. Pooling across series mixes differences between series (some series sit near their own threshold more often) with within-series skill, so the pooled number is inflated relative to the within-series one. The all-disease pooled figures (0.80 / 0.79 for p90) additionally mix diseases and should not be the headline.
- Using raw forecast counts as the score gives pooled AUROC of about 0.41 (p90), so large states do not drive the pooled result.
- Full table with all variants: `data/processed/early_warning_sensitivity.csv`; classification metrics: `early_warning_eval.csv`. Differences between model and persistence were not tested with intervals; say "similar", not "better".

## Other results

1. Metrics module reproduces Person 2: log_model 0.828, log_last 0.846; the backtest harness reproduces persistence and the two-year-average baseline exactly (tests/test_metrics.py, tests/test_backtest.py).
2. Anomaly detector: `data/processed/anomaly_annual.csv`, 1187 rows (state-disease-years with >=5 prior observed years). Columns: state_id, disease, year, observed, expected (median of last 5 observed years), zscore (log1p scale, 1.4826*MAD, floor 0.1), pctile (rank among earlier years). A test checks that rows up to year Y do not change when later years are altered.
3. Sequence baselines (pooled GRU and LSTM, 3-year window, same folds and scored pairs as Person 2): GRU log-MAE 0.846, difference vs persistence +0.001 [-0.044, +0.046]; LSTM 0.838, difference -0.008 [-0.057, +0.042]. Persistence is 0.846. Both are statistically tied with persistence. Two comparisons were run.
4. Monthly seasonality and lead time in months: blocked until Person 1 confirms real monthly data.
