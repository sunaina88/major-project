# Person 3 findings

Comparisons run: ST-GNN vs persistence (1); GRU and LSTM vs persistence (2); early warning = 3 definitions x 2 predictors x 3 diseases = 18 results (plus 6 pooled "all" rows).

Scored pairs follow Person 2's code (target and last year observed). The handoff text says "last two years", but the code only requires the last year.

1. Metrics module reproduces Person 2: log_model 0.828, log_last 0.846 (tests pass).
2. Anomaly detector: data/processed/anomaly_annual.csv, 1187 rows (state-disease-years with >=5 prior observed years). Columns: state_id, disease, year, observed, expected (median of last 5 observed years), zscore (log1p scale, 1.4826*MAD, floor 0.1), pctile (rank among earlier years).
3. Early warning (data/processed/early_warning_eval.csv, year-ahead, lead time = 1 year by construction): the model and persistence are indistinguishable. Pooled AUROC 0.800 vs 0.791 (p90), 0.797 vs 0.779 (p95), 0.815 vs 0.810 (mean+2sd). Outbreak counts: p90 202/936, p95 156/936, mean+2sd 58/936. Malaria under mean+2sd has only 12 events and dengue only 20, so those rows are not evidence. No bootstrap interval was computed for AUROC differences, so say "similar", not "better".
4. Sequence baselines (pooled GRU and LSTM, 3-year window, same folds and scored pairs as Person 2): GRU log-MAE 0.846, difference vs persistence +0.001 [-0.044, +0.046]; LSTM 0.838, difference -0.008 [-0.057, +0.042]. Persistence is 0.846. Both are statistically tied with persistence. Two comparisons were run.
5. Monthly seasonality and lead time in months: blocked until Person 1 confirms real monthly data.