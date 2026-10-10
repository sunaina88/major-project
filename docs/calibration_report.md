# Interval calibration report

Source: `stgnn_annual_geo_w_h1_pairs.csv` (1046 state/disease/year predictions, 2010-2022).

Method: error `e = log1p(model) - log1p(true)`; the 80% interval for a forecast `f` is `expm1(log1p(f) - q90)` to `expm1(log1p(f) - q10)`. For every test year Y the quantiles use only folds before Y, and a year is tested only when at least 100 earlier pairs exist.

## Result

Nominal coverage: **80%**

| Quantiles | Observed coverage | Tested pairs |
|---|---|---|
| pooled | 76.9% | 892 |
| per_disease | 74.2% | 892 |

## Per year: pooled

| Test year | Tested pairs | Earlier pairs | Coverage | Mean width (log) |
|---|---|---|---|---|
| 2012 | 82 | 154 | 73% | 2.35 |
| 2013 | 86 | 236 | 86% | 2.56 |
| 2014 | 51 | 322 | 90% | 2.41 |
| 2015 | 54 | 373 | 74% | 2.24 |
| 2016 | 61 | 427 | 77% | 2.20 |
| 2017 | 65 | 488 | 89% | 2.28 |
| 2018 | 101 | 553 | 76% | 2.19 |
| 2019 | 100 | 654 | 86% | 2.24 |
| 2020 | 97 | 754 | 58% | 2.18 |
| 2021 | 97 | 851 | 73% | 2.44 |
| 2022 | 98 | 948 | 72% | 2.43 |

## Per year: per_disease

| Test year | Tested pairs | Earlier pairs | Coverage | Mean width (log) |
|---|---|---|---|---|
| 2012 | 82 | 154 | 80% | 2.30 |
| 2013 | 86 | 236 | 85% | 2.51 |
| 2014 | 51 | 322 | 80% | 1.75 |
| 2015 | 54 | 373 | 76% | 1.69 |
| 2016 | 61 | 427 | 66% | 1.76 |
| 2017 | 65 | 488 | 89% | 2.04 |
| 2018 | 101 | 553 | 77% | 2.45 |
| 2019 | 100 | 654 | 83% | 2.41 |
| 2020 | 97 | 754 | 49% | 2.31 |
| 2021 | 97 | 851 | 76% | 2.56 |
| 2022 | 98 | 948 | 61% | 2.62 |

## Reading this honestly

- `configs/risk.yaml` currently uses **pooled** quantiles (`uncertainty.by_disease`). The setting closest to nominal in this run is **pooled**.
- Observed coverage is below the nominal 80% in both settings, so the intervals are, if anything, slightly too narrow. 2020 is the weakest year.
- Intervals are wide because the model's typical log error is about 0.8 (a factor of roughly 2), so an 80% interval spans about a factor of 10 end to end. That is the real uncertainty; do not narrow it for display.
- Only horizon 1 has per-pair errors, so horizons 2 and 3 have **no interval** (`null`).
- Each year is tested once with a small number of pairs; yearly coverage is noisy and states are not independent. Treat the pooled figure as approximate.
