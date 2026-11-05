# Interval calibration report

Source: `stgnn_annual_geo_w_h1_pairs.csv` (1318 state/disease/year predictions, 2010-2025).

Method: error `e = log1p(model) - log1p(true)`; the 80% interval for a forecast `f` is `expm1(log1p(f) - q90)` to `expm1(log1p(f) - q10)`. For every test year Y the quantiles use only folds before Y, and a year is tested only when at least 100 earlier pairs exist.

## Result

Nominal coverage: **80%**

| Quantiles | Observed coverage | Tested pairs |
|---|---|---|
| pooled | 78.4% | 1169 |
| per_disease | 76.1% | 1169 |

## Per year: pooled

| Test year | Tested pairs | Earlier pairs | Coverage | Mean width (log) |
|---|---|---|---|---|
| 2012 | 81 | 149 | 73% | 2.39 |
| 2013 | 85 | 230 | 84% | 2.62 |
| 2014 | 51 | 315 | 86% | 2.53 |
| 2015 | 54 | 366 | 85% | 2.44 |
| 2016 | 59 | 420 | 78% | 2.37 |
| 2017 | 63 | 479 | 89% | 2.43 |
| 2018 | 98 | 542 | 77% | 2.32 |
| 2019 | 97 | 640 | 78% | 2.35 |
| 2020 | 94 | 737 | 59% | 2.39 |
| 2021 | 92 | 831 | 77% | 2.56 |
| 2022 | 94 | 923 | 74% | 2.57 |
| 2023 | 98 | 1017 | 83% | 2.62 |
| 2024 | 102 | 1115 | 78% | 2.63 |
| 2025 | 101 | 1217 | 85% | 2.65 |

## Per year: per_disease

| Test year | Tested pairs | Earlier pairs | Coverage | Mean width (log) |
|---|---|---|---|---|
| 2012 | 81 | 149 | 80% | 2.37 |
| 2013 | 85 | 230 | 84% | 2.58 |
| 2014 | 51 | 315 | 82% | 1.82 |
| 2015 | 54 | 366 | 78% | 1.75 |
| 2016 | 59 | 420 | 64% | 1.80 |
| 2017 | 63 | 479 | 90% | 2.16 |
| 2018 | 98 | 542 | 79% | 2.54 |
| 2019 | 97 | 640 | 81% | 2.44 |
| 2020 | 94 | 737 | 49% | 2.34 |
| 2021 | 92 | 831 | 76% | 2.61 |
| 2022 | 94 | 923 | 61% | 2.64 |
| 2023 | 98 | 1017 | 77% | 2.80 |
| 2024 | 102 | 1115 | 77% | 2.79 |
| 2025 | 101 | 1217 | 91% | 2.80 |

## Reading this honestly

- `configs/risk.yaml` currently uses **pooled** quantiles (`uncertainty.by_disease`). The setting closest to nominal in this run is **pooled**.
- Observed coverage is below the nominal 80% in both settings, so the intervals are, if anything, slightly too narrow. 2020 is the weakest year.
- Intervals are wide because the model's typical log error is about 0.8 (a factor of roughly 2), so an 80% interval spans about a factor of 10 end to end. That is the real uncertainty; do not narrow it for display.
- Only horizon 1 has per-pair errors, so horizons 2 and 3 have **no interval** (`null`).
- Each year is tested once with a small number of pairs; yearly coverage is noisy and states are not independent. Treat the pooled figure as approximate.
