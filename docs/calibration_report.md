# Interval calibration report

Source: `stgnn_annual_geo_w_h1_pairs.csv` (1446 state/disease/year predictions, 2010-2025).

Method: error `e = log1p(model) - log1p(true)`; the 80% interval for a forecast `f` is `expm1(log1p(f) - q90)` to `expm1(log1p(f) - q10)`. For every test year Y the quantiles use only folds before Y, and a year is tested only when at least 100 earlier pairs exist.

## Result

Nominal coverage: **80%**

| Quantiles | Observed coverage | Tested pairs |
|---|---|---|
| pooled | 79.3% | 1297 |
| per_disease | 76.4% | 1297 |

## Per year: pooled

| Test year | Tested pairs | Earlier pairs | Coverage | Mean width (log) |
|---|---|---|---|---|
| 2012 | 81 | 149 | 73% | 2.39 |
| 2013 | 85 | 230 | 84% | 2.62 |
| 2014 | 80 | 315 | 75% | 2.54 |
| 2015 | 85 | 395 | 75% | 2.52 |
| 2016 | 93 | 480 | 80% | 2.61 |
| 2017 | 97 | 573 | 89% | 2.63 |
| 2018 | 98 | 670 | 82% | 2.53 |
| 2019 | 97 | 768 | 88% | 2.53 |
| 2020 | 94 | 865 | 55% | 2.45 |
| 2021 | 92 | 959 | 77% | 2.74 |
| 2022 | 94 | 1051 | 74% | 2.71 |
| 2023 | 98 | 1145 | 85% | 2.75 |
| 2024 | 102 | 1243 | 81% | 2.72 |
| 2025 | 101 | 1345 | 89% | 2.71 |

## Per year: per_disease

| Test year | Tested pairs | Earlier pairs | Coverage | Mean width (log) |
|---|---|---|---|---|
| 2012 | 81 | 149 | 80% | 2.37 |
| 2013 | 85 | 230 | 84% | 2.58 |
| 2014 | 80 | 315 | 75% | 2.42 |
| 2015 | 85 | 395 | 74% | 2.34 |
| 2016 | 93 | 480 | 78% | 2.61 |
| 2017 | 97 | 573 | 91% | 2.65 |
| 2018 | 98 | 670 | 76% | 2.52 |
| 2019 | 97 | 768 | 82% | 2.51 |
| 2020 | 94 | 865 | 49% | 2.43 |
| 2021 | 92 | 959 | 74% | 2.60 |
| 2022 | 94 | 1051 | 65% | 2.72 |
| 2023 | 98 | 1145 | 74% | 2.83 |
| 2024 | 102 | 1243 | 76% | 2.82 |
| 2025 | 101 | 1345 | 90% | 2.83 |

## Reading this honestly

- `configs/risk.yaml` currently uses **pooled** quantiles (`uncertainty.by_disease`). The setting closest to nominal in this run is **pooled**.
- Observed coverage is below the nominal 80% in both settings, so the intervals are, if anything, slightly too narrow. 2020 is the weakest year.
- Intervals are wide because the model's typical log error is about 0.8 (a factor of roughly 2), so an 80% interval spans about a factor of 10 end to end. That is the real uncertainty; do not narrow it for display.
- Only horizon 1 has per-pair errors, so horizons 2 and 3 have **no interval** (`null`).
- Each year is tested once with a small number of pairs; yearly coverage is noisy and states are not independent. Treat the pooled figure as approximate.
