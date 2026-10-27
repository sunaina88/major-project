# Risk method

Owner: Person 5. Code: `src/risk/engine.py`. Settings: `configs/risk.yaml`.

## What the risk level is

For one state, one disease and one forecast year:

1. Take the **forecast** (annual cases).
2. Take that state's **previous observed annual values** of the same disease (everything up to the latest observed year, 2022). Years that were not reported are skipped.
3. Compute the **percentile**: the share of those previous years whose value is **strictly smaller** than the forecast, on a 0 to 100 scale. Strict on purpose: a state that always reported 0 and is forecast 0 gets percentile 0, not 50.
4. Map the percentile to a level with the cut-offs in `configs/risk.yaml`:

| Percentile | Level |
|---|---|
| below 50 | Low |
| 50 to below 75 | Moderate |
| 75 to below 90 | High |
| 90 and above | Very high |

These cut-offs are a **configuration choice, not a finding**. Nothing was fitted or validated to pick them.

## What overrides the level

The level is **`Insufficient evidence`** (and `percentile` is `null`) if any of these hold:

- `confidence` is `low` in the forecast file (this covers `no_last_year_value`, `short_history` and `capped_at_historical_max`);
- the state has fewer than `min_prior_years` (5) previous observed years.

The notes that caused it are returned in `risk_reason` and `notes`. Clients must show them and must not draw a risk colour for this level. Currently 28 of the 108 horizon-1 rows are `Insufficient evidence`.

## Growth is a separate field

`growth_vs_persistence_pct = 100 * (forecast - persistence) / persistence`, where persistence is last year's value. It is `null` when persistence is missing or 0. It is **not** blended into the level: two forecasts with the same percentile get the same level whatever their growth.

## What the level does not mean

- It is **not a prediction of an outbreak** and not an early-warning signal. The model is statistically tied with last year's value (`docs/handoff/README.md`, section 4).
- It compares the forecast with the state's own past. A "Low" state can still have more cases than a "High" state elsewhere. It does not compare states with each other.
- The data is annual, up to 2022. There is nothing monthly or live here.
- Horizons 2 and 3 are experimental: no better than persistence, and they have no interval.

## Not included

No weighted composite of trend, anomaly and environment. Person 3's `anomaly_annual.csv` is loaded into the database but does not feed the level, because there is no validation of any weights.

## Interval attached to each horizon-1 forecast

An 80% interval from past forecast errors, see `docs/calibration_report.md`. Observed coverage in a walk-forward check is below 80%, so treat the intervals as slightly optimistic, and very wide.

## Reading the risk map (added by Person 2)

- **A risk level compares a state with its own past, not with other states.** It is the percentile of the forecast among that state's previous observed years. A state with a large burden can show Low, and a small one can show Very high.
- **Trend can drive the level.** If reported cases have fallen over the years, forecasts near last year's value rank low; if they have risen, they rank high. In the current output almost all malaria forecasts are Low while many dengue forecasts are High or Very high. A trend effect is the likely reason, but this has not been verified.
- **A level is not a prediction of an outbreak.** The model is statistically tied with last year's value.
- **Andhra Pradesh history has a rescaling break.** Its values before the cutoff are whole numbers multiplied by 0.5861509524794264 (dengue through 2009, malaria through at least 2010 in the data seen; the cutoff differs by disease). The cause is not confirmed (probably an apportionment after the Telangana split); Person 1 has been asked. Treat Andhra's history, its percentiles and its early backtest folds with caution.
