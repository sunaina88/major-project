"""Compare simple annual forecasters on the ST-GNN rolling-origin folds.

Run from the repository root:
    python scripts/evaluate_simple_baselines.py

The script writes only simple_baselines_{folds,pairs,summary}.csv under
data/processed/. All trend and smoothing settings are fixed in advance; no
test-year observations are used to choose parameters.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.backtest import run_backtest
from src.evaluation.data import get_data
from src.evaluation.metrics import summarize_vs_reference

SES_ALPHAS = (0.2, 0.5, 0.8)
TREND_SHRINKAGES = (0.25, 0.5, 0.75)
HOLT_ALPHA = 0.5
HOLT_BETA = 0.2
HOLT_DAMPING = 0.8
TREND_WINDOW = 5


def _to_cases(log_forecast, history_cases):
    """Keep extrapolations nonnegative and no higher than the training maximum."""
    log_history = np.log1p(history_cases)
    clipped = np.clip(log_forecast, 0.0, float(np.max(log_history)))
    return min(float(np.expm1(clipped)), float(np.max(history_cases)))


def _ses_forecast(log_values, alpha):
    level = float(log_values[0])
    for value in log_values[1:]:
        level = alpha * float(value) + (1.0 - alpha) * level
    return level


def _shrunken_trend_forecast(years, log_values, shrinkage, target_year):
    if len(log_values) < 2:
        return float(log_values[-1])

    x = years[-TREND_WINDOW:].astype(float)
    y = log_values[-TREND_WINDOW:]
    centered_x = x - x.mean()
    slope = float(np.dot(centered_x, y - y.mean()) / np.dot(centered_x, centered_x))
    return float(log_values[-1] + shrinkage * slope * (target_year - years[-1]))


def _damped_holt_forecast(years, log_values, target_year):
    if len(log_values) < 2:
        return float(log_values[-1])

    level = float(log_values[0])
    gap = max(int(years[1] - years[0]), 1)
    trend = float((log_values[1] - log_values[0]) / gap)
    previous_year = int(years[0])

    for year, value in zip(years[1:], log_values[1:]):
        gap = max(int(year - previous_year), 1)
        damped_sum = HOLT_DAMPING * (1.0 - HOLT_DAMPING**gap) / (1.0 - HOLT_DAMPING)
        projected_level = level + damped_sum * trend
        projected_trend = HOLT_DAMPING**gap * trend
        new_level = HOLT_ALPHA * float(value) + (1.0 - HOLT_ALPHA) * projected_level
        new_trend = HOLT_BETA * (new_level - level) + (1.0 - HOLT_BETA) * projected_trend
        level, trend = new_level, new_trend
        previous_year = int(year)

    gap = max(int(target_year - previous_year), 1)
    damped_sum = HOLT_DAMPING * (1.0 - HOLT_DAMPING**gap) / (1.0 - HOLT_DAMPING)
    return float(level + damped_sum * trend)


def make_predictor(method):
    """Build a predict_fn compatible with src.evaluation.backtest.run_backtest."""
    def predict(y_train, years_train, target_year):
        y_train = np.asarray(y_train, dtype=float)
        years_train = np.asarray(years_train, dtype=int)
        forecast = np.full(y_train.shape[1:], np.nan, dtype=float)

        for state_idx in range(y_train.shape[1]):
            for disease_idx in range(y_train.shape[2]):
                series = y_train[:, state_idx, disease_idx]
                observed = ~np.isnan(series)
                if not observed.any():
                    continue
                history = np.log1p(series[observed])
                observed_years = years_train[observed]

                if method == "persistence":
                    forecast[state_idx, disease_idx] = series[observed][-1]
                    continue
                if method.startswith("mean_last_"):
                    count = int(method.rsplit("_", 1)[1])
                    log_prediction = float(np.mean(history[-count:]))
                elif method.startswith("ses_alpha_"):
                    alpha = float(method.rsplit("_", 1)[1])
                    log_prediction = _ses_forecast(history, alpha)
                elif method.startswith("shrink_trend_"):
                    shrinkage = float(method.rsplit("_", 1)[1])
                    log_prediction = _shrunken_trend_forecast(
                        observed_years, history, shrinkage, target_year
                    )
                elif method == "damped_holt":
                    log_prediction = _damped_holt_forecast(
                        observed_years, history, target_year
                    )
                else:
                    raise ValueError(f"Unknown baseline method: {method}")

                forecast[state_idx, disease_idx] = _to_cases(
                    log_prediction, series[observed]
                )

        return forecast

    return predict


def _methods():
    methods = ["persistence", "mean_last_3", "mean_last_5"]
    methods.extend(f"ses_alpha_{alpha:.1f}" for alpha in SES_ALPHAS)
    methods.extend(f"shrink_trend_{value:.2f}" for value in TREND_SHRINKAGES)
    methods.append("damped_holt")
    return methods


def evaluate(data_path, start_year, out_prefix):
    states, years, targets, _ = get_data(data_path)
    years = np.asarray(years, dtype=int)
    if start_year < years[0] or start_year > years[-1]:
        raise ValueError(
            f"Start year {start_year} must be between {years[0]} and {years[-1]}."
        )

    test_years = range(start_year, int(years[-1]) + 1)
    fold_frames = []
    pair_frames = []

    for method in _methods():
        folds, pairs = run_backtest(
            targets, years, states, make_predictor(method), test_years=test_years
        )
        folds = folds.rename(
            columns={
                "mae_model": f"mae_{method}",
                "log_model": f"log_mae_{method}",
            }
        )
        folds = folds.drop(columns=["mae_last", "mae_avg2", "log_last", "log_avg2"])
        if method != "persistence":
            folds = folds.drop(columns=["n"])
        fold_frames.append(folds.set_index("year"))

        pairs = pairs.rename(
            columns={"model": method, "persistence": "persistence"}
        )
        pair_frames.append(pairs.set_index(["year", "state_id", "disease"]))

    fold_results = pd.concat(fold_frames, axis=1).reset_index()
    pair_results = pd.concat(pair_frames, axis=1)
    pair_results = pair_results.loc[:, ~pair_results.columns.duplicated()].reset_index()

    summary_rows = []
    for method in _methods():
        log_scores = fold_results[f"log_mae_{method}"]
        row = {
            "method": method,
            "mean_log_mae": float(log_scores.mean()),
            "mean_mae": float(fold_results[f"mae_{method}"].mean()),
            "folds": int(log_scores.notna().sum()),
        }
        if method == "persistence":
            row.update(
                diff_vs_persistence=0.0,
                ci_95_low=0.0,
                ci_95_high=0.0,
                folds_better_than_persistence=0,
            )
        else:
            comparison = summarize_vs_reference(
                log_scores.to_numpy(),
                fold_results["log_mae_persistence"].to_numpy(),
            )
            differences = log_scores.to_numpy() - fold_results[
                "log_mae_persistence"
            ].to_numpy()
            row.update(
                diff_vs_persistence=comparison["diff"],
                ci_95_low=comparison["ci_lo"],
                ci_95_high=comparison["ci_hi"],
                folds_better_than_persistence=int(np.sum(differences < 0)),
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fold_path = prefix.with_name(f"{prefix.name}_folds.csv")
    pair_path = prefix.with_name(f"{prefix.name}_pairs.csv")
    summary_path = prefix.with_name(f"{prefix.name}_summary.csv")
    fold_results.to_csv(fold_path, index=False)
    pair_results.to_csv(pair_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(summary.round(3).to_string(index=False))
    print(f"\nWrote:\n  {fold_path}\n  {pair_path}\n  {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/processed/pan_india_state_enriched_tensor.csv",
        help="Annual tensor used by the ST-GNN.",
    )
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument(
        "--out-prefix",
        default="data/processed/simple_baselines",
        help="Prefix for the three new CSV outputs.",
    )
    args = parser.parse_args()
    evaluate(args.data, args.start_year, args.out_prefix)


if __name__ == "__main__":
    main()
