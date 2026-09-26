"""
train_baselines.py

Trains and evaluates baseline forecasting models (Seasonal Naive and two
HistGradientBoosting variants) for all states across India using the
enriched pan-India spatiotemporal tensor.

Owner: Person 1 (Disease Data & Forecasting Engineer)

--------------------------------------------------------------------------
Why two Gradient Boosting variants
--------------------------------------------------------------------------
This script reads the *enriched* tensor (disease targets + climate + lag
features) and deliberately trains two Gradient Boosting variants side by
side, using the exact same train/test split and disease-lag features, so
the only difference between them is whether weather/lag features are
included:

    - Gradient_Boosting          : disease-lag features only
                                    (lag_1, lag_2, lag_12, month)
    - Gradient_Boosting_Weather  : disease-lag features + engineered
                                    climate/lag features (temp_mean_c,
                                    rainfall_mm, temp_lag_1/2,
                                    rainfall_lag_1/2, rainfall_rolling_3m)

This makes the value of the weather/lag feature engineering (added by
build_weather_tensor.py / build_lag_features.py) directly measurable
against a fair, matched baseline, rather than leaving those features
unvalidated by any benchmark.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------------------------------
# Config & Paths
# --------------------------------------------------------------------------
INPUT_TENSOR = Path("data/processed/pan_india_state_enriched_tensor.csv")
OUTPUT_METRICS_PATH = Path("data/processed/baseline_model_metrics.csv")

TARGET_DISEASES = ["dengue_cases", "malaria_cases", "chikungunya_cases"]

# Engineered climate/lag features available on the enriched tensor.
WEATHER_FEATURES = [
    "temp_mean_c", "rainfall_mm",
    "temp_lag_1", "temp_lag_2",
    "rainfall_lag_1", "rainfall_lag_2",
    "rainfall_rolling_3m",
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def evaluate_forecast(y_true, y_pred) -> dict:
    """Computes MAE and RMSE, ignoring NaNs."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return {"mae": np.nan, "rmse": np.nan}
    
    yt = y_true[mask]
    yp = y_pred[mask]
    
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    return {"mae": mae, "rmse": rmse}


def run_baselines():
    if not INPUT_TENSOR.exists():
        raise FileNotFoundError(f"Master tensor not found at {INPUT_TENSOR}. Run ingestion first.")

    df = pd.read_csv(INPUT_TENSOR)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df = df.sort_values(["state_name", "start_date"]).reset_index(drop=True)

    all_metrics = []

    logger.info("Starting baseline evaluation across all states and diseases...")

    for state in df["state_name"].unique():
        state_df = df[df["state_name"] == state].copy()
        
        for disease in TARGET_DISEASES:
            series = state_df[disease].values
            if np.isnan(series).all():
                continue

            # Feature Engineering: lags (t-12 for seasonal lag, t-1, t-2)
            state_df["lag_1"] = state_df[disease].shift(1)
            state_df["lag_2"] = state_df[disease].shift(2)
            state_df["lag_12"] = state_df[disease].shift(12)  # Seasonal lag (1 year prior)
            state_df["month"] = state_df["start_date"].dt.month

            # Train / Test split (train on < 2021, test on 2021-2022)
            train_mask = state_df["start_date"].dt.year < 2021
            test_mask = state_df["start_date"].dt.year >= 2021

            train_data = state_df[train_mask].dropna(subset=["lag_12", disease])
            test_data = state_df[test_mask]

            if train_data.empty or test_data.empty:
                continue

            # 1. Seasonal Naive Baseline (predict exactly 12 months prior)
            y_true_test = test_data[disease].values
            y_pred_naive = test_data["lag_12"].values
            naive_metrics = evaluate_forecast(y_true_test, y_pred_naive)

            all_metrics.append({
                "state_name": state,
                "disease": disease,
                "model": "Seasonal_Naive",
                "mae": naive_metrics["mae"],
                "rmse": naive_metrics["rmse"]
            })

            # 2. Scikit-Learn Gradient Boosting Baseline (disease-lags only)
            lag_features = ["lag_1", "lag_2", "lag_12", "month"]
            X_train = train_data[lag_features]
            y_train = train_data[disease]
            X_test = test_data[lag_features]

            # Native missing value support via HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(random_state=42)
            model.fit(X_train, y_train)

            y_pred_gb = model.predict(X_test)
            gb_metrics = evaluate_forecast(y_true_test, y_pred_gb)

            all_metrics.append({
                "state_name": state,
                "disease": disease,
                "model": "Gradient_Boosting",
                "mae": gb_metrics["mae"],
                "rmse": gb_metrics["rmse"]
            })

            # 3. Scikit-Learn Gradient Boosting + Weather/Lag Features
            #    Same split, same disease-lag features, PLUS the engineered
            #    climate/lag columns -- isolates the effect of the weather
            #    feature engineering versus baseline #2 above.
            weather_features = lag_features + WEATHER_FEATURES
            X_train_w = train_data[weather_features]
            X_test_w = test_data[weather_features]

            model_weather = HistGradientBoostingRegressor(random_state=42)
            model_weather.fit(X_train_w, y_train)

            y_pred_gbw = model_weather.predict(X_test_w)
            gbw_metrics = evaluate_forecast(y_true_test, y_pred_gbw)

            all_metrics.append({
                "state_name": state,
                "disease": disease,
                "model": "Gradient_Boosting_Weather",
                "mae": gbw_metrics["mae"],
                "rmse": gbw_metrics["rmse"]
            })

    metrics_df = pd.DataFrame(all_metrics)
    OUTPUT_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUTPUT_METRICS_PATH, index=False)
    
    logger.info(f"Baseline training complete. Metrics saved to {OUTPUT_METRICS_PATH}")
    print("\n--- Baseline Performance Sample ---")
    print(metrics_df.dropna().head(10).to_string(index=False))


if __name__ == "__main__":
    run_baselines()