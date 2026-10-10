"""Empirical intervals from past forecast errors (log scale)."""
import numpy as np


def log_errors(model, true):
    """e = log1p(model) - log1p(true). Positive means the model was too high."""
    return np.log1p(np.asarray(model, float)) - np.log1p(np.asarray(true, float))


def fit_quantiles(errors, nominal=0.80):
    """Lower and upper error quantiles for a central interval, e.g. 10% and 90% for 80%."""
    tail = (1.0 - nominal) / 2.0
    e = np.asarray(errors, float)
    return float(np.quantile(e, tail)), float(np.quantile(e, 1.0 - tail))


def apply_interval(forecast, q_lo, q_hi):
    """true = forecast * exp(-e), so the interval is expm1(log1p(f) - q_hi) .. expm1(log1p(f) - q_lo)."""
    z = np.log1p(float(forecast))
    low = max(0.0, float(np.expm1(z - q_hi)))
    high = max(0.0, float(np.expm1(z - q_lo)))
    return int(round(low)), int(round(high))


def fit_interval_model(pairs, nominal=0.80, by_disease=False):
    """pairs: DataFrame with columns disease, true, model. Returns {key: (q_lo, q_hi)}.

    key is a disease name when by_disease is True, otherwise the single key 'all'.
    """
    models = {}
    if by_disease:
        for d in sorted(pairs["disease"].unique()):
            sub = pairs[pairs["disease"] == d]
            models[d] = fit_quantiles(log_errors(sub["model"], sub["true"]), nominal)
    else:
        models["all"] = fit_quantiles(log_errors(pairs["model"], pairs["true"]), nominal)
    return models


def interval_for(models, disease, forecast):
    key = disease if disease in models else "all"
    q_lo, q_hi = models[key]
    return apply_interval(forecast, q_lo, q_hi)
