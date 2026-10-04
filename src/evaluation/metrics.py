import numpy as np

def mae(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.mean(np.abs(pred - true)))

def log_mae(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.mean(np.abs(np.log1p(pred) - np.log1p(true))))

def paired_diff(model_errors, reference_errors):
    """Per-fold difference, model minus reference. Negative = model better."""
    return np.asarray(model_errors, float) - np.asarray(reference_errors, float)

def bootstrap_ci(values, n=10000, seed=0, alpha=0.05):
    """Mean and (1-alpha) percentile bootstrap CI over folds. Returns (mean, lo, hi)."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))

def summarize_vs_reference(model_errors, reference_errors, n=10000, seed=0):
    d = paired_diff(model_errors, reference_errors)
    m, lo, hi = bootstrap_ci(d, n=n, seed=seed)
    return {"diff": m, "ci_lo": lo, "ci_hi": hi, "tie": bool(lo <= 0 <= hi)}
