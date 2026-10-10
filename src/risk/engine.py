"""Risk engine: a documented rule, not a colour. Method in docs/risk_method.md."""
import numpy as np
import yaml

from src.api.config import REPO_ROOT

LEVELS = ["Low", "Moderate", "High", "Very high"]


def load_config(path=None):
    path = path or REPO_ROOT / "configs" / "risk.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def percentile_rank(value, prior):
    """Share (0-100) of previous observed years that are STRICTLY smaller than the value.

    Strict on purpose: if a state always reported 0 and the forecast is 0, the forecast
    is not 'higher than' any earlier year, so the percentile is 0 and the level is Low.
    """
    prior = np.asarray(prior, dtype=float)
    prior = prior[~np.isnan(prior)]
    if len(prior) == 0:
        return None
    count = 0
    for v in prior:
        if v < value:
            count += 1
    return 100.0 * count / len(prior)


def level_from_percentile(pct, cutoffs):
    if pct < cutoffs["moderate"]:
        return "Low"
    if pct < cutoffs["high"]:
        return "Moderate"
    if pct < cutoffs["very_high"]:
        return "High"
    return "Very high"


def growth_vs_persistence(forecast, persistence):
    """Percent change of the forecast relative to last year's value. Kept separate from the level."""
    if persistence is None or np.isnan(persistence) or persistence <= 0:
        return None
    return 100.0 * (forecast - persistence) / persistence


def assess(forecast, persistence, confidence, notes, prior_values, cfg):
    """Return the risk fields for one state / disease / year.

    notes: list of strings from the forecast file.
    prior_values: that state's previous observed annual values for the same disease.
    """
    r = cfg["risk"]
    prior = np.asarray(prior_values, dtype=float)
    prior = prior[~np.isnan(prior)]
    growth = growth_vs_persistence(forecast, persistence)
    out = {"risk_level": None, "percentile": None, "growth_vs_persistence_pct": growth,
           "n_prior_years": int(len(prior)), "reason": ""}

    reasons = []
    if confidence == "low":
        reasons.append("low_confidence")
        for n in notes:
            reasons.append(n)
    if len(prior) < r["min_prior_years"]:
        reasons.append("fewer_than_min_prior_years")

    if reasons:
        # No level and no percentile: a number next to "Insufficient evidence" invites a colour.
        out["risk_level"] = r["insufficient_label"]
        out["reason"] = ";".join(reasons)
        return out
    out["percentile"] = percentile_rank(forecast, prior)
    out["risk_level"] = level_from_percentile(out["percentile"], r["cutoffs"])
    return out
