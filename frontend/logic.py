"""Display rules. Tested in tests/. Keep UI free of rule logic."""
import pandas as pd

LEVELS = ["Low", "Moderate", "High", "Very high", INSUFF := "Insufficient evidence"]
COLOURS = {"Low": "#9ecae1", "Moderate": "#fdae6b", "High": "#de2d26", "Very high": "#a50f15", INSUFF: "#808080"}
SHORT = {"Low": "L", "Moderate": "M", "High": "H", "Very high": "V", INSUFF: "?"}


# One colour family per disease (light to dark = Low to Very high); grey = insufficient evidence.
PALETTES = {"dengue": ["#fcbba1", "#fb6a4a", "#de2d26", "#a50f15"],
            "malaria": ["#c6dbef", "#6baed6", "#3182bd", "#08519c"],
            "chikungunya": ["#c7e9c0", "#74c476", "#31a354", "#006d2c"]}
DISEASE_COLOURS = {d: p[2] for d, p in PALETTES.items()}


def colour_map(disease=None):
    if disease in PALETTES:
        return {**dict(zip(LEVELS[:4], PALETTES[disease])), INSUFF: COLOURS[INSUFF]}
    return COLOURS


def discrete_scale(levels, disease=None):
    """Plotly colour scale with one flat band per level (first stop 0, last stop exactly 1)."""
    n, cm, cs = len(levels), colour_map(disease), []
    for i, lvl in enumerate(levels):
        cs += [[i / n, cm[lvl]], [(i + 1) / n, cm[lvl]]]
    return cs


def nice(state_id):
    """andaman_and_nicobar_islands -> Andaman and Nicobar Islands"""
    return state_id.replace("_", " ").title().replace(" And ", " and ")


def build_display(rows, horizon=1):
    """Forecast rows -> display rows. Always has a persistence text; low confidence -> no risk colour."""
    df = pd.DataFrame(rows)
    df = df[df["horizon"] == horizon].copy()
    df["risk"] = INSUFF
    for d, g in df[df["confidence"] == "normal"].groupby("disease"):
        r = g["forecast"].rank(pct=True)
        df.loc[g.index, "risk"] = pd.cut(r, [0, 1/3, 2/3, 1.0], labels=["Low", "Moderate", "High"],
                                         include_lowest=True).astype(str)
    if "risk_level" in df.columns:          # real API: use Person 5's risk engine, do not recompute
        df["risk"] = df["risk_level"].fillna(INSUFF)
    df["colour"] = df["risk"].map(COLOURS)
    df["persistence_text"] = df["persistence"].apply(
        lambda v: "not available (no last-year value)" if v is None or pd.isna(v) else f"{float(v):,.0f}")
    df["forecast_text"] = df["forecast"].apply(lambda v: f"{float(v):,.0f}")
    df["notes"] = df["notes"].fillna("")
    return df

# ---- helpers for the research pages (kept here so they can be tested without Streamlit) ----

def is_tie(ci):
    """True when the 95% interval is missing or includes 0: no clear difference from persistence."""
    if ci is None or len(ci) != 2:
        return True
    return ci[0] <= 0 <= ci[1]


def verdict(ci):
    """'tie', 'better than persistence' or 'worse than persistence' (diff is model minus persistence)."""
    if is_tie(ci):
        return "tie"
    return "better than persistence" if ci[1] < 0 else "worse than persistence"


def closer_years(d):
    """d: replay rows of one state and disease (columns true, model, persistence).
    Returns (model_closer, persistence_closer, equal) counted over years, using |log1p(x) - log1p(true)|."""
    model_closer = pers_closer = equal = 0
    for t, m, p in zip(d["true"], d["model"], d["persistence"]):
        e_model = abs(_log1p(m) - _log1p(t))
        e_pers = abs(_log1p(p) - _log1p(t))
        if e_model < e_pers:
            model_closer += 1
        elif e_pers < e_model:
            pers_closer += 1
        else:
            equal += 1
    return model_closer, pers_closer, equal


def _log1p(v):
    import math
    return math.log1p(max(float(v), 0.0))


def reported_grid(hist, disease, states, years):
    """Matrix [state][year] of 1 (a value was reported) or 0 (no value)."""
    h = hist[hist.disease == disease]
    seen = set(zip(h.state_id, h.year))
    return [[1 if (s, y) in seen else 0 for y in years] for s in states]
