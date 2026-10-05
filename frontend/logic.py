"""Display rules. Tested in tests/. Keep UI free of rule logic."""
import pandas as pd
INSUFF = "Insufficient evidence"
COLOURS = {"Lower": "#9ecae1", "Medium": "#fdae6b", "Higher": "#de2d26", INSUFF: "#d9d9d9"}
SHORT = {"Lower": "L", "Medium": "M", "Higher": "H", INSUFF: "?"}

def build_display(rows, horizon=1):
    """Forecast rows -> display rows. Always has a persistence text; low confidence -> no risk colour."""
    df = pd.DataFrame(rows)
    df = df[df["horizon"] == horizon].copy()
    df["risk"] = INSUFF
    for d, g in df[df["confidence"] == "normal"].groupby("disease"):
        r = g["forecast"].rank(pct=True)
        df.loc[g.index, "risk"] = pd.cut(r, [0, 1/3, 2/3, 1.0], labels=["Lower", "Medium", "Higher"],
                                         include_lowest=True).astype(str)
    df["colour"] = df["risk"].map(COLOURS)
    df["persistence_text"] = df["persistence"].apply(
        lambda v: "not available (no last-year value)" if v is None or pd.isna(v) else f"{float(v):,.0f}")
    df["forecast_text"] = df["forecast"].apply(lambda v: f"{float(v):,.0f}")
    df["notes"] = df["notes"].fillna("")
    return df
