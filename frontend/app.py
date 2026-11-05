import json
import os
import pathlib

import api_client as api
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from logic import COLOURS, INSUFF, LEVELS, build_display, discrete_scale

st.set_page_config(page_title="VectorWatch India", layout="wide")
ROOT = pathlib.Path(__file__).resolve().parent.parent
STATES = [s.strip() for s in open(ROOT / "configs/states.txt") if s.strip()]
GEOJSON = json.load(open(os.getenv("VW_GEOJSON", ROOT / "data/processed/india_states.geojson")))
SMALL = ["lakshadweep", "andaman_and_nicobar_islands", "chandigarh", "delhi",
         "dadra_and_nagar_haveli_and_daman_and_diu"]
DIS = ["dengue", "malaria", "chikungunya"]

@st.cache_data
def _load(name): return api.get(name)

def load(name):
    try:
        return _load(name)
    except api.NotAvailable as e:
        st.warning(f"{e} Run with USE_MOCKS=true until it exists.")
        st.stop()

PAGES = ["Home", "State", "Disease explorer", "Model comparison", "Annual replay", "System status", "Limitations"]
page = st.sidebar.radio("Page", PAGES)
st.sidebar.caption("Annual data. Not live.")
st.info("Annual forecasts are statistically tied with last year's value (persistence), so both are always shown. "
        "Pick **Limitations** in the sidebar for details.")
if api.USE_MOCKS: st.caption("Running on mock data generated from Person 2 files.")

fc = load("forecast"); hist = pd.DataFrame(load("history"))
latest_year = int(hist.year.max())
st.sidebar.caption(f"Latest observed year: {latest_year}")

def horizon_picker():
    h = st.radio("Years ahead", [1, 2, 3], horizontal=True, format_func=lambda x: str(x) if x == 1 else f"{x} (experimental)")
    if h > 1: st.warning("Horizons 2 and 3 are experimental; compare them with persistence.")
    return h

if page == "Home":
    st.title("VectorWatch India")
    d = st.tabs(DIS)
    for tab, dis in zip(d, DIS):
        with tab:
            st.caption(f"Annual data, latest observed year {latest_year}. Forecast year: next year (horizon 1).")
            df = build_display(fc, 1); df = df[df.disease == dis].set_index("state_id").reindex(STATES).reset_index()
            df["risk"] = df["risk"].fillna(INSUFF); df["colour"] = df["colour"].fillna(COLOURS[INSUFF])
            df["forecast_text"] = df["forecast_text"].fillna("n/a"); df["persistence_text"] = df["persistence_text"].fillna("n/a")
            hover = [f"{n}<br>Risk: {r}<br>Forecast: {f}<br>Last year: {p}" for n, r, f, p in
                     zip(df.state_id, df.risk, df.forecast_text.fillna("n/a"), df.persistence_text.fillna("n/a"))]
            fig = go.Figure(go.Choropleth(
                geojson=GEOJSON, locations=df.state_id, featureidkey="properties.state_id",
                z=[LEVELS.index(r) for r in df.risk], colorscale=discrete_scale(LEVELS),
                zmin=-0.5, zmax=len(LEVELS) - 0.5, showscale=False, text=hover, hoverinfo="text",
                marker_line_width=0.5))
            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True, key=f"map_{dis}")
            st.caption("Colour scale (text alternative): " + ", ".join(f"{l} ({COLOURS[l]})" for l in LEVELS)
                       + ". Grey = insufficient evidence, no risk is shown. The table below repeats every value in text.")
            st.markdown("**Small territories and islands** (hard to click on the map)")
            st.dataframe(df[df.state_id.isin(SMALL)][["state_id", "risk", "forecast_text", "persistence_text"]], hide_index=True)
            st.dataframe(df[["state_id", "risk", "forecast_text", "persistence_text", "confidence"]], hide_index=True)

elif page == "State":
    s = st.selectbox("State", STATES); dis = st.selectbox("Disease", DIS); h = horizon_picker()
    df = build_display(fc, h); r = df[(df.state_id == s) & (df.disease == dis)]
    g = hist[(hist.state_id == s) & (hist.disease == dis)]
    fig = go.Figure(go.Scatter(x=g.year, y=g.cases, mode="lines+markers", name="Reported (annual)"))
    if len(r):
        r = r.iloc[0]
        fig.add_trace(go.Scatter(x=[r.year], y=[r.forecast], mode="markers", name="Forecast", marker=dict(size=12, symbol="diamond")))
        if r.persistence is not None and not pd.isna(r.persistence):
            fig.add_trace(go.Scatter(x=[r.year], y=[r.persistence], mode="markers", name="Persistence", marker=dict(size=12, symbol="x")))
        c1, c2, c3 = st.columns(3)
        c1.metric("Forecast", r.forecast_text); c2.metric("Persistence (last year)", r.persistence_text)
        c3.metric("Confidence", r.confidence.upper())
        iv = r.get("interval_80")
        st.write(f"**Risk level:** {r.risk}" + (f" (percentile {r.get('percentile'):.0f})" if pd.notna(r.get("percentile")) else ""))
        st.write("**80% interval:** " + (f"{iv[0]:,} to {iv[1]:,} cases" if isinstance(iv, (list, tuple)) else "not available (experimental horizon)"))
        if r.confidence == "low":
            st.warning(f"{INSUFF}. Reason: {r.notes or 'low confidence'}")
        elif isinstance(r.get("risk_reason"), (list, tuple)) and r.get("risk_reason"):
            st.caption("Risk reason: " + ", ".join(r.get("risk_reason")))
    st.plotly_chart(fig, use_container_width=True)

elif page == "Disease explorer":
    dis = st.selectbox("Disease", DIS)
    g = hist[hist.disease == dis]
    st.subheader("National annual total"); st.line_chart(g.groupby("year").cases.sum())
    y = st.slider("Year", int(g.year.min()), int(g.year.max()), int(g.year.max()))
    gy = g[g.year == y].sort_values("cases", ascending=False)
    st.subheader(f"Top states, {y}"); st.bar_chart(gy.head(10).set_index("state_id").cases)
    st.subheader("Distribution across states (log10 cases)")
    st.bar_chart(np.histogram(np.log10(gy.cases + 1), bins=10)[0])

elif page == "Model comparison":
    st.title("Model comparison (log-MAE, lower is better)")
    st.dataframe(pd.DataFrame(load("model_performance")), hide_index=True)
    replay_years = sorted(pd.DataFrame(load("replay")).year.unique())
    st.caption(f"Rolling-origin folds, {replay_years[0]} to {replay_years[-1]}. "
               "If the 95% interval includes 0, the result is a tie.")

elif page == "Annual replay":
    st.title("Annual replay (backtest)")
    rp = pd.DataFrame(load("replay")); y = st.selectbox("Year", sorted(rp.year.unique())); dis = st.selectbox("Disease", DIS)
    d = rp[(rp.year == y) & (rp.disease == dis)].copy()
    lm = lambda c: float(np.abs(np.log1p(d[c]) - np.log1p(d["true"])).mean())
    c1, c2 = st.columns(2); c1.metric("log-MAE model", f"{lm('model'):.3f}"); c2.metric("log-MAE persistence", f"{lm('persistence'):.3f}")
    st.dataframe(d[["state_id", "true", "model", "persistence"]], hide_index=True)
    st.caption("Backtest on past years. Not a live forecast.")

elif page == "System status":
    st.title("System status"); rep = load("drift_report")
    st.write("Flagged" if rep["flagged"] else "No drift flags", rep["flags"]); st.json(rep)

else:
    st.title("Limitations")
    st.markdown(f"""
- Only annual totals are real. Monthly disease values are a fixed split of the annual total, so **no monthly charts or accuracy are shown**.
- On annual totals the model is **statistically tied with last year's value** (persistence). The graph, GAT and weather variants make no significant difference.
- Weather comes from NASA POWER, one centroid per state. Its effect points the right way but is not significant.
- Data runs through {latest_year}. This is annual data, not live or real-time.
- A risk level ranks the forecast against the state's own history. It is not a prediction of an outbreak.
- Rows marked low confidence show *Insufficient evidence* instead of a risk colour, with the reason.
- Horizons 2 and 3 are experimental and no better than persistence.
- Rerunning on real monthly or newer annual data is how this can still give a positive result.
""")
