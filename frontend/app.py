import pathlib, numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
import api_client as api
from logic import build_display, COLOURS, SHORT, INSUFF

st.set_page_config(page_title="VectorWatch India", layout="wide")
STATES = [s.strip() for s in open(pathlib.Path(__file__).parent.parent / "configs/states.txt") if s.strip()]
DIS = ["dengue", "malaria", "chikungunya"]

@st.cache_data
def load(name): return api.get(name)

PAGES = ["Home", "State", "Disease explorer", "Model comparison", "Annual replay", "System status", "Limitations"]
page = st.sidebar.radio("Page", PAGES)
st.sidebar.caption("Annual data, latest year 2022. Not live.")
st.info("Annual data, latest year 2022. The model is statistically tied with last year's value (persistence). "
        "Pick **Limitations** in the sidebar for details.")
if api.USE_MOCKS: st.caption("Running on mock data generated from Person 2 files.")

fc = load("forecast"); hist = pd.DataFrame(load("history"))

def horizon_picker():
    h = st.radio("Years ahead", [1, 2, 3], horizontal=True, format_func=lambda x: str(x) if x == 1 else f"{x} (experimental)")
    if h > 1: st.warning("Horizons 2 and 3 are experimental and no better than persistence.")
    return h

if page == "Home":
    st.title("VectorWatch India")
    d = st.tabs(DIS)
    for tab, dis in zip(d, DIS):
        with tab:
            df = build_display(fc, 1); df = df[df.disease == dis].set_index("state_id").reindex(STATES).reset_index()
            df["risk"] = df["risk"].fillna(INSUFF); df["colour"] = df["colour"].fillna(COLOURS[INSUFF])
            levels = list(COLOURS); z = np.array([levels.index(r) for r in df["risk"]]).reshape(6, 6)
            text = np.array([f"{s[:10]}<br>{SHORT[r]}" for s, r in zip(df.state_id, df.risk)]).reshape(6, 6)
            cs = [[i/4 + o, COLOURS[l]] for i, l in enumerate(levels) for o in (0, 0.25 - 1e-9)]
            fig = go.Figure(go.Heatmap(z=z, text=text, texttemplate="%{text}", colorscale=cs, zmin=0, zmax=3,
                                       showscale=False, xgap=3, ygap=3, hoverinfo="text"))
            fig.update_yaxes(autorange="reversed", visible=False); fig.update_xaxes(visible=False)
            fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Placeholder tile map (one tile per state, alphabetical). Letters: L=Lower, M=Medium, H=Higher, "
                       "?=Insufficient evidence. Replaced by india_states.geojson when Person 4 delivers.")
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
        if r.confidence == "low": st.warning(f"{INSUFF}. Reason: {r.notes or 'low confidence'}")
        st.caption("No prediction interval in the forecast file. Add it when Person 5's API provides one.")
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
    st.caption("13 rolling-origin folds, 2010 to 2022. If the 95% interval includes 0, the result is a tie. About a dozen variants were run.")

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
    st.markdown("""
- Monthly disease values in the source tensor are an annual total split with one fixed curve, so **no monthly charts or accuracy are shown**.
- The real information is annual totals (about 1,600 numbers).
- On annual totals the model is **statistically tied with last year's value**. The graph, GAT and weather variants change nothing significant.
- Weather direction is favourable but not significant.
- Data runs to 2022 only. This is not live or real-time.
- Rows marked low confidence show *Insufficient evidence* instead of a risk colour.
- Rerunning on real monthly or newer annual data is how this can still give a positive result.
""")
