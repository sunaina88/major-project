import json
import os
import pathlib

import api_client as api
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import extra_pages as ex
from logic import DISEASE_COLOURS, INSUFF, LEVELS, build_display, colour_map, discrete_scale, nice

st.set_page_config(page_title="VectorWatch India", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 4.5rem; max-width: 1400px; }
h1 { font-weight: 650; letter-spacing: -0.01em; }
section[data-testid="stSidebar"] { border-right: 1px solid rgba(143,166,186,.18); }
[data-testid="stMetricLabel"] p { font-size: .8rem; color: #8FA6BA; }
.stTabs [data-baseweb="tab-list"] { gap: 1.2rem; }
.key { font-size: .85rem; color: #8FA6BA; margin: .2rem 0 .6rem 0; }
.key span { margin-right: 1.1rem; white-space: nowrap; }
footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)
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

PAGES = ["Home", "How it works", "State", "Disease explorer", "Time-lapse map", "Why this forecast?", "Network",
         "Research results", "Annual replay", "Data story", "Limitations"]
page = st.sidebar.radio("Page", PAGES)

fc = load("forecast"); hist = pd.DataFrame(load("history"))
latest_year = int(hist.year.max())
st.sidebar.caption(f"Annual data through {latest_year}")

def horizon_picker():
    h = st.radio("Years ahead", [1, 2, 3], horizontal=True, format_func=lambda x: str(x) if x == 1 else f"{x} (experimental)")
    if h > 1: st.caption("Horizons 2 and 3 are experimental and carry no interval.")
    return h

if page == "Home":
    st.title("VectorWatch India")
    st.caption(f"Annual surveillance data through {latest_year}. Forecasts are for {latest_year + 1}.")
    tabs = st.tabs([d.capitalize() for d in DIS])
    for tab, dis in zip(tabs, DIS):
        with tab:
            df = build_display(fc, 1); df = df[df.disease == dis].set_index("state_id").reindex(STATES).reset_index()
            df["risk"] = df["risk"].fillna(INSUFF)
            df["forecast_text"] = df["forecast_text"].fillna("n/a"); df["persistence_text"] = df["persistence_text"].fillna("n/a")
            df["State"] = df.state_id.map(nice)
            n = df.risk.value_counts()
            m = st.columns(4)
            m[0].metric("High or very high", int(n.get("High", 0) + n.get("Very high", 0)))
            m[1].metric("Moderate", int(n.get("Moderate", 0)))
            m[2].metric("Low", int(n.get("Low", 0)))
            m[3].metric("Not rated", int(n.get(INSUFF, 0)))
            cm = colour_map(dis)
            st.markdown('<div class="key">' + "".join(
                f'<span><b style="color:{cm[l]}">&#9632;</b> {"Not rated" if l == INSUFF else l}</span>' for l in LEVELS)
                + "</div>", unsafe_allow_html=True)
            hover = [f"<b>{n_}</b><br>Risk: {r}<br>Forecast: {f}<br>Last year: {p}" for n_, r, f, p in
                     zip(df.State, df.risk, df.forecast_text, df.persistence_text)]
            fig = go.Figure(go.Choropleth(
                geojson=GEOJSON, locations=df.state_id, featureidkey="properties.state_id",
                z=[LEVELS.index(r) for r in df.risk], colorscale=discrete_scale(LEVELS, dis),
                zmin=-0.5, zmax=len(LEVELS) - 0.5, showscale=False, text=hover, hoverinfo="text",
                marker_line_width=0.6, marker_line_color="#0A1524"))
            fig.update_geos(visible=False, lonaxis_range=[67, 98], lataxis_range=[6, 37.5], bgcolor="rgba(0,0,0,0)")
            fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", geo_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch", theme=None, key=f"map_{dis}")
            st.caption("Each level compares a state's forecast with that state's own earlier years.")
            q = st.text_input("Find a state", key=f"q_{dis}", placeholder="Type a state name")
            view = df[["State", "risk", "forecast_text", "persistence_text", "confidence"]].rename(columns={
                "risk": "Risk level", "forecast_text": "Forecast", "persistence_text": "Last year", "confidence": "Confidence"})
            view["Confidence"] = view["Confidence"].astype(str).str.capitalize()
            if q:
                view = view[view.State.str.contains(q, case=False)]
            st.dataframe(view, hide_index=True, width="stretch")

elif page == "How it works":
    ex.page_how(hist, STATES, fc, latest_year)

elif page == "State":
    st.title("State")
    sc1, sc2 = st.columns(2)
    scope = sc1.selectbox("State", ["India"] + STATES, index=STATES.index("kerala") + 1,
                          format_func=lambda x: x if x == "India" else nice(x))
    dis = sc2.selectbox("Disease", ["All"] + DIS, index=1)
    title = "All states" if scope == "India" else nice(scope)
    if dis == "All":
        h = hist if scope == "India" else hist[hist.state_id == scope]
        years = sorted(h.year.unique(), reverse=True)
        y = st.selectbox("Year", years)
        totals = h[h.year == y].groupby("disease").cases.sum().reindex(DIS).fillna(0)
        cols = st.columns(3)
        for c, d in zip(cols, DIS):
            c.metric(d.capitalize(), f"{int(totals[d]):,}")
        left, right = st.columns([2, 3])
        with left:
            if totals.sum() > 0:
                fig = go.Figure(go.Bar(x=[x.capitalize() for x in DIS], y=totals.values,
                                       marker_color=[DISEASE_COLOURS[x] for x in DIS]))
                fig.update_layout(title=f"Reported cases, {title}, {y}", height=380, yaxis_title="Cases")
                st.plotly_chart(fig, width="stretch", key="all_bar")
            else:
                st.info("No reported cases for this selection.")
        with right:
            tr = h.groupby(["year", "disease"]).cases.sum().reset_index()
            fig = go.Figure()
            for d in DIS:
                t = tr[tr.disease == d]
                fig.add_trace(go.Scatter(x=t.year, y=t.cases, mode="lines", name=d.capitalize(),
                                         line=dict(color=DISEASE_COLOURS[d])))
            fig.add_vline(x=y, line_dash="dot", line_color="#8FA6BA")
            fig.update_layout(title="Trend by disease (log scale)", height=380, yaxis_type="log", yaxis_title="Cases")
            fig.update_xaxes(tickformat="d")
            st.plotly_chart(fig, width="stretch", key="all_trend")
        tbl = h[h.year == y].pivot_table(index="state_id", columns="disease", values="cases", aggfunc="sum").reindex(columns=DIS)
        tbl = tbl.assign(total=tbl.sum(axis=1)).sort_values("total", ascending=False)
        tbl.index = tbl.index.map(nice); tbl.index.name = "State"
        st.dataframe(tbl.reset_index(), hide_index=True, width="stretch")
        st.caption(f"Reported annual cases in {y} (observed data, not forecasts).")
    elif scope == "India":
        g = hist[hist.disease == dis]
        y = st.slider("Year", int(g.year.min()), int(g.year.max()), int(g.year.max()), key="india_year")
        top = float(np.log10(g.cases.max() + 1))
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(ex.cases_map(g, y, GEOJSON, dis, STATES, top, height=480), width="stretch",
                            theme=None, key="india_map")
            st.caption("White = no cases reported.")
        with right:
            gy = g[g.year == y].sort_values("cases", ascending=False).head(8)
            names = [nice(x) for x in gy.state_id]
            fig = go.Figure(go.Bar(x=gy.cases[::-1], y=names[::-1], orientation="h", marker_color=DISEASE_COLOURS[dis]))
            fig.update_layout(title=f"Highest states, {y}", height=480, xaxis_title="Reported cases",
                              margin=dict(l=0, r=10, t=44, b=10))
            st.plotly_chart(fig, width="stretch", key="india_top")
        nat = g.groupby("year").cases.sum().reset_index()
        fig = go.Figure(go.Scatter(x=nat.year, y=nat.cases, mode="lines+markers", name="Reported (annual)",
                                   line=dict(color=DISEASE_COLOURS[dis])))
        fig.add_vline(x=y, line_dash="dot", line_color="#8FA6BA")
        fig.update_layout(title=f"{dis.capitalize()}: reported cases, all states combined", height=320)
        fig.update_xaxes(tickformat="d")
        st.plotly_chart(fig, width="stretch", key="india_line")
        st.caption("Pick a state above to see its forecast beside last year's value.")
    else:
        s = scope; h = horizon_picker()
        df = build_display(fc, h); r = df[(df.state_id == s) & (df.disease == dis)]
        g = hist[(hist.state_id == s) & (hist.disease == dis)].sort_values("year")
        opt1, opt2 = st.columns([3, 1])
        others = opt1.multiselect("Compare with other states", [x for x in STATES if x != s], format_func=nice)
        logy = opt2.toggle("Log scale", value=False)
        fig = go.Figure()
        for o in others:
            og = hist[(hist.state_id == o) & (hist.disease == dis)].sort_values("year")
            fig.add_trace(go.Scatter(x=og.year, y=og.cases, mode="lines", name=nice(o), line=dict(width=1.2), opacity=0.8))
        fig.add_trace(go.Scatter(x=g.year, y=g.cases, mode="lines+markers", name=f"{nice(s)}, reported",
                                 line=dict(color=DISEASE_COLOURS[dis], width=3)))
        if len(r):
            r = r.iloc[0]
            iv = r.get("interval_80")
            has_iv = isinstance(iv, (list, tuple)) and len(iv) == 2
            if len(g):
                fig.add_trace(go.Scatter(x=[g.year.iloc[-1], r.year], y=[g.cases.iloc[-1], r.forecast], mode="lines",
                                         showlegend=False, hoverinfo="skip", line=dict(dash="dot", color=DISEASE_COLOURS[dis])))
            fig.add_trace(go.Scatter(
                x=[r.year], y=[r.forecast], mode="markers", name="Forecast", marker=dict(size=13, symbol="diamond", color="#16C7B2"),
                error_y=dict(type="data", symmetric=False, array=[iv[1] - r.forecast], arrayminus=[r.forecast - iv[0]],
                             color="#16C7B2", thickness=1.6) if has_iv else None))
            if r.persistence is not None and not pd.isna(r.persistence):
                fig.add_trace(go.Scatter(x=[r.year], y=[r.persistence], mode="markers", name="Last year's value (persistence)",
                                         marker=dict(size=12, symbol="x", color="#E8A34B")))
            m = st.columns(4)
            m[0].metric("Forecast", r.forecast_text)
            m[1].metric("Persistence (last year)", r.persistence_text)
            m[2].metric("Risk level", str(r.risk))
            m[3].metric("Confidence", str(r.confidence).capitalize())
            st.write("**80% interval:** " + (f"{iv[0]:,} to {iv[1]:,} cases" if has_iv else "not available for this horizon")
                     + (f"  |  **Percentile:** {r.get('percentile'):.0f}" if pd.notna(r.get("percentile")) else ""))
            if r.confidence == "low":
                st.warning(f"{INSUFF}. Reason: {r.notes or 'low confidence'}")
            elif isinstance(r.get("risk_reason"), (list, tuple)) and r.get("risk_reason"):
                st.caption("Risk reason: " + ", ".join(r.get("risk_reason")))
        fig.update_xaxes(tickformat="d")
        fig.update_layout(height=460, yaxis_title="Cases", yaxis_type="log" if logy else "linear",
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch", key="state_chart")
        latest = hist[(hist.disease == dis) & (hist.year == latest_year)].sort_values("cases", ascending=False).reset_index(drop=True)
        pos = latest.index[latest.state_id == s]
        if len(pos) and latest.cases.sum() > 0:
            share = 100 * float(latest.cases[pos[0]]) / float(latest.cases.sum())
            st.write(f"In {latest_year}, {nice(s)} reported **{int(latest.cases[pos[0]]):,}** {dis} cases, "
                     f"**{share:.1f}%** of the national total (rank {pos[0] + 1} of {len(latest)} reporting states).")
        with st.expander("About this chart"):
            st.write("The solid line is reported annual cases. The dotted line joins the last observed year to the forecast. "
                     "The diamond is the forecast with its 80% interval; the cross is last year's value, the simplest "
                     "possible forecast, shown beside every forecast for reference.")
        out = g[["year", "cases"]].rename(columns={"year": "Year", "cases": "Reported cases"})
        st.download_button("Download this series (CSV)", out.to_csv(index=False), file_name=f"{s}_{dis}.csv", mime="text/csv")

elif page == "Disease explorer":
    st.title("Disease explorer")
    dis = st.radio("Disease", DIS, horizontal=True)
    g = hist[hist.disease == dis]
    nat = g.groupby("year").cases.sum().reset_index()
    y = st.slider("Year", int(g.year.min()), int(g.year.max()), int(g.year.max()))
    top = float(np.log10(g.cases.max() + 1))
    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"Reported cases by state, {y}")
        st.plotly_chart(ex.cases_map(g, y, GEOJSON, dis, STATES, top), width="stretch", theme=None, key="exp_map")
        st.caption("White = no cases reported. Same colour scale for every year.")
    with right:
        gy = g[g.year == y].sort_values("cases", ascending=False)
        top10 = gy.head(10); names = [nice(x) for x in top10.state_id]
        st.subheader("Top 10 states")
        fig = go.Figure(go.Bar(x=top10.cases[::-1], y=names[::-1], orientation="h", marker_color=DISEASE_COLOURS[dis]))
        fig.update_layout(height=520, xaxis_title="Reported cases", margin=dict(l=0, r=10, t=10, b=0))
        st.plotly_chart(fig, width="stretch", key="top10")
    st.subheader("National annual total")
    fig_n = go.Figure(go.Scatter(x=nat.year, y=nat.cases, mode="lines+markers", line=dict(color=DISEASE_COLOURS[dis])))
    fig_n.add_vline(x=y, line_dash="dot", line_color="#8FA6BA")
    fig_n.update_xaxes(tickformat="d"); fig_n.update_layout(yaxis_title="Reported cases", height=320, margin=dict(t=10))
    st.plotly_chart(fig_n, width="stretch", key="national_total")

elif page == "Time-lapse map":
    ex.page_timelapse(hist, STATES, GEOJSON)

elif page == "Why this forecast?":
    ex.page_why(STATES)

elif page == "Network":
    ex.page_network(STATES, GEOJSON, fc)

elif page == "Research results":
    ex.page_research(load("model_performance"))

elif page == "Data story":
    ex.page_story(hist, STATES, latest_year)

elif page == "Annual replay":
    st.title("Annual replay (backtest)")
    rp = pd.DataFrame(load("replay"))
    tab_over, tab_year, tab_state = st.tabs(["All years", "By year", "One state over time"])
    with tab_over:
        e = rp.assign(Model=np.abs(np.log1p(rp["model"]) - np.log1p(rp["true"])),
                      Persistence=np.abs(np.log1p(rp["persistence"]) - np.log1p(rp["true"])))
        by = e.groupby("year")[["Model", "Persistence"]].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=by.year, y=by.Model, mode="lines+markers", name="Model"))
        fig.add_trace(go.Scatter(x=by.year, y=by.Persistence, mode="lines+markers", name="Persistence"))
        fig.update_layout(yaxis_title="log-MAE (lower is better)", xaxis=dict(tickformat="d", dtick=1), height=400)
        st.plotly_chart(fig, width="stretch", key="replay_overview")
        better = int((by.Model < by.Persistence).sum())
        st.write(f"Model closer in {better} of {len(by)} years, averaged over all states and diseases.")
        st.caption("Each year is forecast using earlier data only.")
    with tab_year:
        y = st.selectbox("Year", sorted(rp.year.unique())); dis = st.selectbox("Disease", DIS)
        d = rp[(rp.year == y) & (rp.disease == dis)].copy()
        lm = lambda c: float(np.abs(np.log1p(d[c]) - np.log1p(d["true"])).mean())
        c1, c2 = st.columns(2); c1.metric("log-MAE model", f"{lm('model'):.3f}"); c2.metric("log-MAE persistence", f"{lm('persistence'):.3f}")
        d["State"] = d.state_id.map(nice)
        st.dataframe(d[["State", "true", "model", "persistence"]], hide_index=True)
        st.caption("Backtest on past years. Not a live forecast.")
    with tab_state:
        ex.backtest_by_state(rp, STATES)

else:
    st.title("Limitations")
    st.markdown(f"""
**Data**
- Only annual totals are real. Monthly disease values in the source are a fixed split of the annual total, so no monthly charts or accuracy are shown, and no monthly environmental data is available.
- Coverage is uneven, dengue is missing for 2014 to 2016 in the original source, and some zeros are probably unreported years.
- Values for 2021 and 2022 were revised, so a clean 2023 to 2025 holdout is not possible. Data runs through {latest_year}; this is annual data, not live or real-time.

**Model**
- In rolling-origin evaluation (2010 to 2025), we did not detect a statistically significant difference from persistence (0.853 vs 0.855; 95% interval for the difference -0.040 to +0.039). This does not establish that the methods are equivalent.
- Graph, GAT and weather variants, and the extra features tested separately (humidity, monsoon, population density, elevation), also showed no detectable difference.
- The weather inputs are synthetic annual climatology, not observed weather.
- Forecasts are capped at each series' historical maximum. Horizons 2 and 3 are experimental and carry no interval.

**Reading the outputs**
- A risk level ranks the forecast against the state's own history. It does not compare states, can reflect trend, and is not a prediction of an outbreak.
- Rows with low confidence show *Insufficient evidence* instead of a risk colour, with the reason.
- Rerunning on real monthly or newer annual data is how this can still give a positive result.
""")