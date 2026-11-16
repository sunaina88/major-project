"""Extra pages. Each function draws one page. Rules kept on every page:
persistence next to every forecast, a tie is called a tie, no overstated wording.
Rule logic lives in logic.py; this file only draws."""
import calendar
import json
import pathlib

import api_client as api
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from logic import DISEASE_COLOURS, closer_years, nice, reported_grid, verdict

STATIC = pathlib.Path(__file__).resolve().parent / "static"
DIS = ["dengue", "malaria", "chikungunya"]
VERDICT_COLOUR = {"tie": "#8FA6BA", "better than persistence": "#2e7d32", "worse than persistence": "#c62828"}
GROUP_LABEL = {"own_history": "Own history", "other_diseases": "Other diseases",
               "neighbours": "Neighbouring states", "weather": "Weather (synthetic)"}


@st.cache_data
def _folds(tag, horizon):
    return api.get_folds(tag, horizon)



# ---------------------------------------------------------------- shared map helpers
SCALES = {"dengue": ["#fde0a8", "#fdae61", "#f46d43", "#d73027", "#a50026", "#5e0b1e"],
          "malaria": ["#c7e9f1", "#7fcdbb", "#41b6c4", "#2c7fb8", "#253494", "#081d58"],
          "chikungunya": ["#d9f0a3", "#addd8e", "#78c679", "#31a354", "#006837", "#00331b"]}
LOG_TICKS = [(0, "0"), (1, "10"), (2, "100"), (3, "1k"), (4, "10k"), (5, "100k"), (6, "1M")]


def _scale(dis):
    """White for no cases reported, then a six-step colour ramp."""
    c = SCALES[dis]
    return [[0, "#ffffff"], [1e-6, c[0]]] + [[i / (len(c) - 1), col] for i, col in enumerate(c) if i > 0]


def _colorbar(top):
    t = [(v, l) for v, l in LOG_TICKS if v <= top + 0.3]
    return dict(title="cases", thickness=10, len=0.7, outlinewidth=0,
                tickvals=[v for v, _ in t], ticktext=[l for _, l in t])


def cases_map(g, year, geojson, dis, states, top, height=520):
    """One-year choropleth of reported cases. White = nothing reported."""
    sr = g[g.year == year].set_index("state_id").cases.reindex(states)
    text = [f"<b>{nice(i)}</b><br>" + ("not reported" if pd.isna(v) else f"{v:,.0f} reported cases") for i, v in sr.items()]
    fig = go.Figure(go.Choropleth(
        geojson=geojson, locations=list(sr.index), featureidkey="properties.state_id",
        z=np.log10(sr.fillna(0).values + 1), zmin=0, zmax=top, colorscale=_scale(dis), text=text, hoverinfo="text",
        marker_line_width=0.5, marker_line_color="#0A1524", colorbar=_colorbar(top)))
    fig.update_geos(visible=False, lonaxis_range=[67, 98], lataxis_range=[6, 37.5], bgcolor="rgba(0,0,0,0)")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", geo_bgcolor="rgba(0,0,0,0)")
    return fig


# ---------------------------------------------------------------- Research results
def page_research(mp):
    st.title("Research results")
    rows = [r for r in mp if r.get("diff") is not None and r.get("horizon", 1) == 1]
    rows.sort(key=lambda r: r["diff"])
    labels = [r["variant"] for r in rows]
    n_tie = sum(verdict(r["ci"]) == "tie" for r in rows)
    tab_a, tab_b, tab_c = st.tabs(["Overview", "Error by year", "All variants"])

    with tab_a:
        st.caption("Difference in log-MAE from persistence. Left of zero = model better, right = model worse. "
                   "A bar that crosses zero is a tie.")
        cols = [VERDICT_COLOUR[verdict(r["ci"])] for r in rows]
        fig = go.Figure(go.Scatter(
            x=[r["diff"] for r in rows], y=labels, mode="markers", marker=dict(size=10, color=cols),
            error_x=dict(type="data", symmetric=False, color="#8FA6BA",
                         array=[r["ci"][1] - r["diff"] for r in rows],
                         arrayminus=[r["diff"] - r["ci"][0] for r in rows]),
            customdata=[verdict(r["ci"]) for r in rows],
            hovertemplate="%{y}<br>difference %{x:+.3f}<br>%{customdata}<extra></extra>"))
        fig.add_vline(x=0, line_dash="dash", line_color="#DCE8F2")
        fig.update_layout(height=max(320, 34 * len(rows) + 120), margin=dict(l=0, r=10, t=10, b=0),
                          xaxis_title="Model minus persistence (log-MAE, 95% interval)")
        fig.update_yaxes(categoryorder="array", categoryarray=labels[::-1])
        st.plotly_chart(fig, width="stretch", key="forest")
        st.write(f"**{n_tie} of {len(rows)} variants are tied with persistence.** "
                 "Many variants were compared; with that many, one can look better by chance.")

    with tab_b:
        head = next((r["variant"] for r in rows if r.get("tag") == "geo_w"), labels[0])
        chosen = st.selectbox("Variant", labels, index=labels.index(head))
        row = next(r for r in rows if r["variant"] == chosen)
        folds = pd.DataFrame(_folds(row["tag"], row["horizon"]))
        if folds.empty:
            st.info("No per-year results for this variant.")
        else:
            folds = folds.sort_values("year")
            folds["diff"] = folds.log_model - folds.log_last
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=folds.year, y=folds.log_model, mode="lines+markers", name="Model"))
            fig2.add_trace(go.Scatter(x=folds.year, y=folds.log_last, mode="lines+markers", name="Persistence"))
            fig2.update_layout(yaxis_title="log-MAE (lower is better)", xaxis=dict(tickformat="d", dtick=1), height=380)
            st.plotly_chart(fig2, width="stretch", key="folds_line")
            better = int((folds["diff"] < 0).sum())
            st.write(f"Model better in {better} of {len(folds)} years; persistence better in {int((folds['diff'] > 0).sum())}.")
            with st.expander("Year-by-year numbers"):
                out = folds[["year", "log_model", "log_last", "diff"]].rename(
                    columns={"year": "Year", "log_model": "Model", "log_last": "Persistence", "diff": "Model minus persistence"})
                st.dataframe(out.round(3), hide_index=True, width="stretch")

    with tab_c:
        t = pd.DataFrame(mp).copy()
        t["variant"] = t["variant"].str.replace(" (", ", ", regex=False).str.replace(")", "", regex=False)
        t["ci"] = t["ci"].map(lambda c: f"{c[0]:+.3f} to {c[1]:+.3f}" if isinstance(c, (list, tuple)) and len(c) == 2
                              else ("" if c is None or (isinstance(c, float) and np.isnan(c)) else str(c)))
        st.dataframe(t, hide_index=True, width="stretch")
        st.caption("log-MAE, lower is better. If the 95% interval includes 0, the result is a tie.")


# ---------------------------------------------------------------- Data story
def page_story(hist, states, latest_year):
    st.title("Data story")
    story = json.load(open(STATIC / "data_story.json"))
    tab_a, tab_b, tab_c = st.tabs(["Monthly values", "Coverage", "Revisions"])

    with tab_a:
        m = story["monthly"]
        fig = go.Figure(go.Bar(x=m["months"], y=[v * 100 for v in m["share"]], marker_color="#6baed6"))
        fig.update_layout(xaxis_title="Month", yaxis_title="Share of the annual total (%)", height=380,
                          xaxis=dict(tickmode="array", tickvals=m["months"]))
        st.plotly_chart(fig, width="stretch", key="story_months")
        st.caption("Every state, year and disease gets this same split (largest spread between any two of them: "
                   f"{m['max_std_across_states_years_diseases']}). So the monthly numbers carry no information beyond "
                   "the annual total, and no monthly charts or accuracy are shown.")

    with tab_b:
        dis = st.radio("Disease", DIS, horizontal=True, key="story_dis")
        years = list(range(1999, latest_year + 1))
        z = np.array(reported_grid(hist, dis, states, years))
        fig = go.Figure(go.Bar(x=years, y=z.sum(axis=0), marker_color=DISEASE_COLOURS[dis],
                               hovertemplate="%{x}: %{y} of " + str(len(states)) + " states<extra></extra>"))
        fig.update_layout(yaxis_title="States with a reported value", yaxis_range=[0, len(states)], height=360,
                          xaxis=dict(tickformat="d", dtick=2))
        st.plotly_chart(fig, width="stretch", key="story_cov")
        st.caption("Dengue has no data for any state in 2014, 2015 and 2016; the model treats those years as missing.")
        with st.expander("State-by-year detail"):
            fig = go.Figure(go.Heatmap(z=z, x=years, y=[nice(s) for s in states], xgap=1, ygap=1, showscale=False,
                                       colorscale=[[0, "#1b2f45"], [1, DISEASE_COLOURS[dis]]], zmin=0, zmax=1,
                                       hovertemplate="%{y}, %{x}: %{z}<extra></extra>"))
            fig.update_layout(height=760, margin=dict(l=0, r=0, t=10, b=0))
            fig.update_xaxes(tickformat="d", dtick=2)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, width="stretch", key="story_heat")
            st.caption("Coloured = a value was reported, dark = no value.")

    with tab_c:
        rev = pd.DataFrame(story["revisions"])
        changed = int(rev.changed.sum())
        st.write(f"**{changed} of {len(rev)} state-disease-year values differ** between the earlier source and the "
                 "current one. Nothing before 2021 changed.")
        rev["State"] = rev.state_id.map(nice)
        fig = px.scatter(rev, x=np.log10(rev.old + 1), y=np.log10(rev.new + 1), color="disease",
                         symbol="changed", hover_data={"State": True, "year": True, "old": True, "new": True},
                         color_discrete_map=DISEASE_COLOURS, labels={"x": "Earlier value (log10)", "y": "Current value (log10)"})
        top = float(max(np.log10(rev.old + 1).max(), np.log10(rev.new + 1).max()))
        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(dash="dash", color="#8FA6BA"),
                                 name="No change", hoverinfo="skip"))
        fig.update_layout(height=460)
        st.plotly_chart(fig, width="stretch", key="story_rev")
        st.caption("Points on the dashed line were not changed. A true holdout on 2023 to 2025 is therefore not clean.")
        with st.expander("Changed values"):
            table = rev[rev.changed][["year", "disease", "State", "old", "new"]].rename(
                columns={"year": "Year", "disease": "Disease", "old": "Earlier", "new": "Current"})
            st.dataframe(table, hide_index=True, width="stretch")


# ---------------------------------------------------------------- Why this forecast?
def page_why(states, default_state="kerala"):
    st.title("Why this forecast?")
    c1, c2 = st.columns(2)
    s = c1.selectbox("State", states, index=states.index(default_state) if default_state in states else 0,
                     format_func=nice, key="why_state")
    d = c2.selectbox("Disease", DIS, key="why_dis")
    try:
        ex = api.get_explanation(s, d, 1)
    except api.NotAvailable as e:
        st.info(str(e))
        return
    m1, m2 = st.columns(2)
    m1.metric(f"Forecast {ex['year']}", f"{ex['base_forecast']:,.0f}")
    pers = ex.get("persistence")
    m2.metric("Persistence (last year)", "not available" if pers is None else f"{pers:,.0f}")
    groups = ex["groups"]
    fig = go.Figure(go.Bar(
        y=[GROUP_LABEL.get(g["group"], g["group"]) for g in groups],
        x=[g["change_cases"] for g in groups], orientation="h", marker_color=DISEASE_COLOURS[d],
        customdata=[g["description"] for g in groups],
        hovertemplate="%{customdata}<br>forecast changes by %{x:+,.0f} cases<extra></extra>"))
    fig.update_layout(xaxis_title="Change in the forecast when this input group is removed (cases)",
                      margin=dict(l=0, r=10, t=10, b=0), height=320)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch", key="why_bar")
    st.caption("Shows how the forecast changes, not that an input helps. " + ex.get("label", ""))
    if "low_confidence_forecast" in ex.get("notes", []):
        st.warning("This forecast is low confidence. Treat the bars with extra care.")
    st.caption(ex.get("method", ""))


# ---------------------------------------------------------------- Time-lapse map
def page_timelapse(hist, states, geojson):
    st.title("Time-lapse map")
    dis = st.radio("Disease", DIS, horizontal=True, key="tl_dis")
    h = hist[hist.disease == dis]
    years = sorted(h.year.unique())
    grid = pd.MultiIndex.from_product([years, states], names=["year", "state_id"]).to_frame(index=False)
    df = grid.merge(h[["state_id", "year", "cases"]], on=["state_id", "year"], how="left")
    df["State"] = df.state_id.map(nice)
    df["log_cases"] = np.log10(df.cases.fillna(0) + 1)
    df["cases_text"] = df.cases.map(lambda v: "not reported" if pd.isna(v) else f"{v:,.0f} reported cases")
    top = float(df.log_cases.max())
    fig = px.choropleth(df, geojson=geojson, locations="state_id", featureidkey="properties.state_id",
                        color="log_cases", animation_frame="year", range_color=(0, top),
                        color_continuous_scale=_scale(dis), custom_data=["State", "cases_text"])
    tpl = "%{customdata[0]}<br>%{customdata[1]}<extra></extra>"
    for tr in list(fig.data) + [t for fr in fig.frames for t in fr.data]:
        tr.hovertemplate = tpl
        tr.marker.line.width = 0.5
        tr.marker.line.color = "#0A1524"
    fig.update_geos(visible=False, lonaxis_range=[67, 98], lataxis_range=[6, 37], bgcolor="rgba(0,0,0,0)")
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      coloraxis_colorbar=_colorbar(top))
    st.plotly_chart(fig, width="stretch", theme=None, key=f"timelapse_{dis}")
    st.caption("Press play or drag the slider. Reported annual cases (observed data, not forecasts). "
               "White = no cases reported that year. The colour scale is logarithmic and fixed across years, "
               "so frames can be compared.")


# ---------------------------------------------------------------- Backtest explorer (second tab of Annual replay)
def backtest_by_state(rp, states):
    c1, c2 = st.columns(2)
    s = c1.selectbox("State", states, index=states.index("kerala") if "kerala" in states else 0,
                     format_func=nice, key="bt_state")
    d = c2.selectbox("Disease", DIS, key="bt_dis")
    g = rp[(rp.state_id == s) & (rp.disease == d)].sort_values("year")
    if g.empty:
        st.info("No backtest rows for this selection.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.year, y=g["true"], mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=g.year, y=g["model"], mode="lines+markers", name="Model"))
    fig.add_trace(go.Scatter(x=g.year, y=g["persistence"], mode="lines+markers", name="Persistence"))
    fig.update_layout(yaxis_title="Cases", xaxis=dict(tickformat="d"))
    st.plotly_chart(fig, width="stretch", key="bt_line")
    a, b, c = closer_years(g)
    st.write(f"**Model closer in {a} of {len(g)} years**, persistence closer in {b}" + (f", equal in {c}." if c else "."))
    st.caption("Closer = smaller gap on the log scale. Backtest on past years, each made with earlier data only. "
               "Not a forecast of the future.")


# ---------------------------------------------------------------- State network
GRAPH_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed" / "graph"
ISLANDS = {"andaman_and_nicobar_islands", "lakshadweep"}
STAGES = ["Nodes only", "Land borders", "Land borders + island links", "kNN graph", "Distance graph", "Hybrid graph"]
STAGE_FILE = {2: "adjacency_raw", 3: "adjacency_knn_raw", 4: "adjacency_dist_raw", 5: "adjacency_hybrid_raw"}
STAGE_NOTE = {
    0: "No connections: each state is forecast from its own history only.",
    1: "States that share a land border. Island territories are not linked yet.",
    2: "Land borders plus island territories attached to the nearest mainland state. This is the graph used by the registered model.",
    3: "Each state linked to its nearest states by distance.",
    4: "Nearly every pair of states linked, with connection strength weighted by distance.",
    5: "Land borders combined with distance weighting.",
}


def _centroid(geom):
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    best, area_best = None, -1.0
    for poly in polys:
        ring = np.array(poly[0])
        x, y = ring[:, 0], ring[:, 1]
        cr = x[:-1] * y[1:] - x[1:] * y[:-1]
        a = cr.sum() / 2
        if abs(a) > area_best:
            area_best = abs(a)
            best = (((x[:-1] + x[1:]) * cr).sum() / (6 * a), ((y[:-1] + y[1:]) * cr).sum() / (6 * a))
    return best


@st.cache_data
def _centroids(geojson_text):
    gj = json.loads(geojson_text)
    return {f["properties"]["state_id"]: _centroid(f["geometry"]) for f in gj["features"]}


@st.cache_data
def _adjacency(name):
    p = GRAPH_DIR / f"{name}.npy"
    return np.load(p) if p.exists() else None


def _edge_list(A, states, drop_islands=False):
    out = []
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            if A[i, j] > 0 or A[j, i] > 0:
                if drop_islands and (states[i] in ISLANDS or states[j] in ISLANDS):
                    continue
                out.append((i, j, float(max(A[i, j], A[j, i]))))
    return out


def _lines(edges, cen, states):
    lon, lat = [], []
    for i, j, _ in edges:
        a, b = cen[states[i]], cen[states[j]]
        lon += [a[0], b[0], None]; lat += [a[1], b[1], None]
    return lon, lat


def page_network(states, geojson, fc):
    from logic import INSUFF, build_display, colour_map
    st.title("State network")
    st.caption("How the model links the 36 states and union territories. Each state is a node; a graph layer lets it "
               "read its neighbours before the model looks at the last three years.")
    if _adjacency("adjacency_raw") is None:
        st.info("Graph files were not found in data/processed/graph.")
        return
    cen = _centroids(json.dumps(geojson))
    c1, c2 = st.columns([1, 2])
    dis = c1.radio("Colour nodes by", DIS, horizontal=True, key="net_dis")
    stage = STAGES.index(c2.select_slider("Build the graph", STAGES, value=STAGES[2], key="net_stage"))
    if stage == 0:
        edges = []
    elif stage == 1:
        edges = _edge_list(_adjacency("adjacency_raw"), states, drop_islands=True)
    else:
        A = _adjacency(STAGE_FILE[stage])
        edges = _edge_list(A, states) if A is not None else []
    st.caption(STAGE_NOTE[stage] + f" {len(edges)} connections.")

    # forecasts per state for node size and colour
    df = build_display(fc, 1); df = df[df.disease == dis].set_index("state_id").reindex(states)
    risk = df["risk"].fillna(INSUFF); fcst = df["forecast"].astype(float)
    fmax = float(np.nanmax(fcst.values)) if fcst.notna().any() else 1.0
    size = [10 + 22 * (float(f) / fmax) ** 0.5 if pd.notna(f) else 8 for f in fcst]
    cm = colour_map(dis)

    # click on a node (from the previous run) picks the state
    ev = st.session_state.get("net_chart")
    try:
        pts = ev["selection"]["points"]
        if pts and pts[0].get("customdata"):
            cid = pts[0]["customdata"][0] if isinstance(pts[0]["customdata"], list) else pts[0]["customdata"]
            if cid != st.session_state.get("net_last_click"):
                st.session_state["net_last_click"] = cid
                st.session_state["net_sel"] = cid
    except (KeyError, TypeError):
        pass
    sel = st.session_state.get("net_sel", "madhya_pradesh")
    if sel not in states:
        sel = "madhya_pradesh"
    sel = st.selectbox("Selected state", states, index=states.index(sel), format_func=nice, key="net_sel")
    si = states.index(sel)

    other = [e for e in edges if si not in e[:2]]
    mine = [e for e in edges if si in e[:2]]
    nbrs = [states[b if a == si else a] for a, b, _ in mine]

    fig = go.Figure(go.Choropleth(geojson=geojson, locations=states, featureidkey="properties.state_id",
                                  z=[0] * len(states), colorscale=[[0, "#16293d"], [1, "#16293d"]], showscale=False,
                                  hoverinfo="skip", marker_line_width=0.5, marker_line_color="#0A1524"))
    dense = len(edges) > 100
    if other:
        lon, lat = _lines(other, cen, states)
        fig.add_trace(go.Scattergeo(lon=lon, lat=lat, mode="lines", hoverinfo="skip",
                                    line=dict(width=0.8 if dense else 1.4, color="rgba(143,166,186,%s)" % (0.28 if dense else 0.6))))
    if mine:
        lon, lat = _lines(mine, cen, states)
        fig.add_trace(go.Scattergeo(lon=lon, lat=lat, mode="lines", hoverinfo="skip", line=dict(width=3, color="#16C7B2")))
    hover = [f"<b>{nice(s)}</b><br>Risk: {r}<br>Forecast: " + ("n/a" if pd.isna(f) else f"{f:,.0f}")
             for s, r, f in zip(states, risk, fcst)]
    fig.add_trace(go.Scattergeo(
        lon=[cen[s][0] for s in states], lat=[cen[s][1] for s in states], mode="markers", text=hover, hoverinfo="text",
        customdata=states, marker=dict(size=size, color=[cm.get(r, "#808080") for r in risk],
                                       line=dict(width=[3 if s == sel else 1 for s in states],
                                                 color=["#16C7B2" if s == sel else "#0A1524" for s in states]))))
    fig.update_geos(visible=False, lonaxis_range=[67, 98], lataxis_range=[6, 37.5], bgcolor="rgba(0,0,0,0)")
    fig.update_layout(height=760, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    left, right = st.columns([5, 2])
    with left:
        st.plotly_chart(fig, width="stretch", theme=None, key="net_chart", on_select="rerun", selection_mode="points")
        st.caption("Node size follows the forecast, colour follows the risk level. Click a node or use the selector.")
    with right:
        st.subheader(nice(sel))
        m1, m2 = st.columns(2)
        m1.metric("Connections", len(nbrs))
        m2.metric("Risk level", str(risk.iloc[si]))
        f0 = fcst.iloc[si]; p0 = df["persistence"].iloc[si]
        m3, m4 = st.columns(2)
        m3.metric("Forecast", "n/a" if pd.isna(f0) else f"{f0:,.0f}")
        m4.metric("Last year", "n/a" if p0 is None or pd.isna(p0) else f"{float(p0):,.0f}")
        if nbrs:
            nb = pd.DataFrame({"Neighbour": [nice(n) for n in nbrs],
                               "Risk": [risk.iloc[states.index(n)] for n in nbrs],
                               "Forecast": ["n/a" if pd.isna(fcst.iloc[states.index(n)]) else f"{fcst.iloc[states.index(n)]:,.0f}" for n in nbrs]})
            st.dataframe(nb, hide_index=True, width="stretch")
        else:
            st.write("No connections at this stage.")
        with st.expander("How a state uses its neighbours"):
            st.write("In each graph layer a state's representation is mixed with those of its connected states, so "
                     "information from neighbours reaches it before the GRU reads the three-year history. "
                     "The 'Why this forecast?' page shows how much the neighbouring states change a forecast.")
        with st.expander("Graph variants tested"):
            st.dataframe(pd.DataFrame({"Graph": ["No graph", "Land borders (registered)", "kNN", "Distance", "Hybrid"],
                                       "log-MAE": [0.843, 0.853, 0.836, 0.836, 0.840]}), hide_index=True, width="stretch")
            st.caption("Lower is better. Persistence scores 0.855; see Research results for the intervals.")


# ---------------------------------------------------------------- How it works
HOW_STAGES = ["Data", "Graph", "Model", "Forecast", "Risk level"]


def _pipeline_figure(active):
    boxes = [("Inputs", "8 features<br>3 years"), ("Graph layer", "reads<br>neighbours"), ("GRU", "reads the<br>3-year window"),
             ("Graph layer", "reads<br>neighbours"), ("Shared head", "3 diseases"), ("Forecast", "last year +<br>bounded correction")]
    w, gap = 128, 30
    fig = go.Figure()
    for i, (a, b) in enumerate(boxes):
        x0 = i * (w + gap)
        hot = (active == "Model" and i in (1, 2, 3, 4)) or (active == "Graph" and i in (1, 3)) or (active == "Forecast" and i == 5) \
            or (active == "Data" and i == 0)
        fig.add_shape(type="rect", x0=x0, x1=x0 + w, y0=20, y1=80, line=dict(color="#16C7B2" if hot else "#2c4660", width=2),
                      fillcolor="#14304a" if hot else "#102438")
        fig.add_annotation(x=x0 + w / 2, y=50, text=f"<b>{a}</b><br><span style='font-size:11px;color:#8FA6BA'>{b}</span>",
                           showarrow=False, font=dict(color="#DCE8F2", size=13))
        if i:
            fig.add_annotation(x=x0, y=50, ax=x0 - gap, ay=50, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowcolor="#8FA6BA", arrowwidth=1.5, text="")
    fig.update_xaxes(visible=False, range=[-4, len(boxes) * (w + gap) - gap + 4])
    fig.update_yaxes(visible=False, range=[0, 100])
    fig.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def page_how(hist, states, fc, latest_year):
    from logic import INSUFF, build_display
    st.title("How it works")
    st.caption("From reported cases to a forecast and a risk level, in five steps.")
    stage = st.radio("Step", HOW_STAGES, horizontal=True, key="how_stage", label_visibility="collapsed")
    st.plotly_chart(_pipeline_figure(stage), width="stretch", key="how_diagram")

    if stage == "Data":
        l, r = st.columns([2, 3])
        with l:
            st.subheader("What goes in")
            st.write("Annual reported cases of dengue, malaria and chikungunya for each state and union territory, "
                     "from national surveillance exports. Missing values stay missing, and the model is told which "
                     "values were observed.")
            a, b, c = st.columns([0.7, 1.5, 1.3])
            a.metric("States", len(states)); b.metric("Years", f"{int(hist.year.min())}\u2013{int(hist.year.max())}")
            c.metric("Reported values", f"{len(hist):,}")
        with r:
            tr = hist.groupby(["year", "disease"]).cases.sum().reset_index()
            fig = go.Figure()
            for d in DIS:
                t = tr[tr.disease == d]
                fig.add_trace(go.Scatter(x=t.year, y=t.cases, mode="lines", name=d.capitalize(), line=dict(color=DISEASE_COLOURS[d])))
            fig.update_layout(height=320, yaxis_type="log", yaxis_title="Reported cases (log scale)", title="National totals")
            fig.update_xaxes(tickformat="d")
            st.plotly_chart(fig, width="stretch", key="how_data")
    elif stage == "Graph":
        A = _adjacency("adjacency_raw")
        l, r = st.columns([2, 3])
        with l:
            st.subheader("States talk to their neighbours")
            st.write("Each state is a node. States that share a land border are connected, and island territories are "
                     "attached to the nearest mainland state. A graph layer lets every state mix in information from its "
                     "neighbours.")
            if A is not None:
                deg = ((A > 0) & ~np.eye(len(A), dtype=bool)).sum(axis=1)
                a, b, c = st.columns(3)
                a.metric("Nodes", len(A)); b.metric("Connections", int(deg.sum() // 2)); c.metric("Average neighbours", f"{deg.mean():.1f}")
            st.write("Explore it on the **Network** page.")
        with r:
            if A is not None:
                deg = ((A > 0) & ~np.eye(len(A), dtype=bool)).sum(axis=1)
                order = np.argsort(deg)[::-1][:10]
                fig = go.Figure(go.Bar(x=deg[order][::-1], y=[nice(states[i]) for i in order][::-1], orientation="h", marker_color="#16C7B2"))
                fig.update_layout(height=340, title="Most connected states", xaxis_title="Neighbours")
                st.plotly_chart(fig, width="stretch", key="how_graph")
    elif stage == "Model":
        st.subheader("Reading space, then time")
        st.write("For every year, a graph layer combines each state with its neighbours. A GRU then reads a three-year "
                 "window of those representations, a second graph layer mixes neighbours again, and one shared head "
                 "produces the forecast for all three diseases. It is plain PyTorch and runs on a CPU.")
        a, b, c = st.columns(3)
        a.metric("Input window", "3 years"); b.metric("Hidden size", "16"); c.metric("Training seeds", "3")
        st.caption("Evaluated with rolling origin: for each test year from 2010 to 2025, the model is trained only on earlier years.")
    elif stage == "Forecast":
        st.subheader("A bounded step from last year")
        st.write("The head does not predict cases from scratch. It predicts a bounded correction to last year's value, "
                 "and forecasts are capped at each series' historical maximum. Every forecast is shown next to last "
                 "year's value so the two can be compared.")
        c1, c2 = st.columns(2)
        s = c1.selectbox("State", states, index=states.index("kerala") if "kerala" in states else 0, format_func=nice, key="how_fs")
        d = c2.selectbox("Disease", DIS, key="how_fd")
        df = build_display(fc, 1); row = df[(df.state_id == s) & (df.disease == d)]
        g = hist[(hist.state_id == s) & (hist.disease == d)].sort_values("year")
        if len(row):
            row = row.iloc[0]
            m = st.columns(3)
            m[0].metric(f"Forecast {int(row.year)}", row.forecast_text); m[1].metric("Last year", row.persistence_text)
            m[2].metric("Confidence", str(row.confidence).capitalize())
            fig = go.Figure(go.Scatter(x=g.year, y=g.cases, mode="lines+markers", name="Reported", line=dict(color=DISEASE_COLOURS[d])))
            fig.add_trace(go.Scatter(x=[row.year], y=[row.forecast], mode="markers", name="Forecast", marker=dict(size=13, symbol="diamond", color="#16C7B2")))
            if row.persistence is not None and not pd.isna(row.persistence):
                fig.add_trace(go.Scatter(x=[row.year], y=[row.persistence], mode="markers", name="Last year's value", marker=dict(size=12, symbol="x", color="#E8A34B")))
            fig.update_layout(height=340, yaxis_title="Cases"); fig.update_xaxes(tickformat="d")
            st.plotly_chart(fig, width="stretch", key="how_fc")
        else:
            st.info("No forecast for this selection.")
    else:
        st.subheader("Compared with the state's own past")
        st.write("A risk level is the percentile of the forecast among that state's own earlier years: Low below the 50th "
                 "percentile, Moderate 50 to 75, High 75 to 90, Very high 90 and above. Forecasts with low confidence "
                 f"show {INSUFF} instead of a level.")
        c1, c2 = st.columns(2)
        s = c1.selectbox("State", states, index=states.index("kerala") if "kerala" in states else 0, format_func=nice, key="how_rs")
        d = c2.selectbox("Disease", DIS, key="how_rd")
        df = build_display(fc, 1); row = df[(df.state_id == s) & (df.disease == d)]
        g = hist[(hist.state_id == s) & (hist.disease == d)]
        if len(row) and len(g) >= 3:
            row = row.iloc[0]
            q = np.percentile(g.cases, [50, 75, 90])
            fig = go.Figure(go.Histogram(x=g.cases, nbinsx=12, marker_color=DISEASE_COLOURS[d], opacity=0.85, name="Earlier years"))
            for v, lab in zip(q, ["50th", "75th", "90th"]):
                fig.add_vline(x=v, line_dash="dot", line_color="#8FA6BA", annotation_text=lab, annotation_font_color="#8FA6BA")
            fig.add_vline(x=float(row.forecast), line_color="#16C7B2", line_width=3, annotation_text="Forecast", annotation_font_color="#16C7B2")
            fig.update_layout(height=340, xaxis_title="Annual reported cases", yaxis_title="Number of years",
                              title=f"{nice(s)}: where the forecast sits among its own years", showlegend=False)
            st.plotly_chart(fig, width="stretch", key="how_risk")
            st.metric("Risk level", str(row.risk))
        else:
            st.info("Not enough history for this selection.")


# ---------------------------------------------------------------- Climate & Geography
CLIMATE = {   # label: (column, unit, colour scale, decimals)
    "Temperature": ("temp_mean_c", "\u00b0C", "Plasma", 1),
    "Rainfall": ("rainfall_mm", "mm per month", [[0, "#14304a"], [0.5, "#1f8a9c"], [1, "#d7f7ef"]], 0),
    "Humidity": ("rh_mean_pct", "%", "Viridis", 0),
    "Wind speed": ("wind_ms", "m/s", "Cividis", 1),
}
GEOGRAPHY = {   # label: (column, unit, log colour scale, decimals, multiplier)
    "Population density": ("population_density_per_km2", "people per km\u00b2", True, 0, 1),
    "Elevation": ("mean_elevation_m", "m", False, 0, 1),
    "Urban share": ("urban_fraction", "%", False, 1, 100),
    "Area": ("area_km2", "km\u00b2", True, 0, 1),
    "Population (Census 2011)": ("population", "people", True, 0, 1),
}
ANNUAL_CHARTS = [("mean_temp_c", "Mean temperature (\u00b0C)"), ("total_rain_mm", "Total rainfall (mm)"),
                 ("monsoon_rain_mm", "Monsoon rainfall (mm)"), ("hottest_month_temp_c", "Hottest-month temperature (\u00b0C)"),
                 ("mean_rh_pct", "Mean humidity (%)"), ("mean_wind_ms", "Mean wind speed (m/s)")]
LOG_LABELS = {1: "10", 2: "100", 3: "1k", 4: "10k", 5: "100k", 6: "1M", 7: "10M", 8: "100M"}


def _fmt(v, dp):
    return "not available" if v is None or pd.isna(v) else f"{v:,.{dp}f}"


def _climate_figure(geojson, states, z, text, scale, zmin, zmax, title, ticks=None):
    cb = dict(title=title, thickness=10, len=0.7, outlinewidth=0)
    if ticks:
        cb.update(tickvals=[v for v in ticks if zmin <= v <= zmax], ticktext=[ticks[v] for v in ticks if zmin <= v <= zmax])
    fig = go.Figure(go.Choropleth(geojson=geojson, locations=states, featureidkey="properties.state_id",
                                  z=z, zmin=zmin, zmax=zmax, colorscale=scale, text=text, hoverinfo="text",
                                  marker_line_width=0.5, marker_line_color="#0A1524", colorbar=cb))
    fig.update_geos(visible=False, lonaxis_range=[67, 98], lataxis_range=[6, 37.5], bgcolor="rgba(0,0,0,0)")
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", geo_bgcolor="rgba(0,0,0,0)")
    return fig


def page_climate(states, geojson, load):
    from api_client import NotAvailable
    st.title("Climate & Geography")
    st.caption("Environmental context for each state. These values are not inputs to the registered forecast; "
               "the feature test is on the Research results page.")
    try:
        mo = pd.DataFrame(load("climate_monthly")); an = pd.DataFrame(load("climate_annual")); pr = pd.DataFrame(load("state_profile"))
    except NotAvailable as e:
        st.info(str(e))
        return
    state = st.selectbox("State (for trends and profile)", states, index=states.index("kerala") if "kerala" in states else 0,
                         format_func=nice, key="clim_state")
    tab_map, tab_trend, tab_prof = st.tabs(["Map", "State trends", "State profile"])

    with tab_map:
        c1, c2, c3 = st.columns([2, 2, 3])
        measure = c1.selectbox("Measure", list(CLIMATE) + list(GEOGRAPHY), key="clim_measure")
        if measure in CLIMATE:
            col, unit, scale, dp = CLIMATE[measure]
            year = c2.slider("Year", int(mo.year.min()), int(mo.year.max()), int(mo.year.max()), key="clim_year")
            month = c3.select_slider("Month", list(range(1, 13)), value=7, key="clim_month", format_func=lambda m: calendar.month_name[m])
            when = f"{calendar.month_name[month]} {year}"
            sub = mo[(mo.year == year) & (mo.month == month)].set_index("state_id").reindex(states)
            vals = sub[col]
            zmin, zmax = float(mo[col].quantile(0.01)), float(mo[col].quantile(0.99))
            text = [f"<b>{nice(s)}</b><br>{measure}: {_fmt(v, dp)} {unit if pd.notna(v) else ''}<br>{when}" for s, v in vals.items()]
            fig = _climate_figure(geojson, states, vals.values, text, scale, zmin, zmax, unit)
            note = f"{measure}, {when}, averaged over each state. The colour scale is the same for every month."
        else:
            col, unit, logscale, dp, mult = GEOGRAPHY[measure]
            vals = pr.set_index("state_id").reindex(states)[col] * mult
            text = [f"<b>{nice(s)}</b><br>{measure}: {_fmt(v, dp)} {unit if pd.notna(v) else ''}" for s, v in vals.items()]
            if logscale:
                z = np.log10(vals.where(vals > 0)); ticks = LOG_LABELS
            else:
                z = vals; ticks = None
            fig = _climate_figure(geojson, states, z.values, text, "Viridis", float(np.nanmin(z)), float(np.nanmax(z)), unit, ticks)
            note = f"{measure} by state." + (" Population is from Census 2011." if "Population" in measure or "density" in measure else "") \
                + (" Colours use a logarithmic scale." if logscale else "")
        st.plotly_chart(fig, width="stretch", theme=None, key="clim_map")
        st.caption(note)

    with tab_trend:
        mcol, mright = st.columns([2, 3])
        tm = mcol.selectbox("Monthly measure", list(CLIMATE), key="clim_trend_measure")
        col, unit, _, _ = CLIMATE[tm]
        ms = mo[mo.state_id == state].sort_values(["year", "month"])
        avg = mo.groupby(["year", "month"])[col].mean().reset_index().sort_values(["year", "month"])
        dates = pd.to_datetime(dict(year=ms.year, month=ms.month, day=1))
        adates = pd.to_datetime(dict(year=avg.year, month=avg.month, day=1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=adates, y=avg[col], mode="lines", name="All-state average",
                                 line=dict(color="#8FA6BA", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=dates, y=ms[col], mode="lines", name=nice(state), line=dict(color="#16C7B2", width=2)))
        fig.update_layout(height=340, yaxis_title=f"{tm} ({unit})", title=f"{nice(state)}: monthly {tm.lower()}",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch", key="clim_monthly_line")
        npts = int(ms.n_points.median()) if len(ms) else 0
        st.caption(f"Each monthly value is the average over {npts} NASA POWER sample point{'s' if npts != 1 else ''} inside the state boundary.")
        a_state = an[an.state_id == state].sort_values("year")
        a_avg = an.groupby("year").mean(numeric_only=True).reset_index()
        cols = st.columns(3)
        for i, (c, title) in enumerate(ANNUAL_CHARTS):
            f2 = go.Figure()
            f2.add_trace(go.Scatter(x=a_avg.year, y=a_avg[c], mode="lines", line=dict(color="#8FA6BA", width=1, dash="dot"), name="All-state average"))
            f2.add_trace(go.Scatter(x=a_state.year, y=a_state[c], mode="lines+markers", line=dict(color="#16C7B2", width=2), name=nice(state)))
            f2.update_layout(height=250, title=dict(text=title, font=dict(size=13)), showlegend=False, margin=dict(l=0, r=10, t=40, b=10))
            f2.update_xaxes(tickformat="d")
            cols[i % 3].plotly_chart(f2, width="stretch", key=f"clim_annual_{c}")

    with tab_prof:
        row = pr[pr.state_id == state]
        if row.empty:
            st.info("No profile for this state.")
        else:
            row = row.iloc[0]
            items = [("Area", "area_km2", "km\u00b2", 0, 1), ("Population (Census 2011)", "population", "", 0, 1),
                     ("Density", "population_density_per_km2", "per km\u00b2", 0, 1), ("Urban share", "urban_fraction", "%", 1, 100),
                     ("Mean elevation", "mean_elevation_m", "m", 0, 1)]
            cols = st.columns(5)
            for cc, (label, c, u, dp, mult) in zip(cols, items):
                v = row[c]
                cc.metric(label, "not available" if pd.isna(v) else f"{v * mult:,.{dp}f} {u}".strip())
                if pd.notna(v):
                    cc.caption(f"Rank {int(pr[c].rank(ascending=False)[row.name])} of {int(pr[c].notna().sum())}")
            pick = st.selectbox("Compare all states", [i[0] for i in items], key="clim_compare")
            label, c, u, dp, mult = next(i for i in items if i[0] == pick)
            d = pr.assign(v=pr[c] * mult).dropna(subset=["v"]).sort_values("v")
            colours = ["#16C7B2" if s == state else "#2c4660" for s in d.state_id]
            fig = go.Figure(go.Bar(x=d.v, y=[nice(s) for s in d.state_id], orientation="h", marker_color=colours,
                                   hovertemplate="%{y}: %{x:,.1f}<extra></extra>"))
            fig.update_layout(height=760, margin=dict(l=0, r=10, t=10, b=10), xaxis_title=f"{label} {u}".strip(),
                              xaxis_type="log" if c in ("area_km2", "population", "population_density_per_km2") else "linear")
            st.plotly_chart(fig, width="stretch", key="clim_compare_bar")

    with st.expander("About these data"):
        st.write("**Weather:** monthly values from NASA POWER (temperature, precipitation, relative humidity and wind at 2 m), "
                 "sampled on a 0.75\u00b0 grid inside each state boundary (up to 30 points) and averaged. Very small states have only "
                 "one or a few sample points. **Geography:** area, mean elevation and urban share per state; population is from "
                 "Census 2011. This page is exploratory: these variables were tested as model inputs and did not give a detectable "
                 "improvement (see Research results).")