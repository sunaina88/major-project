"""Extra pages. Each function draws one page. Rules kept on every page:
persistence next to every forecast, a tie is called a tie, no overstated wording.
Rule logic lives in logic.py; this file only draws."""
import json
import pathlib

import api_client as api
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from logic import DISEASE_COLOURS, PALETTES, closer_years, is_tie, nice, reported_grid, verdict

STATIC = pathlib.Path(__file__).resolve().parent / "static"
DIS = ["dengue", "malaria", "chikungunya"]
VERDICT_COLOUR = {"tie": "#808080", "better than persistence": "#2e7d32", "worse than persistence": "#c62828"}
GROUP_LABEL = {"own_history": "Own history", "other_diseases": "Other diseases",
               "neighbours": "Neighbouring states", "weather": "Weather"}


@st.cache_data
def _folds(tag, horizon):
    return api.get_folds(tag, horizon)


# ---------------------------------------------------------------- Start here
def page_start(mp, latest_year):
    st.title("Start here")
    st.write(f"VectorWatch India forecasts yearly dengue, malaria and chikungunya cases for 36 states and union "
             f"territories. The data is annual and runs to {latest_year}.")
    head = next((r for r in mp if r.get("tag") == "geo_w" and r.get("horizon") == 1), None)
    result = "On yearly totals the model is statistically tied with last year's value."
    if head is not None and head.get("ci"):
        result = (f"Model error {head['log_mae']:.3f} against {mp[0]['log_mae']:.3f} for last year's value "
                  f"(log-MAE, lower is better). The difference is {head['diff']:+.3f}, 95% interval "
                  f"{head['ci'][0]:+.3f} to {head['ci'][1]:+.3f}: " +
                  ("a tie." if is_tie(head["ci"]) else "a clear difference."))
    steps = [
        ("1. The problem", "Health teams plan before the season. Can past yearly counts, neighbouring states and weather "
                           "say anything about next year that \"same as last year\" does not?"),
        ("2. The data", "Yearly reported cases per state. Monthly values in the source are a fixed split of the yearly "
                        "total, so no monthly results are shown. Open the Data story page for the gaps and revisions."),
        ("3. The model", "A graph layer links neighbouring states, then a GRU reads a 3-year window and forecasts "
                         "the next year."),
        ("4. The result", result),
        ("5. What is next", "Rerun on real monthly data or more years. That is the way this can still give a "
                            "positive result."),
    ]
    for title, text in steps:
        st.subheader(title)
        st.write(text)
    st.header("Glossary")
    glossary = {
        "Persistence": "The forecast \"next year equals last year\". Every forecast here is shown next to it.",
        "log-MAE": "Average absolute gap between the logarithm of the forecast and of the true value. Lower is "
                   "better. The log stops large states from dominating.",
        "Interval": "80% interval on a forecast: the range expected to hold the true value about 8 times in 10. "
                    "95% interval on a model comparison: the range of plausible differences from persistence; "
                    "if it includes 0 the result is a tie.",
        "Risk level": "Where the forecast ranks among the state's own earlier years. It is not a prediction "
                      "of an outbreak.",
        "Tie": "The model and persistence cannot be told apart on this data.",
        "Backtest": "A forecast made for a past year using only data before it, then compared with what happened.",
        "Horizon": "How many years ahead. Horizons 2 and 3 are experimental.",
    }
    for term, text in glossary.items():
        st.markdown(f"**{term}.** {text}")


# ---------------------------------------------------------------- Research results
def page_research(mp):
    st.title("Research results")
    st.caption("Difference in log-MAE from persistence. Left of zero = model better, right = model worse. "
               "A bar that crosses zero is a tie.")
    rows = [r for r in mp if r.get("diff") is not None and r.get("horizon", 1) == 1]
    rows.sort(key=lambda r: r["diff"])
    labels = [r["variant"] for r in rows]
    cols = [VERDICT_COLOUR[verdict(r["ci"])] for r in rows]
    fig = go.Figure(go.Scatter(
        x=[r["diff"] for r in rows], y=labels, mode="markers", marker=dict(size=10, color=cols),
        error_x=dict(type="data", symmetric=False, color="#555",
                     array=[r["ci"][1] - r["diff"] for r in rows],
                     arrayminus=[r["diff"] - r["ci"][0] for r in rows]),
        customdata=[verdict(r["ci"]) for r in rows],
        hovertemplate="%{y}<br>difference %{x:+.3f}<br>%{customdata}<extra></extra>"))
    fig.add_vline(x=0, line_dash="dash", line_color="#000")
    fig.update_layout(height=max(320, 34 * len(rows) + 120), margin=dict(l=0, r=10, t=10, b=0),
                      xaxis_title="Model minus persistence (log-MAE, 95% interval)")
    fig.update_yaxes(categoryorder="array", categoryarray=labels[::-1])
    event = st.plotly_chart(fig, width="stretch", key="forest", on_select="rerun", selection_mode="points")
    n_tie = sum(verdict(r["ci"]) == "tie" for r in rows)
    st.write(f"**{n_tie} of {len(rows)} variants are tied with persistence.** "
             "Many variants were compared; with that many, one can look better by chance.")
    st.markdown(" ".join(f":{'gray' if verdict(r['ci']) == 'tie' else 'green' if verdict(r['ci']).startswith('better') else 'red'}"
                         f"-badge[{r['variant']}: {verdict(r['ci'])}]" for r in rows))

    clicked = None
    try:
        pts = event["selection"]["points"]
        if pts:
            clicked = pts[0]["y"]
    except (KeyError, TypeError):
        pass
    pick = st.selectbox("Variant (or click a point above)", labels,
                        index=labels.index(next((r["variant"] for r in rows if r.get("tag") == "geo_w"), labels[0])))
    chosen = clicked if clicked in labels else pick
    row = next(r for r in rows if r["variant"] == chosen)
    folds = pd.DataFrame(_folds(row["tag"], row["horizon"]))
    if folds.empty:
        st.info("No per-year results for this variant.")
        return
    folds = folds.sort_values("year")
    folds["diff"] = folds.log_model - folds.log_last
    st.subheader(f"Error by year: {chosen}")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=folds.year, y=folds.log_model, mode="lines+markers", name="Model"))
    fig2.add_trace(go.Scatter(x=folds.year, y=folds.log_last, mode="lines+markers", name="Persistence"))
    fig2.update_layout(yaxis_title="log-MAE (lower is better)", xaxis=dict(tickformat="d", dtick=1))
    st.plotly_chart(fig2, width="stretch", key="folds_line")
    better = int((folds["diff"] < 0).sum())
    st.write(f"Model better in {better} of {len(folds)} years; persistence better in "
             f"{int((folds['diff'] > 0).sum())}.")
    out = folds[["year", "log_model", "log_last", "diff"]].rename(
        columns={"year": "Year", "log_model": "Model", "log_last": "Persistence", "diff": "Model minus persistence"})
    st.dataframe(out.round(3), hide_index=True)


# ---------------------------------------------------------------- Data story
def page_story(hist, states, latest_year):
    st.title("Data story")
    story = json.load(open(STATIC / "data_story.json"))

    st.header("1. Monthly values are a fixed split")
    m = story["monthly"]
    fig = go.Figure(go.Bar(x=m["months"], y=[v * 100 for v in m["share"]], marker_color="#6baed6"))
    fig.update_layout(xaxis_title="Month", yaxis_title="Share of the annual total (%)",
                      xaxis=dict(tickmode="array", tickvals=m["months"]))
    st.plotly_chart(fig, width="stretch", key="story_months")
    st.caption("Every state, year and disease gets this same split (largest spread between any two of them: "
               f"{m['max_std_across_states_years_diseases']}). So the monthly numbers carry no information beyond "
               "the annual total, and no monthly charts or accuracy are shown.")

    st.header("2. Which values were reported")
    dis = st.radio("Disease", DIS, horizontal=True, key="story_dis")
    years = list(range(1999, latest_year + 1))
    z = reported_grid(hist, dis, states, years)
    fig = go.Figure(go.Heatmap(z=z, x=years, y=[nice(s) for s in states], xgap=1, ygap=1, showscale=False,
                               colorscale=[[0, "#e0e0e0"], [1, DISEASE_COLOURS[dis]]], zmin=0, zmax=1,
                               hovertemplate="%{y}, %{x}: %{z}<extra></extra>"))
    fig.update_layout(height=760, margin=dict(l=0, r=0, t=10, b=0))
    fig.update_xaxes(tickformat="d", dtick=2)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch", key="story_heat")
    st.caption("Coloured = a value was reported, grey = no value. Dengue has no data for any state in 2014, 2015 "
               "and 2016; the model treats those years as missing.")

    st.header("3. Values revised in 2021 and 2022")
    rev = pd.DataFrame(story["revisions"])
    changed = int(rev.changed.sum())
    st.write(f"**{changed} of {len(rev)} state-disease-year values differ** between the earlier source and the "
             "current one. Nothing before 2021 changed.")
    rev["State"] = rev.state_id.map(nice)
    fig = px.scatter(rev, x=np.log10(rev.old + 1), y=np.log10(rev.new + 1), color="disease",
                     symbol="changed", hover_data={"State": True, "year": True, "old": True, "new": True},
                     color_discrete_map=DISEASE_COLOURS, labels={"x": "Earlier value (log10)", "y": "Current value (log10)"})
    top = float(max(np.log10(rev.old + 1).max(), np.log10(rev.new + 1).max()))
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(dash="dash", color="#000"),
                             name="No change", hoverinfo="skip"))
    st.plotly_chart(fig, width="stretch", key="story_rev")
    st.caption("Points on the dashed line were not changed. A true holdout on 2023 to 2025 is therefore not clean.")
    table = rev[rev.changed][["year", "disease", "State", "old", "new"]].rename(
        columns={"year": "Year", "disease": "Disease", "old": "Earlier", "new": "Current"})
    st.dataframe(table, hide_index=True)


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
    df["log_cases"] = np.log10(df.cases + 1)
    df["cases_text"] = df.cases.map(lambda v: "not reported" if pd.isna(v) else f"{v:,.0f} reported cases")
    top = float(df.log_cases.max())
    fig = px.choropleth(df, geojson=geojson, locations="state_id", featureidkey="properties.state_id",
                        color="log_cases", animation_frame="year", range_color=(0, top),
                        color_continuous_scale=["#f7f7f7"] + PALETTES[dis], custom_data=["State", "cases_text"])
    tpl = "%{customdata[0]}<br>%{customdata[1]}<extra></extra>"
    for tr in list(fig.data) + [t for fr in fig.frames for t in fr.data]:
        tr.hovertemplate = tpl
        tr.marker.line.width = 0.5
        tr.marker.line.color = "#000000"
    fig.update_geos(visible=False, lonaxis_range=[67, 98], lataxis_range=[6, 37], bgcolor="rgba(0,0,0,0)")
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      coloraxis_colorbar=dict(title="log10 cases"))
    st.plotly_chart(fig, width="stretch", theme=None, key=f"timelapse_{dis}")
    st.caption("Press play or drag the slider. Reported annual cases (observed data, not forecasts). Colour is "
               "log10 of cases so small and large states are both visible. Blank = no value reported that year.")


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
