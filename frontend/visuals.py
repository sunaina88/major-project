"""Visual identity and reusable chart/HTML helpers. No data rules here (those live in logic.py)."""
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from logic import LEVELS, PALETTES, colour_map, discrete_scale, nice

NAVY, PANEL, PANEL2 = "#0A1524", "#102438", "#14304A"
TEAL, AMBER, RED = "#16C7B2", "#E8A34B", "#E5534B"
TEXT, MUTED = "#DCE8F2", "#8FA6BA"
GRID = "rgba(143,166,186,0.16)"
BASE_LAND = "#1b2f45"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(1100px 560px at 88% -8%, #12405a 0%, #0A1524 58%) fixed; }
.block-container { padding-top: 1.4rem; max-width: 1500px; }
section[data-testid="stSidebar"] { background: #0b1a2b; border-right: 1px solid rgba(143,166,186,.15); }
h1, h2, h3 { letter-spacing: -0.02em; }
.vw-brand { display:flex; gap:12px; align-items:center; padding:6px 0 14px 0; }
.vw-logo { width:40px; height:40px; border-radius:11px; background:linear-gradient(135deg,#16C7B2,#0b7f8f);
           display:flex; align-items:center; justify-content:center; font-weight:800; color:#06202a; }
.vw-brand-t { font-weight:700; font-size:1.02rem; color:#fff; line-height:1.1; }
.vw-brand-s { font-size:.72rem; color:#8FA6BA; }
.vw-foot { font-size:.74rem; color:#8FA6BA; border-top:1px solid rgba(143,166,186,.15); padding-top:10px; margin-top:14px; }
.vw-hero { margin: 0 0 14px 0; }
.vw-eyebrow { font-size:.72rem; letter-spacing:.14em; color:#16C7B2; font-weight:600; }
.vw-title { font-size:2.15rem; font-weight:800; color:#fff; line-height:1.15; margin:2px 0 4px 0; }
.vw-sub { color:#8FA6BA; font-size:.98rem; max-width:900px; }
.vw-kpi { background:linear-gradient(160deg,#133049,#0f2236); border:1px solid rgba(22,199,178,.20);
          border-radius:14px; padding:14px 18px; height:100%; }
.vw-kpi .l { font-size:.70rem; text-transform:uppercase; letter-spacing:.09em; color:#8FA6BA; font-weight:600; }
.vw-kpi .v { font-size:1.65rem; font-weight:700; color:#fff; line-height:1.25; }
.vw-kpi .s { font-size:.76rem; color:#8FA6BA; }
.vw-card { background:#102438; border:1px solid rgba(143,166,186,.16); border-radius:14px; padding:16px 18px; }
.vw-card h4 { margin:0 0 8px 0; font-size:1.05rem; color:#fff; }
.vw-row { display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid rgba(143,166,186,.10); font-size:.88rem; }
.vw-row span:first-child { color:#8FA6BA; }
.vw-note { border-left:3px solid #16C7B2; background:rgba(22,199,178,.08); padding:10px 14px;
           border-radius:0 10px 10px 0; color:#DCE8F2; font-size:.9rem; margin:6px 0 12px 0; }
.vw-warn { border-left-color:#E8A34B; background:rgba(232,163,75,.09); }
.vw-pill { display:inline-block; padding:3px 11px; border-radius:99px; font-size:.78rem; font-weight:600; margin:2px 4px 2px 0; }
.vw-stage { background:linear-gradient(160deg,#133049,#0f2236); border:1px solid rgba(22,199,178,.25); border-radius:12px;
            padding:12px 10px; text-align:center; min-height:96px; }
.vw-stage b { color:#fff; font-size:.92rem; }
.vw-stage div { color:#8FA6BA; font-size:.74rem; margin-top:4px; }
.vw-stage.on { border-color:#16C7B2; box-shadow:0 0 0 1px #16C7B2, 0 0 18px rgba(22,199,178,.35); }
.vw-arrow { text-align:center; color:#16C7B2; font-size:1.4rem; padding-top:30px; }
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] { background:#102438; border-radius:10px 10px 0 0; padding:8px 18px; }
.stTabs [aria-selected="true"] { background:#16C7B2 !important; }
.stTabs [aria-selected="true"] p { color:#06202a !important; font-weight:700; }
div[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }
footer, #MainMenu { visibility:hidden; }
</style>
"""


def apply_theme():
    pio.templates["vw"] = go.layout.Template(layout=dict(
        font=dict(family="Inter, sans-serif", color=TEXT),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        colorway=[TEAL, "#fb6a4a", "#6baed6", "#74c476", AMBER, "#b39ddb"],
        legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=10, r=10, t=44, b=10),
        hoverlabel=dict(bgcolor=PANEL2, font_color=TEXT)))
    pio.templates.default = "vw"
    st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ small HTML helpers
def hero(title, subtitle="", eyebrow=""):
    st.markdown(f'<div class="vw-hero"><div class="vw-eyebrow">{eyebrow}</div>'
                f'<div class="vw-title">{title}</div><div class="vw-sub">{subtitle}</div></div>',
                unsafe_allow_html=True)


def kpi(label, value, sub=""):
    st.markdown(f'<div class="vw-kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def kpi_row(items):
    for col, (label, value, sub) in zip(st.columns(len(items)), items):
        with col:
            kpi(label, value, sub)


def note(text, kind="info"):
    cls = "vw-note vw-warn" if kind == "warn" else "vw-note"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def card(title, rows):
    """rows: list of (label, value)."""
    body = "".join(f'<div class="vw-row"><span>{a}</span><span>{b}</span></div>' for a, b in rows)
    st.markdown(f'<div class="vw-card"><h4>{title}</h4>{body}</div>', unsafe_allow_html=True)


def pills(items, colour=TEAL, text_colour="#06202a"):
    st.markdown("".join(f'<span class="vw-pill" style="background:{colour};color:{text_colour}">{i}</span>'
                        for i in items), unsafe_allow_html=True)


def risk_legend(disease):
    cm = colour_map(disease)
    st.markdown("".join(f'<span class="vw-pill" style="background:{cm[l]};color:#06131f">{l}</span>' for l in LEVELS),
                unsafe_allow_html=True)


def show(fig, key, height=None, **kw):
    if height:
        fig.update_layout(height=height)
    return st.plotly_chart(fig, width="stretch", theme=None, key=key, **kw)


# ------------------------------------------------------------------ maps
LOG_TICKS = [(0, "0"), (1, "10"), (2, "100"), (3, "1k"), (4, "10k"), (5, "100k"), (6, "1M")]


def _land_layout(fig, height):
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=6, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      geo_bgcolor="rgba(0,0,0,0)")


def _case_text(name, v):
    return f"<b>{name}</b><br>" + ("not reported" if np.isnan(v) else f"{v:,.0f} reported cases")


def cases_choropleth(cases, geojson, disease, height=560):
    """cases: pandas Series indexed by state_id; NaN = not reported (drawn as dark land, not as zero)."""
    ids = list(cases.index)
    vals = cases.astype(float).values
    z = np.log10(np.nan_to_num(vals, nan=0.0) + 1)
    text = [_case_text(nice(s), v) for s, v in zip(ids, vals)]
    pal = PALETTES[disease]
    scale = [[i / (len(pal) - 1), c] for i, c in enumerate(pal)]
    top = max(1.0, float(z.max()))
    ticks = [(v, t) for v, t in LOG_TICKS if v <= top + 0.3]
    fig = go.Figure(go.Choropleth(
        geojson=geojson, locations=ids, featureidkey="properties.state_id", z=[0] * len(ids),
        colorscale=[[0, BASE_LAND], [1, BASE_LAND]], showscale=False, text=text, hoverinfo="text",
        marker_line_width=0.6, marker_line_color=NAVY))
    mask = ~np.isnan(vals)
    fig.add_trace(go.Choropleth(
        geojson=geojson, locations=[i for i, m in zip(ids, mask) if m], featureidkey="properties.state_id",
        z=z[mask], text=[t for t, m in zip(text, mask) if m], hoverinfo="text", colorscale=scale, zmin=0, zmax=top,
        marker_line_width=0.6, marker_line_color=NAVY,
        colorbar=dict(title="cases", thickness=12, len=0.65, tickvals=[v for v, _ in ticks],
                      ticktext=[t for _, t in ticks], outlinewidth=0)))
    _land_layout(fig, height)
    return fig


def risk_choropleth(df, geojson, disease, height=560):
    """df has state_id, risk, forecast_text, persistence_text (one row per state)."""
    hover = [f"<b>{nice(s)}</b><br>Risk: {r}<br>Forecast: {f}<br>Last year: {p}"
             for s, r, f, p in zip(df.state_id, df.risk, df.forecast_text, df.persistence_text)]
    fig = go.Figure(go.Choropleth(
        geojson=geojson, locations=df.state_id, featureidkey="properties.state_id",
        z=[LEVELS.index(r) for r in df.risk], colorscale=discrete_scale(LEVELS, disease),
        zmin=-0.5, zmax=len(LEVELS) - 0.5, showscale=False, text=hover, hoverinfo="text",
        marker_line_width=0.6, marker_line_color=NAVY))
    _land_layout(fig, height)
    return fig


def land_base(geojson, states):
    """A dark base layer of all states, used under graph overlays."""
    return go.Choropleth(geojson=geojson, locations=states, featureidkey="properties.state_id", z=[0] * len(states),
                         colorscale=[[0, BASE_LAND], [1, BASE_LAND]], showscale=False, hoverinfo="skip",
                         marker_line_width=0.6, marker_line_color="#0f2438")