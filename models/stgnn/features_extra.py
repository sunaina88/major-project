import numpy as np
import pandas as pd

LOG = {"area_km2", "population", "population_density_per_km2"}   # spans orders of magnitude, so use logs


def static_channels(states, years, cols, path="data/processed/static_state_features.csv"):
    df = pd.read_csv(path).set_index("state_id").reindex(states)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"static columns not found: {missing}. Available: {list(df.columns)}")
    out = []
    for c in cols:
        v = df[c].astype(float).values
        if np.isnan(v).all():
            raise SystemExit(f"static column '{c}' is empty")
        if c in LOG:
            v = np.log1p(v)
        out.append(np.broadcast_to(v[None, :], (len(years), len(states))))
    return np.stack(out, -1).astype(np.float64)


def env_channels(states, years, cols, path="data/processed/annual_environment.csv", min_cover=0.5):
    df = pd.read_csv(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"environment columns not found: {missing}. Available: {list(df.columns)}")
    out = []
    for c in cols:
        a = df.pivot(index="year", columns="state_id", values=c).reindex(index=list(years), columns=list(states)).values
        cover = 1 - np.isnan(a[-1]).mean()
        if cover < min_cover:
            raise SystemExit(f"'{c}' covers only {cover:.0%} of states in {years[-1]}: the weather file stops too early")
        out.append(a)
    return np.stack(out, -1)