"""Sensitivity checks for the early-warning AUROC.
(a) pooled AUROC with RAW forecast counts as the score (shows how much big states drive it)
(b) AUROC computed inside each state-disease series, averaged over series that have both outcomes."""
import numpy as np, pandas as pd
from .data import get_data, DISEASES
from .metrics import bootstrap_ci
from .early_warning import threshold, auroc, DEFS, MIN_PRIOR

def build(pairs, Y, years, states):
    years = list(years); sidx = {s: i for i, s in enumerate(states)}; out = {}
    for dfn in DEFS:
        recs = []
        for r in pairs.itertuples():
            t = years.index(r.year)
            x = Y[:t, sidx[r.state_id], DISEASES.index(r.disease)]; x = x[~np.isnan(x)]
            if len(x) < MIN_PRIOR: continue
            thr, is_log = threshold(x, dfn); lt = thr if is_log else np.log1p(thr)
            tv = np.log1p(r.true) if is_log else r.true
            recs.append(dict(state_id=r.state_id, disease=r.disease, label=int(tv >= thr),
                             rel_model=np.log1p(r.model) - lt, rel_pers=np.log1p(r.persistence) - lt,
                             raw_model=r.model, raw_pers=r.persistence))
        out[dfn] = pd.DataFrame(recs)
    return out

if __name__ == "__main__":
    states, years, Y, _ = get_data()
    pairs = pd.read_csv("data/processed/stgnn_annual_geo_w_h1_pairs.csv")
    rows = []
    for dfn, df in build(pairs, Y, years, states).items():
        for dis in DISEASES + ["all"]:
            sub = df if dis == "all" else df[df.disease == dis]
            for pred, rel, raw in [("model", "rel_model", "raw_model"), ("persistence", "rel_pers", "raw_pers")]:
                within = []
                for _, g in sub.groupby(["state_id", "disease"]):
                    a = auroc(g[rel], g.label)
                    if not np.isnan(a): within.append(a)
                lo = hi = np.nan
                if len(within) >= 3: _, lo, hi = bootstrap_ci(within, n=5000, seed=0)
                rows.append(dict(definition=dfn, disease=dis, predictor=pred, n=len(sub), n_outbreak=int(sub.label.sum()),
                                 auroc_relative_pooled=auroc(sub[rel], sub.label),
                                 auroc_raw_count_pooled=auroc(sub[raw], sub.label),
                                 auroc_within_series_mean=float(np.mean(within)) if within else np.nan,
                                 within_ci_lo=lo, within_ci_hi=hi,
                                 within_series_median=float(np.median(within)) if within else np.nan,
                                 frac_series_above_chance=float(np.mean(np.array(within) > 0.5)) if within else np.nan,
                                 n_series_usable=len(within), n_series_total=sub.groupby(["state_id", "disease"]).ngroups))
    res = pd.DataFrame(rows); res.to_csv("data/processed/early_warning_sensitivity.csv", index=False)
    print(res.round(3).to_string())
