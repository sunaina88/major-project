import numpy as np, pandas as pd
from .data import get_data, DISEASES

DEFS = ["p90", "p95", "mean+2sd"]
MIN_PRIOR = 5

def auroc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, int)
    npos, nneg = label.sum(), (1 - label).sum()
    if npos == 0 or nneg == 0:
        return np.nan
    r = pd.Series(score).rank().to_numpy()
    return float((r[label == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))

def threshold(prior, definition):
    """Returns (thr, is_log). prior = earlier observed values only."""
    if definition == "p90": return np.percentile(prior, 90), False
    if definition == "p95": return np.percentile(prior, 95), False
    lp = np.log1p(prior)
    return lp.mean() + 2 * lp.std(ddof=1), True

def evaluate(pairs, Y, years, states, out="data/processed/early_warning_eval.csv"):
    years = list(years); sidx = {s: i for i, s in enumerate(states)}
    rows = []
    for definition in DEFS:
        recs = []
        for r in pairs.itertuples():
            t = years.index(r.year)
            x = Y[:t, sidx[r.state_id], DISEASES.index(r.disease)]
            x = x[~np.isnan(x)]
            if len(x) < MIN_PRIOR:
                continue
            thr, is_log = threshold(x, definition)
            lg = (lambda v: np.log1p(v)) if True else None
            tv = np.log1p(r.true) if is_log else r.true
            lthr = thr if is_log else np.log1p(thr)
            recs.append(dict(disease=r.disease, label=int(tv >= thr),
                             s_model=np.log1p(r.model) - lthr,
                             s_pers=np.log1p(r.persistence) - lthr))
        df = pd.DataFrame(recs)
        for dis in DISEASES + ["all"]:
            sub = df if dis == "all" else df[df.disease == dis]
            for pred, col in [("model", "s_model"), ("persistence", "s_pers")]:
                y, sc = sub.label.to_numpy(), sub[col].to_numpy()
                f = (sc >= 0).astype(int)
                tp, fp = int(((f == 1) & (y == 1)).sum()), int(((f == 1) & (y == 0)).sum())
                fn, tn = int(((f == 0) & (y == 1)).sum()), int(((f == 0) & (y == 0)).sum())
                prec = tp / (tp + fp) if tp + fp else np.nan
                rec = tp / (tp + fn) if tp + fn else np.nan
                f1 = 2 * prec * rec / (prec + rec) if tp else (0.0 if tp + fp + fn else np.nan)
                rows.append(dict(definition=definition, disease=dis, predictor=pred,
                                 n=len(sub), n_outbreak=int(y.sum()), precision=prec, recall=rec, f1=f1,
                                 auroc=auroc(sc, y), false_alarm_rate=fp / (fp + tn) if fp + tn else np.nan))
    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)
    return res

if __name__ == "__main__":
    states, years, Y, _ = get_data()
    pairs = pd.read_csv("data/processed/stgnn_annual_geo_w_h1_pairs.csv")
    res = evaluate(pairs, Y, years, states)
    print(res.round(3).to_string())
    print("\nComparisons: 3 definitions x 2 predictors x 3 diseases = 18 (plus pooled 'all' rows). Check n_outbreak before trusting AUROC.")
