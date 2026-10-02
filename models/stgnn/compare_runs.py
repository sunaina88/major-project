import sys, numpy as np, pandas as pd
rng = np.random.default_rng(0)

def load(spec):
    tag, _, h = spec.partition("@"); h = int(h or 1)
    return h, pd.read_csv(f"data/processed/stgnn_annual_{tag}_h{h}.csv")

def ci(d):
    bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    return np.percentile(bs, [2.5, 97.5])

specs = sys.argv[1:]
_, ref = load(specs[0])
print("diff < 0 means the model is better. CI = 95% bootstrap over folds.\n")
for s in specs:
    h, r = load(s)
    d = (r.log_model - r.log_last).values; lo, hi = ci(d)
    line = (f"{s:12s} h={h} | log-MAE {r.log_model.mean():.3f} vs persistence {r.log_last.mean():.3f} "
            f"| diff {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] | MAE {r.mae_model.mean():.0f} vs {r.mae_last.mean():.0f}")
    if h == 1 and s != specs[0]:
        m = r.merge(ref, on="year", suffixes=("", "_ref"))
        d2 = (m.log_model - m.log_model_ref).values; l2, h2 = ci(d2)
        line += f" | vs {specs[0]}: {d2.mean():+.3f} [{l2:+.3f},{h2:+.3f}]"
    print(line)