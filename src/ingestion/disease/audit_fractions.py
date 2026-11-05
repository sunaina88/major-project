import glob, os, re
import pandas as pd

C_AP = 0.5861509524794264
RAW = "data/raw/pan_india_state_tabular"

for path in sorted(glob.glob(f"{RAW}/*_state.csv")):
    name = os.path.basename(path).replace("_state.csv", "")
    df = pd.read_csv(path)
    print("=" * 90)
    print(name, df.shape)
    print("columns:", list(df.columns))
    scol = next(c for c in df.columns if "state" in c.lower())
    ycol = next(c for c in df.columns if "year" in c.lower())
    df["_year"] = df[ycol].astype(str).str.extract(r"(\d{4})")[0].astype(float)
    skip = {scol, ycol, "_year"} | {c for c in df.columns if c.lower() in ("country", "additional info")}
    for c in [c for c in df.columns if c not in skip]:
        v = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
        nonint = v.notna() & ((v - v.round()).abs() > 1e-9)
        print(f"\n[{c[:70]}] non-numeric: {int(v.isna().sum())} | fractional values: {int(nonint.sum())}")
        if nonint.any():
            sub = df.loc[nonint, [scol, "_year"]].assign(value=v[nonint])
            sub["x_ap"] = (sub.value / C_AP)
            sub["x_tel"] = (sub.value / (1 - C_AP))
            sub["ap_multiple"] = (sub.x_ap - sub.x_ap.round()).abs() < 1e-5
            sub["tel_multiple"] = (sub.x_tel - sub.x_tel.round()).abs() < 1e-5
            print(sub.groupby(scol).agg(n=("value", "size"), year_min=("_year", "min"), year_max=("_year", "max"),
                                        ap_multiple=("ap_multiple", "mean"),
                                        tel_multiple=("tel_multiple", "mean")).to_string())
        for st in df[scol].astype(str).unique():
            if re.search(r"andhra|telangana", st, re.I):
                m = df[scol].astype(str) == st
                s = df.loc[m, ["_year"]].assign(value=v[m]).sort_values("_year")
                s["whole_number"] = (s.value - s.value.round()).abs() < 1e-9
                print(f"\n  {st}:")
                print(s.to_string(index=False))