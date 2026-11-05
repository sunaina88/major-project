import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "models/stgnn")
from annual_data import load_annual

states, years, Y, _ = load_annual("data/processed/pan_india_state_enriched_tensor.csv")
t = pd.DataFrame((~np.isnan(Y)).sum(1), index=years, columns=["dengue", "malaria", "chikungunya"])
print("states with a reported value, per year (36 = complete):")
print(t.to_string())
for s in (1999, 2005, 2010, 2015):
    sub = t.loc[s:]
    print(f"start {s}: {int(sub.values.sum())} real state-years ({sub.values.sum() / (len(sub) * 36 * 3):.0%} of the full grid)")