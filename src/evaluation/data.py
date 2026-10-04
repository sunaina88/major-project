import sys
sys.path.insert(0, "models/stgnn")
DISEASES = ["dengue", "malaria", "chikungunya"]
DEFAULT_TENSOR = "data/processed/pan_india_state_enriched_tensor_real.csv"

def get_data(path=DEFAULT_TENSOR):
    from annual_data import load_annual
    states, years, Y, W = load_annual(path)
    import numpy as np
    return list(states), np.asarray(years), np.asarray(Y, dtype=float), W
