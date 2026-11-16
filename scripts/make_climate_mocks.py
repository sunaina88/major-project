"""Build frontend/mocks/climate_monthly.json, climate_annual.json and state_profile.json from the processed
environment files, using the same code as the API. Run from the repo root."""
import json
import sys
import os

sys.path.insert(0, ".")
from src.api import environment  # noqa: E402

OUT = "frontend/mocks"
os.makedirs(OUT, exist_ok=True)
for name, rows in [("climate_monthly", environment.monthly()), ("climate_annual", environment.annual()),
                   ("state_profile", environment.profile())]:
    with open(f"{OUT}/{name}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, separators=(",", ":"))
    print(name, len(rows), "rows")