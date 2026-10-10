"""Write the OpenAPI document and example responses for Person 6 (frontend mocks).

Run from the repository root after ingest:  python -m src.api.export_examples
Output: docs/api_examples/*.json and docs/openapi.json
"""
import json

from fastapi.testclient import TestClient

from src.api.config import REPO_ROOT
from src.api.main import app

EXAMPLES = {
    "health": "/health",
    "states": "/states",
    "historical_kerala": "/historical/kerala",
    "forecast_kerala_dengue": "/forecast/kerala/dengue",
    "forecast_kerala_dengue_with_experimental": "/forecast/kerala/dengue?include_experimental=true",
    "forecast_low_confidence_andhra_pradesh_dengue": "/forecast/andhra_pradesh/dengue",
    "risk_map_dengue_2023": "/risk-map?disease=dengue&year=2023",
    "risk_map_malaria_2023": "/risk-map?disease=malaria&year=2023",
    "risk_map_chikungunya_2023": "/risk-map?disease=chikungunya&year=2023",
    "model_performance": "/model-performance",
    "explanation_kerala_dengue": "/explanation/kerala/dengue",
}


def main():
    out = REPO_ROOT / "docs" / "api_examples"
    out.mkdir(parents=True, exist_ok=True)
    client = TestClient(app)
    for name, url in EXAMPLES.items():
        r = client.get(url)
        r.raise_for_status()
        with open(out / f"{name}.json", "w") as f:
            json.dump(r.json(), f, indent=2)
            f.write("\n")
    with open(REPO_ROOT / "docs" / "openapi.json", "w") as f:
        json.dump(app.openapi(), f, indent=2)
        f.write("\n")
    print(f"wrote {len(EXAMPLES)} examples to {out} and docs/openapi.json")


if __name__ == "__main__":
    main()
