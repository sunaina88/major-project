# Frontend

Decision: Streamlit (fast, team is short on time; Plotly tile map now, GeoJSON later).
Run: `pip install -r requirements-frontend.txt`, `python scripts/make_mocks.py`, `streamlit run frontend/app.py`.
Switch to the real API: `USE_MOCKS=false API_URL=http://localhost:8000`.
Pages map to endpoints in `frontend/api_client.py` (ENDPOINTS). Update after reading person5.md.
