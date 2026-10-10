"""Paths and small helpers shared by the API, ingest and explainability code."""
import contextlib
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DISEASES = ["dengue", "malaria", "chikungunya"]
DEFAULT_DB = REPO_ROOT / "data" / "vectorwatch.db"


def db_path():
    """Connection string for development = path of a SQLite file, from the environment."""
    return os.getenv("VW_DB_PATH", str(DEFAULT_DB))


@contextlib.contextmanager
def in_repo_root():
    """Person 2's helpers use paths like configs/states.txt, so they must run from the repo root."""
    old = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(old)


def state_name(state_id):
    """andaman_and_nicobar_islands -> Andaman and Nicobar Islands"""
    words = state_id.split("_")
    out = []
    for w in words:
        out.append(w if w == "and" else w.capitalize())
    return " ".join(out)
