"""
build_pan_india_state_tensor.py

Builds the national, state-wise VectorWatch India spatiotemporal tensor from
three NDAP-style raw annual state tables (Dengue, Malaria, Chikungunya),
covering all ~36 Indian States/UTs.

Pipeline:
    1. Load the three raw CSVs.
    2. Extract a clean 4-digit year from the free-text 'Year' column.
    3. Standardize state names (whitespace/casing/aliases) and drop
       non-state rows (national totals, blank states).
    4. Full outer-merge the three diseases on [State, Year] so a state
       missing a report for one disease in a given year is preserved with
       NaN for that disease rather than dropped.
    5. Clean numeric Cases/Deaths columns (coerce non-numeric -> NaN).
    6. Disaggregate each annual total into 12 monthly rows using a fixed
       monsoon-weighted curve (peaking Jul-Sep), for every (state, year)
       row present in the merged data, capped at year <= 2022.
    7. Assemble the final standardized schema.
    8. Save to data/processed/pan_india_state_spatiotemporal_tensor.csv.
    9. Print a summary of the result.

--------------------------------------------------------------------------
Design notes / assumptions (flagged explicitly rather than buried)
--------------------------------------------------------------------------
- NUMERIC MISSING VALUES ARE KEPT AS NaN, NOT FILLED WITH 0.
  A state simply absent from a disease's source file for a given year (or
  a cell that couldn't be parsed as a number, e.g. 'NR') means "no report",
  not "zero cases". Silently converting that to 0 would fabricate a
  negative epidemiological signal. If your downstream modeling genuinely
  wants 0-filled counts, do that explicitly and separately -- don't bake
  the assumption into ingestion.
- CHIKUNGUNYA HAS NO DEATHS COLUMN AT SOURCE. chikungunya_deaths is set to
  pd.NA throughout (not 0), for the same "unreported != zero" reasoning.
- MONTHLY DISAGGREGATION PRODUCES FLOATS, NOT INTEGERS.
  annual_total * monthly_weight is a modeled estimate, not an original
  reported count, and forcing it to Int64 would either lose precision or
  make the 12 months not sum back exactly to the annual total. Cases/Deaths
  columns in the final monthly tensor are float64. If integer months are
  required downstream, round explicitly at that stage.
- COVERAGE OF "from their earliest available year through 2022": this
  script disaggregates every (state, year) row that actually exists in the
  merged annual data (capped at year <= 2022) -- it does not fabricate
  additional years a state has zero source coverage for. That would be
  interpolation, not disaggregation, and is a materially different
  modeling choice best made explicitly by the forecasting team, not
  silently inside an ingestion script.
- HEADER MATCHING IS KEYWORD-BASED, NOT EXACT-STRING. The raw column names
  in the prompt contain irregular whitespace (e.g. a double space in
  '...Due To Dengue  (UOM...'). Columns are located by keyword ('case' /
  'death') rather than exact string equality, so minor whitespace/wording
  drift across NDAP exports doesn't silently break the pipeline.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
# Paths relative to the working directory at run time, consistent with the
# rest of this ingestion pipeline. Run as:
#     python src/ingestion/disease/build_pan_india_state_tensor.py
RAW_DIR = Path("data/raw/pan_india_state_tabular")
OUTPUT_PATH = Path("data/processed/pan_india_state_spatiotemporal_tensor.csv")

RAW_FILES = {
    "dengue": RAW_DIR / "dengue_state.csv",
    "malaria": RAW_DIR / "malaria_state.csv",
    "chikungunya": RAW_DIR / "chikungunya_state.csv",
}
# Which diseases have a deaths column at source at all.
DISEASE_HAS_DEATHS = {"dengue": True, "malaria": True, "chikungunya": False}

MAX_YEAR = 2022

# Monthly monsoon-weighted disaggregation curve (Jan..Dec), sums to 1.0,
# peaking Jul-Aug-Sep -- the core Indian monsoon vector-breeding season.
MONSOON_MONTHLY_WEIGHTS = {
    1: 0.02, 2: 0.02, 3: 0.03, 4: 0.04, 5: 0.05, 6: 0.08,
    7: 0.18, 8: 0.22, 9: 0.18, 10: 0.10, 11: 0.05, 12: 0.03,
}
assert abs(sum(MONSOON_MONTHLY_WEIGHTS.values()) - 1.0) < 1e-9, "Monthly weights must sum to 1.0"

# Non-state / non-UT rows to drop after standardization.
NON_STATE_ENTITIES = {
    "India", "All India", "Total", "All-India Total", "Grand Total", "All India Total",
}

# Common raw-name variants -> canonical name, keyed by the Title-Case form
# the raw value takes after whitespace/ampersand normalization.
STATE_NAME_ALIASES = {
    "Orissa": "Odisha",
    "Pondicherry": "Puducherry",
    "Uttaranchal": "Uttarakhand",
    "Chattisgarh": "Chhattisgarh",
    "Nct Of Delhi": "Delhi",
    "Delhi (Nct)": "Delhi",
    "National Capital Territory Of Delhi": "Delhi",
    "A And N Islands": "Andaman And Nicobar Islands",
    "Andaman And Nicobar": "Andaman And Nicobar Islands",
    "D And N Haveli": "Dadra And Nagar Haveli And Daman And Diu",
    "Dadra And Nagar Haveli": "Dadra And Nagar Haveli And Daman And Diu",
    "Daman And Diu": "Dadra And Nagar Haveli And Daman And Diu",
    "The Dadra And Nagar Haveli And Daman And Diu": "Dadra And Nagar Haveli And Daman And Diu",
    "Jammu And Kashmir (Ut)": "Jammu And Kashmir",
}

YEAR_PATTERN = re.compile(r"(\d{4})")


# --------------------------------------------------------------------------
# Step 1: Load raw CSVs
# --------------------------------------------------------------------------
def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    return pd.read_csv(path, dtype=str)


# --------------------------------------------------------------------------
# Column detection (keyword-based, tolerant of whitespace/wording drift)
# --------------------------------------------------------------------------
def find_column(df: pd.DataFrame, must_include: list, must_exclude: list = None) -> str:
    must_exclude = must_exclude or []
    for col in df.columns:
        col_lower = col.lower()
        if all(kw in col_lower for kw in must_include) and not any(kw in col_lower for kw in must_exclude):
            return col
    raise ValueError(
        f"Could not find a column matching include={must_include}, exclude={must_exclude}. "
        f"Available columns: {list(df.columns)}"
    )


# --------------------------------------------------------------------------
# Step 2: Extract clean 4-digit year
# --------------------------------------------------------------------------
def extract_year(year_series: pd.Series) -> pd.Series:
    extracted = year_series.astype(str).str.extract(YEAR_PATTERN, expand=False)
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")


# --------------------------------------------------------------------------
# Step 3: Standardize state names, drop non-state / NaN rows
# --------------------------------------------------------------------------
def standardize_state_name(raw_state) -> str:
    if pd.isna(raw_state):
        return np.nan
    s = str(raw_state).strip()
    s = re.sub(r"\s*&\s*", " and ", s)   # "A & N Islands" -> "A and N Islands"
    s = re.sub(r"\s+", " ", s).strip()
    s_title = s.title()
    return STATE_NAME_ALIASES.get(s_title, s_title)


def clean_states(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["State"] = df["State"].map(standardize_state_name)
    df = df[df["State"].notna()]
    df = df[~df["State"].isin(NON_STATE_ENTITIES)]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Step 5: Clean numeric columns (coerce to numeric, NaN on failure)
# --------------------------------------------------------------------------
def clean_numeric_column(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


# --------------------------------------------------------------------------
# Per-disease loader: standardizes one raw file down to
# [State, Year, <disease>_cases, <disease>_deaths]
# --------------------------------------------------------------------------
def load_disease(disease: str, path: Path) -> pd.DataFrame:
    df = load_raw(path)

    if "State" not in df.columns or "Year" not in df.columns:
        raise ValueError(f"{path.name} is missing required 'State'/'Year' columns.")

    cases_col = find_column(df, must_include=["case"], must_exclude=["death"])
    df["Year"] = extract_year(df["Year"])
    df = clean_states(df)

    out = pd.DataFrame({
        "State": df["State"],
        "Year": df["Year"],
        f"{disease}_cases": clean_numeric_column(df[cases_col]),
    })

    if DISEASE_HAS_DEATHS[disease]:
        deaths_col = find_column(df, must_include=["death"])
        out[f"{disease}_deaths"] = clean_numeric_column(df[deaths_col])
    else:
        out[f"{disease}_deaths"] = np.nan  # e.g. Chikungunya: no deaths column at source

    # A (State, Year) should be unique per source file; if a source has
    # duplicate rows for the same state-year, sum them defensively.
    out = out.groupby(["State", "Year"], as_index=False).sum(min_count=1)
    return out


# --------------------------------------------------------------------------
# Step 4: Full outer merge on [State, Year]
# --------------------------------------------------------------------------
def merge_diseases(dengue: pd.DataFrame, malaria: pd.DataFrame, chikungunya: pd.DataFrame) -> pd.DataFrame:
    merged = dengue.merge(malaria, on=["State", "Year"], how="outer")
    merged = merged.merge(chikungunya, on=["State", "Year"], how="outer")
    return merged


# --------------------------------------------------------------------------
# Step 6: Disaggregate annual totals into monthly rows via the monsoon curve
# --------------------------------------------------------------------------
def disaggregate_to_monthly(annual_df: pd.DataFrame) -> pd.DataFrame:
    annual_df = annual_df[annual_df["Year"] <= MAX_YEAR].copy()

    value_cols = [
        "dengue_cases", "dengue_deaths",
        "malaria_cases", "malaria_deaths",
        "chikungunya_cases", "chikungunya_deaths",
    ]

    monthly_frames = []
    for month, weight in MONSOON_MONTHLY_WEIGHTS.items():
        chunk = annual_df.copy()
        chunk["start_date"] = pd.to_datetime(
            dict(year=chunk["Year"].astype(int), month=month, day=1)
        )
        for col in value_cols:
            chunk[col] = chunk[col] * weight
        monthly_frames.append(chunk)

    monthly = pd.concat(monthly_frames, ignore_index=True)
    return monthly


# --------------------------------------------------------------------------
# Step 7: Assemble final standardized schema
# --------------------------------------------------------------------------
def build_state_id(state_name: str) -> str:
    s = state_name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def finalize_schema(monthly: pd.DataFrame) -> pd.DataFrame:
    df = monthly.copy()
    df["state_name"] = df["State"]
    df["state_id"] = df["state_name"].map(build_state_id)

    final_cols = [
        "start_date", "state_name", "state_id",
        "dengue_cases", "dengue_deaths",
        "malaria_cases", "malaria_deaths",
        "chikungunya_cases", "chikungunya_deaths",
    ]
    df = df[final_cols]
    df = df.sort_values(["state_name", "start_date"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Step 9: Print a summary
# --------------------------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("PAN-INDIA STATE SPATIOTEMPORAL TENSOR -- SUMMARY")
    print("=" * 70)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Unique states: {df['state_name'].nunique()}")
    print(f"Date range: {df['start_date'].min().date()} to {df['start_date'].max().date()}")
    print("\nStates included:")
    print(sorted(df['state_name'].unique().tolist()))
    print("\nHead sample:")
    print(df.head(10).to_string(index=False))
    print("=" * 70)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def process() -> pd.DataFrame:
    dengue = load_disease("dengue", RAW_FILES["dengue"])
    malaria = load_disease("malaria", RAW_FILES["malaria"])
    chikungunya = load_disease("chikungunya", RAW_FILES["chikungunya"])

    merged = merge_diseases(dengue, malaria, chikungunya)
    monthly = disaggregate_to_monthly(merged)
    final_df = finalize_schema(monthly)
    return final_df


if __name__ == "__main__":
    result = process()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    to_save = result.copy()
    to_save["start_date"] = to_save["start_date"].dt.strftime("%Y-%m-%d")
    to_save.to_csv(OUTPUT_PATH, index=False)

    print_summary(result)
    print(f"\nSaved pan-India spatiotemporal tensor to: {OUTPUT_PATH}")