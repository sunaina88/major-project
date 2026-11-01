import argparse, json, os, re
import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid

STATES = [s.strip() for s in open("configs/states.txt") if s.strip()]
DNH = "dadra_and_nagar_haveli_and_daman_and_diu"
ALIASES = {
    "orissa": "odisha", "pondicherry": "puducherry", "uttaranchal": "uttarakhand",
    "nct of delhi": "delhi", "national capital territory of delhi": "delhi",
    "dadra and nagar haveli": DNH, "daman and diu": DNH,
    "dadra and nagar haveli and daman and diu": DNH,
    "andaman and nicobar": "andaman_and_nicobar_islands",
    "andaman nicobar islands": "andaman_and_nicobar_islands",
    "jammu kashmir": "jammu_and_kashmir",
    "tamilnadu": "tamil_nadu", "chhattishgarh": "chhattisgarh", "telengana": "telangana",
    "daman and diu and dadra and nagar haveli": DNH,
}
NAME_CANDIDATES = ["STATE", "State", "ST_NM", "NAME_1", "st_nm", "STATE_NAME", "State_Name",
                   "NAME", "Name", "name", "stname", "STNAME"]


def norm(s):
    s = str(s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"^the ", "", s)


def to_id(name):
    n = norm(name)
    return ALIASES.get(n, n.replace(" ", "_"))


def round_coords(o, nd=4):
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, list):
        return [round_coords(x, nd) for x in o]
    return o


ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--name-col")
ap.add_argument("--inspect", action="store_true")
ap.add_argument("--drop", nargs="*", default=[], help="source names to ignore")
ap.add_argument("--tolerance", type=float, default=0.01, help="simplification in degrees (~1 km)")
ap.add_argument("--out", default="data/processed/india_states.geojson")
a = ap.parse_args()

gdf = gpd.read_file(a.src)
print("rows:", len(gdf), "| CRS:", gdf.crs)
print("columns:", [c for c in gdf.columns if c != "geometry"])
col = a.name_col or next((c for c in NAME_CANDIDATES if c in gdf.columns), None)
if col is None:
    raise SystemExit("Could not guess the name column. Use --name-col <column>.")
print("name column:", col)
gdf = gdf[~gdf[col].isin(a.drop)].copy()

if a.inspect:
    for n in gdf[col]:
        sid = to_id(n)
        print(f"{str(n):50s} -> {sid:45s} {'' if sid in STATES else 'UNMATCHED'}")
    raise SystemExit

gdf["state_id"] = gdf[col].map(to_id)
bad = sorted(set(gdf.loc[~gdf.state_id.isin(STATES), col].astype(str)))
if bad:
    raise SystemExit(f"Unmatched names {bad}. Add them to ALIASES, or ignore them with --drop, then rerun.")
missing = sorted(set(STATES) - set(gdf.state_id))
if missing:
    raise SystemExit(f"No polygon for {missing}. This source lacks them (outdated?); pick another source.")

if gdf.crs is None:
    print("WARNING: no CRS in the file, assuming EPSG:4326 (lon/lat)")
    gdf = gdf.set_crs(4326)
gdf = gdf.to_crs(4326)
gdf["geometry"] = gdf.geometry.apply(make_valid)

os.makedirs("data/raw/geography", exist_ok=True)
pd.DataFrame({"source_name": gdf[col].astype(str), "state_id": gdf.state_id}).to_csv(
    "data/raw/geography/state_name_map.csv", index=False)

g = gdf.dissolve(by="state_id", as_index=False)[["state_id", "geometry"]]
area = g.to_crs(6933).geometry.area / 1e6          # equal-area projection, from the UNSIMPLIFIED shapes
pd.DataFrame({"state_id": g.state_id, "area_km2": area.round(1)}).to_csv(
    "data/processed/state_area_km2.csv", index=False)

g["geometry"] = g.geometry.simplify(a.tolerance, preserve_topology=True).apply(make_valid)
js = json.loads(g.to_json())
for f in js["features"]:
    f["geometry"]["coordinates"] = round_coords(f["geometry"]["coordinates"])
    f["properties"]["state_name"] = f["properties"]["state_id"].replace("_", " ").title().replace(" And ", " and ")
json.dump(js, open(a.out, "w"), separators=(",", ":"))

print(f"\nfeatures: {len(js['features'])} | unique ids: {g.state_id.nunique()} | size: {os.path.getsize(a.out)/1e6:.2f} MB")
print("bounds (lon/lat min/max):", [round(x, 1) for x in g.total_bounds], "(India is roughly 68, 6, 98, 37)")
print("saved", a.out, "and data/processed/state_area_km2.csv")