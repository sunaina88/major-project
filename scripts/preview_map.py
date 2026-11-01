import json
import pandas as pd
import plotly.express as px

gj = json.load(open("data/processed/india_states.geojson"))
ids = [f["properties"]["state_id"] for f in gj["features"]]
df = pd.DataFrame({"state_id": ids, "value": range(len(ids))})
fig = px.choropleth(df, geojson=gj, locations="state_id", featureidkey="properties.state_id",
                    color="value", hover_name="state_id")
fig.update_geos(fitbounds="locations", visible=False)
fig.write_html("data/processed/map_preview.html")
print("wrote data/processed/map_preview.html")
