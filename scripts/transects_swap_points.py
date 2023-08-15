import geopandas as gpd
from shapely.geometry import Point, LineString

# STEP 1: Enter in the name of the file to read from
gdf = gpd.read_file(
    r"C:\development\doodleverse\coastseg\CoastSeg\RM_config_gdf.geojson"
)


# Drop features whose "type" is not "transect"
gdf = gdf[gdf["type"] == "transect"]

# Reverse the coordinates of each LineString to swap origin and end points
gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom.coords[::-1])

# Create new GeoDataFrame for the reversed transects
reversed_transects_gdf = gpd.GeoDataFrame(
    gdf, geometry=gdf["geometry"].apply(lambda coords: LineString(coords))
)
# STEP 2: Enter in the names of the files to save to
# Save the reversed transects GeoDataFrame to a GeoJSON file
reversed_transects_gdf.to_file("reversed_RM_transects.geojson", driver="GeoJSON")

print("Reversed transects saved to 'reversed_transects.geojson'!")
