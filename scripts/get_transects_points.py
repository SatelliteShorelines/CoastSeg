import geopandas as gpd
from shapely.geometry import Point

# STEP 1: Enter in the name of the file to read from
# Read the GeoJSON file
gdf = gpd.read_file(
    r"C:\development\doodleverse\coastseg\CoastSeg\shortened_transects_JB.geojson"
)

# Drop features whose "type" is not "transect"
gdf = gdf[gdf["type"] == "transect"]


# Create GeoDataFrames for the origin and end points
origin_gdf = gpd.GeoDataFrame(
    gdf[["type"]], geometry=gdf["geometry"].apply(lambda geom: Point(geom.coords[0]))
)
end_gdf = gpd.GeoDataFrame(
    gdf[["type"]], geometry=gdf["geometry"].apply(lambda geom: Point(geom.coords[-1]))
)
# STEP 2: Enter in the names of the files to save to
# Save the origin and end GeoDataFrames to separate GeoJSON files
origin_gdf.to_file("origin_points.geojson", driver="GeoJSON")
end_gdf.to_file("end_points.geojson", driver="GeoJSON")

print(origin_gdf)
print(end_gdf)
