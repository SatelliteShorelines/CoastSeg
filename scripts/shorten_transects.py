import geopandas as gpd
from shapely.geometry import LineString


def utm_zone_from_lonlat(lon, lat):
    """
    Get the UTM zone for a given longitude and latitude.
    Returns the EPSG code as a string.
    """
    zone_number = int((lon + 180) / 6) + 1
    if lat < 0:
        return f"EPSG:327{zone_number}"
    else:
        return f"EPSG:326{zone_number}"


def shorten_transect(line, distance):
    """
    Shorten a transect by moving the origin closer to the end point
    by the specified distance.
    """
    start_point = line.coords[0]
    end_point = line.coords[-1]

    # Calculate the direction vector
    direction = (end_point[0] - start_point[0], end_point[1] - start_point[1])

    # Normalize the direction vector
    magnitude = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
    normalized_direction = (direction[0] / magnitude, direction[1] / magnitude)

    # Calculate the new start point
    new_start_point = (
        start_point[0] + normalized_direction[0] * distance,
        start_point[1] + normalized_direction[1] * distance,
    )

    # Return the shortened LineString
    return LineString([new_start_point, end_point])


def lengthen_transect(line, distance):
    """
    Lengthen a transect by pushing the end point out by the specified distance.
    """
    # Extract start and end points
    start_point = line.coords[0]
    end_point = line.coords[-1]

    # Calculate the direction vector
    direction = (end_point[0] - start_point[0], end_point[1] - start_point[1])

    # Normalize the direction vector
    magnitude = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
    normalized_direction = (direction[0] / magnitude, direction[1] / magnitude)

    # Calculate the new endpoint
    new_end_point = (
        end_point[0] + normalized_direction[0] * distance,
        end_point[1] + normalized_direction[1] * distance,
    )

    return LineString([start_point, new_end_point])


# STEP 1: Enter in the name of the file to read from
# Read the GeoJSON file
gdf = gpd.read_file(
    r"C:\development\doodleverse\coastseg\CoastSeg\reversed_RM_transects.geojson"
)

# Drop features whose "type" is not "transect"
gdf = gdf[gdf["type"] == "transect"]


original_crs = gdf.crs

# Determine the appropriate UTM zone for the centroid of the data
centroid = gdf.unary_union.centroid
utm_epsg = utm_zone_from_lonlat(centroid.x, centroid.y)

# Convert to the determined UTM CRS
gdf_projected = gdf.to_crs(utm_epsg)
print(gdf_projected)
# Specify the distance by which you want to shorten the transect in meters.
distance_to_shorten = 820  # e.g., 100 meters
distance_to_length = 150  # e.g., 100 meters
# Apply the shortening function to each geometry
gdf_projected["geometry"] = gdf_projected["geometry"].apply(
    lambda geom: shorten_transect(geom, distance_to_shorten)
)

# Apply the shortening function to each geometry
gdf_projected["geometry"] = gdf_projected["geometry"].apply(
    lambda geom: lengthen_transect(geom, distance_to_length)
)

print(gdf_projected)

# Convert the GeoDataFrame back to EPSG:4326 if needed
gdf_shortened = gdf_projected.to_crs(original_crs)

# STEP 2: Enter in the names of the files to save to
# Save the shortened transects GeoDataFrame to a GeoJSON file
filename = "shortened_transects_RM.geojson"
gdf_shortened.to_file(filename, driver="GeoJSON")

print(f"Shortened transects saved to {filename}!")
