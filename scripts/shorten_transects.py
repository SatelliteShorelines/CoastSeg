import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
import argparse


# Example 1: python shorten_transects.py  -i "C:\development\doodleverse\coastseg\CoastSeg\reversed_transects.geojson" -s 500
# -s shorten the length of the transect by 500 meters by moving the origin seaward
# Example 2: python shorten_transects.py -i "C:\development\doodleverse\coastseg\CoastSeg\reversed_transects.geojson" -s  500 -l 100
# -s shorten the length of the transect by 500 meters by moving the origin towards the end point
# -l lengthen the length of the transect by 100 meters by moving the end point more seaward
# Example 3: python shorten_transects.py -i "C:\development\doodleverse\coastseg\CoastSeg\reversed_transects.geojson" -s  500 -l 100 -o "shortened_transects2.geojson"
# -s shorten the length of the transect by 500 meters by moving the origin towards the end point
# -l lengthen the length of the transect by 100 meters by moving the end point more seaward
# -o save the new transects to file "shortened_transects2.geojson"
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


def extract_origin(geom):
    if isinstance(geom, MultiLineString):
        # Access the first LineString's first point
        return geom.geoms[0].coords[0]
    else:  # Assuming it's a LineString
        return geom.coords[0]


def extract_end(geom):
    if isinstance(geom, MultiLineString):
        # Access the first LineString's last point
        return geom.geoms[0].coords[-1]
    else:  # Assuming it's a LineString
        return geom.coords[-1]


def shorten_transect(line, distance):
    """
    Shorten a transect by moving the origin closer to the end point
    by the specified distance.
    """
    start_point = extract_origin(line)
    end_point = extract_end(line)

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
    start_point = extract_origin(line)
    end_point = extract_end(line)

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


def main(
    input_file: str, shorten_distance: float, lengthen_distance: float, output_file: str
):
    # Read the GeoJSON file
    # input_file = r"C:\development\doodleverse\coastseg\CoastSeg\reversed_RM_transects.geojson"
    gdf = gpd.read_file(input_file)

    # Drop features whose "type" is not "transect"
    if "type" in gdf.columns:
        gdf = gdf[gdf["type"] == "transect"]

    original_crs = gdf.crs

    # Determine the appropriate UTM zone for the centroid of the data
    centroid = gdf.unary_union.centroid
    utm_epsg = utm_zone_from_lonlat(centroid.x, centroid.y)

    # Convert to the determined UTM CRS
    gdf_projected = gdf.to_crs(utm_epsg)

    # Apply the shortening function to each geometry
    gdf_projected["geometry"] = gdf_projected["geometry"].apply(
        lambda geom: shorten_transect(geom, shorten_distance)
    )

    # Apply the shortening function to each geometry
    gdf_projected["geometry"] = gdf_projected["geometry"].apply(
        lambda geom: lengthen_transect(geom, lengthen_distance)
    )

    # Convert the GeoDataFrame back to EPSG:4326 if needed
    gdf_shortened = gdf_projected.to_crs(original_crs)

    # Save the shortened transects GeoDataFrame to a GeoJSON file
    gdf_shortened.to_file(output_file, driver="GeoJSON")
    print(f"Shortened transects saved to {output_file}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a geojson file to shorten or lengthen transects."
    )
    parser = argparse.ArgumentParser(
        description="Process a geojson file to shorten or lengthen transects. Shorten will move the origin seaward and lengthen will move the end point seaward  Example: python shorten_transects.py -i input_file.geojson -s 300 -l 100  "
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input geojson file."
    )
    parser.add_argument(
        "-s",
        "--shorten",
        type=float,
        default=0,
        help="Distance by which to shorten the transect in meters.This will shorten the transect by moving the origin towards the end point.",
    )
    parser.add_argument(
        "-l",
        "--lengthen",
        type=float,
        default=0,
        help="Distance by which to lengthen the transect in meters.This will lengthen the transect by moving the end points seaward.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="shortened_transects.geojson",
        help="Filename for the output geojson file.",
    )
    args = parser.parse_args()
    main(args.input, args.shorten, args.lengthen, args.output)
