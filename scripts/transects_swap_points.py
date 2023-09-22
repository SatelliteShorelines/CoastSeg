import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
import argparse
import os
import warnings

# Ignore UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


# Example : python transects_swap_points.py -i "C:\development\doodleverse\coastseg\CoastSeg\RM_config_gdf.geojson"
# Example :  python get_transects_points.py  -i "C:\development\doodleverse\coastseg\CoastSeg\shortened_transects_JB.geojson" -o "output3.geojson"
# -o provides the name of the output geojson file to save the transect to


def ReversePoints(geom):
    if isinstance(geom, MultiLineString):
        # Access the first LineString's first point
        return geom.geoms[0].coords[::-1]
    else:  # Assuming it's a LineString
        return geom.coords[::-1]


def main(input_file, output_file):
    # STEP 1: Read the file
    # input_file = r"C:\development\doodleverse\coastseg\CoastSeg\RM_config_gdf.geojson"
    if not os.path.exists(input_file):
        raise FileNotFoundError(input_file)
    gdf = gpd.read_file(input_file)

    # Drop features whose "type" is not "transect"
    if "type" in gdf.columns:
        gdf = gdf[gdf["type"] == "transect"]

    # Reverse the coordinates of each LineString to swap origin and end points
    # gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom.coords[::-1])
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: ReversePoints(geom))

    # Create new GeoDataFrame for the reversed transects
    reversed_transects_gdf = gpd.GeoDataFrame(
        gdf, geometry=gdf["geometry"].apply(lambda coords: LineString(coords))
    )

    # STEP 2: Save the reversed transects GeoDataFrame to a GeoJSON file
    reversed_transects_gdf.to_file(output_file, driver="GeoJSON")
    print(f"Reversed transects saved to {output_file}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Provide a geojson file containing transects, then this script will reverse the origin and end point for each transects.  Example: python transects_swap_points.py -i input_file.geojson -o reversed_transects.geojson "
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input geojson file."
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        default="reversed_transects.geojson",
        help="Filename for the output geojson file.",
    )
    args = parser.parse_args()
    main(args.input, args.output)
