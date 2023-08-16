import geopandas as gpd
from shapely.geometry import Point
import argparse


# Example 1: python get_transects_points.py  -i "C:\development\doodleverse\coastseg\CoastSeg\shortened_transects_JB.geojson"
# Example 2: python get_transects_points.py -i "C:\development\doodleverse\coastseg\CoastSeg\shortened_transects_JB.geojson" -o "origin.geojson" -e "end.geojson"
def main(input_file, origin_filename, end_filename):
    # Read the GeoJSON file
    # input_file = r"C:\development\doodleverse\coastseg\CoastSeg\shortened_transects_JB.geojson"
    gdf = gpd.read_file(input_file)

    # Drop features whose "type" is not "transect"
    gdf = gdf[gdf["type"] == "transect"]

    # Create GeoDataFrames for the origin and end points
    origin_gdf = gpd.GeoDataFrame(
        gdf[["type"]],
        geometry=gdf["geometry"].apply(lambda geom: Point(geom.coords[0])),
    )
    end_gdf = gpd.GeoDataFrame(
        gdf[["type"]],
        geometry=gdf["geometry"].apply(lambda geom: Point(geom.coords[-1])),
    )

    # Save the origin and end GeoDataFrames to separate GeoJSON files
    origin_gdf.to_file(origin_filename, driver="GeoJSON")
    end_gdf.to_file(end_filename, driver="GeoJSON")

    print(f"Saved origin points for all the transects to {origin_filename}")
    print(f"Saved end points for all the transects to {end_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract origin and end points from geojson transects. Example: python get_transects_points.py -i input_file.geojson -o origin.geojson -e end.geojson"
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input geojson file."
    )
    parser.add_argument(
        "-o",
        "--origin",
        default="origin_points.geojson",
        help="Filename for the origin points geojson file.",
    )
    parser.add_argument(
        "-e",
        "--end",
        default="end_points.geojson",
        help="Filename for the end points geojson file.",
    )
    args = parser.parse_args()
    main(args.input, args.origin, args.end)
