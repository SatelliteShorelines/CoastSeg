import os
import argparse
from functools import reduce

import numpy as np
import geopandas as gpd
from coastsat import SDS_transects

from coastseg import merge_utils, file_utilities
from coastseg.common import (
    stringify_datetime_columns,
    get_cross_distance_df,
)


def main(args):
    # Use args to access the command-line arguments
    session_locations = args.session_locations
    save_location = args.save_location
    merged_session_name = args.merged_session_name
    output_epsg = args.crs
    settings_transects = {
        "along_dist": args.along_dist,  # along-shore distance to use for computing the intersection
        "min_points": args.min_points,  # minimum number of shoreline points to calculate an intersection
        "max_std": args.max_std,  # max std for points around transect
        "max_range": args.max_range,  # max range for points around transect
        "min_chainage": args.min_chainage,  # largest negative value along transect (landwards of transect origin)
        "multiple_inter": args.multiple_inter,  # mode for removing outliers ('auto', 'nan', 'max')
        "prc_multiple": args.prc_multiple,  # percentage of the time that multiple intersects are present to use the max
    }
    # @DEBUG only
    # print(f"settings_transects: {settings_transects}")

    merged_session_location = os.path.join(save_location, merged_session_name)
    # make the location to store the merged session
    os.makedirs(merged_session_location, exist_ok=True)

    # Merge the config_gdf.geojson files from each session into a single geodataframe
    #    - if the shorelines or transects are at the exact same location, they will be merged into one
    #    - if transects have different ids for the same location, they will be merged into one and both ids will be saved
    merged_config = merge_utils.merge_geojson_files(
        session_locations, merged_session_location, crs=output_epsg
    )

    # read the extracted shorelines from the session locations
    gdfs = merge_utils.process_geojson_files(
        session_locations,
        ["extracted_shorelines_points.geojson", "extracted_shorelines.geojson"],
        merge_utils.convert_lines_to_multipoints,
        merge_utils.read_first_geojson_file,
        crs=output_epsg,
    )

    # get all the ROIs from all the sessions
    roi_rows = merged_config[merged_config["type"] == "roi"]

    # Determine if any of the extracted shorelines are in the overlapping regions between the ROIs
    overlap_list = merge_utils.get_overlapping_features(roi_rows, gdfs)

    if len(overlap_list) > 0:
        print("No overlapping ROIs found. Sessions can be merged.")
    else:
        print(
            "Overlapping ROIs found. Overlapping regions may have double shorelines if the shorelines were detected on the same dates."
        )

    # merge the extracted shoreline geodataframes on date and satname, then average the cloud_cover and geoaccuracy for the merged rows

    # Perform a full outer join and average the numeric columns across all GeoDataFrames
    merged_shorelines = reduce(merge_utils.merge_and_average, gdfs)
    # sort by date and reset the index
    merged_shorelines.sort_values(by="date", inplace=True)
    merged_shorelines.reset_index(drop=True, inplace=True)

    # Save the merged extracted shorelines to `extracted_shorelines_dict.json`
    # --------------------------------------------------------------------------
    # mapping of dictionary keys to dataframe columns
    keymap = {
        "shorelines": "geometry",
        "dates": "date",
        "satname": "satname",
        "cloud_cover": "cloud_cover",
        "geoaccuracy": "geoaccuracy",
    }

    # shoreline dict should have keys: dates, satname, cloud_cover, geoaccuracy, shorelines
    shoreline_dict = merge_utils.dataframe_to_dict(merged_shorelines, keymap)
    # save the extracted shoreline dictionary to json file
    file_utilities.to_file(
        shoreline_dict,
        os.path.join(merged_session_location, "extracted_shorelines_dict.json"),
    )

    print("Extracted shorelines merged and saved to extracted_shorelines_dict.json")
    print(f"Saved {len(shoreline_dict['shorelines'])} extracted shorelines")

    # Save extracted shorelines to GeoJSON file
    # -----------------------------------------

    # 1. convert datetime columns to strings
    merged_shorelines = stringify_datetime_columns(merged_shorelines)

    # 2. Save the shorelines that are formatted as mulitpoints a to GeoJSON file
    # Save extracted shorelines as mulitpoints GeoJSON file
    merged_shorelines.to_file(
        os.path.join(merged_session_location, "extracted_shorelines_points.geojson"),
        driver="GeoJSON",
    )
    print("Extracted shorelines saved to extracted_shorelines_points.geojson")
    # 3. Convert the multipoints to linestrings and save to GeoJSON file
    es_lines_gdf = merge_utils.convert_multipoints_to_linestrings(merged_shorelines)
    # save extracted shorelines as interpolated linestrings
    es_lines_gdf.to_file(
        os.path.join(merged_session_location, "extracted_shorelines_lines.geojson"),
        driver="GeoJSON",
    )
    print("Extracted shorelines saved to extracted_shorelines_lines.geojson")

    # Compute the timeseries of where transects and new merged shorelines intersect
    # ---------------------------------------------------------------------

    # 1. load transects for from all the sessions
    transect_rows = merged_config[merged_config["type"] == "transect"]
    transects_dict = {
        row["id"]: np.array(row["geometry"].coords)
        for i, row in transect_rows.iterrows()
    }
    # 2. compute the intersection between the transects and the extracted shorelines
    cross_distance = SDS_transects.compute_intersection_QC(
        shoreline_dict, transects_dict, settings_transects
    )

    # use coastseg.common to get the cross_distance_df
    transects_df = get_cross_distance_df(shoreline_dict, cross_distance)
    # 3. save the timeseries of where all the transects and shorelines intersected to a csv file
    filepath = os.path.join(merged_session_location, "transect_time_series.csv")
    transects_df.to_csv(filepath, sep=",")

    # 4. Save a CSV file for each transect
    #   - Save the timeseries of intersections between the shoreline and a single tranesct to csv file
    merge_utils.create_csv_per_transect(
        merged_session_location,
        cross_distance,
        shoreline_dict,
    )


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Merge sessions script.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Add mandatory arguments
    parser.add_argument(
        "-i",
        "--session_locations",
        nargs="+",
        required=True,
        help='Locations of the ROI sessions to be merged. \n- USE DOUBLE QUOTES or it will not work \n Example: -i "C:\CoastSeg\sessions\session_2022\ID_ewr1_datetime12-20-23__03_25_23"  "C:\CoastSeg\sessions\session_2022\ID_ewr3_datetime12-20-23__03_25_23"  \n  ',
    )

    parser.add_argument(
        "-c",
        "--crs",
        required=True,
        help='Coordinate reference system (CRS) for the merged session. \n You can find the EPSG code for your CRS in your config.json at "output_epsg": or set a new one.\n The CRS must be in a cartesian coordinate system (i.e. projected) and not a geographic coordinate system (i.e. lat/lon). \n Example: "EPSG:32610" is the EPSG code for UTM Zone 10N.\n  ',
    )

    parser.add_argument(
        "-n",
        "--merged_session_name",
        required=True,
        help='Name for the merged session folder that will be created at save_location. \n Example: "merged_session_2022" \n   ',
    )

    # Add optional argument with default value
    parser.add_argument(
        "-s",
        "--save_location",
        default=os.path.join(os.getcwd(), "merged_sessions"),
        help='Location to save the merged session (default: coastseg\scripts\merged_sessions) \n Example: -s "C:\CoastSeg\merged_sessions" \n  ',
    )

    # Settings for transects
    parser.add_argument(
        "-ad",
        "--along_dist",
        type=int,
        default=25,
        help="\n Along-shore distance (in meters) for computing the intersection (default: 25m)\n Example: --ad 30 \n   ",
    )
    parser.add_argument(
        "-mp",
        "--min_points",
        type=int,
        default=3,
        help="Minimum number of shoreline points to calculate an intersection (default: 3)\n Example: --mp 5 \n   ",
    )
    parser.add_argument(
        "-ms",
        "--max_std",
        type=int,
        default=15,
        help="Maximum standard deviation for points around transect (default: 15)\n Example: --ms 20\n   ",
    )
    parser.add_argument(
        "-mr",
        "--max_range",
        type=int,
        default=30,
        help="Maximum range for points around transect (default: 30)\n Example: --mr 40\n   ",
    )
    parser.add_argument(
        "-mc",
        "--min_chainage",
        type=int,
        default=-100,
        help="Largest negative value along transect (landwards of transect origin) (default: -100)\n Example: --mc -200 \n   ",
    )
    parser.add_argument(
        "-mi",
        "--multiple_inter",
        default="auto",
        choices=["auto", "nan", "max"],
        help='Mode for removing outliers ("auto", "nan", "max") (default: "auto")\n Example: --mi "nan"\n   ',
    )
    parser.add_argument(
        "-pm",
        "--prc_multiple",
        type=float,
        default=0.1,
        help="Percentage of the time that multiple intersects are present to use the max shoreline point for intersection value along transect (default: 0.1)\n Example: --pm 0.20 \n   ",
    )

    # Parse the arguments
    return parser.parse_args()


if __name__ == "__main__":
    # Parse the arguments
    args = parse_arguments()
    main(args)
