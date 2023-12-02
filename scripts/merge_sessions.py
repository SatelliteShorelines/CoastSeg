import os
from coastseg import merge_utils, file_utilities
from coastseg.common import (
    convert_linestrings_to_multipoints,
    stringify_datetime_columns,
    get_cross_distance_df,
)
from functools import reduce
import geopandas as gpd
from coastsat import SDS_transects
import numpy as np

settings_transects = {
    "along_dist": 25,  # along-shore distance to use for computing the intersection
    "min_points": 3,  # minimum number of shoreline points to calculate an intersection
    "max_std": 15,  # max std for points around transect
    "max_range": 30,  # max range for points around transect
    "min_chainage": -100,  # largest negative value along transect (landwards of transect origin)
    "multiple_inter": "auto",  # mode for removing outliers ('auto', 'nan', 'max')
    "prc_multiple": 0.1,  # percentage of the time that multiple intersects are present to use the max
}

# Enter ROI session locations here
# session_locations = [
#     r"C:\development\doodleverse\coastseg\CoastSeg\test_data\test_case4_overlapping\ID_gac6_datetime10-30-23__01_44_50",
#     r"C:\development\doodleverse\coastseg\CoastSeg\test_data\test_case4_overlapping\ID_gac1_datetime10-30-23__01_44_50",
# ]

session_locations = [
    r"C:\Users\sf230\Downloads\AK_shoreline1-20231127T221704Z-001\AK_shoreline1\ID_egw1_datetime09-19-23__11_37_20",
    r"C:\Users\sf230\Downloads\AK_shoreline1-20231127T221704Z-001\AK_shoreline1\ID_egw3_datetime09-19-23__11_37_20",
    r"C:\Users\sf230\Downloads\AK_shoreline1-20231127T221704Z-001\AK_shoreline1\ID_egw2_datetime09-19-23__11_37_20",
    r"C:\Users\sf230\Downloads\AK_shoreline1-20231127T221704Z-001\AK_shoreline1\ID_egw4_datetime09-19-23__11_37_20",
]

# enter directory to save the merged session
save_location = r"C:\development\doodleverse\coastseg\CoastSeg\test_results"
# enter the name of the merged session
merged_session_name = "large_dataset"

# Script execution begins here
# ----------------------------

merged_session_location = os.path.join(save_location, merged_session_name)
# make the location to store the merged session
os.makedirs(merged_session_location, exist_ok=True)

# Merge the config_gdf.geojson files from each session into a single geodataframe
#    - if the shorelines or transects are at the exact same location, they will be merged into one
#    -  if transects have different ids for the same location, they will be merged into one and both ids will be saved
merged_config = merge_utils.merge_geojson_files(
    session_locations, merged_session_location
)

# read the extracted shorelines from the session locations
gdfs = merge_utils.process_geojson_files(
    session_locations,
    ["extracted_shorelines_points.geojson", "extracted_shorelines.geojson"],
    merge_utils.convert_lines_to_multipoints,
    merge_utils.read_first_geojson_file,
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

# merge the extracted shorelin geodataframes on date and satname, then average the cloud_cover and geoaccuracy for the merged rows

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
    row["id"]: np.array(row["geometry"].coords) for i, row in transect_rows.iterrows()
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
