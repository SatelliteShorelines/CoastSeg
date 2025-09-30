from coastseg import coastseg_logs
from coastseg.common import initialize_gee
from coastseg import coastseg_map
from coastseg import file_utilities
import geopandas as gpd
import os


# This script is intended to be run to validate the CoastSeg workflow for downloading imagery and extracting shorelines is
#  functioning as expected. The script will download imagery for a sample ROI, extract shorelines, and validate the download
# and extraction processes.
# The script validates the contents of the downloaded data and the extracted shorelines exist and are the expected outputs.


# FUNCTIONS TO VALIDATE DOWNLOAD
# _______________________________________________________________________________________


def directory_exists(directory_path):
    """Check if a directory exists."""
    return os.path.exists(directory_path) and os.path.isdir(directory_path)


def validate_directory_contents(directory_path, folder_name):
    """Validate the contents of a given directory."""
    required_folders = ["jpg_files", "L9"]
    required_files = [
        "config.json",
        "config_gdf.geojson",
        "download_report.txt",
        f"{folder_name}_metadata.json",
    ]
    # Check for required folders
    missing_folders = [
        folder
        for folder in required_folders
        if not os.path.isdir(os.path.join(directory_path, folder))
    ]
    if missing_folders:
        print(f"Missing folders: {', '.join(missing_folders)}")

    # Check for required files
    missing_files = [
        file
        for file in required_files
        if not os.path.isfile(os.path.join(directory_path, file))
    ]
    if missing_files:
        print(f"Missing files: {', '.join(missing_files)}")

    return len(missing_folders) == 0 and len(missing_files) == 0


def validate_jpg_files_subdirectory(directory_path):
    """Validate the 'jpg_files' directory for 'preprocessed' folder."""
    jpg_files_path = os.path.join(directory_path, "jpg_files")
    preprocessed_path = os.path.join(jpg_files_path, "preprocessed")
    if not directory_exists(preprocessed_path):
        print(f"Missing directory: {preprocessed_path}")
    return directory_exists(preprocessed_path)


def validate_satname_subdirectory(directory_path, satname="L9"):
    """Validate the 'L9' directory for required sub-folders."""
    satname_path = os.path.join(directory_path, satname)
    sat_folders = {
        "L5": ["mask", "ms", "meta"],
        "L7": ["mask", "ms", "pan", "meta"],
        "L9": ["mask", "ms", "pan", "meta"],
        "L8": ["mask", "ms", "pan", "meta"],
        "S2": ["swir", "ms", "meta"],
    }

    required_subfolders = sat_folders[satname]

    missing_subfolders = [
        folder
        for folder in required_subfolders
        if not directory_exists(os.path.join(satname_path, folder))
    ]
    if missing_subfolders:
        print(f"Missing subdirectories in L9: {', '.join(missing_subfolders)}")

    return len(missing_subfolders) == 0


def validate_download(data_directory, folder_name, satnames=set(["L9"])):
    if not directory_exists(data_directory):
        print("Data directory does not exist.")
        return

    ROI_path = os.path.join(data_directory, folder_name)
    if not validate_directory_contents(ROI_path, folder_name):
        print(f"Failed to validate directory contents for {folder_name}.")
        return

    if not validate_jpg_files_subdirectory(ROI_path):
        print(f"'jpg_files' sub-directory validation failed for {folder_name}.")
        return

    for satname in satnames:
        if not validate_satname_subdirectory(ROI_path, satname=satname):
            print(
                f"Folders under {satname} sub-directory validation failed for {folder_name}."
            )
            return

    print("All downloaded data validations passed.")


# ---------------------------------------------------------------------------------------


# FUNCTIONS TO VALIDATE EXTRACT SHORELINES
# _______________________________________________________________________________________
def validate_config(actual_config: dict, roi_settings: dict, settings: dict) -> None:
    """
    Validate the configuration against the ROI settings and general settings.

    Args:
        actual_config (dict): The actual configuration to validate.
        roi_settings (dict): The ROI settings to compare against.
            Contains each ROI IDs as the keys and a dictionary of each ROI's settings as the values.
        settings (dict): The general settings to compare against.

    Raises:
        AssertionError: If any of the ROI IDs in the actual configuration are not found in the ROI settings.
        AssertionError: If any of the key-value pairs in the actual configuration do not match the corresponding
                        key-value pairs in the ROI settings.
        AssertionError: If any of the key-value pairs in the 'settings' section of the actual configuration do not
                        match the corresponding key-value pairs in the general settings.

    """
    for roi_id in actual_config["roi_ids"]:
        assert (
            roi_id in roi_settings
        ), f"ROI ID not found in roi settings {roi_settings.keys()}"
        for key in actual_config[roi_id]:
            assert roi_settings[roi_id][key] == actual_config[roi_id][key]
    for roi_id, item in actual_config.get("settings", {}).items():
        assert settings[roi_id] == item


def session_exists(session_name, folder):
    """Check if the session exists within the given folder."""
    session_path = os.path.join(folder, session_name)
    return os.path.exists(session_path) and os.path.isdir(session_path)


def roi_exists(session_name, folder, roi_name):
    """Check if the ROI exists within the session folder."""
    roi_path = os.path.join(folder, session_name, roi_name)
    return os.path.exists(roi_path) and os.path.isdir(roi_path)


def validate_roi_contents(
    session_name, folder, roi_name, required_files, required_folders
):
    """Validate the ROI folder contains the required files and folders."""
    roi_path = os.path.join(folder, session_name, roi_name)

    missing_files = [
        file
        for file in required_files
        if not os.path.isfile(os.path.join(roi_path, file))
    ]
    if missing_files:
        print(f"Missing files: {', '.join(missing_files)}")

    missing_folders = [
        folder
        for folder in required_folders
        if not os.path.isdir(os.path.join(roi_path, folder))
    ]
    if missing_folders:
        print(f"Missing folders: {', '.join(missing_folders)}")

    return len(missing_files) == 0 and len(missing_folders) == 0


def validate_jpg_files_subfolders(session_name, folder, roi_name, required_subfolders):
    """Validate the jpg_files folder contains the required subfolders and that those contain JPEGs."""
    jpg_files_path = os.path.join(folder, session_name, roi_name, "jpg_files")
    subfolders_valid = all(
        os.path.isdir(os.path.join(jpg_files_path, subfolder))
        for subfolder in required_subfolders
    )

    # Check if each subfolder contains at least one JPEG file
    jpgs_exist = all(
        any(
            file.endswith(".jpg")
            for file in os.listdir(os.path.join(jpg_files_path, subfolder))
        )
        for subfolder in required_subfolders
    )

    return subfolders_valid and jpgs_exist


def validate_feature_type(gdf1, gdf2, feature_name):
    joined_gdf = gdf1.sjoin(gdf2[gdf2["type"] == feature_name])

    # Check if the joined_gdf is empty or not to determine if any geometry from gdf1 is within gdf2
    if not joined_gdf.empty:
        # print(f"There are geometries from {feature_name} within config_gdf")
        return True
    else:
        print(f"No geometries from gdf1 are within config_gdf")
        return False


def validate_config_gdf(config_path, shoreline_path, roi_path, transect_path):
    gdf2 = gpd.read_file(config_path)
    paths = [
        (shoreline_path, "shoreline"),
        (roi_path, "roi"),
        (transect_path, "transect"),
    ]
    invalid_features = []
    for path, feature_name in paths:
        gdf1 = gpd.read_file(path)
        valid_feature = (
            validate_feature_type(gdf1, gdf2, feature_name),
            f"{feature_name} not found in {config_path}",
        )
        assert valid_feature, f"{feature_name} not found in {config_path}"
        if not valid_feature:
            invalid_features.append(valid_feature)
    if invalid_features:
        print(f"Invalid features found: {invalid_features}")
        return False
    return True


# Example usage
def validate_extract_shorelines_session(
    session_directory, session_name, roi_name, rois_path, shoreline_path, transect_path
):

    required_files = [
        "config.json",
        "config_gdf.geojson",
        "extracted_shorelines_dict.json",
        "extracted_shorelines_lines.geojson",
        "extracted_shorelines_points.geojson",
        "extract_shorelines_report.txt",
        "raw_transect_time_series.csv",
        "raw_transect_time_series_merged.csv",
        "raw_transect_time_series_points.geojson",
        "raw_transect_time_series_vectors.geojson",
        "shoreline_settings.json",
        "transects_cross_distances.json",
        "transects_settings.json",
    ]

    required_folders = ["jpg_files"]
    required_subfolders = ["detection"]

    if not session_exists(session_name, session_directory):
        print("Session does not exist.")
        return

    if not roi_exists(session_name, session_directory, roi_name):
        print("ROI does not exist within the session.")
        return

    if not validate_roi_contents(
        session_name, session_directory, roi_name, required_files, required_folders
    ):
        print("ROI folder does not contain the required files and folders.")
        return

    # Check if extracted_shorelines_lines.geojson exist and if it does make sure it contains only LineStrings and MultiLineStrings
    extracted_shorelines_lines_path = os.path.join(
        session_directory, session_name, roi_name, "extracted_shorelines_lines.geojson"
    )
    if not os.path.isfile(extracted_shorelines_lines_path):
        raise FileNotFoundError("extracted_shorelines_lines.geojson not found.")

    extracted_shorelines_lines = gpd.read_file(extracted_shorelines_lines_path)
    if not extracted_shorelines_lines.empty:
        if not all(
            geom.type in ["LineString", "MultiLineString"]
            for geom in extracted_shorelines_lines.geometry
        ):
            print("extracted_shorelines_lines.geojson contains invalid geometries.")
            raise ValueError(
                "Invalid geometries found in extracted_shorelines_lines.geojson."
            )

    # Now test if extracted_shorelines_points.geojson exist and if it does make sure it contains only Points
    extracted_shorelines_points_path = os.path.join(
        session_directory, session_name, roi_name, "extracted_shorelines_points.geojson"
    )
    if not os.path.isfile(extracted_shorelines_points_path):
        raise FileNotFoundError("extracted_shorelines_points.geojson not found.")
    extracted_shorelines_points = gpd.read_file(extracted_shorelines_points_path)
    if not extracted_shorelines_points.empty:
        if not all(
            geom.type in ["Point", "MultiPoint"]
            for geom in extracted_shorelines_points.geometry
        ):
            print("extracted_shorelines_points.geojson contains invalid geometries.")
            raise ValueError(
                "Invalid geometries found in extracted_shorelines_points.geojson."
            )

    if not validate_jpg_files_subfolders(
        session_name, session_directory, roi_name, required_subfolders
    ):
        print(
            "jpg_files folder does not contain the required subfolders or lacks JPEGs."
        )
        return

    # validate the contents of the config.json file
    config_path = os.path.join(
        session_directory, session_name, roi_settings["sitename"], "config.json"
    )
    actual_config = file_utilities.read_json_file(config_path)
    settings = coastsegmap.get_settings()
    validate_config(actual_config, coastsegmap.rois.get_roi_settings(), settings)

    # validate the contents of the config_gdf.geojson file contain the ROI, ref shoreline and transects
    config_path = os.path.join(
        session_directory, session_name, roi_settings["sitename"], "config_gdf.geojson"
    )
    validate_config_gdf(config_path, shoreline_path, rois_path, transect_path)

    print(
        "Validation of extracted shorelines session passed. All files and folders are present."
    )


# _______________________________________________________________________________________

# Main script: Downloads imagery and extract shorelines


# if you get an error here, enter your project id
initialize_gee(auth_mode="localhost", project="")

coastsegmap = coastseg_map.CoastSeg_Map(create_map=False)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"The script directory is {script_dir}")
# Construct the path to rois.geojson
rois_path = os.path.join(script_dir, "examples", "rois.geojson")
print(f"Loading ROIs from {rois_path}")

# path to Coastseg/data
data_folder = os.path.join(script_dir, "data")
print(f"The data folder is {data_folder}")

# sample ROI (Region of Interest) file
roi = coastsegmap.load_feature_from_file("roi", rois_path)
print(roi)
# get the select all the ROI IDs from the file and store them in a list
roi_ids = list(roi.gdf.id)
print(f"Downloading imagery for ROI with ID {roi_ids}")
# customize the settings for the imagery download
settings = {
    "sat_list": [
        "L9",
        "L5",
    ],  # list of satellites to download imagery from. Options: 'L5', 'L7', 'L8', 'L9','S2'
    "dates": ["2023-12-01", "2024-02-01"],  # Start and end date to download imagery
    "landsat_collection": "C02",  # GEE collection to use. CoastSeg uses the Landsat Collection 2 (C02) by default
    "apply_cloud_mask": True,  # apply cloud mask to the imagery. If False, the cloud mask will not be applied.
}

# download the imagery for that ROI to the /data folder
coastsegmap.download_imagery(
    rois=roi.gdf, selected_ids=roi_ids, settings=settings, file_path=data_folder
)

# name the session where you want to save the extracted shorelines
session_name = "sample_session_workflow"
coastsegmap.set_session_name(session_name)

# Modify the settings for the shoreline extraction here
# These settings will only extract shorelines with:
# - a minimum beach area of 500 m^2
# - a minimum length of 20 m
# - a maximum distance from the reference shoreline of 300 m
coastsegmap.set_settings(min_beach_area=500, min_length_sl=20, max_dist_ref=300)

# load a shoreline file from the examples folder
script_dir = os.path.dirname(os.path.abspath(__file__))
shoreline_path = os.path.join(script_dir, "examples", "shoreline.geojson")
shoreline = coastsegmap.load_feature_from_file("shoreline", shoreline_path)

# load transects from the examples folder
transect_path = os.path.join(script_dir, "examples", "transects.geojson")
transects = coastsegmap.load_feature_from_file("transects", transect_path)

# extract the shorelines for the selected ROI and save them to the /sessions/session_name folder
coastsegmap.extract_all_shorelines(roi_ids=roi_ids)

# Validate the download and extraction processes
#  ---------------------------------------------

# 1.  Validate the download process worked
roi_settings = coastsegmap.rois.get_roi_settings(roi_ids[0])
ROI_path = os.path.join(roi_settings["filepath"], roi_settings["sitename"])
validate_download(roi_settings["filepath"], os.path.basename(ROI_path))

# 2. Validate the shoreline extraction process worked
session_directory = os.path.join(os.path.dirname(roi_settings["filepath"]), "sessions")
validate_extract_shorelines_session(
    session_directory,
    session_name,
    os.path.basename(ROI_path),
    rois_path,
    shoreline_path,
    transect_path,
)
