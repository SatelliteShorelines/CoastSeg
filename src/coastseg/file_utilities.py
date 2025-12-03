import datetime
import json
import logging
import os
import pathlib
import re
import shutil
from contextlib import contextmanager

# Specific classes/functions from modules
from json import JSONEncoder
from typing import Collection, List, Optional, Union

# Third-party imports
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from coastseg import common, core_utilities

# Logger setup
logger = logging.getLogger(__name__)


@contextmanager
def progress_bar_context(
    use_progress_bar: bool, total: int = 6, description: str = "", **kwargs
):
    """
    Context manager for handling progress bar creation and updates.

    Args:
        use_progress_bar (bool): If True, a tqdm progress bar will be displayed. Otherwise, it's a no-op.
        total (int): Total number of expected iterations for the progress bar. Defaults to 6.
        description (str): Description text to display with the progress bar. Defaults to "".
        **kwargs: Additional keyword arguments passed to tqdm.

    Returns:
        Callable[[str, float], None]: A function to call for updating the progress bar.
            Takes a message string and optional update value. If use_progress_bar is False,
            it's a no-op function.
    """
    if use_progress_bar:
        progress_bar = tqdm(total=total, dynamic_ncols=True, desc=description, **kwargs)

        def update(message: str, update_value: float = 1):
            progress_bar.set_description(message)
            progress_bar.update(update_value)

        yield update
        progress_bar.close()
    else:
        # If not using a progress bar, just yield a no-op function
        yield lambda message: None


def join_model_scores_to_time_series(
    time_series_csv_path: str,
    good_bad_csv: Optional[str] = None,
    good_bad_seg_csv: Optional[str] = None,
) -> str:
    """
    Joins model scores to a time series CSV file.

    Reads a time series CSV, merges classifier and/or segmentation model scores,
    and saves the updated file back to the same location.

    Args:
        time_series_csv_path (str): Path to the time series CSV file to update.
        good_bad_csv (Optional[str]): Path to image classifier scores CSV. Defaults to None.
        good_bad_seg_csv (Optional[str]): Path to segmentation model scores CSV. Defaults to None.

    Returns:
        str: Path to the updated time series CSV file.
    """
    df = pd.read_csv(time_series_csv_path)
    df["dates"] = pd.to_datetime(df["dates"], utc=True)

    if good_bad_csv:
        df = merge_model_scores(
            df, good_bad_csv, model_type="classifier", date_col="dates"
        )
        df.drop(
            columns=[col for col in df.columns if "Unnamed:" in col],
            errors="ignore",
            inplace=True,
        )

    if good_bad_seg_csv:
        df = merge_model_scores(
            df, good_bad_seg_csv, model_type="segmentation", date_col="dates"
        )

    df.to_csv(time_series_csv_path, index=False)
    print(f"Saved updated transect time series to {time_series_csv_path}")
    return time_series_csv_path


def merge_model_scores(
    df: Union[pd.DataFrame, gpd.GeoDataFrame],
    score_csv: str,
    model_type: str,
    date_col: str = "dates",
    merge_on_col: str = "dates",
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Process a model score CSV and merge into a provided DataFrame/GeoDataFrame based on date.

    Merges model score data from a CSV into a DataFrame/GeoDataFrame based on date.
    Drops the merge_on_col if it is not the same as date_col after merging. Example:
    If date_col is "date" and merge_on_col is "dates", then "dates" will be dropped.

    Args:
        df (Union[pd.DataFrame, gpd.GeoDataFrame]): The target DataFrame (Pandas or GeoPandas).
        score_csv (str): Path to model score CSV.
        model_type (str): 'classifier' or 'segmentation'.
        date_col (str): Name of the datetime column in the target df. Defaults to "dates".
        merge_on_col (str): Name to use for joining on date from the CSV. Defaults to "dates".

    Returns:
        Union[pd.DataFrame, gpd.GeoDataFrame]: Merged DataFrame with model scores added.
                                              Returns same type as input df.
    """
    score_data = pd.read_csv(score_csv)

    if "im_paths" not in score_data or "model_scores" not in score_data:
        raise ValueError(f"Missing required columns in {score_csv}")

    # Extract datetime from image path
    score_data[merge_on_col] = score_data["im_paths"].apply(
        lambda x: pd.to_datetime(
            os.path.basename(x).split("_")[0], utc=True, format="%Y-%m-%d-%H-%M-%S"
        )
    )

    score_col = f"{model_type}_model_score"
    threshold_col = f"{model_type}_threshold"

    df.drop(columns=[score_col, threshold_col], errors="ignore", inplace=True)

    merge_cols = [merge_on_col, "model_scores"]
    if "threshold" in score_data.columns:
        merge_cols.append("threshold")

    score_subset = score_data[merge_cols]

    df = df.merge(
        score_subset,
        left_on=date_col,
        right_on=merge_on_col,
        how="left",
        suffixes=("", f"_{model_type}"),
    )

    df.rename(columns={"model_scores": score_col}, inplace=True)
    if "threshold" in df.columns:
        df.rename(columns={"threshold": threshold_col}, inplace=True)

    # Only drop the merge_on_col if it is NOT the same as date_col
    if merge_on_col != date_col:
        df.drop(columns=[merge_on_col], errors="ignore", inplace=True)
    return df


def join_model_scores_to_geodataframe(
    geodataframe_path: str,
    good_bad_csv: Optional[str] = None,
    good_bad_seg_csv: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Joins model scores to features in the geodataframe from optional CSVs.
    All features are retained; missing scores are marked NaN.

    Parameters:
        geodataframe_path (str): Path to the GeoJSON file containing the features.
        good_bad_csv (str, optional): Path to image classifier scores.
        good_bad_seg_csv (str, optional): Path to segmentation scores.
        output_path (str, optional): Output path to save the updated GeoJSON file.
            By default, it overwrites the input file, geodataframe_path

    Returns:
        str: Path to updated GeoJSON with model scores.
    """
    if not os.path.exists(geodataframe_path):
        raise FileNotFoundError(f"Shoreline file not found: {geodataframe_path}")

    geodataframe = gpd.read_file(geodataframe_path)
    geodataframe["date"] = pd.to_datetime(geodataframe["date"], utc=True)

    if good_bad_csv:
        result = merge_model_scores(
            geodataframe, good_bad_csv, model_type="classifier", date_col="date"
        )
        geodataframe = result if isinstance(result, gpd.GeoDataFrame) else geodataframe
    if good_bad_seg_csv:
        result = merge_model_scores(
            geodataframe, good_bad_seg_csv, model_type="segmentation", date_col="date"
        )
        geodataframe = result if isinstance(result, gpd.GeoDataFrame) else geodataframe

    geodataframe["date"] = geodataframe["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    geodataframe = common.stringify_datetime_columns(geodataframe)

    output_path = output_path or geodataframe_path
    geodataframe.to_file(output_path, driver="GeoJSON")  # type: ignore
    return output_path


def find_file_path_in_roi(
    roi_id: str, roi_settings: dict, filename: str = "image_classification_results.csv"
) -> str:
    """
    Finds the file path of a specified file within a region of interest (ROI) directory.

    Args:
        roi_id (str): The identifier for the region of interest.
        roi_settings (dict): A dictionary containing settings for each ROI, including file paths and site names.
        filename (str): The name of the file to find. Defaults to "image_classification_results.csv".

    Returns:
        str: The path to the file if found, otherwise an empty string.
    """
    # Read the ROI Settings to get the location of the original downloaded data
    roi_data_location = os.path.join(
        roi_settings[roi_id]["filepath"], roi_settings[roi_id]["sitename"]
    )
    logger.info(f"roi_data_location: {roi_data_location}")

    # Construct the path to the expected file location
    expected_csv_path = os.path.join(
        roi_data_location, "jpg_files", "preprocessed", "RGB", filename
    )
    if os.path.exists(expected_csv_path):
        return expected_csv_path
    else:
        # Try to find the file recursively in the ROI data location
        try:
            found_file_path = find_file_recursively(roi_data_location, filename)
            return found_file_path
        except FileNotFoundError:
            print(f"File not found: {filename}")
            return ""  # Return an empty string if the file is not found


def get_ROI_ID_from_session(session_name: str) -> str:
    """
    Retrieves the ROI ID from the config.json file in the extracted shoreline session directory.

    Args:
        session_name (str): The name of the session.

    Returns:
        str: The ROI ID.

    Raises:
        Exception: If the session directory does not exist.
    """
    # need to read the ROI ID from the config.json file found in the extracted shoreline session directory
    session_directory = os.path.join(
        core_utilities.get_base_dir(), "sessions", session_name
    )
    if not os.path.exists(session_directory):
        raise Exception(f"The session directory {session_directory} does not exist")
    config_json_location = find_file_recursively(session_directory, "config.json")
    config = load_data_from_json(config_json_location)
    roi_id = config.get("roi_id", "")
    return roi_id


def load_package_resource(
    resource_name: str,
    file_name: str = "",
    pkg_name: str = "coastseg",
) -> str:
    """
    Loads a resource from a package.

    Args:
        resource_name (str): Name of the resource to load.
        file_name (str): Specific file name within the resource, if applicable.
        pkg_name (str): Name of the package where the resource is located.

    Returns:
        str: Absolute path to the resource or file.

    Raises:
        ImportError: If necessary importlib resources are not available.
        FileNotFoundError: If the resource or file is not found.
    """
    try:
        from importlib import resources
    except (ImportError, AttributeError):
        try:
            # Use importlib backport for Python older than 3.9
            import importlib_resources as resources
        except ImportError:
            raise ImportError(
                "Both importlib.resources and importlib_resources are unavailable. Ensure you have the necessary packages installed."
            )
    # Get the resource location
    resource_location = resources.files(pkg_name).joinpath(resource_name)

    # If a file name is provided, update the resource location
    if file_name:
        resource_location = resource_location.joinpath(file_name)

    return str(resource_location)


def generate_datestring() -> str:
    """
    Generate a datetime string in the format %m-%d-%y__%I_%M_%S

    Returns:
        str: A datetime string in the format %m-%d-%y__%I_%M_%S
             (e.g., "01-31-22__12_19_45").
    """
    date = datetime.datetime.now()
    return date.strftime("%m-%d-%y__%I_%M_%S")


def read_json_file(
    json_file_path: str, raise_error: bool = False, encoding: str = "utf-8"
) -> dict:
    """
    Reads a JSON file and returns the parsed data as a dictionary.

    Args:
        json_file_path (str): The path to the JSON file.
        raise_error (bool): Set to True if an error should be raised if the file doesn't exist. Defaults to False.
        encoding (str): The encoding of the file. Defaults to "utf-8".

    Returns:
        dict: The parsed JSON data as a dictionary. Returns empty dict if file doesn't exist and raise_error is False.

    Raises:
        FileNotFoundError: If the file does not exist and `raise_error` is True.
    """
    if not os.path.exists(json_file_path):
        if raise_error:
            raise FileNotFoundError(
                f"Model settings file does not exist at {json_file_path}"
            )
        else:
            return {}
    with open(json_file_path, "r", encoding=encoding) as f:
        data = json.load(f)
    return data


def get_session_contents_location(session_name: str, roi_id: str = "") -> str:
    """
    Get the location of the session folder, optionally filtered by ROI ID.

    Args:
        session_name (str): The name of the session.
        roi_id (str): Optional ROI ID to find a specific ROI directory within the session. Defaults to "".

    Returns:
        str: Path to the session directory or ROI-specific directory within the session.

    Raises:
        Exception: If the session directory doesn't exist or doesn't contain a config.json file.
    """
    session_path = get_session_location(session_name)
    if roi_id:
        roi_location = find_matching_directory_by_id(session_path, roi_id)
        if roi_location is not None:
            session_path = roi_location
        # check if a config file exists in the session if it doesn't then this isn't correct
        try:
            find_file_by_regex(session_path, r"^config\.json$")
        except FileNotFoundError:
            raise Exception(
                f"Session Directory didn't contains config json file {session_path}"
            )
    if not os.path.isdir(session_path):
        raise Exception(f"Session Directory didn't exist {session_path}")
    return session_path


def find_file_by_regex(
    search_path: str, search_pattern: str = r"^config\.json$"
) -> str:
    """Searches for a file with matching regex in the specified directory

    Args:
        search_path (str): the directory path to search for the  file matching the search pattern
        search_pattern (str): the regular expression pattern to search for the config file

    Returns:
        str: the file path that matched the search_pattern

    Raises:
        FileNotFoundError: if a file is not found in the specified directory
    """
    logger.info(f"searching directory for config : {search_path}")
    config_regex = re.compile(search_pattern, re.IGNORECASE)
    for file in os.listdir(search_path):
        if config_regex.match(file):
            logger.info(f"{file} matched regex")
            file_path = os.path.join(search_path, file)
            return file_path

    raise FileNotFoundError(
        f"file matching pattern {search_pattern} was not found at {search_path}"
    )


def validate_config_files_exist(src: str) -> bool:
    """
    Check if config files exist in the source directory.

    Validates the presence of both a geojson file starting with "config_gdf"
    and a file named "config.json" in the source directory.

    Args:
        src (str): The source directory to check.

    Returns:
        bool: True if both config files exist, False otherwise.
    """
    files = os.listdir(src)
    config_gdf_exists = False
    config_json_exists = False
    for file in files:
        if file.startswith("config_gdf") and file.endswith(".geojson"):
            config_gdf_exists = True
        elif file == "config.json":
            config_json_exists = True
        if config_gdf_exists and config_json_exists:
            return True
    return False


def move_files(
    src: Union[str, Collection[str]], dst_dir: str, delete_src: bool = False
) -> None:
    """
    Moves every file from either a source directory or a list of source files
    to a destination directory and optionally deletes the source when finished.

    Args:
    - src (str or list): The source directory path or a list of source file paths.
    - dst_dir (str): The destination directory path.
    - delete_src (bool): Whether to delete the source directory or files after moving. Defaults to False.

    Returns:
    - None
    """

    # Validate src: it must be a directory path or a list of file paths.
    if isinstance(src, str):
        if os.path.isdir(src):
            src_files = [os.path.join(src, filename) for filename in os.listdir(src)]
            logger.info(
                f"Moving all files from directory {src} to {dst_dir}. Delete Source: {delete_src}"
            )
        elif os.path.isfile(src):
            src_files = [src]
            logger.info(f"Moving file {src} to {dst_dir}. Delete Source: {delete_src}")
        else:
            logger.error(
                f"Provided src is a string but not a valid directory path: {src}"
            )
            return
    elif isinstance(src, list):
        src_files = src
        logger.info(f"Moving listed files to {dst_dir}. Delete Source: {delete_src}")
    else:
        logger.error(
            "The src parameter must be a directory path (str) or a list of file paths (list)."
        )
        return

    # Ensure the destination directory exists.
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Check whether the destination directory is the same as any source directory to avoid data loss.
    for src_file in src_files:
        src_dir = os.path.dirname(src_file)
        if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
            logger.error(
                f"Cannot move files; the source directory {src_dir} is the same as the destination directory {dst_dir}."
            )
            return

    # Move the files from src to dst_dir.
    for src_file in src_files:
        if os.path.isfile(src_file):
            dst_file = os.path.join(dst_dir, os.path.basename(src_file))
            shutil.move(src_file, dst_file)
        else:
            logger.warning(f"{src_file} is not a valid file path. Skipping...")

    # Optionally, delete the source directory or source directories of moved files.
    if delete_src:
        if isinstance(src, str) and os.path.exists(src):
            shutil.rmtree(src)
            logger.info(f"Deleted source directory {src}")
        elif isinstance(src, list):
            for src_file in src_files:
                src_dir = os.path.dirname(src_file)
                if os.path.exists(src_dir) and not os.listdir(src_dir):
                    os.rmdir(src_dir)
                    logger.info(f"Deleted source directory {src_dir}")


def find_parent_directory(
    path: str, directory_name: str, stop_directory: str = ""
) -> Optional[str]:
    """
    Find the path to the parent directory that contains the specified directory name.

    Args:
        path (str): The path to start the search from.
        directory_name (str): Part of the name of the directory to find.
            For example, directory_name = 'ID' will return the first directory
            that contains 'ID' in its name.
        stop_directory (str): Optional directory name to stop the search at.
            If this is specified, the search will stop when this
            directory is reached. If not specified, the search will
            continue until the top-level directory is reached. Defaults to "".

    Returns:
        Optional[str]: The path to the parent directory containing the directory with
                      the specified name, or None if the directory is not found.
    """
    while True:
        # check if the current directory name contains the target directory name
        if directory_name in os.path.basename(path):
            return path

        # get the parent directory
        parent_dir = os.path.dirname(path)

        # check if the parent directory is the same as the current directory
        if parent_dir == path or os.path.basename(path) == stop_directory:
            print(
                f"Reached top-level directory without finding '{directory_name}':", path
            )
            return None

        # update the path to the parent directory and continue the loop
        path = parent_dir


def config_to_file(config: Union[dict, gpd.GeoDataFrame], filepath: str) -> None:
    """
    Save config to config.json or config_gdf.geojson.

    The config type determines the file type: dict saves to config.json,
    geodataframe saves to config_gdf.geojson.

    Args:
        config (Union[dict, gpd.GeoDataFrame]): Data to save to config file.
        filepath (str): Full path to directory to save config file.
                       NOT INCLUDING THE FILENAME unless it ends with config.json or config_gdf.geojson.

    Returns:
        None
    """
    # default save path
    filepath = str(filepath)
    save_path = filepath
    # check if config.json or config_gdf.geojson in the filepath
    if filepath.endswith("config.json"):
        filename = "config.json"
        if isinstance(config, dict):
            write_to_json(filepath, config)
    elif filepath.endswith("config_gdf.geojson"):
        filename = "config_gdf.geojson"
        if isinstance(config, gpd.GeoDataFrame):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            config.to_file(filepath, driver="GeoJSON")
    elif isinstance(config, dict):
        filename = "config.json"
        save_path = os.path.abspath(os.path.join(filepath, filename))
        write_to_json(save_path, config)
    elif isinstance(config, gpd.GeoDataFrame):
        filename = "config_gdf.geojson"
        save_path = os.path.abspath(os.path.join(filepath, filename))
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        config.to_file(save_path, driver="GeoJSON")

    logger.info(f"Saved {filename} saved to {save_path}")


def filter_files(files: List[str], avoid_patterns: List[str]) -> List[str]:
    """
    Filter a list of filepaths based on a list of avoid patterns.

    Args:
        files: A list of filepaths to filter.
        avoid_patterns: A list of regular expression patterns to avoid.

    Returns:
        A list of filepaths whose filenames do not match any of the patterns in avoid_patterns.

    Examples:
        >>> files = ['/path/to/file1.txt', '/path/to/file2.txt', '/path/to/avoid_file.txt']
        >>> avoid_patterns = ['.*avoid.*']
        >>> filtered_files = filter_files(files, avoid_patterns)
        >>> print(filtered_files)
        ['/path/to/file1.txt', '/path/to/file2.txt']

    """
    filtered_files = []
    for file in files:
        # Check if the file's name matches any of the avoid patterns
        for pattern in avoid_patterns:
            if re.match(pattern, os.path.basename(file)):
                break
        else:
            # If the file's name does not match any of the avoid patterns, add it to the filtered files list
            filtered_files.append(file)
    return filtered_files


def get_session_location(
    session_name: str = "", base_path: str = "", raise_error: bool = False
) -> str:
    """
    Gets the session location path based on the provided session name.
    If the session directory doesn't exist, it creates one.

    Args:
        session_name (str, optional): Name of the session. Defaults to "".
        base_path (str, optional): location of parent directory containing sessions directory. Defaults to "".
        raise_error (bool,optional): Raise a FileNotFound error is session_path does not exist. Defaults to False
    Returns:
        str: Path to the session location.

    Note:
    - This function assumes the presence of a function named `create_directory`.
    """
    if not (base_path and os.path.exists(base_path)):
        base_path = os.path.abspath(core_utilities.get_base_dir())
    session_dir = "sessions"
    session_path = (
        os.path.join(base_path, session_dir, session_name)
        if session_name
        else os.path.join(base_path, session_dir)
    )
    # if raise error is True and the session_path does not exist raise FileNotFound
    if not os.path.exists(session_path) and raise_error:
        raise FileNotFoundError(session_path)

    return session_path


def extract_roi_id(path: str) -> Optional[str]:
    """
    Extract the ROI ID from the path.

    Args:
        path (str): Path containing ROI directory.

    Returns:
        Optional[str]: ID of the ROI within the path, or None if not found.
    """
    pattern = r"ID_([A-Za-z0-9]+)"
    match = re.search(pattern, path)
    if match:
        return match.group(1)
    else:
        return None


def find_matching_directory_by_id(base_directory: str, roi_id: str) -> Optional[str]:
    """
    Find a directory with a matching ROI ID in the given base directory.

    Loops through all directories in the given base_directory to find a directory
    with a matching ROI ID.

    Args:
        base_directory (str): Path to the directory containing subdirectories to check.
        roi_id (str): ROI ID to match against directories.

    Returns:
        Optional[str]: Path to the matching directory if found, otherwise None.
    """

    # Iterate over each directory in the base_directory
    for dir_name in os.listdir(base_directory):
        dir_path = os.path.join(base_directory, dir_name)

        # Check if it's a directory and has the matching ROI ID
        if os.path.isdir(dir_path) and extract_roi_id(dir_name) == roi_id:
            return dir_path

    # If no matching directory is found, return None
    return None


def create_session_path(session_name: str, ROI_directory_name: str) -> str:
    """
    Creates a session path by joining the current working directory, a fixed "sessions" directory, and the provided session name.
    After constructing the path, it further creates the sub directory provided ROI_directory.

    Parameters:
    - session_name (str): Name of the session for which the path has to be created.
    - ROI_directory_name (str): The name of the directory related to the region of interest (ROI) that will be appended to the session path.

    Returns:
    - str: The path to the newly created session directory.

    Note:
    - This function assumes the presence of a function named `create_directory` and a logger object named `logger`.
    """
    base_dir = os.path.abspath(core_utilities.get_base_dir())
    session_path = os.path.join(base_dir, "sessions", session_name)
    session_path = create_directory(session_path, ROI_directory_name)
    logger.info(f"Created a session folder at {session_path}")
    return session_path


def create_directory(file_path: Union[os.PathLike, str], name: str) -> str:
    """Creates a new directory with the given name at the specified file path.

    Args:
        file_path (os.PathLike): The file path where the new directory will be created.
        name (str): The name of the new directory to be created.

    Returns:
        str: The full path of the new directory created or existing directory if it already existed.

    Raises:
        OSError: If there was an error creating the new directory.
    """
    # if file_path is not a path then convert it to a path
    if not isinstance(file_path, os.PathLike):
        file_path = os.path.abspath(file_path)
    new_directory = os.path.join(file_path, name)
    # If the directory named 'name' does not exist, create it
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    return new_directory


def write_to_json(filepath: str, settings: dict) -> None:
    """
    Write the settings dictionary to a JSON file.

    Args:
        filepath (str): The file path where the JSON will be written.
        settings (dict): Dictionary containing the settings to save.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    to_file(settings, filepath)


def to_file(data: dict, filepath: str) -> None:
    """
    Serializes a dictionary to a JSON file, handling special serialization for datetime and numpy ndarray objects.

    The function handles two special cases:
    1. If the data contains datetime.date or datetime.datetime objects, they are serialized to their ISO format.
    2. If the data contains numpy ndarray objects, they are converted to lists before serialization.

    Parameters:
    - data (dict): Dictionary containing the data to be serialized to a JSON file.
    - filepath (str): Path (including filename) where the JSON file should be saved.

    Returns:
    - None

    Note:
    - This function requires the json, datetime, and numpy modules to be imported.
    """

    class DateTimeEncoder(JSONEncoder):
        # Override the default method
        def default(self, obj):
            if isinstance(obj, (datetime.date, datetime.datetime)):
                return obj.isoformat()
            # Check for numpy arrays
            if isinstance(obj, np.ndarray):
                # Check if the dtype is 'object', which indicates it might have mixed types including datetimes
                if obj.dtype == "object":
                    # Convert each element of the array
                    return [self.default(item) for item in obj]
                else:
                    # If it's not an object dtype, simply return the array as a list
                    return obj.tolist()

    with open(filepath, "w") as fp:
        json.dump(data, fp, cls=DateTimeEncoder)


def find_directory_recursively(path: str = ".", name: str = "RGB") -> str:
    """
    Recursively search for a directory with the given name.

    Args:
        path (str): The starting directory to search in. Defaults to current directory.
        name (str): The name of the directory to find. Defaults to "RGB".

    Returns:
        str: The path of the first directory with the given name found.

    Raises:
        Exception: If the directory is empty or no directory matching the name is found.
    """
    dir_location = None
    if os.path.basename(path) == name:
        dir_location = path
    else:
        for dirpath, dirnames, filenames in os.walk(path):
            if name in dirnames:
                dir_location = os.path.join(dirpath, name)
                break  # stop searching once the first directory is found

    if not os.listdir(dir_location):
        raise Exception(f"{name} directory is empty.")

    if not dir_location:
        raise Exception(f"No directroy matching {name} could be found at {path}")

    return dir_location


def find_file_recursively(path: str = ".", name: str = "RGB") -> str:
    """
    Recursively search for a file with the given name.

    Args:
        path (str): The starting directory to search in. Defaults to current directory.
        name (str): The name of the file to find. Defaults to "RGB".

    Returns:
        str: The path of the first file with the given name found.

    Raises:
        Exception: If the file location is empty or no file matching the name is found.
    """
    file_location = None
    if os.path.basename(path) == name:
        file_location = path
    else:
        for dirpath, dirnames, filenames in os.walk(path):
            if name in filenames:
                file_location = os.path.join(dirpath, name)
                return file_location

    if not os.listdir(file_location):
        raise Exception(f"{file_location} is empty.")

    if not file_location:
        raise Exception(f" No file matching {name} could be found at {path}")

    return file_location


def load_json_data_from_file(search_location: str, filename: str) -> dict:
    """
    Load JSON data from a file by searching for the file in the specified location.

    The function searches recursively in the provided search location for a file with
    the specified filename. Once the file is found, it loads the JSON data from the file
    and returns it as a dictionary.

    Args:
        search_location (str): Directory or path to search for the file.
        filename (str): Name of the file to load.

    Returns:
        dict: Data read from the JSON file as a dictionary.

    Raises:
        FileNotFoundError: If the file cannot be found in the search location.
    """
    file_path = find_file_recursively(search_location, filename)
    json_data = load_data_from_json(file_path)
    return json_data


def load_data_from_json(filepath: Union[str, pathlib.Path]) -> dict:
    """
    Read data from a JSON file and return it as a dictionary.

    The function reads the data from the specified JSON file using the provided filepath.
    It applies a custom object hook, `DecodeDateTime`, to decode the datetime and shoreline
    data if they exist in the dictionary.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: Data read from the JSON file as a dictionary with decoded datetime and shoreline data.
    """

    def DecodeDateTime(readDict):
        """
        Helper function to decode datetime and shoreline data in the dictionary.

        Args:
            readDict (dict): Dictionary to decode.

        Returns:
            dict: Decoded dictionary.

        """
        if "dates" in readDict:
            tmp = [
                datetime.datetime.fromisoformat(dates) for dates in readDict["dates"]
            ]
            readDict["dates"] = tmp
        if "shorelines" in readDict:
            tmp = [
                np.array(shoreline) if len(shoreline) > 0 else np.empty((0, 2))
                for shoreline in readDict["shorelines"]
            ]
            readDict["shorelines"] = tmp
        return readDict

    with open(filepath, "r") as fp:
        data = json.load(fp, object_hook=DecodeDateTime)
    return data
