import os
import re
import glob
import shutil
import json
import math
import logging
from typing import Callable, List
from typing import Union
from json import JSONEncoder
import datetime

# Internal dependencies imports
from coastseg import exceptions


from tqdm.auto import tqdm
import requests
from area import area
import geopandas as gpd
import numpy as np
import geojson
from leafmap import check_file_path
import pandas as pd
import shapely
from ipyfilechooser import FileChooser

from ipywidgets import ToggleButton
from ipywidgets import HBox
from ipywidgets import VBox
from ipywidgets import Layout
from ipywidgets import HTML

# widget icons from https://fontawesome.com/icons/angle-down?s=solid&f=classic

logger = logging.getLogger(__name__)


def create_file_chooser(
    callback: Callable[[FileChooser], None],
    title: str = None,
    filter_pattern: str = None,
):
    """
    This function creates a file chooser and a button to close the file chooser.
    It takes a callback function and an optional title as arguments.
    It only searches for .geojson files.

    Args:
        callback (Callable[[FileChooser],None]): A callback function that which is called
        when a file is selected.
        title (str): Optional title for the file chooser.

    Returns:
        chooser (HBox): A HBox containing the file chooser and close button.
    """
    padding = "0px 0px 0px 5px"  # upper, right, bottom, left
    # creates a unique instance of filechooser and button to close filechooser
    geojson_chooser = FileChooser(os.getcwd())
    geojson_chooser.dir_icon = os.sep

    geojson_chooser.filter_pattern = ["*.geojson"]
    if filter_pattern:
        geojson_chooser.filter_pattern = [filter_pattern]

    geojson_chooser.title = "<b>Select a geojson file</b>"
    if title is not None:
        geojson_chooser.title = f"<b>{title}</b>"
    # callback function is called when a file is selected
    geojson_chooser.register_callback(callback)

    close_button = ToggleButton(
        value=False,
        tooltip="Close File Chooser",
        icon="times",
        button_style="primary",
        layout=Layout(height="28px", width="28px", padding=padding),
    )

    def close_click(change: dict):
        if change["new"]:
            geojson_chooser.close()
            close_button.close()

    close_button.observe(close_click, "value")
    chooser = HBox([geojson_chooser, close_button], layout=Layout(width="100%"))
    return chooser


def create_dir_chooser(callback, title: str = None, starting_directory: str = "data"):
    padding = "0px 0px 0px 5px"  # upper, right, bottom, left
    inital_path = os.path.join(os.getcwd(), starting_directory)
    if not os.path.exists(inital_path):
        inital_path = os.getcwd()
    # creates a unique instance of filechooser and button to close filechooser
    dir_chooser = FileChooser(inital_path)
    dir_chooser.dir_icon = os.sep
    # Switch to folder-only mode
    dir_chooser.show_only_dirs = True
    if title is not None:
        dir_chooser.title = f"<b>{title}</b>"
    dir_chooser.register_callback(callback)

    close_button = ToggleButton(
        value=False,
        tooltip="Close Directory Chooser",
        icon="times",
        button_style="primary",
        layout=Layout(height="28px", width="28px", padding=padding),
    )

    def close_click(change):
        if change["new"]:
            dir_chooser.close()
            close_button.close()

    close_button.observe(close_click, "value")
    chooser = HBox([dir_chooser, close_button])
    return chooser


def get_transect_settings(settings: dict) -> dict:
    transect_settings = {}
    transect_settings["max_std"] = settings.get("max_std")
    transect_settings["min_points"] = settings.get("min_points")
    transect_settings["along_dist"] = settings.get("along_dist")
    transect_settings["max_range"] = settings.get("max_range")
    transect_settings["min_chainage"] = settings.get("min_chainage")
    transect_settings["multiple_inter"] = settings.get("multiple_inter")
    transect_settings["prc_multiple"] = settings.get("prc_multiple")
    return transect_settings


def get_session_path(session_name: str, ROI_directory: str) -> str:
    session_path = os.path.join(os.getcwd(), "sessions", session_name)
    session_path = create_directory(session_path, ROI_directory)
    logger.info(f"session_path: {session_path}")
    return session_path


def create_directory_in_google_drive(path: str, name: str) -> str:
    """
    Creates a new directory with the provided name in the given path.
    Raises FileNotFoundError if the given path does not exist.

    Parameters:
    path (str): path to the directory where the new directory will be created
    name (str): name of the new directory

    Returns:
    new_path (str): path to the newly created directory
    """
    new_path = os.path.join(path, name)
    if os.path.exists(path):
        if not os.path.isdir(new_path):
            os.mkdir(new_path)
    else:
        raise FileNotFoundError(new_path)
    return new_path


def is_in_google_colab() -> bool:
    """
    Returns True if the code is running in Google Colab, False otherwise.
    """
    if os.getenv("COLAB_RELEASE_TAG"):
        return True
    else:
        return False


def to_file(data: dict, filepath: str) -> None:
    class DateTimeEncoder(JSONEncoder):
        # Override the default method
        def default(self, obj):
            if isinstance(obj, (datetime.date, datetime.datetime)):
                return obj.isoformat()
            if isinstance(obj, (np.ndarray)):
                new_obj = [array.tolist() for array in obj]
                return new_obj

    with open(filepath, "w") as fp:
        json.dump(data, fp, cls=DateTimeEncoder)


def get_ids_with_invalid_area(
    geometry: gpd.GeoDataFrame, max_area: float = 98000000, min_area: float = 0
) -> set:
    if isinstance(geometry, gpd.GeoDataFrame):
        geometry = json.loads(geometry.to_json())
    if isinstance(geometry, dict):
        if "features" in geometry.keys():
            rows_drop = set()
            for i, feature in enumerate(geometry["features"]):
                roi_area = get_area(feature["geometry"])
                if roi_area >= max_area or roi_area <= min_area:
                    rows_drop.add(i)
            return rows_drop
    else:
        raise TypeError("Must be geodataframe")


def load_cross_distances_from_file(dir_path):
    transect_dict = None
    glob_str = os.path.join(dir_path, "*transects_cross_distances.json*")
    for file in glob.glob(glob_str):
        if os.path.basename(file) == "transects_cross_distances.json":
            transect_dict = from_file(file)

    if transect_dict is None:
        logger.warning(
            f"No transect cross shore distances could be loaded from {dir_path}"
        )
        return None

    # convert lists to np.array for each transect
    for key in transect_dict.keys():
        tmp = np.array(transect_dict[key])
        transect_dict[key] = tmp
    logger.info(f"Loaded transect cross shore distances from: {dir_path}")
    return transect_dict


def from_file(filepath: str) -> dict:
    def DecodeDateTime(readDict):
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


def mount_google_drive(name: str = "CoastSeg") -> None:
    """
    If the user is running in Google Colab, the Google Drive will be mounted to the root directory
    "/content/drive/MyDrive" and a new directory will be created with the provided name.

    Parameters:
    name (str): The name of the directory to be created. Default is 'CoastSeg'.

    Returns:
    None
    """
    if is_in_google_colab():
        from google.colab import drive

        # default location google drive is mounted to
        root_dir = "/content/drive/MyDrive"
        # mount google drive to default home directory
        drive.mount("/content/drive", force_remount=True)
        # create directory with provided name in google drive
        new_path = create_directory_in_google_drive(root_dir, name)
        # change working directory to directory with name
        os.chdir(new_path)
    else:
        print("Not running in Google Colab.")


def create_hover_box(title: str, feature_html: HTML = None) -> VBox:
    """
    Creates a box with a title and optional HTML containing information about the feature that was
    last hovered over.
    The hover box has two buttons, an 'uncollapse' and 'collapse' button.
    The 'uncollapse' button opens the hover box to reveal details about the feature that was
    last hovered over, whereas the 'collapse' button hides the feature_html and just shows the default messages of
    'Hover over a feature' or 'Hover Data Available'.

    Parameters:
    title (str): The title of the hover box
    feature_html (HTML, optional): HTML of the feature to be displayed in the hover box

    Returns:
    container (VBox): Box with the given title and details about the feature given by feature_html
    """
    padding = "0px 0px 0px 5px"  # upper, right, bottom, left
    # create title
    title = HTML(f"  <h4>{title} Hover  </h4>")
    # Default message shown when nothing has been hovered
    msg = HTML(f"Hover over a feature</br>")
    # open button allows user to see hover data
    uncollapse_button = ToggleButton(
        value=False,
        tooltip="Show hover data",
        icon="angle-down",
        button_style="info",
        layout=Layout(height="28px", width="28px", padding=padding),
    )

    # collapse_button collapses hover data
    collapse_button = ToggleButton(
        value=False,
        tooltip="Show hover data",
        icon="angle-up",
        button_style="info",
        layout=Layout(height="28px", width="28px", padding=padding),
    )

    # message tells user that data is available on hover
    container_content = VBox([msg])
    if feature_html.value == "":
        container_content.children = [msg]
    elif feature_html.value != "":
        container_content.children = [feature_html]

    # default configuration for container is in collapsed mode
    container_header = HBox([title, uncollapse_button])
    container = VBox([container_header])

    def uncollapse_click(change: dict):
        logger.info(change)
        if feature_html.value == "":
            container_content.children = [msg]
        elif feature_html.value != "":
            container_content.children = [feature_html]
        container_header.children = [title, collapse_button]
        container.children = [container_header, container_content]

    def collapse_click(change: dict):
        logger.info(change)
        container_header.children = [title, uncollapse_button]
        container.children = [container_header]

    collapse_button.observe(collapse_click, "value")
    uncollapse_button.observe(uncollapse_click, "value")
    return container


def filter_dict_by_keys(original_dict: dict, keys: list) -> dict:
    filtered_dict = {k: v for k, v in original_dict.items() if k in keys}
    return filtered_dict


def create_warning_box(title: str = None, msg: str = None) -> HBox:
    """
    Creates a warning box with a title and message that can be closed with a close button.

    Parameters:
    title (str, optional): The title of the warning box. Default is 'Warning'.
    msg (str, optional): The message of the warning box. Default is 'Something went wrong...'.

    Returns:
    HBox: The warning box containing the title, message and close button.
    """
    padding = "0px 0px 0px 5px"  # upper, right, bottom, left
    # create title
    if title is None:
        title = "Warning"
    warning_title = HTML(f"<b>⚠️<u>{title}</u></b>")
    # create msg
    if msg is None:
        msg = "Something went wrong..."
    warning_msg = HTML(
        f"_______________________________________\
                   </br>⚠️{msg}"
    )
    # create vertical box to hold title and msg
    warning_content = VBox(
        [warning_title, warning_msg],
        layout=Layout(width="70%", max_width="75%", padding="5px 5px 5px 5px"),
    )

    # define a close button
    close_button = ToggleButton(
        value=False,
        tooltip="Close Warning Box",
        icon="times",
        button_style="danger",
        layout=Layout(height="28px", width="28px", padding=padding),
    )

    def close_click(change):
        if change["new"]:
            warning_content.close()
            close_button.close()

    close_button.observe(close_click, "value")
    warning_box = HBox([warning_content, close_button])
    return warning_box


def clear_row(row: HBox):
    """close widgets in row/column and clear all children
    Args:
        row (HBox)(VBox): row or column
    """
    for index in range(len(row.children)):
        row.children[index].close()
    row.children = []


def save_to_geojson_file(out_file: str, geojson: dict, **kwargs) -> None:
    """save_to_geojson_file Saves given geojson to a geojson file at outfile
    Args:
        out_file (str): The output file path
        geojson (dict): geojson dict containing FeatureCollection for all geojson objects in selected_set
    """
    # Save the geojson to a file
    out_file = check_file_path(out_file)
    ext = os.path.splitext(out_file)[1].lower()
    if ext == ".geojson":
        out_geojson = out_file
    else:
        out_geojson = os.path.splitext(out_file)[1] + ".geojson"
    with open(out_geojson, "w") as f:
        json.dump(geojson, f, **kwargs)


def download_url(url: str, save_path: str, filename: str = None, chunk_size: int = 128):
    """Downloads the data from the given url to the save_path location.
    Args:
        url (str): url to data to download
        save_path (str): directory to save data
        chunk_size (int, optional):  Defaults to 128.
    """
    with requests.get(url, stream=True) as r:
        logger.info(r)
        logger.info(r.content)
        logger.info(r.headers)
        if r.status_code == 404:
            logger.error(f"Error {r.status_code}. DownloadError: {save_path}")
            raise exceptions.DownloadError(os.path.basename(save_path))
        if r.status_code == 429:
            logger.error(f"Error {r.status_code}.DownloadError: {save_path}")
            raise Exception(
                "Zenodo has denied the request. You may have requested too many files at once."
            )
        # check header to get content length, in bytes
        content_length = r.headers.get("Content-Length")
        if content_length:
            total_length = int(content_length)
            with open(save_path, "wb") as fd:
                with tqdm(
                    total=total_length,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {filename}",
                    initial=0,
                    ascii=True,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        fd.write(chunk)
                        pbar.update(len(chunk))
        else:
            logger.warning("Content length not found in response headers")


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


def get_center_rectangle(coords: list) -> tuple:
    """returns the center point of rectangle specified by points coords
    Args:
        coords list[tuple(float,float)]: lat,lon coordinates
    Returns:
        tuple[float]: (center x coordinate, center y coordinate)
    """
    x1, y1 = coords[0][0], coords[0][1]
    x2, y2 = coords[2][0], coords[2][1]
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    return center_x, center_y


def get_epsg_from_geometry(geometry: "shapely.geometry.polygon.Polygon") -> int:
    """Uses geometry of shapely rectangle in crs 4326 to return the most accurate
    utm code as a string of format 'epsg:utm_code'
    example: 'espg:32610'

    Args:
        geometry (shapely.geometry.polygon.Polygon): geometry of a rectangle

    Returns:
        int: most accurate epsg code based on lat lon coordinates of given geometry
    """
    rect_coords = geometry.exterior.coords
    center_x, center_y = get_center_rectangle(rect_coords)
    utm_code = convert_wgs_to_utm(center_x, center_y)
    return int(utm_code)


def convert_wgs_to_utm(lon: float, lat: float) -> str:
    """return most accurate utm epsg-code based on lat and lng
    convert_wgs_to_utm function, see https://stackoverflow.com/a/40140326/4556479
    Args:
        lon (float): longitude
        lat (float): latitude
    Returns:
        str: new espg code
    """
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == "1":
        utm_band = "0" + utm_band
    if lat >= 0:
        epsg_code = "326" + utm_band  # North
        return epsg_code
    epsg_code = "327" + utm_band  # South
    return epsg_code


def extract_roi_by_id(gdf: gpd.geodataframe, roi_id: int) -> gpd.geodataframe:
    """Returns geodataframe with a single ROI whose id matches roi_id.
       If roi_id is None returns gdf

    Args:
        gdf (gpd.geodataframe): ROI geodataframe to extract ROI with roi_id from
        roi_id (int): id of the ROI to extract
    Raises:
        exceptions.Id_Not_Found: if id doesn't exist in ROI's geodataframe or self.rois.gdf is empty
    Returns:
        gpd.geodataframe: ROI with id matching roi_id
    """
    if roi_id is None:
        single_roi = gdf
    else:
        # Select a single roi by id
        single_roi = gdf[gdf["id"].astype(str) == str(roi_id)]
        # if the id was not found in the geodataframe raise an exception
    if single_roi.empty:
        logger.error(f"Id: {id} was not found in {gdf}")
        raise exceptions.Id_Not_Found(id)
    logger.info(f"single_roi: {single_roi}")
    return single_roi


def get_area(polygon: dict) -> float:
    "Calculates the area of the geojson polygon using the same method as geojson.io"
    return round(area(polygon), 3)


def read_json_file(filename: str) -> dict:
    logger.info(f"read_json_file {filename}")
    with open(filename, "r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    return data


def find_config_json(search_path) -> bool:
    logger.info(f"searching directory for config.json: {search_path}")

    def use_regex(input_text):
        pattern = re.compile(r"config.*\.json", re.IGNORECASE)
        if pattern.match(input_text) is not None:
            return True
        return False

    for item in os.listdir(search_path):
        if use_regex(item):
            logger.info(f"{item} matched regex")
            return item

    raise FileNotFoundError(f"config.json file was not found at {search_path}")


def config_to_file(config: Union[dict, gpd.GeoDataFrame], file_path: str):
    """Saves config to config.json or config_gdf.geojson
    config's type is dict or geodataframe respectively

    Args:
        config (Union[dict, gpd.GeoDataFrame]): data to save to config file
        file_path (str): full path to directory to save config file
    """
    if isinstance(config, dict):
        filename = f"config.json"
        save_path = os.path.abspath(os.path.join(file_path, filename))
        write_to_json(save_path, config)
        logger.info(f"Saved config json: {filename} \nSaved to {save_path}")
    elif isinstance(config, gpd.GeoDataFrame):
        filename = f"config_gdf.geojson"
        save_path = os.path.abspath(os.path.join(file_path, filename))
        logger.info(f"Saving config gdf:{config} \nSaved to {save_path}")
        config.to_file(save_path, driver="GeoJSON")


def create_directory(file_path: str, name: str) -> str:
    """Creates a new directory with the given name at the specified file path.

    Args:
        file_path (str): The file path where the new directory will be created.
        name (str): The name of the new directory to be created.

    Returns:
        str: The full path of the new directory created or existing directory if it already existed.

    Raises:
        OSError: If there was an error creating the new directory.
    """
    new_directory = os.path.join(file_path, name)
    # If the directory named 'name' does not exist, create it
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    return new_directory


def replace_column(df, new_name="id", replace_col=None) -> None:
    """Renames the column named replace_col with new_name. If column named
    new_name does not exist a new column named new_name is created with the row index.

    NOTE: replace_col is case insensitive, so if replace_col = 'NAME' then any column with 'name','NAME','NaME','Name',etc.
    will be replaced with new name.

    Args:
        df (geodataframe): geodataframe with columns
        new_name (str, optional): new column nam. Defaults to 'id'.
        replace_col (_type_, optional): column name to replace. Defaults to None.
    """
    # name  of column to replace with new_name
    if replace_col is not None:
        if new_name not in df.columns:
            if replace_col in df.columns.str.lower():
                col_idx = df.columns.str.lower().get_loc(replace_col)
                col_name = df.columns[col_idx]
                df.rename(columns={col_name: new_name}, inplace=True)
    elif replace_col is None:
        if new_name not in df.columns:
            # create a new column with new_name and row index
            df[new_name] = df.index


def get_transect_points_dict(feature: gpd.geodataframe) -> dict:
    """Returns dict of np.arrays of transect start and end points
    Example
    {
        'usa_CA_0289-0055-NA1': array([[-13820440.53165404,   4995568.65036405],
        [-13820940.93156407,   4995745.1518021 ]]),
        'usa_CA_0289-0056-NA1': array([[-13820394.24579453,   4995700.97802925],
        [-13820900.16320004,   4995862.31860808]])
    }
    Args:
        feature (gpd.geodataframe): clipped transects within roi
    Returns:
        dict: dict of np.arrays of transect start and end points
        of form {
            '<transect_id>': array([[start point],
                        [end point]]),}
    """
    features = []
    # Use explode to break multilinestrings in linestrings
    feature_exploded = feature.explode(ignore_index=True)
    # For each linestring portion of feature convert to lat,lon tuples
    lat_lng = feature_exploded.apply(
        lambda row: {str(row.id): np.array(np.array(row.geometry.coords).tolist())},
        axis=1,
    )
    features = list(lat_lng)
    new_dict = {}
    for item in list(features):
        new_dict = {**new_dict, **item}
    return new_dict


def move_files(src_dir: str, dst_dir: str, delete_src: bool = False) -> None:
    """
    Moves every file in a source directory to a destination directory, and has the option to delete the source directory when finished.

    The function uses the `shutil` library to move the files from the source directory to the destination directory. If the `delete_src` argument is set to `True`, the function will delete the source directory after all the files have been moved.

    Args:
    - src_dir (str): The path of the source directory.
    - dst_dir (str): The path of the destination directory.
    - delete_src (bool, optional): A flag indicating whether to delete the source directory after the files have been moved. Default is `False`.

    Returns:
    - None
    """
    logger.info(f"Moving files from {src_dir} to dst_dir. Delete Source:{delete_src}")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        dst_file = os.path.join(dst_dir, filename)
        shutil.move(src_file, dst_file)
    if delete_src:
        os.rmdir(src_dir)


def get_cross_distance_df(
    extracted_shorelines: dict, cross_distance_transects: dict
) -> pd.DataFrame:
    transects_csv = {}
    # copy dates from extracted shoreline
    transects_csv["dates"] = extracted_shorelines["dates"]
    # add cross distances for each transect within the ROI
    transects_csv = {**transects_csv, **cross_distance_transects}
    return pd.DataFrame(transects_csv)


def remove_z_axis(geodf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """If the geodataframe has z coordinates in any rows, the z coordinates are dropped.
    Otherwise the original geodataframe is returned.

    Additionally any multi part geometeries will be exploded into single geometeries.
    eg. MutliLineStrings will be converted into LineStrings.
    Args:
        geodf (gpd.GeoDataFrame): geodataframe to check for z-axis

    Returns:
        gpd.GeoDataFrame: original dataframe if there is no z axis. If a z axis is found
        a new geodataframe is returned with z axis dropped.
    """
    if geodf.empty:
        logger.warning(f"Empty geodataframe has no z-axis")
        return geodf

    # if any row has a z coordinate then remove the z_coordinate
    logger.info(f"Has Z axis: {geodf['geometry'].has_z.any()}")
    if geodf["geometry"].has_z.any():

        def remove_z_from_row(row):
            if row.geometry.has_z:
                row.geometry = shapely.ops.transform(
                    lambda x, y, z=None: (x, y), row.geometry
                )
                return row
            else:
                return row

        # Use explode to break multilinestrings in linestrings
        feature_exploded = geodf.explode(ignore_index=True)
        # For each linestring portion of feature convert to lat,lon tuples
        no_z_gdf = feature_exploded.apply(remove_z_from_row, axis=1)
        return no_z_gdf
    else:
        return geodf


def create_json_config(inputs: dict, settings: dict) -> dict:
    """returns config dictionary with the settings, currently selected_roi ids, and
    each of the inputs specified by roi id.
    sample config:
    {
        'roi_ids': ['17','20']
        'settings':{ 'dates': ['2018-12-01', '2019-03-01'],
                    'cloud_thresh': 0.5,
                    'dist_clouds': 300,
                    'output_epsg': 3857,}
        '17':{
            'sat_list': ['L8'],
            'landsat_collection': 'C01',
            'dates': ['2018-12-01', '2019-03-01'],
            'sitename':'roi_17',
            'filepath':'C:\\Home'
        }
        '20':{
            'sat_list': ['L8'],
            'landsat_collection': 'C01',
            'dates': ['2018-12-01', '2019-03-01'],
            'sitename':'roi_20',
            'filepath':'C:\\Home'
        }
    }

    Args:
        inputs (dict): json style dictionary with roi ids at the keys with inputs as values
        settings (dict):  json style dictionary containing map settings
    Returns:
        dict: json style dictionary, config
    """
    roi_ids = list(inputs.keys())
    config = {**inputs}
    config["roi_ids"] = roi_ids
    config["settings"] = settings
    logger.info(f"config_json: {config} ")
    return config


def create_config_gdf(
    rois: gpd.GeoDataFrame,
    shorelines_gdf: gpd.GeoDataFrame = None,
    transects_gdf: gpd.GeoDataFrame = None,
    bbox_gdf: gpd.GeoDataFrame = None,
) -> gpd.GeoDataFrame():
    if shorelines_gdf is None:
        shorelines_gdf = gpd.GeoDataFrame()
    if transects_gdf is None:
        transects_gdf = gpd.GeoDataFrame()
    if bbox_gdf is None:
        bbox_gdf = gpd.GeoDataFrame()
    # create new column 'type' to indicate object type
    rois["type"] = "roi"
    shorelines_gdf["type"] = "shoreline"
    transects_gdf["type"] = "transect"
    bbox_gdf["type"] = "bbox"
    new_gdf = gpd.GeoDataFrame(pd.concat([rois, shorelines_gdf], ignore_index=True))
    new_gdf = gpd.GeoDataFrame(pd.concat([new_gdf, transects_gdf], ignore_index=True))
    new_gdf = gpd.GeoDataFrame(pd.concat([new_gdf, bbox_gdf], ignore_index=True))
    return new_gdf


def write_to_json(filepath: str, settings: dict):
    """ "Write the  settings dictionary to json file"""
    with open(filepath, "w", encoding="utf-8") as output_file:
        json.dump(settings, output_file)


def read_geojson_file(geojson_file: str) -> dict:
    """Returns the geojson of the selected ROIs from the file specified by geojson_file"""
    with open(geojson_file) as f:
        data = geojson.load(f)
    return data


def read_gpd_file(filename: str) -> gpd.GeoDataFrame:
    """
    Returns geodataframe from geopandas geodataframe file
    """
    if os.path.exists(filename):
        logger.info(f"Opening \n {filename}")
        with open(filename, "r") as f:
            gpd_data = gpd.read_file(f)
    else:
        raise FileNotFoundError
    return gpd_data


def get_jpgs_from_data() -> str:
    """Returns the folder where all jpgs were copied from the data folder in coastseg.
    This is where the model will save the computed segmentations."""
    # Data folder location
    src_path = os.path.abspath(os.getcwd() + os.sep + "data")
    if os.path.exists(src_path):
        rename_jpgs(src_path)
        # Create a new folder to hold all the data
        location = os.getcwd()
        name = "segmentation_data"
        # new folder "segmentation_data_datetime"
        new_folder = mk_new_dir(name, location)
        # create subdirectories for each image type
        file_types = ["RGB", "SWIR", "NIR"]
        for file_type in file_types:
            new_path = os.path.join(new_folder, file_type)
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            glob_str = (
                src_path
                + str(os.sep + "**" + os.sep) * 2
                + "preprocessed"
                + os.sep
                + file_type
                + os.sep
                + "*.jpg"
            )
            copy_files_to_dst(src_path, new_path, glob_str)
            RGB_path = os.path.join(new_folder, "RGB")
        return RGB_path
    else:
        print("ERROR: Cannot find the data directory in coastseg")
        raise Exception("ERROR: Cannot find the data directory in coastseg")


def rename_jpgs(src_path: str) -> None:
    """Renames all the jpgs in the data directory in coastseg
    Args:
        src_path (str): full path to the data directory in coastseg
    """
    files_renamed = False
    for folder in os.listdir(src_path):
        folder_path = src_path + os.sep + folder
        # Split the folder name at the first _
        folder_id = folder.split("_")[0]
        folder_path = folder_path + os.sep + "jpg_files" + os.sep + "preprocessed"
        jpgs = glob.glob1(folder_path + os.sep, "*jpg")
        # Append folder id to basename of jpg if not already there
        for jpg in jpgs:
            if folder_id not in jpg:
                files_renamed = True
                base, ext = os.path.splitext(jpg)
                new_name = folder_path + os.sep + base + "_" + folder_id + ext
                old_name = folder_path + os.sep + jpg
                os.rename(old_name, new_name)
        if files_renamed:
            print(f"Renamed files in {src_path} ")


def do_rois_filepaths_exist(roi_settings: dict, roi_ids: list) -> bool:
    """Returns true if all rois have filepaths that exist
    Args:
        roi_settings (dict): settings of all rois on map
        roi_ids (list): ids of rois selected on map
    Returns:
        bool: True if all rois have filepaths that exist
    """
    # by default assume all filepaths exist
    does_filepath_exist = True
    for roi_id in roi_ids:
        filepath = str(roi_settings[roi_id]["filepath"])
        if not os.path.exists(filepath):
            # if filepath does not exist stop checking
            does_filepath_exist = False
            logger.info(f"filepath did not exist{filepath}")
            print("Some ROIs contained filepaths that did not exist")
            break
    logger.info(f"{does_filepath_exist} All rois filepaths exist")
    return does_filepath_exist


def do_rois_have_sitenames(roi_settings: dict, roi_ids: list) -> bool:
    """Returns true if all rois have "sitename" with non-empty string
    Args:
        roi_settings (dict): settings of all rois on map
        roi_ids (list): ids of rois selected on map

    Returns:
        bool: True if all rois have "sitename" with non-empty string
    """
    # by default assume all sitenames are not empty
    is_sitename_not_empty = True
    for roi_id in roi_ids:
        if roi_settings[roi_id]["sitename"] == "":
            # if sitename is empty means user has not downloaded ROI data
            is_sitename_not_empty = False
            break
    logger.info(f"{is_sitename_not_empty} All rois have non-empty sitenames")
    return is_sitename_not_empty


def were_rois_downloaded(roi_settings: dict, roi_ids: list) -> bool:
    """Returns true if rois were downloaded before. False if they have not
    Uses 'sitename' key for each roi to determine if roi was downloaded.
    And checks if filepath were roi is saved is valid
    If each roi's 'sitename' is not empty string returns true
    Args:
        roi_settings (dict): settings of all rois on map
        roi_ids (list): ids of rois selected on map

    Returns:
        bool: True means rois were downloaded before
    """
    # by default assume rois were downloaded
    is_downloaded = True
    if roi_settings is None:
        # if rois do not have roi_settings this means they were never downloaded
        is_downloaded = False
    elif roi_settings == {}:
        # if rois do not have roi_settings this means they were never downloaded
        is_downloaded = False
    elif roi_settings != {}:
        all_sitenames_exist = do_rois_have_sitenames(roi_settings, roi_ids)
        all_filepaths_exist = do_rois_filepaths_exist(roi_settings, roi_ids)
        is_downloaded = all_sitenames_exist and all_filepaths_exist
    # print correct message depending on whether ROIs were downloaded
    if is_downloaded:
        logger.info(f"Located previously downloaded ROI data.")
    elif is_downloaded == False:
        print(
            "Did not locate previously downloaded ROI data. To download the imagery for your ROIs click Download Imagery"
        )
        logger.info(
            f"Did not locate previously downloaded ROI data. To download the imagery for your ROIs click Download Imagery"
        )
    return is_downloaded


def create_roi_settings(
    settings: dict,
    selected_rois: dict,
    filepath: str,
    date_str: str = "",
) -> dict:
    """returns a dict of settings for each roi with roi id as the key.
    Example:
    "2": {
            "dates": ["2018-12-01", "2019-03-01"],
            "sat_list": ["L8"],
            "sitename": "ID_2_datetime10-19-22__04_00_34",
            "filepath": "C:\\CoastSeg\\data",
            "roi_id": "2",
            "polygon": [
                [
                    [-124.16930255115336, 40.8665390046026],
                    [-124.16950858759564, 40.878247531017706],
                    [-124.15408259844114, 40.878402930533994],
                    [-124.1538792781699, 40.8666943403763],
                    [-124.16930255115336, 40.8665390046026],
                ]
            ],
            "landsat_collection": "C01",
        },
        "3": {
            "dates": ["2018-12-01", "2019-03-01"],
            "sat_list": ["L8"],
            "sitename": "ID_3_datetime10-19-22__04_00_34",
            "filepath": "C:\\CoastSeg\\data",
            "roi_id": "3",
            "polygon": [
                [
                    [-124.16950858759564, 40.878247531017706],
                    [-124.16971474532464, 40.88995603272874],
                    [-124.15428603840094, 40.890111496009816],
                    [-124.15408259844114, 40.878402930533994],
                    [-124.16950858759564, 40.878247531017706],
                ]
            ],
            "landsat_collection": "C01",
        },

    Args:
        settings (dict): settings from coastseg_map.
        Must have keys ["sat_list","landsat_collection","dates"]
        selected_rois (dict): geojson dict of rois selected
        filepath (str): file path to directory to hold roi data
        date_str (str, optional): datetime formatted string. Defaults to "".

    Returns:
        dict: settings for each roi with roi id as the key
    """

    roi_settings = {}
    sat_list = settings["sat_list"]
    landsat_collection = settings["landsat_collection"]
    dates = settings["dates"]
    for roi in selected_rois["features"]:
        roi_id = str(roi["properties"]["id"])
        sitename = (
            "" if date_str == "" else "ID_" + str(roi_id) + "_datetime" + date_str
        )
        polygon = roi["geometry"]["coordinates"]
        inputs_dict = {
            "dates": dates,
            "sat_list": sat_list,
            "roi_id": roi_id,
            "polygon": polygon,
            "landsat_collection": landsat_collection,
            "sitename": sitename,
            "filepath": filepath,
        }
        roi_settings[roi_id] = inputs_dict
    return roi_settings


def generate_datestring() -> str:
    """Returns a datetime string in the following format %m-%d-%y__%I_%M_%S
    EX: "ID_0__01-31-22_12_19_45"""
    date = datetime.datetime.now()
    return date.strftime("%m-%d-%y__%I_%M_%S")


def mk_new_dir(name: str, location: str):
    """Create new folder with name_datetime stamp at location
    Args:
        name (str): name of folder to create
        location (str): full path to location to create folder
    """
    if os.path.exists(location):
        new_folder = location + os.sep + name + "_" + generate_datestring()
        os.mkdir(new_folder)
        return new_folder
    else:
        raise Exception("Location provided does not exist.")


def get_RGB_in_path(current_path: str) -> str:
    """
    Returns the full path to the RGB directory relative to the given current path, or raises an error if no RGB directory
    is found or if the RGB directory is empty.

    Args:
        current_path (str): The full path to the directory of images to segment.

    Raises:
        Exception: Raised if no RGB directory is found or if the RGB directory is empty.

    Returns:
        str: The full path to the RGB directory relative to the current path.
    """
    rgb_jpgs = glob.glob(current_path + os.sep + "*RGB*")
    logger.info(f"rgb_jpgs: {rgb_jpgs}")
    if rgb_jpgs:
        return current_path

    parent_dir = os.path.dirname(current_path)
    logger.info(f"parent_dir: {parent_dir}")
    dirs = os.listdir(parent_dir)
    logger.info(f"child dirs: {dirs}")
    if "RGB" not in dirs:
        raise Exception("Invalid directory. Please select an RGB directory.")
    rgb_path = os.path.join(parent_dir, "RGB")
    if not os.listdir(rgb_path):
        raise Exception("RGB directory is empty.")
    logger.info(f"returning path:{rgb_path}")
    return rgb_path


def copy_files_to_dst(src_path: str, dst_path: str, glob_str: str) -> None:
    """Copies all files from src_path to dest_path
    Args:
        src_path (str): full path to the data directory in coastseg
        dst_path (str): full path to the images directory in Sniffer
    """
    if not os.path.exists(dst_path):
        raise FileNotFoundError(f"dst_path: {dst_path} doesn't exist.")
    elif not os.path.exists(src_path):
        raise FileNotFoundError(f"src_path: {src_path} doesn't exist.")
    else:
        for file in glob.glob(glob_str):
            shutil.copy(file, dst_path)
        logger.info(f"\nCopied files that matched {glob_str}  \nto {dst_path}")


def scale(matrix: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """returns resized matrix with shape(rows,cols)
        for 2d discrete labels
        for resizing 2d integer arrays
    Args:
        im (np.ndarray): 2d matrix to resize
        nR (int): number of rows to resize 2d matrix to
        nC (int): number of columns to resize 2d matrix to

    Returns:
        np.ndarray: resized matrix with shape(rows,cols)
    """
    src_rows = len(matrix)  # source number of rows
    src_cols = len(matrix[0])  # source number of columns
    tmp = [
        [
            matrix[int(src_rows * r / rows)][int(src_cols * c / cols)]
            for c in range(cols)
        ]
        for r in range(rows)
    ]
    return np.array(tmp).reshape((rows, cols))


def rescale_array(dat, mn, mx):
    """
    rescales an input dat between mn and mx
    Code from doodleverse_utils by Daniel Buscombe
    source: https://github.com/Doodleverse/doodleverse_utils
    """
    m = min(dat.flatten())
    M = max(dat.flatten())
    return (mx - mn) * (dat - m) / (M - m) + mn
