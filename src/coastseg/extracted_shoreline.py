# Standard library imports
import copy
import datetime
import fnmatch
import json
import logging
import os
import re
from glob import glob
from itertools import islice
from time import perf_counter
from typing import Dict, List, Optional, Union, Tuple
from datetime import timezone

# External dependencies imports
import dask
import pytz
import geopandas as gpd
import numpy as np
import pandas as pd
from ipyleaflet import GeoJSON
import matplotlib

matplotlib.use("Agg")  # Use a non-GUI backend
from skimage import measure, morphology
from shapely.geometry import LineString, MultiPoint
from tqdm.auto import tqdm
from osgeo import gdal

# Local project imports
from coastseg import common, exceptions, file_utilities, geodata_processing, plotting
from coastseg import core_utilities
from coastseg.model_info import ModelInfo
from coastseg.intersections import split_line
from coastsat import SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects

gdal.UseExceptions()

# Set pandas option
pd.set_option("mode.chained_assignment", None)

# Logger setup
logger = logging.getLogger(__name__)
__all__ = ["Extracted_Shoreline"]


class SegmentationFilter:
    """
    Handles optional filtering of segmentation files based on classifier model predictions.
    """

    def __init__(self, session_path: str):
        """
        Args:
            session_path (str): Path to the saved session directory containing segmentation results.
        """
        self.session_path = session_path

    def apply_filter(self, enable_filtering: bool = True) -> str:
        """
        Applies segmentation filtering using a classifier model if enabled.

        Args:
            enable_filtering (bool): Whether to apply filtering. Defaults to True.

        Returns:
            str: Directory path to the filtered segmentation files (could be same as session_path if not filtered).

        Logs:
            Warnings if filtering is skipped due to missing dependencies or errors.
        """
        if not enable_filtering:
            logger.info(
                "Segmentation filtering is disabled. Returning original session path."
            )
            return self.session_path

        try:
            from coastseg import classifier

            classifier.check_tensorflow()

            logger.info(
                f"Applying segmentation filter for session: {self.session_path}"
            )
            filtered_dir = classifier.filter_segmentations(self.session_path)
            logger.info(
                f"Segmentation filtering complete. Filtered files in: {filtered_dir}"
            )
            return filtered_dir

        except ImportError as e:
            logger.warning(
                f"Segmentation filter skipped: TensorFlow or 'classifier' module not available. Error: {e}"
            )
        except Exception as e:
            logger.warning(f"Segmentation filtering failed due to runtime error: {e}")

        # Return original path if filtering could not be applied
        return self.session_path


class MetadataManager:
    """
    Handles loading and filtering of metadata based on shoreline settings and session contents.
    """

    def __init__(self, shoreline_settings: dict):
        """
        Args:
            shoreline_settings (dict): Configuration for shoreline extraction.
                Requires 'inputs' key with 'filepath' and 'sitename'.
                Example:
                    {
                        "inputs": {
                            "filepath": "/path/to/data", # path to the data folder
                            "sitename": "example_site"   # Name of folder in the data folder
                        }
                    }

        """
        self.shoreline_settings = shoreline_settings

    def load_metadata(self) -> dict:
        """
        Loads the metadata from the session specified in the shoreline settings, then
        filters it based on the available RBG files.

        Returns:
            dict: Filtered metadata grouped by satellite.
        """
        inputs = self.shoreline_settings.get("inputs", {})
        if not inputs:
            inputs = self.shoreline_settings

        filepath = inputs.get("filepath")
        if not filepath:
            raise ValueError("The 'filepath' key in 'inputs' is missing or None.")
        filepath_data = get_data_folder(filepath)
        metadata = get_metadata(inputs, filepath_data)

        sitename = inputs.get("sitename")
        rgb_dir = os.path.join(
            filepath_data, sitename, "jpg_files", "preprocessed", "RGB"
        )
        # filter out files that were removed from RGB directory
        try:
            logger.info(f"Looking for RGB files in: {rgb_dir}")
            metadata = common.filter_metadata_with_dates(
                metadata, rgb_dir, file_type="jpg"
            )
        except FileNotFoundError:
            logger.warning(f"No RGB files found at {rgb_dir}")
            return {}

        self._log_metadata_summary(metadata)
        return metadata

    def filter_metadata_by_segmentations(
        self, metadata: dict, segmentation_dir: str, file_type: str = "npz"
    ) -> dict:
        """
        Filters metadata entries to match available segmentation files.

        Args:
            metadata (dict): Full metadata by satellite.
            segmentation_dir (str): Directory containing valid segmentations.
            file_type (str): File extension to filter by (default "npz").

        Returns:
            dict: Filtered metadata.
        """
        filtered = common.filter_metadata_with_dates(
            metadata, segmentation_dir, file_type
        )
        logger.info(
            f"Metadata filtered to {len(filtered)} satellites using files in '{segmentation_dir}'"
        )
        return filtered

    def _log_metadata_summary(self, metadata: dict):
        """
        Logs key stats for each satellite's metadata.

        Args:
            metadata (dict): Metadata by satellite.
        """
        for satname, satdata in metadata.items():
            if not satdata:
                logger.warning(f"metadata['{satname}'] is empty")
                continue

            def sample(lst, n=5):
                return list(islice(lst, n))

            logger.info(f"Satellite: {satname}")
            logger.info(f"  EPSG: {np.unique(satdata.get('epsg', []))}")
            logger.info(f"  Dates: Sample {sample(satdata.get('dates', []))}")
            logger.info(f"  Filenames: Sample {sample(satdata.get('filenames', []))}")
            logger.info(
                f"  Image Dimensions: {np.unique(satdata.get('im_dimensions', []))}"
            )
            logger.info(
                f"  Accuracy Georef: {np.unique(satdata.get('acc_georef', []))}"
            )
            logger.info(f"  Image Quality: {np.unique(satdata.get('im_quality', []))}")


def get_filepath(filepath_data, sitename, satname):
    """
    Create filepath to the different folders containing the satellite images.

    Originally by KV WRL 2018
    Modified by Sharon Batiste 2025

    Arguments:
    -----------
    'filepath_data': str
        filepath to the directory where the images are downloaded. Typically this is coastseg/data
    'sitename': str
        name of the site
    satname: str
        short name of the satellite mission ('L5','L7','L8','S2')

    Returns:
    -----------
    filepath: str or list of str
        contains the filepath(s) to the folder(s) containing the satellite images

    """
    # access the images
    if satname == "L5":
        # access downloaded Landsat 5 images
        fp_ms = os.path.join(filepath_data, sitename, satname, "ms")
        fp_mask = os.path.join(filepath_data, sitename, satname, "mask")
        filepath = [fp_ms, fp_mask]
    elif satname in ["L7", "L8", "L9"]:
        # access downloaded Landsat 7 images
        fp_ms = os.path.join(filepath_data, sitename, satname, "ms")
        fp_pan = os.path.join(filepath_data, sitename, satname, "pan")
        fp_mask = os.path.join(filepath_data, sitename, satname, "mask")
        filepath = [fp_ms, fp_pan, fp_mask]
    elif satname == "S2":
        # access downloaded Sentinel 2 images
        fp_ms = os.path.join(filepath_data, sitename, satname, "ms")
        fp_swir = os.path.join(filepath_data, sitename, satname, "swir")
        fp_mask = os.path.join(filepath_data, sitename, satname, "mask")
        filepath = [fp_ms, fp_swir, fp_mask]
    elif satname == "S1":
        # access downloaded Sentinel 1 images
        fp_vh = os.path.join(filepath_data, sitename, satname, "VH")
        filepath = [fp_vh]

    return filepath


def get_data_folder(data_path: str = "") -> str:
    """
    Returns the path to the data folder. If the provided data_path does not exist,
    it checks if the data folder exists in the current coastseg directory.

    Args:
        data_path (str): The path to the data folder. Default is an empty string.

    Returns:
        str: The path to the data folder.

    Raises:
        FileNotFoundError: If the provided data_path does not exist and the data folder
                            is not found in the current coastseg directory.
    """
    base_path = os.path.abspath(core_utilities.get_base_dir())
    data_folder = os.path.join(base_path, "data")

    if not os.path.exists(data_path):
        # check if the data folder exists in the current coastseg directory (this is for docker containers/ data downloaded on someone elses computer)
        if os.path.exists(data_folder):
            # convert the data folder to an absolute path
            return data_folder
            # return os.path.abspath(data_folder)
        raise FileNotFoundError(f"The directory {data_path} does not exist.")
    return data_path


def time_func(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds to run.")
        # logger.debug(f"{func.__name__} took {end - start:.6f} seconds to run.")
        return result

    return wrapper


def extract_timestamp_and_id(filename: str) -> Optional[Tuple[str, str]]:
    """
    Extracts the timestamp and Landsat ID from a given filename.

    Args:
        filename (str): The image filename in the format 'YYYYMMDD_LandsatID_rest.tif'.

    Returns:
        tuple[str, str] or None: A tuple containing (timestamp, landsat_id) if parsing is successful, else None.

    Example:
        >>> extract_timestamp_and_id("20220101_L8_xyz.tif")
        ('20220101', 'L8')
    """
    parts = filename.split("_")
    if len(parts) < 2:
        logger.warning(f"Invalid filename format: {filename}")
        return None
    return parts[0], parts[1]


def find_matching_npz(filename: str, directory: str) -> Optional[str]:
    """
    Finds a .npz file in the specified directory that matches the timestamp and Landsat ID from the filename.

    If the directory does not exist, it logs a debug message and returns None.
    If the filename does not contain a valid timestamp and Landsat ID, it logs a warning and returns None.

    Args:
        filename (str): The filename containing timestamp and Landsat ID.
        directory (str): The directory path where .npz files are located.

    Returns:
        str or None: Full path to the matching .npz file if found, else None.

    Example:
        >>> find_matching_npz("20220101_L8_xyz.tif", "/path/to/npz_files")
        '/path/to/npz_files/20220101_someinfo_L8_otherinfo.npz'
    """
    if not os.path.isdir(directory):
        logger.debug(f"Directory does not exist: {directory}")
        return None

    extracted = extract_timestamp_and_id(filename)
    if not extracted:
        return None

    timestamp, landsat_id = extracted
    pattern = f"{timestamp}*{landsat_id}*.npz"

    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, pattern):
            return os.path.join(directory, file)

    return None


def get_model_output_npz_path(filename: str, session_path: str) -> str:
    """
    Retrieves the path to a .npz file corresponding to the given image file.

    Search order:
    1. <session_path>/good/
    2. <session_path>/

    Args:
        filename (str): Image filename to find a matching .npz for.
        session_path (str): Base session directory containing 'good' subfolder or .npz files.

    Returns:
        str: Full path to the matched .npz file.

    Raises:
        FileNotFoundError: If no matching file is found and the session path is invalid.

    Example:
        >>> get_model_output_npz_path("20220101_L8_xyz.tif", "/data/session_01")
        '/data/session_01/good/20220101_abc_L8_xyz.npz'
    """
    search_paths = [
        os.path.join(session_path, "good"),
        session_path,
    ]

    for path in search_paths:
        npz_path = find_matching_npz(filename, path)
        if npz_path:
            logger.info(f"Found .npz file: {npz_path}")
            return npz_path

    if not os.path.isdir(session_path):
        raise FileNotFoundError(f"Session path does not exist: {session_path}")

    raise FileNotFoundError(
        f"No .npz file found for {filename} in expected directories."
    )


def parse_date_from_filename(filename: str) -> datetime.datetime:
    """
    Extracts and parses a UTC datetime from the start of a filename.

    Args:
        filename (str): The filename containing a datetime string at the start.

    Returns:
        datetime.datetime: The extracted datetime in UTC.

    Raises:
        ValueError: If the filename does not contain a valid date format.
    """
    try:
        date_str = filename[0:19]  # Extract first 19 characters (YYYY-MM-DDTHH:MM:SS)
        parsed_date = datetime.datetime(
            int(date_str[:4]),  # Year
            int(date_str[5:7]),  # Month
            int(date_str[8:10]),  # Day
            int(date_str[11:13]),  # Hour
            int(date_str[14:16]),  # Minute
            int(date_str[17:19]),  # Second
        )
        return pytz.utc.localize(parsed_date)
    except (ValueError, IndexError):
        raise ValueError(f"Invalid datetime format in filename: {filename}")


def read_metadata_file(filepath: str) -> Dict[str, Union[str, int, float]]:
    metadata_keys = [
        "filename",
        "epsg",
        "acc_georef",
        "im_quality",
        "im_width",
        "im_height",
    ]

    # Mapping of actual file keys to metadata keys
    key_mapping = {"image_quality": "im_quality"}

    # Initialize the metadata dictionary with default values.
    metadata = {
        "filename": "",
        "epsg": "",
        "acc_georef": -1,
        "im_quality": -1,
        "im_width": -1,
        "im_height": -1,
    }

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            parts = line.split("\t")
            if len(parts) < 2:
                continue  # Skip lines without a tab character

            key = parts[0].strip()
            value = parts[1].strip()

            # Map the actual key in the file to the metadata key
            key = key_mapping.get(key, key)

            # If the mapped key is not in metadata_keys, then skip it.
            if key not in metadata_keys:
                continue

            # Convert value to the appropriate type based on the key
            if key in ["epsg", "im_width", "im_height"]:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        print(
                            f"Error: Unable to convert {key} {value} to a numeric value."
                        )
            elif key in ["acc_georef", "im_quality"]:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep the value as a string if conversion to float fails

            # Update the metadata dictionary with the extracted key-value pair.
            metadata[key] = value

    return metadata


def format_date(date_str: Union[str, datetime.datetime]) -> datetime.datetime:
    """
    Converts a date string or datetime object to a datetime object in UTC timezone.

    Args:
        date_str (Union[str, datetime.datetime]): The date string or datetime object to be converted.

    Returns:
        datetime.datetime: The converted datetime object in UTC timezone.

    Raises:
        ValueError: If the date string is in an invalid format.
    """

    date_formats = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]

    # convert datetime object to string
    if isinstance(date_str, datetime.datetime):
        # converts the datetime object to a string
        date_str = date_str.strftime("%Y-%m-%dT%H:%M:%S")

    # format the string to a datetime object
    for date_format in date_formats:
        try:
            # creates a datetime object from a string with the date in UTC timezone
            start_date = datetime.datetime.strptime(date_str, date_format).replace(
                tzinfo=timezone.utc
            )
            return start_date
        except ValueError:
            pass  # Try the next format
    else:
        raise ValueError(f"Invalid date format: {date_str} not in {date_formats}")


def get_metadata(inputs, data_folder_location: str = ""):
    """
    Gets the metadata from the downloaded images by parsing .txt files located
    in the \meta subfolder.

    KV WRL 2018
    modified by Sharon Fitzpatrick 2023

    Arguments:
    -----------
    inputs: dict with the following fields
        'sitename': str
            name of the site
        'filepath_data': str
            filepath to the directory where the images are downloaded
    data_folder_location: str
        location of the data folder

    Returns:
    -----------
    metadata: dict
        contains the information about the satellite images that were downloaded:
        date, filename, georeferencing accuracy and image coordinate reference system

    """
    # Construct the directory path containing the images
    filepath = os.path.join(inputs["filepath"], inputs["sitename"])
    if not os.path.exists(filepath):
        if data_folder_location:
            filepath = os.path.join(data_folder_location, inputs["sitename"])
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"The directory {filepath} does not exist.")
        else:
            raise FileNotFoundError(f"The directory {filepath} does not exist.")
    # initialize metadata dict
    metadata = dict([])
    # loop through the satellite missions that were specified in the inputs
    satellite_list = inputs.get("sat_list", ["L5", "L7", "L8", "L9", "S2", "S1"])
    for satname in satellite_list:
        sat_path = os.path.join(filepath, satname)
        # if a folder has been created for the given satellite mission
        if satname in os.listdir(filepath):
            # update the metadata dict
            metadata[satname] = {
                "filenames": [],
                "dates": [],
                "epsg": [],
                "acc_georef": [],
                "im_quality": [],
                "im_dimensions": [],
            }
            # directory where the metadata .txt files are stored
            filepath_meta = os.path.join(sat_path, "meta")
            # get the list of filenames and sort it chronologically
            if not os.path.exists(filepath_meta):
                continue
            # Get the list of filenames and sort it chronologically
            filenames_meta = sorted(os.listdir(filepath_meta))
            # loop through the .txt files
            for im_meta in filenames_meta:
                if inputs.get("dates", None) is None:
                    raise ValueError("The 'dates' key is missing from the inputs.")

                start_date = format_date(inputs["dates"][0])
                end_date = format_date(inputs["dates"][1])
                input_date = parse_date_from_filename(im_meta)
                # if the image date is outside the specified date range, skip it
                if input_date < start_date or input_date > end_date:
                    continue
                meta_filepath = os.path.join(filepath_meta, im_meta)
                meta_info = read_metadata_file(meta_filepath)

                # Append meta info to the appropriate lists in the metadata dictionary
                metadata[satname]["filenames"].append(meta_info["filename"])
                metadata[satname]["acc_georef"].append(meta_info["acc_georef"])
                metadata[satname]["epsg"].append(meta_info["epsg"])
                metadata[satname]["dates"].append(
                    parse_date_from_filename(meta_info["filename"])
                )
                metadata[satname]["im_quality"].append(meta_info["im_quality"])
                # if the metadata file didn't contain im_height or im_width set this as an empty list
                if meta_info["im_height"] == -1 or meta_info["im_height"] == -1:
                    metadata[satname]["im_dimensions"].append([])
                else:
                    metadata[satname]["im_dimensions"].append(
                        [meta_info["im_height"], meta_info["im_width"]]
                    )

    # save a json file containing the metadata dict
    metadata_json = os.path.join(filepath, f"{inputs['sitename']}_metadata.json")
    SDS_preprocess.write_to_json(metadata_json, metadata)

    return metadata


def log_contents_of_shoreline_dict(extracted_shorelines_dict):
    # Check and log 'reference shoreline' if it exists
    shorelines_array = extracted_shorelines_dict.get("shorelines", np.array([]))
    if isinstance(shorelines_array, np.ndarray):
        logger.info(f"shorelines.shape: {shorelines_array.shape}")
    logger.info(f"Number of 'shorelines': {len(shorelines_array)}")
    # ------------------------------------------------

    logger.info(
        f"extracted_shorelines_dict length {len(extracted_shorelines_dict.get('dates',[]))} of dates: {list(islice(extracted_shorelines_dict.get('dates',[]),3))}"
    )
    logger.info(
        f"extracted_shorelines_dict length {len(extracted_shorelines_dict.get('satname',[]))} of satname: {np.unique(extracted_shorelines_dict.get('satname',[]))}"
    )
    logger.info(
        f"extracted_shorelines_dict length {len(extracted_shorelines_dict.get('geoaccuracy',[]))} of geoaccuracy: {np.unique(extracted_shorelines_dict.get('geoaccuracy',[]))}"
    )
    logger.info(
        f"extracted_shorelines_dict length {len(extracted_shorelines_dict.get('cloud_cover',[]))} of cloud_cover: {np.unique(extracted_shorelines_dict.get('cloud_cover',[]))}"
    )
    logger.info(
        f"extracted_shorelines_dict length {len(extracted_shorelines_dict.get('filename',[]))} of filename[:3]: {list(islice(extracted_shorelines_dict.get('filename',[]),3))}"
    )


# preprocess_single
# Main function to preprocess a satellite image (L5, L7, L8, L9 or S2)
def preprocess_single(
    fn,
    satname,
    cloud_mask_issue,
    pan_off,
    collection="C02",
    do_cloud_mask=True,
    s2cloudless_prob=60,
):
    """
    Reads the image and outputs the pansharpened/down-sampled multispectral bands,
    the georeferencing vector of the image (coordinates of the upper left pixel),
    the cloud mask, the QA band and a no_data image.
    For Landsat 7-8 it also outputs the panchromatic band and for Sentinel-2 it
    also outputs the 20m SWIR band.

    KV WRL 2018
    Modified by Sharon Fitzpatrick Batiste 2025

    Arguments:
    -----------
    fn: str or list of str
        filename of the .TIF file containing the image. For L7, L8 and S2 this
        is a list of filenames, one filename for each band at different
        resolution (30m and 15m for Landsat 7-8, 10m, 20m, 60m for Sentinel-2)
    satname: str
        name of the satellite mission (e.g., 'L5')
    cloud_mask_issue: boolean
        True if there is an issue with the cloud mask and sand pixels are being masked on the images
    pan_off : boolean
        if True, disable panchromatic sharpening and ignore pan band
    collection: str
        Landsat collection 'C02'
    do_cloud_mask: boolean
        if True, apply the cloud mask to the image. If False, the cloud mask is not applied.
    s2cloudless_prob: float [0,100)
            threshold to identify cloud pixels in the s2cloudless probability mask

    Returns:
    -----------
    im_ms: np.array
        3D array containing the pansharpened/down-sampled bands (B,G,R,NIR,SWIR1)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale] defining the
        coordinates of the top-left pixel of the image
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_extra : np.array
        2D array containing the 20m resolution SWIR band for Sentinel-2 and the 15m resolution
        panchromatic band for Landsat 7 and Landsat 8. This field is empty for Landsat 5.
    im_QA: np.array
        2D array containing the QA band, from which the cloud_mask can be computed.
    im_nodata: np.array
        2D array with True where no data values (-inf) are located

    """
    if isinstance(fn, list):
        fn_to_split = fn[0]
    elif isinstance(fn, str):
        fn_to_split = fn
    # split by os.sep and only get the filename at the end then split again to remove file extension
    fn_to_split = fn_to_split.split(os.sep)[-1].split(".")[0]
    # search for the year the tif was taken with regex and convert to int
    year = int(re.search("[0-9]+", fn_to_split).group(0))
    # after 2022 everything is automatically from Collection 2
    if collection == "C01" and year >= 2022:
        collection = "C02"

    # =============================================================================================#
    # L5 images
    # =============================================================================================#
    if satname == "L5":
        # filepaths to .tif files
        fn_ms = fn[0]
        fn_mask = fn[1]
        # read ms bands
        data = gdal.Open(fn_ms, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = SDS_preprocess.read_bands(fn_ms)
        im_ms = np.stack(bands, 2)
        # read cloud mask
        im_QA = SDS_preprocess.read_bands(fn_mask)[0]
        if not do_cloud_mask:
            cloud_mask = SDS_preprocess.create_cloud_mask(
                im_QA, satname, cloud_mask_issue, collection
            )
            # add pixels with -inf or nan values on any band to the nodata mask
            im_nodata = SDS_preprocess.get_nodata_mask(im_ms, cloud_mask.shape)
            # cloud mask is the same as the no data mask
            cloud_mask = im_nodata.copy()
            # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
            # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
            im_zeros = SDS_preprocess.get_zero_pixels(im_ms, cloud_mask.shape)
            # add zeros to im nodata
            im_nodata = np.logical_or(im_zeros, im_nodata)
        else:
            cloud_mask = SDS_preprocess.create_cloud_mask(
                im_QA, satname, cloud_mask_issue, collection
            )
            # add pixels with -inf or nan values on any band to the nodata mask
            im_nodata = SDS_preprocess.get_nodata_mask(im_ms, cloud_mask.shape)
            # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
            # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
            im_zeros = SDS_preprocess.get_zero_pixels(im_ms, cloud_mask.shape)
            # add zeros to im nodata
            im_nodata = np.logical_or(im_zeros, im_nodata)
            # update cloud mask with all the nodata pixels
            cloud_mask = np.logical_or(cloud_mask, im_nodata)

        # no extra image for Landsat 5 (they are all 30 m bands)
        im_extra = []
    # =============================================================================================#
    # L7, L8 and L9 images
    # =============================================================================================#
    elif satname in ["L7", "L8", "L9"]:
        # filepaths to .tif files
        fn_ms = fn[0]
        fn_pan = fn[1]
        fn_mask = fn[2]
        # read ms bands
        data = gdal.Open(fn_ms, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = SDS_preprocess.read_bands(fn_ms)
        im_ms = np.stack(bands, 2)
        # read cloud mask and get the QA from the first band
        im_QA = SDS_preprocess.read_bands(fn_mask)[0]

        if not do_cloud_mask:
            cloud_mask = SDS_preprocess.create_cloud_mask(
                im_QA, satname, cloud_mask_issue, collection
            )
            # add pixels with -inf or nan values on any band to the nodata mask
            im_nodata = SDS_preprocess.get_nodata_mask(im_ms, cloud_mask.shape)
            # cloud mask is the no data mask
            cloud_mask = im_nodata.copy()
            # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
            # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
            im_zeros = SDS_preprocess.get_zero_pixels(im_ms, cloud_mask.shape)
            # add zeros to im nodata
            im_nodata = np.logical_or(im_zeros, im_nodata)
        else:
            cloud_mask = SDS_preprocess.create_cloud_mask(
                im_QA, satname, cloud_mask_issue, collection
            )
            # add pixels with -inf or nan values on any band to the nodata mask
            im_nodata = SDS_preprocess.get_nodata_mask(im_ms, cloud_mask.shape)
            # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
            # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
            im_zeros = SDS_preprocess.get_zero_pixels(im_ms, cloud_mask.shape)
            # add zeros to im nodata
            im_nodata = np.logical_or(im_zeros, im_nodata)
            # update cloud mask with all the nodata pixels
            cloud_mask = np.logical_or(cloud_mask, im_nodata)

        # if panchromatic sharpening is turned off
        if pan_off:
            # ms bands are untouched and the extra image is empty
            im_extra = []

        # otherwise perform panchromatic sharpening
        else:
            # read panchromatic band
            data = gdal.Open(fn_pan, gdal.GA_ReadOnly)
            georef = np.array(data.GetGeoTransform())
            bands = [
                data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)
            ]
            im_pan = bands[0]

            # pansharpen all of the landsat 7 bands
            if satname == "L7":
                try:
                    im_ms_ps = SDS_preprocess.pansharpen(
                        im_ms[:, :, [0, 1, 2, 3, 4]], im_pan, cloud_mask
                    )
                except:  # if pansharpening fails, keep downsampled bands (for long runs)
                    print("\npansharpening of image %s failed." % fn[0])
                    im_ms_ps = im_ms[:, :, [1, 2, 3]]
                    # add downsampled Blue and SWIR1 bands
                    im_ms_ps = np.append(im_ms[:, :, [0]], im_ms_ps, axis=2)
                    im_ms_ps = np.append(im_ms_ps, im_ms[:, :, [4]], axis=2)

                im_ms = im_ms_ps.copy()
                # the extra image is the 15m panchromatic band
                im_extra = im_pan

            # pansharpen Blue, Green, Red for Landsat 8 and 9
            elif satname in ["L8", "L9"]:
                try:
                    im_ms_ps = SDS_preprocess.pansharpen(
                        im_ms[:, :, [0, 1, 2, 3, 4]], im_pan, cloud_mask
                    )
                except:  # if pansharpening fails, keep downsampled bands (for long runs)
                    print("\npansharpening of image %s failed." % fn[0])
                    im_ms_ps = im_ms[:, :, [0, 1, 2]]
                    # add downsampled NIR and SWIR1 bands
                    im_ms_ps = np.append(im_ms_ps, im_ms[:, :, [3, 4]], axis=2)

                im_ms = im_ms_ps.copy()
                # the extra image is the 15m panchromatic band
                im_extra = im_pan

    # =============================================================================================#
    # S2 images
    # =============================================================================================#
    if satname == "S2":
        # read 10m bands (R,G,B,NIR)
        fn_ms = fn[0]
        data = gdal.Open(fn_ms, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = SDS_preprocess.read_bands(fn_ms, satname)
        im_ms = np.stack(bands, 2)
        im_ms = im_ms / 10000  # TOA scaled to 10000
        # read s2cloudless cloud probability (last band in ms image)
        cloud_prob = data.GetRasterBand(data.RasterCount).ReadAsArray()

        # image size
        nrows = im_ms.shape[0]
        ncols = im_ms.shape[1]
        # if image contains only zeros (can happen with S2), skip the image
        if sum(sum(sum(im_ms))) < 1:
            im_ms = []
            georef = []
            # skip the image by giving it a full cloud_mask
            cloud_mask = np.ones((nrows, ncols)).astype("bool")
            return im_ms, georef, cloud_mask, [], [], []

        # read 20m band (SWIR1) from the first band
        fn_swir = fn[1]
        im_swir = SDS_preprocess.read_bands(fn_swir)[0] / 10000  # TOA scaled to 10000
        im_swir = np.expand_dims(im_swir, axis=2)

        # append down-sampled SWIR1 band to the other 10m bands
        im_ms = np.append(im_ms, im_swir, axis=2)

        # create cloud mask using 60m QA band (not as good as Landsat cloud cover)
        fn_mask = fn[2]
        im_QA = SDS_preprocess.read_bands(fn_mask)[0]
        if not do_cloud_mask:
            cloud_mask_QA60 = SDS_preprocess.create_cloud_mask(
                im_QA, satname, cloud_mask_issue, collection
            )
            # compute cloud mask using s2cloudless probability band
            cloud_mask_s2cloudless = SDS_preprocess.create_s2cloudless_mask(
                cloud_prob, s2cloudless_prob
            )
            # combine both cloud masks
            cloud_mask = np.logical_or(cloud_mask_QA60, cloud_mask_s2cloudless)
            # add pixels with -inf or nan values on any band to the nodata mask
            im_nodata = SDS_preprocess.get_nodata_mask(im_ms, cloud_mask.shape)
            im_nodata = SDS_preprocess.pad_edges(im_swir, im_nodata)
            # cloud mask is the no data mask
            cloud_mask = im_nodata.copy()
            # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
            # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
            im_zeros = SDS_preprocess.get_zero_pixels(im_ms, cloud_mask.shape)
            # add zeros to im nodata
            im_nodata = np.logical_or(im_zeros, im_nodata)
            if "merged" in fn_ms:
                im_nodata = morphology.dilation(im_nodata, morphology.square(5))
            # update cloud mask with all the nodata pixels
            # v0.1.40 change : might be bug
            cloud_mask = np.logical_or(cloud_mask, im_nodata)
        else:  # apply the cloud mask
            # compute cloud mask using QA60 band
            cloud_mask_QA60 = SDS_preprocess.create_cloud_mask(
                im_QA, satname, cloud_mask_issue, collection
            )
            # compute cloud mask using s2cloudless probability band
            cloud_mask_s2cloudless = SDS_preprocess.create_s2cloudless_mask(
                cloud_prob, s2cloudless_prob
            )
            # combine both cloud masks
            cloud_mask = np.logical_or(cloud_mask_QA60, cloud_mask_s2cloudless)
            # add pixels with -inf or nan values on any band to the nodata mask
            im_nodata = SDS_preprocess.get_nodata_mask(im_ms, cloud_mask.shape)
            im_nodata = SDS_preprocess.pad_edges(im_swir, im_nodata)
            # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
            # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
            im_zeros = SDS_preprocess.get_zero_pixels(im_ms, cloud_mask.shape)
            # add zeros to im nodata
            im_nodata = np.logical_or(im_zeros, im_nodata)
            # update cloud mask with all the nodata pixels
            cloud_mask = np.logical_or(cloud_mask, im_nodata)
            if "merged" in fn_ms:
                im_nodata = morphology.dilation(im_nodata, morphology.square(5))
            # move cloud mask to above if statement to avoid bug in v0.1.40

        # no extra image
        im_extra = []

    return im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata


def check_percent_no_data_allowed(
    percent_no_data_allowed: float, cloud_mask: np.ndarray, im_nodata: np.ndarray
) -> bool:
    """
    Checks if the percentage of no data pixels in the image exceeds the allowed percentage.

    Args:
        settings (dict): A dictionary containing settings for the shoreline extraction.
        cloud_mask (numpy.ndarray): A binary mask indicating cloud cover in the image.
        im_nodata (numpy.ndarray): A binary mask indicating no data pixels in the image.

    Returns:
        bool: True if the percentage of no data pixels is less than or equal to the allowed percentage, False otherwise.
    """
    if percent_no_data_allowed is not None:
        num_total_pixels = cloud_mask.shape[0] * cloud_mask.shape[1]
        percentage_no_data = np.sum(im_nodata) / num_total_pixels
        if percentage_no_data > percent_no_data_allowed:
            logger.info(
                f"percent_no_data_allowed exceeded {percentage_no_data} > {percent_no_data_allowed}"
            )
            return False
    return True


def convert_linestrings_to_multipoints(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert LineString geometries in a GeoDataFrame to MultiPoint geometries.

    Args:
    - gdf (gpd.GeoDataFrame): The input GeoDataFrame.

    Returns:
    - gpd.GeoDataFrame: A new GeoDataFrame with MultiPoint geometries. If the input GeoDataFrame
                        already contains MultiPoints, the original GeoDataFrame is returned.
    """

    # Check if the gdf already contains MultiPoints
    if any(gdf.geometry.type == "MultiPoint"):
        return gdf

    def linestring_to_multipoint(linestring):
        if isinstance(linestring, LineString):
            return MultiPoint(linestring.coords)
        return linestring

    # Convert each LineString to a MultiPoint
    gdf["geometry"] = gdf["geometry"].apply(linestring_to_multipoint)

    return gdf


def transform_gdf_to_crs(gdf, crs=4326):
    """Convert the GeoDataFrame to the specified CRS."""
    return gdf.to_crs(crs)


def select_and_stringify(gdf, row_number):
    """Select a single shoreline and stringify its datetime columns."""
    single_shoreline = gdf.iloc[[row_number]]
    return common.stringify_datetime_columns(single_shoreline)


def convert_gdf_to_json(gdf):
    """Convert a GeoDataFrame to a JSON representation."""
    return json.loads(gdf.to_json())


def style_layer(
    geojson: dict, layer_name: str, color: str, style_dict: dict = {}
) -> GeoJSON:
    """Return styled GeoJson object with layer name
    Args:
        geojson (dict): geojson dictionary to be styled
        layer_name(str): name of the GeoJSON layer
        color(str): hex code or name of color render shorelines
        style_dict (dict, optional): Additional style attributes to be merged with the default style.
    Returns:
        "ipyleaflet.GeoJSON": shoreline as GeoJSON layer styled with color
    """
    assert geojson != {}, "ERROR.\n Empty geojson cannot be drawn onto  map"
    # Default style dictionary
    default_style = {
        "color": color,  # Outline color
        "opacity": 1,  # opacity 1 means no transparency
        "weight": 3,  # Width
        "fillColor": color,  # Fill color
        "fillOpacity": 0.8,  # Fill opacity.
        "radius": 1,
    }

    # If a style_dict is provided, merge it with the default style
    default_style.update(style_dict)
    return GeoJSON(
        data=geojson, name=layer_name, style=default_style, point_style=default_style
    )


def read_from_dict(d: dict, keys_of_interest: list | set | tuple):
    """
    Function to extract the value from the first matching key in a dictionary.

    Parameters:
    d (dict): The dictionary from which to extract the value.
    keys_of_interest (list | set | tuple): Iterable of keys to look for in the dictionary.
    The function returns the value of the first matching key it finds.

    Returns:
    The value from the dictionary corresponding to the first key found in keys_of_interest,
    or None if no matching keys are found.
    Raises:
    KeyError if the keys_of_interest were not in d
    """
    for key in keys_of_interest:
        if key in d:
            return d[key]
    raise KeyError(f"{keys_of_interest} were not in {d}")


def remove_small_objects_and_binarize(merged_labels, min_size):
    # Ensure the image is binary
    binary_image = merged_labels > 0

    # Remove small objects from the binary image
    filtered_image = morphology.remove_small_objects(
        binary_image, min_size=min_size, connectivity=2
    )

    return filtered_image


def compute_transects_from_roi(
    extracted_shorelines: dict,
    transects_gdf: gpd.GeoDataFrame,
    settings: dict,
) -> dict:
    """Computes the intersection between the 2D shorelines and the shore-normal.
        transects. It returns time-series of cross-shore distance along each transect.
    Args:
        extracted_shorelines (dict): contains the extracted shorelines and corresponding metadata
        transects_gdf (gpd.GeoDataFrame): transects in ROI with crs = output_crs in settings
        settings (dict): settings dict with keys
                    'along_dist': int
                        alongshore distance considered calculate the intersection
    Returns:
        dict:  time-series of cross-shore distance along each of the transects.
               Not tidally corrected.
    """
    # create dict of numpy arrays of transect start and end points

    transects = common.get_transect_points_dict(transects_gdf)
    # cross_distance: along-shore distance over which to consider shoreline points to compute median intersection (robust to outliers)
    cross_distance = SDS_transects.compute_intersection_QC(
        extracted_shorelines, transects, settings
    )
    return cross_distance


def combine_satellite_data(satellite_data: dict) -> dict:
    """
    Function to merge the satellite_data dictionary, which has one key per satellite mission
    into a dictionnary containing all the shorelines and dates ordered chronologically.

    Arguments:
    -----------
    satellite_data: dict
        contains the extracted shorelines and corresponding dates, organised by
        satellite mission

    Returns:
    -----------
    merged_satelllite_data: dict
        contains the extracted shorelines in a single list sorted by date

    """
    # Initialize merged_satellite_data dict
    merged_satellite_data = {
        "dates": [],
        "geoaccuracy": [],
        "shorelines": [],
        "idx": [],
        "satname": [],
    }

    # Iterate through satellite_data keys (satellite names)
    for satname in satellite_data:
        # Iterate through each key in the nested dictionary
        for key in satellite_data[satname].keys():
            # Add the key to merged_satellite_data if it doesn't already exist
            if key not in merged_satellite_data:
                merged_satellite_data[key] = []

    # Add an additional key for the satellite name
    merged_satellite_data["satname"] = []

    # Fill the satellite_data dict
    for satname, sat_data in satellite_data.items():
        satellite_data[satname].setdefault("dates", [])
        satellite_data[satname].setdefault("geoaccuracy", [])
        satellite_data[satname].setdefault("shorelines", [])
        satellite_data[satname].setdefault("cloud_cover", [])
        satellite_data[satname].setdefault("filename", [])
        satellite_data[satname].setdefault("idx", [])
        satellite_data[satname].setdefault("dates", [])
        satellite_data[satname].setdefault("geoaccuracy", [])
        satellite_data[satname].setdefault("shorelines", [])
        satellite_data[satname].setdefault("cloud_cover", [])
        satellite_data[satname].setdefault("filename", [])
        satellite_data[satname].setdefault("idx", [])
        # For each key in the nested dictionary
        for key, value in sat_data.items():
            # Wrap non-list values in a list and concatenate to merged_satellite_data
            if not isinstance(value, list):
                merged_satellite_data[key] += [value]
            else:
                merged_satellite_data[key] += value
            # Add the satellite name to the satellite name list
        if "dates" in satellite_data[satname].keys():
            merged_satellite_data["satname"] += [
                _ for _ in np.tile(satname, len(satellite_data[satname]["dates"]))
            ]
    # Sort dates chronologically
    if "dates" in merged_satellite_data.keys():
        idx_sorted = sorted(
            range(len(merged_satellite_data["dates"])),
            key=lambda i: merged_satellite_data["dates"][i],
        )
        for key in merged_satellite_data.keys():
            merged_satellite_data[key] = [
                merged_satellite_data[key][i] for i in idx_sorted
            ]

    return merged_satellite_data


def process_satellite(
    satname: str,
    settings: dict,
    metadata: dict,
    session_path: str,
    class_indices: Optional[List[int]] = None,
    class_mapping: Optional[Dict[int, str]] = None,
    save_location: str = "",
    batch_size: int = 10,
    shoreline_extraction_area: Optional[gpd.GeoDataFrame] = None,
    **kwargs: dict,
):
    """
    Processes a satellite's imagery to extract shorelines.

    Args:
        satname (str): The name of the satellite.
        settings (dict): A dictionary containing settings for the shoreline extraction.
            Settings needed to extract shorelines
            Must contain the following keys
            'min_length_sl': int
                minimum length of shoreline to be considered
            'min_beach_area': int
                minimum area of beach to be considered
            'cloud_thresh': float
                maximum cloud cover allowed
            'cloud_mask_issue': bool
                whether to apply the cloud mask or not
            'along_dist': int
                alongshore distance considered calculate the intersection
        metadata (dict): A dictionary containing metadata for the satellite imagery.
            Metadata is a dictionary created by reading the metadata files in the meta folder for each satellite.
            The metadata dictionary should have the following structure:
            ex.
            metadata = {
                "l8": {
                    "dates": ["2019-01-01", "2019-01-02"],
                    "filenames": ["2019-01-01_123456789.tif", "2019-01-02_123456789.tif", "2019-01-03_123456789.tif"],
                    "epsg": [32601, 32601, 32601],
                    "acc_georef": [True, True, True]
                },
        session_path (str): The path to the session directory.
        class_indices (list, optional): A list of class indices to extract. Defaults to None.
        class_mapping (dict, optional): A dictionary mapping class indices to class names. Defaults to None.
        save_location (str, optional): The path to save the extracted shorelines. Defaults to "".
        batch_size (int, optional): The number of images to process in each batch. Defaults to 10.
        shoreline_extraction_area (gpd.GeoDataFrame, optional): A GeoDataFrame containing the extraction area for the shorelines. Defaults to None.
    Returns:
        dict: A dictionary containing the extracted shorelines for the satellite.
    """
    # filenames of tifs (ms) for this satellite
    filenames = metadata[satname]["filenames"]
    output = {}
    output.setdefault(satname, {})
    output[satname].setdefault("dates", [])
    output[satname].setdefault("geoaccuracy", [])
    output[satname].setdefault("shorelines", [])
    output[satname].setdefault("cloud_cover", [])
    output[satname].setdefault("filename", [])
    output[satname].setdefault("idx", [])

    if len(filenames) == 0:
        logger.warning(f"Satellite {satname} had no imagery")
        return output

    collection = settings["inputs"]["landsat_collection"]
    # deep copy settings
    settings = copy.deepcopy(settings)

    # this will get the location of the data folder where the downloaded data exists
    data_location = get_data_folder(settings.get("inputs", {}).get("filepath", ""))
    logger.info(f"Data location {data_location}")
    sitename = settings["inputs"]["sitename"]
    filepath = get_filepath(data_location, sitename, satname)
    logger.info(f"Loading data from {filepath}")
    pixel_size = SDS_shoreline.get_pixel_size_for_satellite(satname)

    # get the minimum beach area in number of pixels depending on the satellite
    settings["min_length_sl"] = get_min_shoreline_length(
        satname, settings["min_length_sl"]
    )

    # loop through the images
    espg_list = []
    geoaccuracy_list = []
    timestamps = []
    tasks = []

    # compute number of batches
    num_batches = len(filenames) // batch_size
    if len(filenames) % batch_size != 0:
        num_batches += 1

    # initialize progress bar
    pbar = tqdm(
        total=len(filenames),
        desc=f"Mapping Shorelines for {satname}",
        leave=True,
        position=0,
    )

    for batch in range(num_batches):
        espg_list = []
        geoaccuracy_list = []
        timestamps = []
        tasks = []

        # generate tasks for the current batch
        for index in range(
            batch * batch_size, min((batch + 1) * batch_size, len(filenames))
        ):
            image_epsg = metadata[satname]["epsg"][index]
            espg_list.append(image_epsg)
            geoaccuracy_list.append(metadata[satname]["acc_georef"][index])
            timestamps.append(metadata[satname]["dates"][index])
            tasks.append(
                dask.delayed(process_satellite_image)(
                    filenames[index],
                    filepath,
                    settings,
                    satname,
                    collection,
                    image_epsg,
                    pixel_size,
                    session_path,
                    class_indices,
                    class_mapping,
                    save_location,
                    settings.get("apply_cloud_mask", True),
                    shoreline_extraction_area,
                )
            )

        # compute tasks in batches
        results = dask.compute(*tasks)
        # update progress bar
        num_tasks_computed = len(tasks)
        pbar.update(num_tasks_computed)

        for index, result in enumerate(results):
            if result is None:
                continue
            output.setdefault(satname, {})
            output[satname].setdefault("dates", []).append(timestamps[index])
            output[satname].setdefault("geoaccuracy", []).append(
                geoaccuracy_list[index]
            )
            output[satname].setdefault("shorelines", []).append(result["shorelines"])
            output[satname].setdefault("cloud_cover", []).append(result["cloud_cover"])
            output[satname].setdefault("filename", []).append(filenames[index])
            output[satname].setdefault("idx", []).append(index)

    pbar.close()
    return output


def get_cloud_cover_combined(cloud_mask: np.ndarray):
    """
    Calculate the cloud cover percentage of a cloud_mask.
    Note: The way that cloud_mask is created in SDS_preprocess.preprocess_single() means that it will contain 1's where no data pixels were detected.
    TLDR the cloud mask is combined with the no data mask. No idea why.

    Parameters:
    cloud_mask (numpy.ndarray): A 2D numpy array with 0s (clear) and 1s (cloudy) representing the cloud mask.

    Returns:
    float: The percentage of cloud_cover_combined in the cloud_mask.
    """
    # Convert cloud_mask to integer and calculate the sum of all elements (number of cloudy pixels)
    num_cloudy_pixels = sum(sum(cloud_mask.astype(int)))

    # Calculate the total number of pixels in the cloud_mask
    num_total_pixels = cloud_mask.shape[0] * cloud_mask.shape[1]

    # Divide the number of cloudy pixels by the total number of pixels to get the cloud_cover_combined percentage
    cloud_cover_combined = np.divide(num_cloudy_pixels, num_total_pixels)

    return cloud_cover_combined


def get_cloud_cover(cloud_mask: np.ndarray, im_nodata: np.ndarray) -> float:
    """
    Calculate the cloud cover percentage in an image, considering only valid (non-no-data) pixels.

    Args:
    cloud_mask (numpy.array): A boolean 2D numpy array where True represents a cloud pixel,
        and False a non-cloud pixel.
    im_nodata (numpy.array): A boolean 2D numpy array where True represents a no-data pixel,
        and False a valid (non-no-data) pixel.

    Returns:
    float: The cloud cover percentage in the image (0-1), considering only valid (non-no-data) pixels.
    """

    # Remove no data pixels from the cloud mask, as they should not be included in the cloud cover calculation
    cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)

    # Compute updated cloud cover percentage without considering no data pixels
    cloud_cover = np.divide(
        sum(sum(cloud_mask_adv.astype(int))),
        (sum(sum((~im_nodata).astype(int)))),
    )

    return cloud_cover


def process_satellite_image(
    filename: str,
    filepath: str,
    settings: Dict[str, Union[str, int, float, bool]],
    satname: str,
    collection: str,
    image_epsg: int,
    pixel_size: float,
    session_path: str,
    class_indices: Optional[List[int]] = None,
    class_mapping: Optional[Dict[int, str]] = None,
    save_location: str = "",
    apply_cloud_mask: bool = True,
    shoreline_extraction_area: Optional[gpd.GeoDataFrame] = None,
) -> Optional[Dict[str, Union[gpd.GeoDataFrame, float]]]:
    """
    Processes a single satellite image to extract the shoreline.

    Args:
        filename (str): The filename of the image.
        filepath (str): The path to the directory containing the image.
        settings (dict): A dictionary containing settings for the shoreline extraction.
        satname (str): The name of the satellite.
        collection (str): The name of the Landsat collection.
        image_epsg (int): The EPSG code of the image.
        pixel_size (float): The pixel size of the image.
        session_path (str): The path to the session directory.
        class_indices (list, optional): A list of class indices to extract. Defaults to None.
        class_mapping (dict, optional): A dictionary mapping class indices to class names. Defaults to None.
        save_location (str, optional): The path to save the extracted shorelines. Defaults to "".
        apply_cloud_mask (bool, optional): Whether to apply the cloud mask. Defaults to True.
        shoreline_extraction_area (gpd.GeoDataFrame, optional): A GeoDataFrame containing the extraction area for the shorelines. Defaults to None.

    Returns:
        dict: A dictionary containing the extracted shoreline and cloud cover percentage.
         Returns None if the image is skipped due to high cloud cover or no data percentage.
    """
    # get image date
    date = filename[:19]
    # get the filenames for each of the tif files (ms, pan, qa)
    fn = SDS_tools.get_filenames(filename, filepath, satname)
    settings = settings.copy()
    settings["apply_cloud_mask"] = apply_cloud_mask

    # get the image as an array, its georeference,cloud mask and nodata mask
    im_ms, georef, cloud_mask, im_nodata = SDS_preprocess.preprocess_image(
        fn, satname, settings=settings, collection=collection
    )

    # if percentage of no data pixels are greater than allowed, skip
    percent_no_data_allowed = settings.get(
        "percent_no_data", None
    )  # @todo I think this should be 1.0 (aka 100%)  if nothing is set
    if not check_percent_no_data_allowed(
        percent_no_data_allowed, cloud_mask, im_nodata
    ):
        logger.info(
            f"percent_no_data_allowed > {settings.get('percent_no_data', None)}: {filename}"
        )
        return None

    # compute cloud_cover percentage (with no data pixels)
    cloud_cover_combined = get_cloud_cover_combined(cloud_mask)
    if cloud_cover_combined > 0.99:  # if 99% of cloudy pixels in image skip
        logger.info(f"cloud_cover_combined > 0.99 : {filename} ")
        return None

    # compute cloud cover percentage (without no data pixels)
    cloud_cover = get_cloud_cover(cloud_mask, im_nodata)
    # skip image if cloud cover is above user-defined threshold
    cloud_thresh = float(settings.get("cloud_thresh", 1.0))
    if cloud_cover > cloud_thresh:
        logger.info(f"Cloud thresh exceeded for {filename}")
        return None
    # calculate a buffer around the reference shoreline (if any has been digitised)
    # buffer is dilated by 5 pixels
    ref_shoreline_buffer = SDS_shoreline.create_shoreline_buffer(
        cloud_mask.shape, georef, image_epsg, pixel_size, settings
    )

    # read the model outputs from the npz file for this image
    try:
        npz_file = get_model_output_npz_path(filename, session_path)
    except FileNotFoundError as e:
        logger.warning(f"npz file not found for {filename} due to {e}")
        return None

    # get the labels for water and land
    land_mask = load_merged_image_labels(npz_file, class_indices=class_indices)
    all_labels = load_image_labels(npz_file)

    min_beach_area = settings["min_beach_area"]
    land_mask = remove_small_objects_and_binarize(land_mask, min_beach_area)

    # get the shoreline from the image
    shoreline = find_shoreline(
        fn,
        image_epsg,
        settings,
        np.logical_xor(cloud_mask, im_nodata),
        cloud_mask,
        im_nodata,
        georef,
        land_mask,
        ref_shoreline_buffer,
    )
    if shoreline is None:
        logger.warning(f"\nShoreline not found for {fn}")
        return None

    # convert the polygon coordinates of ROI to gdf
    height, width = im_ms.shape[:2]
    output_epsg = settings["output_epsg"]
    # create a geodataframe from the image extent
    roi_gdf = SDS_preprocess.create_gdf_from_image_extent(
        height, width, georef, image_epsg, output_epsg
    )
    # filter shorelines within the extraction area

    shoreline = SDS_shoreline.filter_shoreline(
        shoreline, shoreline_extraction_area, output_epsg
    )
    shoreline_extraction_area_array = (
        SDS_shoreline.get_extract_shoreline_extraction_area_array(
            shoreline_extraction_area, output_epsg, roi_gdf
        )
    )

    # plot the results
    plotting.plot_detection(
        im_ms,
        all_labels,
        land_mask,
        shoreline,
        image_epsg,
        georef,
        settings,
        date,
        satname=satname,
        class_mapping=class_mapping,
        im_ref_buffer=ref_shoreline_buffer,
        save_location=save_location,
        cloud_mask=cloud_mask,
        shoreline_extraction_area=shoreline_extraction_area_array,
        is_sar=True if "S1" in satname.upper() else False,
    )
    # create dictionary of output
    output = {
        "shorelines": shoreline,
        "cloud_cover": cloud_cover,
    }
    return output


def get_model_card_classes(
    model_card_path: str, default_class_mapping: Optional[dict[int, str]] = None
) -> dict:
    """
    Return the classes dictionary from the model card

    Example classes dictionary {0: 'sand', 1: 'sediment', 2: 'whitewater', 3: 'water'}

    Args:
        model_card_path (str): Path to the model card JSON file.
        default_class_mapping (dict[int | str, str], optional): A fallback class mapping to use
            if the model card doesn't include one. Keys can be str or int and will be cast to int.

    Returns:
        dict[int, str]: Dictionary mapping class indices (as integers) to class names.
    """
    if default_class_mapping is None:
        default_class_mapping = {
            0: "water",
            1: "whitewater",
            2: "sediment",
            3: "other",
        }

    # Ensure default_class_mapping keys are ints
    default_class_mapping = {int(k): v for k, v in default_class_mapping.items()}

    model_card_data = file_utilities.read_json_file(model_card_path, raise_error=True)

    # read the classes the model was trained with from either the dictionary under key "DATASET" or "DATASET1"
    try:
        dataset_info = common.get_value_by_key_pattern(
            model_card_data, ("DATASET", "DATASET1")
        )
        classes = dataset_info["CLASSES"]
    except KeyError:
        try:
            classes = common.get_value_by_key_pattern(model_card_data, ("CLASSES",))
        except KeyError:
            # use the default classes below if the model card does not have the classes
            # This is the case for the ak only model and the global models (11/05/2024)
            classes = default_class_mapping

    return {int(k): v for k, v in classes.items()}


def get_class_mapping(
    model_card_path: str, default_class_mapping: Optional[dict[int, str]] = None
) -> dict[int, str]:
    """
    Retrieves the class index-to-name mapping from a model card.

    Args:
        model_card_path (str): Path to the model card JSON file.
        default_class_mapping (dict[int, str], optional): A fallback mapping to use
            if the model card doesn't define one. Keys will be cast to integers.


    Returns:
        dict[int, str]: A dictionary mapping class indices to class names.
                        Example: {0: 'sand', 1: 'sediment', 2: 'whitewater', 3: 'water'}
    """
    # example dictionary {0: 'sand', 1: 'sediment', 2: 'whitewater', 3: 'water'}
    model_class_mapping = get_model_card_classes(model_card_path, default_class_mapping)

    return model_class_mapping


def get_indices_of_classnames(
    class_mapping: Dict[int, str],
    selected_class_names: List[str],
) -> List[int]:
    """
    Given a class mapping and a list of selected class names, returns a list of indices
    corresponding to the selected classes, in the same order as the input list.

    Args:
        class_mapping (Dict[int, str]): Dictionary mapping class indices to class names.
        selected_class_names (List[str]): List of class names to retrieve indices for.

    Returns:
        List[int]: List of indices for the selected class names, preserving the input order.

    Example:
        class_mapping = {
            0: 'sand',
            1: 'sediment',
            2: 'whitewater',
            3: 'water'
        }
        selected_class_names = ['whitewater', 'water']

        get_indices_of_classnames(class_mapping, selected_class_names)
        # Returns: [2, 3]
    """
    # Create a reverse lookup from class name to index
    name_to_index = {name: index for index, name in class_mapping.items()}

    # Build result list in order of selected_class_names
    return [
        name_to_index[name] for name in selected_class_names if name in name_to_index
    ]


def load_image_labels(npz_file: str) -> np.ndarray:
    """
    Load in image labels from a .npz file. Loads in the "grey_label" array from the .npz file and returns it as a 2D

    Parameters:
    npz_file (str): The path to the .npz file containing the image labels.

    Returns:
    np.ndarray: A 2D numpy array containing the image labels from the .npz file.
    """
    if not os.path.isfile(npz_file) or not npz_file.endswith(".npz"):
        raise ValueError(f"{npz_file} is not a valid .npz file.")

    data = np.load(npz_file)
    return data["grey_label"]


def merge_classes(im_labels: np.ndarray, classes_to_merge: list) -> np.ndarray:
    """
    Merge the specified classes in the given numpy array of class labels by creating a new numpy array with 1 values
    for the merged classes and 0 values for all other classes.

    :param im_labels: a numpy array of class labels.
    :param classes_to_merge: a list of class labels to merge.
    :return: an integer numpy array with 1 values for the merged classes and 0 values for all other classes.
    """
    # Create an integer numpy array with 1 values for the merged classes and 0 values for all other classes
    updated_labels = np.zeros(shape=(im_labels.shape[0], im_labels.shape[1]), dtype=int)

    # Set 1 values for merged classes
    for idx in classes_to_merge:
        updated_labels = np.logical_or(updated_labels, im_labels == idx).astype(int)

    return updated_labels


def load_merged_image_labels(
    npz_file: str, class_indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Load and process image labels from a .npz file.
    Pass in the indexes of the classes to merge. For instance, if you want to merge the water and white water classes, and
    the indexes of water is 0 and white water is 1, pass in [0, 1] as the class_indices parameter.

    Parameters:
    npz_file (str): The path to the .npz file containing the image labels.
    class_indices (list): The indexes of the classes to merge. If None, defaults to [2, 1, 0] which merges water, whitewater and sediment classes.

    Returns:
    np.ndarray: A 2D numpy array containing the image labels as 1 for the merged classes and 0 for all other classes.
    """
    if not class_indices:
        class_indices = [2, 1, 0]

    if not os.path.isfile(npz_file) or not npz_file.endswith(".npz"):
        raise ValueError(f"{npz_file} is not a valid .npz file.")

    data = np.load(npz_file)
    # 1 for water, 0 for anything else (land, other, sand, etc.)
    im_labels = merge_classes(data["grey_label"], class_indices)

    return im_labels


def simplified_find_contours(
    im_labels: np.ndarray,
    cloud_mask: np.ndarray,
    reference_shoreline_buffer: np.ndarray,
) -> List[np.ndarray]:
    """Find contours in a binary image using skimage.measure.find_contours and processes out contours that contain NaNs.
    Parameters:
    -----------
    im_labels: np.nd.array
        binary image with 0s and 1s
    cloud_mask: np.array
        boolean array indicating cloud mask
    Returns:
    -----------
    processed_contours: list of arrays
        processed image contours (only the ones that do not contains NaNs)
    """
    # make a copy of the im_labels array as a float (this allows find contours to work))
    im_labels_masked = im_labels.copy().astype(float)
    # Apply the cloud mask by setting masked pixels to NaN
    im_labels_masked[cloud_mask] = np.nan
    # only keep the pixels inside the reference shoreline buffer
    im_labels_masked[~reference_shoreline_buffer] = np.nan

    # 0 or 1 labels means 0.5 is the threshold
    contours = measure.find_contours(im_labels_masked, 0.5)

    # remove contour points that are NaNs (around clouds and nodata intersections)
    processed_contours = SDS_shoreline.process_contours(contours)

    return processed_contours


def find_shoreline(
    filename: str,
    image_epsg: int,
    settings: dict,
    cloud_mask_adv: np.ndarray,
    cloud_mask: np.ndarray,
    im_nodata: np.ndarray,
    georef: float,
    im_labels: np.ndarray,
    reference_shoreline_buffer: np.ndarray,
) -> np.ndarray:
    """
    Finds the shoreline in an image.

    Args:
        fn (str): The filename of the image.
        image_epsg (int): The EPSG code of the image.
        settings (dict): A dictionary containing settings for the shoreline extraction.
        cloud_mask_adv (numpy.ndarray): A binary mask indicating advanced cloud cover in the image.
        cloud_mask (numpy.ndarray): A binary mask indicating cloud cover in the image.
        im_nodata (numpy.ndarray): A binary mask indicating no data pixels in the image.
        georef (flat): A the georeference code for the image.
        im_labels (numpy.ndarray): A labeled array indicating the water and land pixels in the image.
        reference_shoreline_buffer (numpy.ndarray,): A buffer around the reference shoreline.

    Returns:
        numpy.ndarray or None: The shoreline as a numpy array, or None if the shoreline could not be found.
    """

    try:
        contours = simplified_find_contours(
            im_labels, cloud_mask, reference_shoreline_buffer
        )
    except Exception as e:
        logger.error(f"{e}\nCould not map shoreline for this image: {filename}")
        return None
    # process the water contours into a shoreline
    shoreline = SDS_shoreline.process_shoreline(
        contours, cloud_mask_adv, im_nodata, georef, image_epsg, settings
    )
    return shoreline


@time_func
def extract_shorelines_with_dask(
    session_path: str,
    metadata: dict,
    settings: dict,
    class_indices: list = None,
    class_mapping: dict = None,
    save_location: str = "",
    shoreline_extraction_area: gpd.GeoDataFrame = None,
    **kwargs: dict,
) -> dict:
    """
    Extracts shorelines from satellite imagery using a Dask-based implementation.

    Args:
        session_path (str): The path to the session directory.
        metadata (dict): A dictionary containing metadata for the satellite imagery.
        settings (dict): A dictionary containing settings for the shoreline extraction.
        class_indices (list, optional): A list of class indices to extract. Defaults to None.
        class_mapping (dict, optional): A dictionary mapping class indices to class names. Defaults to None.
        save_location (str, optional): The path to save the extracted shorelines. Defaults to "".
        shoreline_extraction_area (gpd.GeoDataFrame, optional): A GeoDataFrame containing the area where the shoreline was extracted. Defaults to None.
        **kwargs (dict): Additional keyword arguments.

    Returns:
        dict: A dictionary containing the extracted shorelines for each satellite.
    """
    # create a subfolder to store the .jpg images showing the extracted shoreline detection
    if not save_location:
        sitename = settings["inputs"]["sitename"]
        filepath_data = settings["inputs"]["filepath"]
        filepath_jpg = os.path.join(filepath_data, sitename, "jpg_files", "detection")
        os.makedirs(filepath_jpg, exist_ok=True)

    logger.info(
        f"Satellites in metadata that will have their shorelines extracted: {metadata.keys()}"
    )

    shoreline_dict = {}
    for satname in metadata.keys():
        satellite_dict = process_satellite(
            satname,
            settings,
            metadata,
            session_path,
            class_indices,
            class_mapping,
            save_location,
            batch_size=10,
            shoreline_extraction_area=shoreline_extraction_area,
            **kwargs,
        )
        if not satellite_dict:
            shoreline_dict[satname] = {}
        elif not satname in satellite_dict.keys():
            shoreline_dict[satname] = {}
        else:
            shoreline_dict[satname] = satellite_dict[satname]

    for satname in shoreline_dict.keys():
        # Check and log 'reference shoreline' if it exists
        ref_sl = shoreline_dict[satname].get("shorelines", np.array([]))
        if isinstance(ref_sl, np.ndarray):
            logger.info(f"shorelines.shape: {ref_sl.shape}")
        logger.info(f"Number of 'shorelines' for {satname}: {len(ref_sl)}")
        if shoreline_dict[satname] == {}:
            logger.info(f"No shorelines found for {satname}")
        else:
            logger.info(
                f"result_dict['{satname}'] length {len(shoreline_dict[satname].get('dates',[]))} of dates[:3] {list(islice(shoreline_dict[satname].get('dates',[]),3))}"
            )
            logger.info(
                f"result_dict['{satname}'] length {len(shoreline_dict[satname].get('geoaccuracy',[]))} of geoaccuracy: {np.unique(shoreline_dict[satname].get('geoaccuracy',[]))}"
            )
            logger.info(
                f"result_dict['{satname}'] length {len(shoreline_dict[satname].get('cloud_cover',[]))} of cloud_cover: {np.unique(shoreline_dict[satname].get('cloud_cover',[]))}"
            )
            logger.info(
                f"result_dict['{satname}'] length {len(shoreline_dict[satname].get('filename',[]))} of filename[:3]{list(islice(shoreline_dict[satname].get('filename',[]),3))}"
            )
    # combine the extracted shorelines for each satellite
    return combine_satellite_data(shoreline_dict)


def get_min_shoreline_length(satname: str, default_min_length_sl: float) -> int:
    """
    Given a satellite name and a default minimum shoreline length, returns the minimum shoreline length
    for the specified satellite.

    If the satellite name is "L7", the function returns a minimum shoreline length of 200, as this
    satellite has diagonal bands that require a shorter minimum length. For all other satellite names,
    the function returns the default minimum shoreline length.

    Args:
    - satname (str): A string representing the name of the satellite to retrieve the minimum shoreline length for.
    - default_min_length_sl (float): A float representing the default minimum shoreline length to be returned if
                                      the satellite name is not "L7".

    Returns:
    - An integer representing the minimum shoreline length for the specified satellite.

    Example usage:
    >>> get_min_shoreline_length("L5", 500)
    500
    >>> get_min_shoreline_length("L7", 500)
    200
    """
    # reduce min shoreline length for L7 because of the diagonal bands
    if satname == "L7":
        return 200
    else:
        return default_min_length_sl


def load_extracted_shoreline_from_files(
    dir_path: str,
) -> Optional["Extracted_Shoreline"]:
    """
    Load the extracted shoreline from the given directory.

    The function searches the directory for the extracted shoreline GeoJSON file, the shoreline settings JSON file,
    and the extracted shoreline dictionary JSON file. If any of these files are missing, the function returns None.

    Args:
        dir_path: The path to the directory containing the extracted shoreline files.

    Returns:
        An instance of the Extracted_Shoreline class containing the extracted shoreline data, or None if any of the
        required files are missing.
    """
    required_files = {
        "geojson": "*shoreline*.geojson",
        "settings": "*shoreline*settings*.json",
        "dict": "*shoreline*dict*.json",
    }

    extracted_files = {}
    logger.info(f"Loading extracted shorelines from: {dir_path}")
    for file_type, file_pattern in required_files.items():
        file_paths = glob(os.path.join(dir_path, file_pattern))
        if not file_paths:
            logger.warning(f"No {file_type} file could be loaded from {dir_path}")
            return None

        file_path = file_paths[0]  # Use the first file if there are multiple matches
        if file_type == "geojson":
            extracted_files[file_type] = geodata_processing.read_gpd_file(file_path)
        else:
            extracted_files[file_type] = file_utilities.load_data_from_json(file_path)

    extracted_shorelines = Extracted_Shoreline()
    # attempt to load the extracted shorelines from the files. If there is an error, return None
    try:
        extracted_shorelines = extracted_shorelines.load_extracted_shorelines(
            extracted_files["dict"],
            extracted_files["settings"],
            extracted_files["geojson"],
        )
    except ValueError as e:
        logger.error(f"Error loading extracted shorelines: {e}")
        del extracted_shorelines
        return None

    return extracted_shorelines


class Extracted_Shoreline:
    """Extracted_Shoreline: contains the extracted shorelines within a Region of Interest (ROI)"""

    LAYER_NAME = "extracted_shoreline"
    FILE_NAME = "extracted_shorelines.geojson"

    def __init__(
        self,
    ):
        # gdf: geodataframe containing extracted shoreline for ROI_id
        self.gdf = gpd.GeoDataFrame()
        # Use roi id to identify which ROI extracted shorelines derive from
        self.roi_id = ""
        # dictionary : dictionary of extracted shorelines
        # contains keys 'dates', 'shorelines', 'filename', 'cloud_cover', 'geoaccuracy', 'idx', 'MNDWI_threshold', 'satname'
        self.dictionary = {}
        # shoreline_settings: dictionary of settings used to extract shoreline
        self.shoreline_settings = {}

    def __str__(self):
        # Get column names and their data types
        col_info = self.gdf.dtypes.apply(lambda x: x.name).to_string()
        # Get first 3 rows as a string
        first_rows = self.gdf
        geom_str = ""
        if isinstance(self.gdf, gpd.GeoDataFrame):
            if "geometry" in self.gdf.columns:
                first_rows = self.gdf.head(3).drop(columns="geometry").to_string()
            if not self.gdf.empty:
                geom_str = str(self.gdf.iloc[0]["geometry"])[:100] + "...)"
        # Get CRS information
        crs_info = f"CRS: {self.gdf.crs}" if self.gdf.crs else "CRS: None"
        return f"Extracted Shoreline:\nROI ID: {self.roi_id}\ngdf:\n{crs_info}\nColumns and Data Types:\n{col_info}\n\nFirst 3 Rows:\n{first_rows}\n geometry: {geom_str}"

    def __repr__(self):
        # Get column names and their data types
        col_info = self.gdf.dtypes.apply(lambda x: x.name).to_string()
        # Get first 5 rows as a string
        first_rows = self.gdf
        geom_str = ""
        if isinstance(self.gdf, gpd.GeoDataFrame):
            if "geometry" in self.gdf.columns:
                first_rows = self.gdf.head(3).drop(columns="geometry").to_string()
            if not self.gdf.empty:
                geom_str = str(self.gdf.iloc[0]["geometry"])[:100] + "...)"
        # Get CRS information
        crs_info = f"CRS: {self.gdf.crs}" if self.gdf.crs else "CRS: None"
        return f"Extracted Shoreline:\nROI ID: {self.roi_id}\ngdf:\n{crs_info}\nColumns and Data Types:\n{col_info}\n\nFirst 3 Rows:\n{first_rows}\n geometry: {geom_str}"

    def get_roi_id(self) -> Optional[str]:
        """
        Extracts the region of interest (ROI) ID from the shoreline settings.

        shoreline_settings:
        {
            'inputs' {
                "sitename": 'ID_0_datetime03-22-23__07_29_15',
                "roi_id": 'ID_0',
                'polygon': [[[-160.8452704000395, 63.897979992144656], [-160.8452704000395, 63.861546670361975],...]]]
                'dates':[<start_date>,<end_date>],
                'sat_list': ['S2', 'L8', 'L9'],
                ''filepath': 'path/to/Coastseg/data',
                }
        }

        Returns:
            The ROI ID as a string, or None if the sitename field is not present or is not in the expected format.
        """
        if self.shoreline_settings is None:
            return None
        inputs = self.shoreline_settings.get("inputs", {})
        # inputs has an 'roi_id' key thats the roi id as a string
        roi_id = inputs.get("roi_id", None)
        return roi_id

    def remove_selected_shorelines(
        self, dates: list[datetime.datetime], satellites: list[str]
    ) -> None:
        """
        Removes selected shorelines based on the provided dates and satellites.

        Args:
            dates (list[datetime.datetime]): A list of dates to filter the shorelines.
            satellites (list[str]): A list of satellites to filter the shorelines.

        Returns:
            None
        """
        if hasattr(self, "dictionary"):
            self._remove_from_dict(dates, satellites)
        if hasattr(self, "gdf"):
            if not self.gdf.empty:
                self.gdf = self._remove_from_gdf(dates, satellites)

    def _remove_from_dict(
        self, dates: list["datetime.datetime"], satellites: list[str]
    ) -> dict:
        """
        Remove selected indexes from the dictionary based on the dates and satellites passed in for a specific region of interest.

        Args:
            dates (list['datetime.datetime']): The list of dates to filter.
            satellites (list[str]): The list of satellites to filter.

        Returns:
            dict: The updated dictionary for the specified region of interest.
        """
        selected_indexes = common.get_selected_indexes(
            self.dictionary, dates, satellites
        )
        self.dictionary = common.delete_selected_indexes(
            self.dictionary, selected_indexes
        )
        return self.dictionary

    def _remove_from_gdf(
        self, dates: list["datetime.datetime"], satellites: list[str]
    ) -> gpd.GeoDataFrame:
        """
        Remove rows from the GeoDataFrame based on the specified dates and satellites.

        Args:
            dates (list[datetime.datetime]): A list of datetime objects representing the dates to filter.
            satellites (list[str]): A list of satellite names to filter.

        Returns:
            gpd.GeoDataFrame: The updated GeoDataFrame after removing the matching rows.
        """
        if all(isinstance(date, datetime.date) for date in dates):
            dates = [date.strftime("%Y-%m-%d %H:%M:%S") for date in dates]

        for sat, date in zip(satellites, dates):
            matching_rows = self.gdf[
                (self.gdf["satname"] == sat) & (self.gdf["date"] == date)
            ]
            self.gdf = self.gdf.drop(matching_rows.index)
        return self.gdf

    def load_extracted_shorelines(
        self,
        extracted_shoreline_dict: dict = None,
        shoreline_settings: dict = None,
        extracted_shorelines_gdf: gpd.GeoDataFrame = None,
    ):
        """Loads extracted shorelines into the Extracted_Shoreline class.
        Intializes the class with the extracted shorelines dictionary, shoreline settings, and the extracted shorelines geodataframe

        Args:
            extracted_shoreline_dict (dict, optional): A dictionary containing the extracted shorelines. Defaults to None.
            shoreline_settings (dict, optional): A dictionary containing the shoreline settings. Defaults to None.
            extracted_shorelines_gdf (GeoDataFrame, optional): The extracted shorelines in a GeoDataFrame. Defaults to None.

        Returns:
            object: The Extracted_Shoreline class with the extracted shorelines loaded.

        Raises:
            ValueError: If the input arguments are invalid.
        """

        if not isinstance(extracted_shoreline_dict, dict):
            raise ValueError(
                f"extracted_shoreline_dict must be dict. not {type(extracted_shoreline_dict)}"
            )
        if extracted_shoreline_dict == {}:
            raise ValueError("extracted_shoreline_dict cannot be empty.")

        if extracted_shorelines_gdf is not None:
            if not isinstance(extracted_shorelines_gdf, gpd.GeoDataFrame):
                raise ValueError(
                    f"extracted_shorelines_gdf must be valid geodataframe. not {type(extracted_shorelines_gdf)}"
                )
            if extracted_shorelines_gdf.empty:
                raise ValueError("extracted_shorelines_gdf cannot be empty.")
            self.gdf = extracted_shorelines_gdf

        if not isinstance(shoreline_settings, dict):
            raise ValueError(
                f"shoreline_settings must be dict. not {type(shoreline_settings)}"
            )
        if shoreline_settings == {}:
            raise ValueError("shoreline_settings cannot be empty.")

        # dictionary : dictionary of extracted shorelines
        self.dictionary = extracted_shoreline_dict
        # shoreline_settings: dictionary of settings used to extract shoreline
        self.shoreline_settings = shoreline_settings
        # Use roi id to identify which ROI extracted shorelines derive from
        self.roi_id = shoreline_settings["inputs"]["roi_id"]
        return self

    def create_extracted_shorelines(
        self,
        roi_id: str = None,
        shoreline: gpd.GeoDataFrame = None,
        roi_settings: dict = None,
        settings: dict = None,
        output_directory: str = None,
        shoreline_extraction_area: gpd.GeoDataFrame = None,
    ) -> "Extracted_Shoreline":
        """
        Extracts shorelines for a specified region of interest (ROI) and returns an Extracted_Shoreline class instance.

        Args:
        - self: The object instance.
        - roi_id (str): The ID of the region of interest for which shorelines need to be extracted.
        - shoreline (GeoDataFrame): A GeoDataFrame of shoreline features.
        - roi_settings (dict): A dictionary of region of interest settings.
        - settings (dict): A dictionary of extraction settings.
        - output_directory (str): The path to the directory where the extracted shorelines will be saved.
           - detection figures will be saved in a subfolder called 'jpg_files' within the output_directory.
           - extract_shoreline reports will be saved within the output_directory.

        Returns:
        - object: The Extracted_Shoreline class instance.
        """
        # validate input parameters are not empty and are of the correct type
        self._validate_input_params(roi_id, shoreline, roi_settings, settings)

        logger.info(f"Extracting shorelines for ROI id{roi_id}")
        # extract the shorelines using the settings doing so returns a dictionary with all the shorelines
        self.dictionary = self.extract_shorelines(
            shoreline,
            roi_settings,
            settings,
            output_directory=output_directory,
            shoreline_extraction_area=shoreline_extraction_area,
        )
        if self.dictionary == {}:
            logger.warning(f"No extracted shorelines for ROI {roi_id}")
            raise exceptions.No_Extracted_Shoreline(roi_id)

        if is_list_empty(self.dictionary["shorelines"]):
            logger.warning(f"No extracted shorelines for ROI {roi_id}")
            raise exceptions.No_Extracted_Shoreline(roi_id)

        self.gdf = self.create_geodataframe(
            self.shoreline_settings["output_epsg"],
            output_crs="EPSG:4326",
            geomtype="lines",
        )
        return self

    def create_extracted_shorelines_from_session(
        self,
        model_info: ModelInfo,
        roi_id: str = None,
        shoreline: gpd.GeoDataFrame = None,
        roi_settings: dict = None,
        settings: dict = None,
        session_path: str = None,
        new_session_path: str = None,
        shoreline_extraction_area: gpd.GeoDataFrame = None,
        apply_segmentation_filter: bool = True,
        **kwargs: dict,
    ) -> "Extracted_Shoreline":
        """
        Extracts shorelines for a specified region of interest (ROI) from a saved session and returns an Extracted_Shoreline class instance.

        Args:
        - self: The object instance.
        - roi_id (str): The ID of the region of interest for which shorelines need to be extracted.
        - shoreline (GeoDataFrame): A GeoDataFrame of shoreline features.
        - roi_settings (dict): Dictionary containing settings for the ROI. It must have the following keys:
            {
                "dates": ["2018-12-01", "2019-03-01"],
                "sat_list": ["L8", "L9", "S2"],
                "roi_id": "lyw1",
                "polygon": [
                [
                    [-73.94584118213996, 40.57245559853209],
                    [-73.94584118213996, 40.52844804565595],
                    [-73.87282173497694, 40.52844804565595],
                    [-73.87282173497694, 40.57245559853209],
                    [-73.94584118213996, 40.57245559853209]
                ]
                ],
                "landsat_collection": "C02",
                "sitename": "ID_lyw1_datetime01-18-24__12_26_51",
                "filepath": "C:\\CoastSeg\\data",
            },
        - settings (dict): A dictionary of extraction settings.
        - session_path (str): The path of the saved session from which the shoreline extraction needs to be resumed.
        - new_session_path (str) :The path of the new session where the extreacted shorelines extraction will be saved
            - detection figures will be saved in a subfolder called 'jpg_files' within the new_session_path.
            - extract_shoreline reports will be saved within the new_session_path.
        - apply_segmentation_filter (bool): Whether to apply the segmentation filter to the session. Defaults to True.
        - shoreline_extraction_area (gpd.GeoDataFrame, optional): A GeoDataFrame containing the area to extract shorelines from. Defaults to None.
        Returns:
        - object: The Extracted_Shoreline class instance.
        """
        # validate input parameters are not empty and are of the correct type
        self._validate_input_params(roi_id, shoreline, roi_settings, settings)

        logger.info(f"Extracting shorelines for ROI id: {roi_id}")

        class_mapping = model_info.class_mapping
        water_classes_indices = model_info.water_class_indices

        # # get the reference shoreline
        reference_shoreline = get_reference_shoreline(
            shoreline, settings["output_epsg"]
        )
        # Add reference shoreline to shoreline_settings
        self.shoreline_settings = self.create_shoreline_settings(
            settings, roi_settings, reference_shoreline
        )

        logger.info(
            f"self.shoreline_settings['inputs'] {self.shoreline_settings['inputs']}"
        )
        # Log all items except 'reference shoreline' and handle 'reference shoreline' separately
        logger.info(
            "self.shoreline_settings : "
            + ", ".join(
                f"{key}: {value}"
                for key, value in settings.items()
                if key != "reference_shoreline"
            )
        )

        # Check and log 'reference_shoreline' if it exists
        ref_sl = self.shoreline_settings.get("reference_shoreline", np.array([]))
        if isinstance(ref_sl, np.ndarray):
            logger.info(f"reference_shoreline.shape: {ref_sl.shape}")
        logger.info(f"Number of 'reference_shoreline': {len(ref_sl)} for ROI {roi_id}")

        # Load then filter the metadata based on the session's jpg files (typically CoastSeg/data/ROI_id/jpg_files/RGB)
        metadata_manager = MetadataManager(self.shoreline_settings)
        metadata = metadata_manager.load_metadata()

        # The metadata may be empty if there are no jpg files in at the location given by
        # shoreline_settings['inputs']['filepath'] (this is typically CoastSeg/data/ROI_id/jpg_files/RGB)
        if not metadata:
            logger.warning(f"Metadata was empty after filtering for session jpg files.")
            print(
                f"Metadata was empty after filtering for session jpg files. Check the RGB folders in the folder loaded"
            )
            self.dictionary = {}
            return self

        # Filter the segmentations into good/bad using the segmentation filter model if apply_segmentation_filter is True
        segmentation_filter = SegmentationFilter(session_path)
        good_directory = segmentation_filter.apply_filter(
            apply_segmentation_filter
        )  # applies filter to the session folder

        # Filter the metadata to only include the files with segmentations that are in the good_directory
        metadata = metadata_manager.filter_metadata_by_segmentations(
            metadata, good_directory, file_type="npz"
        )

        extracted_shorelines_dict = extract_shorelines_with_dask(
            session_path,
            metadata,
            self.shoreline_settings,
            class_indices=water_classes_indices,
            class_mapping=class_mapping,
            save_location=new_session_path,
            shoreline_extraction_area=shoreline_extraction_area,
        )
        if extracted_shorelines_dict == {}:
            logger.error(f"Failed to extract any shorelines.")
            raise Exception(f"Failed to extract any shorelines.")

        # postprocessing by removing duplicates and removing in inaccurate georeferencing (set threshold to 10 m)
        extracted_shorelines_dict = SDS_tools.remove_duplicates(
            extracted_shorelines_dict
        )  # removes duplicates (images taken on the same date by the same satellite
        extracted_shorelines_dict = SDS_tools.remove_inaccurate_georef(
            extracted_shorelines_dict, 10
        )  # remove inaccurate georeferencing (set threshold to 10 m)

        log_contents_of_shoreline_dict(extracted_shorelines_dict)

        self.dictionary = extracted_shorelines_dict

        if is_list_empty(self.dictionary.get("shorelines", [])):
            logger.warning(f"No extracted shorelines for ROI {roi_id}")
            raise exceptions.No_Extracted_Shoreline(roi_id)

        # extracted shorelines have map crs so they can be displayed on the map
        self.gdf = self.create_geodataframe(
            self.shoreline_settings["output_epsg"],
            output_crs="EPSG:4326",
            geomtype="lines",
        )

        # break up the shoreline vectors & smooth
        self.gdf = split_line(self.gdf, "linestring", smooth=True)

        return self

    def _validate_input_params(
        self,
        roi_id: str,
        shoreline: gpd.GeoDataFrame,
        roi_settings: dict,
        settings: dict,
    ) -> None:
        """
        Validates that the input parameters for shoreline extraction are not empty and are of the correct type.

        Args:
        - self: The object instance.
        - roi_id (str): The ID of the region of interest for which shorelines need to be extracted.
        - shoreline (GeoDataFrame): A GeoDataFrame of shoreline features.
        - roi_settings (dict): A dictionary of region of interest settings.
        - settings (dict): A dictionary of extraction settings.

        Raises:
        - ValueError: If any of the input parameters are empty or not of the correct type.
        """
        if not isinstance(roi_id, str):
            raise ValueError(f"ROI id must be string. not {type(roi_id)}")

        if not isinstance(shoreline, gpd.GeoDataFrame):
            raise ValueError(
                f"shoreline must be valid geodataframe. not {type(shoreline)}"
            )
        if shoreline.empty:
            raise ValueError("shoreline cannot be empty.")

        if not isinstance(roi_settings, dict):
            raise ValueError(f"roi_settings must be dict. not {type(roi_settings)}")
        if roi_settings == {}:
            raise ValueError("roi_settings cannot be empty.")

        if not isinstance(settings, dict):
            raise ValueError(f"settings must be dict. not {type(settings)}")
        if settings == {}:
            raise ValueError("settings cannot be empty.")

    def extract_shorelines(
        self,
        shoreline_gdf: gpd.GeoDataFrame,
        roi_settings: dict,
        settings: dict,
        output_directory: str = None,
        shoreline_extraction_area: gpd.GeoDataFrame = None,
    ) -> dict:
        """
        Extracts shorelines for a specified region of interest (ROI).
        Args:
            shoreline_gdf (gpd.GeoDataFrame): GeoDataFrame containing the shoreline data.
            roi_settings (dict): Dictionary containing settings for the ROI. It must have the following keys:
            {
                "dates": ["2018-12-01", "2019-03-01"],
                "sat_list": ["L8", "L9", "S2"],
                "roi_id": "lyw1",
                "polygon": [
                [
                    [-73.94584118213996, 40.57245559853209],
                    [-73.94584118213996, 40.52844804565595],
                    [-73.87282173497694, 40.52844804565595],
                    [-73.87282173497694, 40.57245559853209],
                    [-73.94584118213996, 40.57245559853209]
                ]
                ],
                "landsat_collection": "C02",
                "sitename": "ID_lyw1_datetime01-18-24__12_26_51",
                "filepath": "C:\\development\\doodleverse\\coastseg\\CoastSeg\\data",
            },
            settings (dict): Dictionary containing general settings.

            session_path (str, optional): Path to the session. Defaults to None.
            class_indices (list, optional): List of class indices. Defaults to None.
            class_mapping (dict, optional): Dictionary mapping class indices to class labels. Defaults to None.
            output_directory (str): The path to the directory where the extracted shorelines will be saved.
                - detection figures will be saved in a subfolder called 'jpg_files' within the output_directory.
                - extract_shoreline reports will be saved within the output_directory.
            shoreline_extraction_area (gpd.GeoDataFrame, optional): A GeoDataFrame containing the area to extract shorelines from. Defaults to None.
        Returns:
            dict: Dictionary containing the extracted shorelines for the specified ROI.
        """
        # project shorelines's crs from map's crs to output crs given in settings
        # create a reference shoreline as a numpy array containing lat, lon, and mean sea level for each point
        reference_shoreline = get_reference_shoreline(
            shoreline_gdf, settings["output_epsg"]
        )
        # Add reference shoreline to shoreline_settings
        self.shoreline_settings = self.create_shoreline_settings(
            settings, roi_settings, reference_shoreline
        )
        # gets metadata used to extract shorelines
        filepath_data = get_data_folder(self.shoreline_settings["inputs"]["filepath"])
        filepath = self.shoreline_settings["inputs"].get("filepath", None)
        if filepath is None:
            self.shoreline_settings["inputs"]["filepath"] = filepath_data
        # data_folder =  os.path.join(core_utilities.get_base_dir(),'data')
        metadata = get_metadata(self.shoreline_settings["inputs"], filepath_data)
        sitename = self.shoreline_settings["inputs"]["sitename"]
        # filter out files that were removed from RGB directory
        try:
            RGB_directory = os.path.join(
                filepath_data, sitename, "jpg_files", "preprocessed", "RGB"
            )
            metadata = common.filter_metadata_with_dates(
                metadata, RGB_directory, file_type="jpg"
            )
        except FileNotFoundError as e:
            logger.warning(f"No RGB files existed so no metadata.")
            print(
                f"No shorelines were extracted because no RGB files were found at {os.path.join(filepath_data,sitename)}"
            )
            return {}

        for satname in metadata.keys():
            if not metadata[satname]:
                logger.warning(f"metadata['{satname}'] is empty")
            else:
                logger.info(
                    f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('epsg',[]))} of epsg: {np.unique(metadata[satname].get('epsg',[]))}"
                )
                logger.info(
                    f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('dates',[]))} of dates Sample first five: {list(islice(metadata[satname].get('dates',[]),5))}"
                )
                logger.info(
                    f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('filenames',[]))} of filenames Sample first five: {list(islice(metadata[satname].get('filenames',[]),5))}"
                )
                logger.info(
                    f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('im_dimensions',[]))} of im_dimensions: {np.unique(metadata[satname].get('im_dimensions',[]))}"
                )
                logger.info(
                    f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('acc_georef',[]))} of acc_georef: {np.unique(metadata[satname].get('acc_georef',[]))}"
                )
                logger.info(
                    f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('im_quality',[]))} of im_quality: {np.unique(metadata[satname].get('im_quality',[]))}"
                )

        # extract shorelines with coastsat's models
        extracted_shorelines = SDS_shoreline.extract_shorelines(
            metadata,
            self.shoreline_settings,
            output_directory=output_directory,
            shoreline_extraction_area=shoreline_extraction_area,
        )
        logger.info(f"extracted_shoreline_dict: {extracted_shorelines}")
        # postprocessing by removing duplicates and removing in inaccurate georeferencing (set threshold to 10 m)
        extracted_shorelines = SDS_tools.remove_duplicates(
            extracted_shorelines
        )  # removes duplicates (images taken on the same date by the same satellite)
        extracted_shorelines = SDS_tools.remove_inaccurate_georef(
            extracted_shorelines, 10
        )  # remove inaccurate georeferencing (set threshold to 10 m)
        return extracted_shorelines

    def create_shoreline_settings(
        self,
        settings: dict,
        roi_settings: dict,
        reference_shoreline: dict,
    ) -> dict:
        """Create and return a dictionary containing settings for shoreline.


        Args:
            settings (dict): settings used to control how shorelines are extracted
            settings = {

            "cloud_thresh" (float): percentage of cloud cover allowed
            "cloud_mask_issue" (bool): whether to apply coastsat fix for incorrect cloud masking
            "min_beach_area" (float): minimum area of beach allowed
            "min_length_sl" (int): minimum length (m) of shoreline allowed
            "output_epsg" (int): coordinate reference system of output
            "sand_color" (str): color of sand in RGB image
            "pan_off" (bool): whether to use panchromatic band (always False)
            "max_dist_ref" (int): maximum distance (m) from reference shoreline
            "dist_clouds" (int): distance (m) from clouds to remove
            "percent_no_data" (float): percentage of no data allowed
            "model_session_path" (str): path to model session file
            "apply_cloud_mask" (bool): whether to apply cloud mask
            }
            roi_settings (dict): Dictionary containing settings for the ROI.
            It must have the following keys:
            {
                "dates": ["2018-12-01", "2019-03-01"],
                "sat_list": ["L8", "L9", "S2"],
                "roi_id": "lyw1",
                "polygon": [
                [
                    [-73.94584118213996, 40.57245559853209],
                    [-73.94584118213996, 40.52844804565595],
                    [-73.87282173497694, 40.52844804565595],
                    [-73.87282173497694, 40.57245559853209],
                    [-73.94584118213996, 40.57245559853209]
                ]
                ],
                "landsat_collection": "C02",
                "sitename": "ID_lyw1_datetime01-18-24__12_26_51",
                "filepath": "C:\\development\\doodleverse\\coastseg\\CoastSeg\\data",
            },
            reference_shoreline (dict): reference shoreline

        Example)
        shoreline_settings =
        {
            "reference_shoreline":reference_shoreline,
            "inputs": roi_settings,
            "adjust_detection": False,
            "check_detection": False,
            ...
            rest of items from settings
        }

        Returns:
            dict: The created shoreline settings.
        """
        SHORELINE_KEYS = [
            "cloud_thresh",
            "cloud_mask_issue",
            "min_beach_area",
            "min_length_sl",
            "output_epsg",
            "sand_color",
            "pan_off",
            "max_dist_ref",
            "dist_clouds",
            "percent_no_data",
            "model_session_path",  # path to model session file
            "apply_cloud_mask",
        ]
        shoreline_settings = {k: v for k, v in settings.items() if k in SHORELINE_KEYS}
        shoreline_settings.update(
            {
                "reference_shoreline": reference_shoreline,
                "adjust_detection": False,  # disable adjusting shorelines manually
                "check_detection": False,  # disable adjusting shorelines manually
                "save_figure": True,  # always save a matplotlib figure of shorelines
                "inputs": roi_settings,  # copy settings for ROI shoreline will be extracted from
            }
        )
        return shoreline_settings

    def create_geodataframe(
        self, input_crs: str, output_crs: str = None, geomtype: str = "lines"
    ) -> gpd.GeoDataFrame:
        """Creates a geodataframe with the crs specified by input_crs. Converts geodataframe crs
        to output_crs if provided.

        Converts the internal dictionary of extracted shorelines to a geodataframe and returns it.

        Args:
            input_crs (str ): coordinate reference system string. Format 'EPSG:4326'.
            output_crs (str, optional): coordinate reference system string. Defaults to None.
        Returns:
            gpd.GeoDataFrame: geodataframe with columns = ['geometery','date','satname','geoaccuracy','cloud_cover']
            converted to output_crs if provided otherwise geodataframe's crs will be
            input_crs
        """
        extract_shoreline_gdf = SDS_tools.output_to_gdf(self.dictionary, geomtype)
        if not extract_shoreline_gdf.crs:
            extract_shoreline_gdf.set_crs(input_crs, inplace=True)
        if output_crs is not None:
            extract_shoreline_gdf = extract_shoreline_gdf.to_crs(output_crs)
        return extract_shoreline_gdf

    def to_file(
        self, filepath: str, filename: str, data: Union[gpd.GeoDataFrame, dict]
    ):
        """Save geopandas dataframe to file, or save data to file with to_file().

        Args:
            filepath (str): The directory where the file should be saved.
            filename (str): The name of the file to be saved.
            data (Any): The data to be saved to file.

        Raises:
            ValueError: Raised when data is not a geopandas dataframe and cannot be saved with tofile()
        """
        file_location = os.path.abspath(os.path.join(filepath, filename))

        if isinstance(data, gpd.GeoDataFrame):
            data.to_file(
                file_location,
                driver="GeoJSON",
                encoding="utf-8",
            )
        elif isinstance(data, dict):
            if data != {}:
                file_utilities.to_file(data, file_location)

    def get_layer_name(self) -> list:
        """returns name of extracted shoreline layer"""
        layer_name = "extracted_shoreline"
        return layer_name

    def get_styled_layer(
        self, gdf, row_number: int = 0, map_crs: int = 4326, style: dict = {}
    ) -> dict:
        """
        Returns a single shoreline feature as a GeoJSON object with a specified style.

        Args:
        - gdf: The input GeoDataFrame.
        - row_number (int): The index of the shoreline feature to select from the GeoDataFrame.
        - map_crs (int): The desired coordinate reference system.
        - style (dict) default {} :
            Additional style attributes to be merged with the default style.

        Returns:
        - dict: A styled GeoJSON feature.
        """
        if gdf.empty:
            return []

        projected_gdf = transform_gdf_to_crs(gdf, map_crs)
        single_shoreline = select_and_stringify(projected_gdf, row_number)
        features_json = convert_gdf_to_json(single_shoreline)
        layer_name = self.get_layer_name()

        # Ensure there are features to process.
        if not features_json.get("features"):
            return []

        styled_feature = style_layer(
            features_json["features"][0], layer_name, "red", style
        )
        return styled_feature


def get_reference_shoreline(
    shoreline_gdf: gpd.GeoDataFrame, output_crs: str
) -> np.ndarray:
    """
    Converts a GeoDataFrame of shoreline features into a numpy array of latitudes, longitudes, and zeroes representing the mean sea level.

    Args:
    - shoreline_gdf (GeoDataFrame): A GeoDataFrame of shoreline features.
    - output_crs (str): The output CRS to which the shoreline features need to be projected.

    Returns:
    - np.ndarray: A numpy array of latitudes, longitudes, and zeroes representing the mean sea level.
    """
    # project shorelines's espg from map's espg to output espg given in settings
    reprojected_shorlines = shoreline_gdf.to_crs(output_crs)
    # convert shoreline_in_roi gdf to coastsat compatible format np.array([[lat,lon,0],[lat,lon,0]...])
    shorelines = make_coastsat_compatible(reprojected_shorlines)
    # shorelines = [([lat,lon],[lat,lon],[lat,lon]),([lat,lon],[lat,lon],[lat,lon])...]
    # Stack all the tuples into a single list of n rows X 2 columns
    shorelines = np.vstack(shorelines)
    # Add third column of 0s to represent mean sea level
    shorelines = np.insert(shorelines, 2, np.zeros(len(shorelines)), axis=1)

    return shorelines


def make_coastsat_compatible(feature: gpd.GeoDataFrame) -> list:
    """Return the feature as an np.array in the form:
        [([lat,lon],[lat,lon],[lat,lon]),([lat,lon],[lat,lon],[lat,lon])...])
    Args:
        feature (gpd.GeoDataFrame): clipped portion of shoreline within a roi
    Returns:
        list: shorelines in form:
            [([lat,lon],[lat,lon],[lat,lon]),([lat,lon],[lat,lon],[lat,lon])...])
    """
    features = []
    # Use explode to break multilinestrings in linestrings
    feature_exploded = feature.explode(index_parts=True)
    # For each linestring portion of feature convert to lat,lon tuples
    lat_lng = feature_exploded.apply(
        lambda row: tuple(np.array(row.geometry.coords).tolist()), axis=1
    )
    features = list(lat_lng)
    return features


def is_list_empty(main_list: list) -> bool:
    all_empty = True
    for np_array in main_list:
        if len(np_array) != 0:
            all_empty = False
    return all_empty
