import glob
import json
import logging

# Standard library imports
import os
import re
import shutil
from shutil import SameFileError
from typing import Any, Collection, Dict, List, Optional, Set, Tuple

# Third-party imports
import geopandas as gpd
import numpy as np
import scipy
import skimage
import skimage.io as io
import tensorflow as tf
import tqdm
from doodleverse_utils.model_imports import (
    custom_resunet,
    custom_unet,
    dice_coef_loss,
    segformer,
    simple_resunet,
    simple_satunet,
    simple_unet,
)
from osgeo import gdal
from PIL import Image
from tensorflow.keras import mixed_precision  # type: ignore

from coastseg import (
    common,
    core_utilities,
    extracted_shoreline,
    file_utilities,
    geodata_processing,
    sessions,
)
from coastseg.intersections import save_transects, transect_timeseries
from coastseg.ml import do_seg
from coastseg.model_info import ModelInfo

# Local imports
from . import __version__

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

# Logger setup
logger = logging.getLogger(__name__)


def create_new_session_path(session_name: str) -> str:
    """
    Create a new session directory for storing extracted shorelines and transects.

    The session path is created under the base directory returned by core_utilities.get_base_dir(),
    inside a 'sessions' folder. If the directory already exists, it is not recreated.

    Parameters:
        session_name (str): The name of the session to create a directory for.

    Returns:
        str: The absolute path to the newly created (or existing) session directory.
    """
    base_path = core_utilities.get_base_dir()
    session_path = base_path / "sessions" / session_name
    session_path.mkdir(parents=True, exist_ok=True)
    return str(session_path.resolve())


def load_roi_gdf_from_session(
    session_path: str, roi_id: str, config_geojson_location: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Loads GeoDataFrame for a specific ROI from config_gdf.geojson file.

    Args:
        session_path (str): Path to session directory for locating geojson file.
        roi_id (str): Region of Interest identifier to extract.
        config_geojson_location (Optional[str]): Full path to config_gdf.geojson file.
            If None, searches in session_path.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing only the row for the ROI ID.

    Raises:
        KeyError: If 'id' column is missing in the GeoDataFrame.
        ValueError: If no matching ROI ID is found.
    """
    # Use provided geojson location or discover it
    if config_geojson_location is None:
        config_geojson_location = file_utilities.find_file_recursively(
            session_path, "config_gdf.geojson"
        )
    logger.info(f"config_geojson_location: {config_geojson_location}")

    config_gdf = geodata_processing.read_gpd_file(config_geojson_location)

    # Check for required 'id' column
    if "id" not in config_gdf.columns:
        logger.error(
            f"'id' column missing in config_gdf.geojson: {config_geojson_location}"
        )
        raise KeyError(
            f"'id' column missing in config_gdf.geojson: {config_geojson_location}"
        )

    # Filter for the specified ROI ID
    roi_gdf = config_gdf[config_gdf["id"] == roi_id]

    if roi_gdf.empty:
        logger.error(
            f"{roi_id} ROI ID did not exist in config_gdf.geojson: {config_geojson_location}"
        )
        raise ValueError(
            f"{roi_id} ROI ID did not exist in config_gdf.geojson: {config_geojson_location}"
        )

    return roi_gdf


def load_good_bad_csv(roi_id: str, roi_settings: Dict[str, Any]) -> Optional[str]:
    """
    Locates image classification results CSV file for a given ROI.

    Args:
        roi_id (str): Region of Interest identifier, the ROI ID.
        roi_settings (Dict[str, Any]): ROI settings dictionary from session config.

    Returns:
        Optional[str]: Path to 'image_classification_results.csv' if found, None otherwise.
    """
    try:
        return file_utilities.find_file_path_in_roi(
            roi_id, roi_settings, "image_classification_results.csv"
        )
    except Exception as e:
        logger.error(
            f"Failed to locate 'image_classification_results.csv' for ROI '{roi_id}': {e}"
        )
        return None


def load_roi_settings(
    session_path: str, roi_id: Optional[str] = None
) -> Tuple[Dict[str, Any], str]:
    """
    Loads ROI settings from session configuration file.

    Args:
        session_path (str): Path to session directory containing config.json.
        roi_id (Optional[str]): Specific ROI ID to load. If None, uses first ROI ID.

    Returns:
        Tuple[Dict[str, Any], str]: ROI settings dictionary and loaded ROI ID.

    Raises:
        Exception: If ROI ID is missing, not found, or config is malformed.
    """
    try:
        # Load configuration from config.json
        config = file_utilities.load_json_data_from_file(session_path, "config.json")

        # If roi_id is not provided, try to get the first one from the config
        if roi_id is None:
            roi_ids = config.get("roi_ids")
            if not roi_ids:
                raise KeyError("No ROI IDs found in configuration.")
            roi_id = roi_ids[0]

        # Ensure the specified roi_id exists in the config
        if roi_id is None or not isinstance(roi_id, str) or roi_id not in config:
            raise KeyError(f"ROI ID '{roi_id}' not found in config.")

        roi_settings: Dict[str, Any] = {str(roi_id): config[roi_id]}
        return roi_settings, str(roi_id)

    except (KeyError, ValueError) as e:
        logger.error(f"Error loading ROI settings: {e}")
        if roi_id is None:
            logger.error(f"roi_id was None. Full config: {config}")
            raise Exception(f"The session loaded had no valid roi_id. Config: {config}")
        else:
            logger.error(
                f"ROI ID '{roi_id}' not found in config. Full config: {config}"
            )
            raise Exception(
                f"The ROI ID '{roi_id}' did not exist in config.json.\nConfig: {config}"
            )


def apply_smooth_otsu_to_folder(folder: str) -> str:
    """
    Applies median filter to JPEG images, converts to grayscale, and saves to new folder.

    Args:
        folder (str): Path to folder containing JPEG images to process.

    Returns:
        str: Path to new folder containing smoothed images.
    """
    new_folder_name = os.path.basename(folder) + "_smooth"
    new_folder = os.path.join(os.path.dirname(folder), new_folder_name)
    os.makedirs(new_folder, exist_ok=True)
    # get all files in folder
    files = glob.glob(os.path.join(folder, "*jpg"))
    for file in files:
        img = Image.open(file)
        img = img.convert("L")  # convert to grayscale
        img_arr = np.array(img)
        img_arr_median_filter = scipy.ndimage.median_filter(img_arr, size=15)
        # save the median filtered image to the new folder
        new_file = os.path.join(new_folder, os.path.basename(file))
        io.imsave(new_file, img_arr_median_filter)

    return new_folder


def download_url_dict(url_dict: Dict[str, str]) -> Optional[bool]:
    """
    Downloads files from URLs and saves to specified paths.

    Args:
        url_dict (Dict[str, str]): Dictionary mapping file paths to download URLs.

    Returns:
        Optional[bool]: False if response status is not 200, None if successful.

    Raises:
        Exception: If response status is 404 or 429.
    """
    for save_path, url in url_dict.items():
        # get a response from the url
        response = common.get_response(url, stream=True)
        with response:
            logger.info(f"response: {response}")
            logger.info(f"response.status_code: {response.status_code}")
            logger.info(f"response.headers: {response.headers}")
            if response.status_code == 404:
                logger.info(f"404 response for {url}")
                raise Exception(
                    f"404 response for {url}. Please raise an issue on GitHub."
                )

            # too many requests were made to the API
            if response.status_code == 429:
                content = response.text
                print(
                    f"Response from API for status_code: {response.status_code}: {content}"
                )
                logger.info(
                    f"Response from API for status_code: {response.status_code}: {content}"
                )
                raise Exception(
                    f"Response from API for status_code: {response.status_code}: {content}"
                )

            # raise an exception if the response status_code is not 200
            if response.status_code != 200:
                print(f"response.status_code {response.status_code} for {url}")
                logger.info(f"response.status_code {response.status_code} for {url}")
                return False

            response.raise_for_status()

            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                content_length = int(content_length)
                with open(save_path, "wb") as fd:
                    with tqdm.auto.tqdm(
                        total=content_length,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading {os.path.basename(save_path)}",
                        initial=0,
                        ascii=False,
                        position=0,
                    ) as pbar:
                        for chunk in response.iter_content(1024):
                            if not chunk:
                                break
                            fd.write(chunk)
                            pbar.update(len(chunk))
            else:
                with open(save_path, "wb") as fd:
                    for chunk in response.iter_content(1024):
                        fd.write(chunk)


def add_classifer_scores_to_shorelines(
    good_bad_csv, good_bad_seg_csv, files: Collection
):
    """Adds new columns to the geojson file with the model scores from the image_classification_results.csv and segmentation_classification_results.csv files

    Args:
        geojson_path (gpd.GeoDataFrame): A GeoDataFrame of extracted shorelines that contains the date column
        good_bad_csv (str): The path to the image_classification_results.csv file
        good_bad_seg_csv (str): The path to the segmentation_classification_results.csv file
        files (Collection): A collection of files to add the model scores to
            These files should be geojson files that contain a column called 'date'
    """
    for file in files:
        if os.path.exists(file):
            file_utilities.join_model_scores_to_geodataframe(
                file, good_bad_csv, good_bad_seg_csv
            )


def filter_no_data_pixels(files: list[str], percent_no_data: float = 0.50) -> list[str]:
    """
    Filters out image files that have a percentage of black (no data) pixels greater than the specified threshold.
    Args:
        files (list[str]): A list of file paths to image files.
        percent_no_data (float, optional): The maximum allowed percentage of black pixels in an image.
                                           Images with a higher percentage of black pixels will be filtered out.
                                           Defaults to 0.50 (50%).
    Returns:
        list[str]: A list of file paths to images that have a percentage of black pixels less than or equal to the specified threshold.
    """

    def percentage_of_black_pixels(img: Image.Image) -> float:
        # Calculate the total number of pixels in the image
        num_total_pixels = img.size[0] * img.size[1]
        img_array = np.array(img)
        # Count the number of black pixels in the image
        black_pixels = np.count_nonzero(np.all(img_array == 0, axis=-1))
        # Calculate the percentage of black pixels
        percentage = black_pixels / num_total_pixels
        return percentage

    valid_images = []

    def has_image_files(file_list, extensions):
        return any(file.lower().endswith(extensions) for file in file_list)

    extensions = (".jpg", ".png", ".jpeg")
    contains_images = has_image_files(files, extensions)
    if not contains_images:
        logger.warning(
            f"Cannot filter no data pixels no images found with {extensions} in {files}."
        )
        return files

    for file in files:
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            img = Image.open(file)
            percentage = percentage_of_black_pixels(img)
            if percentage <= percent_no_data:
                valid_images.append(file)

    return valid_images


def get_files_to_download(
    available_files: List[dict],
    filenames: List[str],
    model_id: str,
    model_path: Optional[str],
) -> dict:
    """Constructs a dictionary of file paths and their corresponding download links, based on the available files and a list of desired filenames.

    Args:
    - available_files: A list of dictionaries representing the metadata of available files, including the file key and download links.
    - filenames: A list of strings representing the desired filenames.
    - model_id: A string representing the ID of the model being downloaded.
    - model_path: A string representing the path to the directory where the files will be downloaded.

    Returns:
    A dictionary with file paths as keys and their corresponding download links as values.
    Raises a ValueError if any of the desired filenames are not available in the available_files list.
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    url_dict = {}
    for filename in filenames:
        response = next((f for f in available_files if f["key"] == filename), None)
        if response is None:
            raise ValueError(f"Cannot find {filename} at {model_id}")
        link = response["links"]["self"]
        file_path = os.path.join(model_path, filename)
        url_dict[file_path] = link
    return url_dict


def check_if_files_exist(files_dict: dict) -> dict:
    """Checks if each file in a given dictionary of file paths and download links already exists in the local filesystem.

    Args:
    - files_dict: A dictionary with file paths as keys and their corresponding download links as values.

    Returns:
    A dictionary with file paths as keys and their corresponding download links as values, for any files that do not exist in the local filesystem.
    """
    url_dict = {}
    for save_path, link in files_dict.items():
        if not os.path.isfile(save_path):
            url_dict[save_path] = link
    return url_dict


def get_zenodo_release(zenodo_id: str) -> Dict[str, Any]:
    """
    Retrieves JSON data for Zenodo release with given ID.

    Args:
        zenodo_id (str): The Zenodo record ID.

    Returns:
        Dict[str, Any]: JSON data from Zenodo API response.

    Raises:
        requests.HTTPError: If the API request fails.
    """
    root_url = f"https://zenodo.org/api/records/{zenodo_id}"
    # get a response from the url
    response = common.get_response(root_url, stream=False)
    response.raise_for_status()
    return response.json()


def get_imagery_directory(img_type: str, RGB_path: str) -> str:
    """
    Returns directory of the newly created imagery. Available imagery conversions:

    1. 'NDWI' for 'NIR'
    2. 'MNDWI' for 'SWIR'
    3. 'RGB' for 'RGB'

    Note:
        Directories containing 'NIR','NIR' and 'RGB' imagery must be at the same level as the 'RGB' imagery.
        ex.
        home/
            RGB
            NIR
            SWIR

    Args:
        img_type (str): The type of imagery to generate. Available options: 'RGB', 'NDWI', 'MNDWI'
        RGB_path (str): The path to the RGB imagery directory.

    Returns:
        str: The path to the output directory for the specified imagery type.
    """
    img_type = img_type.upper()
    output_path = os.path.dirname(RGB_path)
    if img_type == "RGB":
        output_path = RGB_path
    # default filetype is NIR and if NDWI is selected else filetype to SWIR
    elif img_type == "NDWI":
        NIR_path = os.path.join(output_path, "NIR")
        output_path = RGB_to_infrared(RGB_path, NIR_path, output_path, "NDWI")
    elif img_type == "MNDWI":
        SWIR_path = os.path.join(output_path, "SWIR")
        output_path = RGB_to_infrared(RGB_path, SWIR_path, output_path, "MNDWI")
    else:
        raise ValueError(
            f"{img_type} not reconigzed as one of the valid types 'RGB', 'NDWI', 'MNDWI'"
        )
    return output_path


def matching_datetimes_files(dir1: str, dir2: str) -> Set[str]:
    """
    Get the matching datetimes from the filenames in two directories.

    Args:
        dir1 (str): Path to the first directory.
        dir2 (str): Path to the second directory.

    Returns:
        Set[str]: A set of strings representing the common datetimes.
    """
    # Get the filenames in each directory
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)

    # Define a pattern to match the date-time part of the filenames
    pattern = re.compile(
        r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
    )  # Matches YYYY-MM-DD-HH-MM-SS

    # Create sets of the date-time parts of the filenames in each directory
    files1_dates = {
        match.group(0) for filename in files1 if (match := re.search(pattern, filename))
    }
    files2_dates = {
        match.group(0) for filename in files2 if (match := re.search(pattern, filename))
    }

    # Find the intersection of the two sets
    matching_files = files1_dates & files2_dates

    return matching_files


def get_full_paths(
    dir1: str, dir2: str, common_dates: Set[str]
) -> Tuple[List[str], List[str]]:
    """
    Get the full paths of the files with matching datetimes.

    Args:
        dir1 (str): Path to the first directory.
        dir2 (str): Path to the second directory.
        common_dates (Set[str]): A set of strings representing the common datetimes.

    Returns:
        Tuple[List[str], List[str]]: Two lists of strings representing the full paths of the matching files in dir1 and dir2.
    """
    # Get the filenames in each directory
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)

    # Define a pattern to match the date-time part of the filenames
    pattern = re.compile(
        r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
    )  # Matches YYYY-MM-DD-HH-MM-SS

    # Find the full paths of the files with the matching date-times
    matching_files_dir1 = [
        os.path.join(dir1, filename)
        for filename in files1
        if (match := re.search(pattern, filename)) and match.group(0) in common_dates
    ]
    matching_files_dir2 = [
        os.path.join(dir2, filename)
        for filename in files2
        if (match := re.search(pattern, filename)) and match.group(0) in common_dates  # type: ignore
    ]

    return matching_files_dir1, matching_files_dir2


def get_files(RGB_dir_path: str, img_dir_path: str) -> np.ndarray:
    """returns matrix of files in RGB_dir_path and img_dir_path
    creates matrix: RGB x number of samples in img_dir_path
    Example:
    [['full_RGB_path.jpg','full_NIR_path.jpg'],
    ['full_jpg_path.jpg','full_NIR_path.jpg']....]
    Args:
        RGB_dir_path (str): full path to directory of RGB images
        img_dir_path (str): full path to directory of non-RGB images
        usually NIR and SWIR

    Raises:
        FileNotFoundError: raised if directory is not found
    Returns:
        np.ndarray: A matrix of matching files, shape (bands, number of samples).
    """
    if not os.path.exists(RGB_dir_path):
        raise FileNotFoundError(f"{RGB_dir_path} not found")
    if not os.path.exists(img_dir_path):
        raise FileNotFoundError(f"{img_dir_path} not found")

    # get the dates in both directories
    common_dates = matching_datetimes_files(RGB_dir_path, img_dir_path)
    # get the full paths to the dates that exist in each directory
    matching_files_RGB_dir, matching_files_img_dir = get_full_paths(
        RGB_dir_path, img_dir_path, common_dates
    )
    # the order must be RGB dir then not RGB dir for other code to work
    # matching_files = sorted(matching_files_RGB_dir) + sorted(matching_files_img_dir)
    files = []
    files.append(sorted(matching_files_RGB_dir))
    files.append(sorted(matching_files_img_dir))
    # creates matrix:  matrix: RGB x number of samples in img_dir_path
    matching_files = np.vstack(files).T
    return matching_files


def RGB_to_infrared(
    RGB_path: str, infrared_path: str, output_path: str, output_type: str
) -> str:
    """Converts two directories of RGB and (NIR/SWIR) imagery to (NDWI/MNDWI) imagery in a directory named
     'NDWI' created at output_path.
     imagery saved as jpg

     to generate NDWI imagery set infrared_path to full path of NIR images
     to generate MNDWI imagery set infrared_path to full path of SWIR images

    Args:
        RGB_path (str): full path to directory containing RGB images
        infrared_path (str): full path to directory containing NIR or SWIR images
        output_path (str): full path to directory to create NDWI/MNDWI directory in
        output_type (str): 'MNDWI' or 'NDWI'
    Based on code from doodleverse_utils by Daniel Buscombe
    source: https://github.com/Doodleverse/doodleverse_utils

    Returns:
        str: full path to directory containing NDWI/MNDWI images
    """
    if output_type.upper() not in ["MNDWI", "NDWI"]:
        logger.error(
            f"Invalid output_type given must be MNDWI or NDWI. Cannot be {output_type}"
        )
        raise Exception(
            f"Invalid output_type given must be MNDWI or NDWI. Cannot be {output_type}"
        )
    # matrix:bands(RGB) x number of samples(NIR)
    files = get_files(RGB_path, infrared_path)
    # output_path: directory to store MNDWI or NDWI outputs
    output_path = os.path.join(output_path, output_type.upper())

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for file in files:
        # Read green band from RGB image and cast to float
        green_band = skimage.io.imread(file[0])[:, :, 1].astype("float")
        # Read infrared(SWIR or NIR) and cast to float
        infrared = skimage.io.imread(file[1]).astype("float")
        # Transform 0 to np.nan
        green_band[green_band == 0] = np.nan
        infrared[infrared == 0] = np.nan
        # Mask out NaNs
        green_band = np.ma.filled(green_band)
        infrared = np.ma.filled(infrared)

        # ensure both matrices have equivalent size
        if not np.shape(green_band) == np.shape(infrared):
            gx, gy = np.shape(green_band)
            nx, ny = np.shape(infrared)
            # resize both matrices to have equivalent size
            green_band = common.scale(
                green_band, np.maximum(gx, nx), np.maximum(gy, ny)
            )
            infrared = common.scale(infrared, np.maximum(gx, nx), np.maximum(gy, ny))

        # output_img(MNDWI/NDWI) imagery formula (Green - SWIR) / (Green + SWIR)
        output_img = (green_band - infrared) / (green_band + infrared)
        # Convert the NaNs to -1
        output_img[np.isnan(output_img)] = -1
        # Rescale to be between 0 - 255
        output_img = common.rescale_array(output_img, 0, 255)
        # create new filenames by replacing image type(SWIR/NIR) with output_type
        if output_type.upper() == "MNDWI":
            new_filename = file[1].split(os.sep)[-1].replace("SWIR", output_type)
        if output_type.upper() == "NDWI":
            new_filename = file[1].split(os.sep)[-1].replace("NIR", output_type)

        # save output_img(MNDWI/NDWI) as .jpg in output directory
        skimage.io.imsave(
            output_path + os.sep + new_filename,
            output_img.astype("uint8"),
            check_contrast=False,
            quality=100,
        )

    return output_path


def get_GPU(num_GPU: str) -> None:
    """
    Configures GPU usage for TensorFlow based on provided parameter.

    Args:
        num_GPU (str): Number of GPUs to use. "0" for CPU only, "1" for single GPU.
    """
    num_GPU = str(num_GPU)
    if num_GPU == "0":
        logger.info("Not using GPU")
        print("Not using GPU")
        # use CPU (not recommended):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif num_GPU == "1":
        print("Using single GPU")
        logger.info("Using 1 GPU")
        # use first available GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if int(num_GPU) == 1:
        # read physical GPUs from machine
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        print(f"physical_devices (GPUs):{physical_devices}")
        logger.info(f"physical_devices (GPUs):{physical_devices}")
        if physical_devices:
            # Restrict TensorFlow to only use the first GPU
            try:
                tf.config.experimental.set_visible_devices(physical_devices, "GPU")
            except RuntimeError as e:
                # Visible devices must be set at program startup
                logger.error(e)
                print(e)
        # set mixed precision
        mixed_precision.set_global_policy("mixed_float16")
        # disable memory growth on all GPUs
        for i in physical_devices:
            tf.config.experimental.set_memory_growth(i, True)
            print(f"visible_devices: {tf.config.get_visible_devices()}")
            logger.info(f"visible_devices: {tf.config.get_visible_devices()}")
        # if multiple GPUs are used use mirror strategy
        if int(num_GPU) > 1:
            # Create a MirroredStrategy.
            strategy = tf.distribute.MirroredStrategy(
                [p.name.split("/physical_device:")[-1] for p in physical_devices],
                cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(),
            )
            print(f"Number of distributed devices: {strategy.num_replicas_in_sync}")
            logger.info(
                f"Number of distributed devices: {strategy.num_replicas_in_sync}"
            )


def get_sorted_files_with_extension(
    sample_direc: str, file_extensions: List[str]
) -> List[str]:
    """
    Get a sorted list of paths to files that have one of the file_extensions.
    It will return the first set of files that matches the first file_extension, so put the
    file_extension list in order of priority

    Args:
        sample_direc: A string representing the directory path to search for images.
        file_extensions: A list of file extensions to search for.

    Returns:
        A list of file paths for sample images found in the directory.

    """
    sample_filenames = []
    for ext in file_extensions:
        filenames = sorted(tf.io.gfile.glob(os.path.join(sample_direc, f"*{ext}")))
        sample_filenames.extend(filenames)
        if sample_filenames:
            break
    return sample_filenames


class Zoo_Model:
    """Machine learning model manager for coastal image segmentation and shoreline extraction."""

    def __init__(self) -> None:
        """
        Initializes Zoo_Model with default configuration.

        Sets up GDAL exceptions, initializes model attributes, and creates default settings.
        """
        gdal.UseExceptions()
        self.weights_directory = None
        self.model_types = []
        self.model_list = []
        self.metadata_dict = {}
        self.settings = {}
        # create default settings
        self.set_settings()

    def clear_zoo_model(self) -> None:
        """Resets all model attributes to initial state."""
        self.weights_directory = None
        self.model_types = []
        self.model_list = []
        self.metadata_dict = {}
        self.settings = {}

    def set_settings(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Updates settings dictionary with provided key-value pairs.

        Sets missing keys to default values as specified in default_settings.

        Args:
            **kwargs: Key-value pairs to add or update in settings.

        Returns:
            Dict[str, Any]: Copy of updated settings dictionary.

        Example:
            >>> model.set_settings(sat_list=['L8'], dates=['2020-01-01', '2020-12-31'])
        """
        # Check if any of the keys are missing
        # if any keys are missing set the default value
        default_settings = {
            "sample_direc": None,
            "use_GPU": "0",
            "implementation": "BEST",
            "model_type": "global_segformer_RGB_4class_14036903",
            "local_model_path": "",  # local path to the directory containing the model
            "use_local_model": False,  # Use local model (not one from zeneodo)
            "otsu": False,
            "tta": False,
            "cloud_thresh": 0.5,  # threshold on maximum cloud cover
            "dist_clouds": 300,  # ditance around clouds where shoreline can't be mapped
            "output_epsg": 4326,  # epsg code of spatial reference system desired for the output
            "save_figure": True,  # if True, saves a figure showing the mapped shoreline for each image
            # minimum area (in metres^2) for an object to be labelled as a beach
            "min_beach_area": 4500,
            # minimum length (in metres) of shoreline perimeter to be valid
            "min_length_sl": 100,
            # switch this parameter to True if sand pixels are masked (in black) on many images
            "cloud_mask_issue": False,
            # 'default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
            "sand_color": "default",
            "pan_off": "False",  # if True, no pan-sharpening is performed on Landsat 7,8 and 9 imagery
            "max_dist_ref": 25,
            "along_dist": 25,  # along-shore distance to use for computing the intersection
            "min_points": 3,  # minimum number of shoreline points to calculate an intersection
            "max_std": 15,  # max std for points around transect
            "max_range": 30,  # max range for points around transect
            "min_chainage": -100,  # largest negative value along transect (landwards of transect origin)
            "multiple_inter": "auto",  # mode for removing outliers ('auto', 'nan', 'max')
            "prc_multiple": 0.1,  # percentage of the time that multiple intersects are present to use the max
            "percent_no_data": 0.50,  # percentage of no data pixels allowed in an image (doesn't work for npz)
            "model_session_path": "",  # path to model session file
            "apply_cloud_mask": True,  # whether to apply cloud mask to images or not
            "drop_intersection_pts": False,  # whether to drop intersection points not on the transect
            "coastseg_version": __version__,  # version of coastseg used to generate the data
            "apply_segmentation_filter": True,  # whether to apply to sort the segmentations as good or bad
        }
        if kwargs:
            self.settings.update({key: value for key, value in kwargs.items()})

        for key, value in default_settings.items():
            self.settings.setdefault(key, value)

        logger.info(f"Settings: {self.settings}")
        return self.settings.copy()

    def get_settings(self) -> Dict[str, Any]:
        """
        Retrieves the current model settings.

        Returns:
            Dict[str, Any]: Dictionary containing model configuration settings.

        Raises:
            Exception: If no settings are found or settings is empty.
        """
        SETTINGS_NOT_FOUND = (
            "No settings found. Click save settings or load a config file."
        )
        logger.info(f"self.settings: {self.settings}")
        if self.settings is None or self.settings == {}:
            raise Exception(SETTINGS_NOT_FOUND)
        return self.settings

    def preprocess_data(
        self, src_directory: str, model_dict: dict, img_type: str, functions: list
    ) -> dict:
        """
        Preprocesses the data in the source directory and updates the model dictionary with the processed data.

        Args:
            src_directory (str): The path to the source directory containing the ROI's data
            model_dict (dict): The dictionary containing the model configuration and parameters.
            img_type (str): The type of imagery to generate. Must be one of "RGB", "NDWI", "MNDWI".
            functions (list): A list of preprocessing functions to apply sequentially to the data directory.

        Returns:
            dict: The updated model dictionary containing the paths to the processed data.
        """
        # Step 1: Get full path to directory named 'RGB' containing RGBs
        RGB_path = file_utilities.find_directory_recursively(src_directory, name="RGB")

        # Step 2: Convert RGB to required imagery
        current_path = get_imagery_directory(img_type, RGB_path)

        # Step 3: Apply each function in sequence
        for func in functions:
            current_path = func(current_path)

        # Step 4: Store the final processed directory in the model_dict
        model_dict["sample_direc"] = current_path
        return model_dict

    def run_model_and_extract_shorelines(
        self,
        input_directory: str,
        session_name: str,
        shoreline_path: str = "",
        transects_path: str = "",
        shoreline_extraction_area_path: str = "",
        coregistered: bool = False,
    ):
        """
        Runs the model and extracts shorelines using the segmented imagery.

        Assumes the settings have been set using `set_settings` method.

        Args:
            input_directory (str): The directory containing the input images to run the model on. Typically the RGB directory.
            session_name (str): The name of the session to save the model outputs and extracted shorelines.
                - This will create a new session directory in the sessions directory.
            shoreline_path (str, optional): The path to save the extracted shorelines. Defaults to "".
            transects_path (str, optional): The path to save the extracted transects. Defaults to "".
            shoreline_extraction_area_path (str, optional): The path to the shoreline extraction area. Defaults to "".
            coregistered (bool, optional): Whether the input images are coregistered. Defaults to False.

        Raises:
            ValueError: If the model type is not set in the settings.
            ValueError: If the input image type is not set in the settings.

        """

        settings = self.get_settings()

        progress_bar = tqdm.auto.tqdm(
            range(2),  # type: ignore
            leave=True,
        )

        # Step 1: Run model
        self.run_model_step(
            settings=settings,
            input_directory=input_directory,
            session_name=session_name,
            coregistered=coregistered,
            progress=progress_bar,
        )

        # Step 2: Load model info
        model_info = self.load_model_info(settings)

        # Step 3: Extract shorelines
        sessions_path = os.path.join(core_utilities.get_base_dir(), "sessions")
        session_directory = file_utilities.create_directory(sessions_path, session_name)
        # extract shorelines using the segmented imagery
        self.extract_shorelines_with_unet(
            settings,
            session_directory,
            session_name,
            model_info=model_info,
            shoreline_path=shoreline_path,
            transects_path=transects_path,
            shoreline_extraction_area_path=shoreline_extraction_area_path,
        )
        progress_bar.update(1)
        progress_bar.set_description_str(
            desc="Finished running model and extracting shorelines", refresh=True
        )

    def extract_shorelines_with_unet(
        self,
        settings: dict,
        session_path: str,
        session_name: str,
        model_info: ModelInfo,
        shoreline_path: str = "",
        transects_path: str = "",
        shoreline_extraction_area_path: str = "",
        **kwargs: dict,
    ) -> None:
        """
        Extracts shorelines using the outputs of running any of the Zoo models.
        This function saves the extracted shorelines and transects to a new session directory.

        Args:
            settings (dict): A dictionary containing the settings for shoreline extraction.
            session_path (str): The path to the model session directory containing the model outputs and configuration files.
            session_name (str): The name of the session to save the extracted shorelines to.
            shoreline_path (str, optional): The path to the shoreline data. Defaults to "".
            - If a geojson file is not provided, the program will attempt to load default shorelines and if that fails it will raise an error.
            transects_path (str, optional): The path to the transects data. Defaults to "".
            - If a geojson file is not provided, the program will attempt to load default transects and if that fails it will raise an error.
            **kwargs (dict): Additional keyword arguments.
            shoreline_extraction_area_path (str, optional): The path to the shoreline extraction area. Defaults to "".
        Returns:
            None
        """
        logger.info(f"extract_shoreline_settings: {settings}")

        # save the selected model session
        settings["model_session_path"] = session_path
        self.set_settings(**settings)
        settings = self.get_settings()
        good_bad_csv = None

        # create session path to store extracted shorelines and transects
        new_session_path = create_new_session_path(session_name)

        # load the ROI settings from the config file
        # @todo by default only the first ROI is loaded change this in the future to allow any ROI ID to be loaded
        roi_settings, roi_id = load_roi_settings(
            session_path,
            roi_id=(
                str(kwargs.get("roi_id")) if kwargs.get("roi_id") is not None else None
            ),
        )
        good_bad_csv = load_good_bad_csv(roi_id, roi_settings)
        logger.info(f"roi_settings: {roi_settings}")

        # read ROI from config geojson file
        roi_gdf = load_roi_gdf_from_session(session_path, roi_id)

        # get roi_id from source directory path in model settings
        model_settings = file_utilities.load_json_data_from_file(
            session_path, "model_settings.json"
        )

        # save model settings to session path
        model_settings_path = os.path.join(new_session_path, "model_settings.json")
        file_utilities.to_file(model_settings, model_settings_path)

        # load transects and shorelines
        output_epsg = settings["output_epsg"]
        transects_gdf = geodata_processing.create_geofeature_geodataframe(
            transects_path, roi_gdf, output_epsg, "transect"
        )
        shoreline_gdf = geodata_processing.create_geofeature_geodataframe(
            shoreline_path, roi_gdf, output_epsg, "shoreline"
        )
        shoreline_extraction_area_gdf = None
        # load the shoreline extraction area from the geojson file
        logger.info(f"shoreline_extraction_area_path: {shoreline_extraction_area_path}")
        logger.info(
            f"shoreline_extraction_area_path exists: {os.path.exists(shoreline_extraction_area_path)}"
        )
        if os.path.exists(shoreline_extraction_area_path):
            shoreline_extraction_area_gdf = geodata_processing.load_feature_from_file(
                shoreline_extraction_area_path, "shoreline_extraction_area"
            )
            logger.info(
                f"shoreline_extraction_area_gdf: {shoreline_extraction_area_gdf}"
            )

        # Update the CRS to the most accurate crs for the ROI this makes extracted shoreline more accurate
        new_espg = common.get_most_accurate_epsg(output_epsg, roi_gdf)
        settings["output_epsg"] = new_espg
        self.set_settings(output_epsg=new_espg)
        # convert the ROI to the new CRS
        roi_gdf = roi_gdf.to_crs(output_epsg)

        # save the config files to the new session location
        common.save_config_files(
            new_session_path,
            roi_ids=[roi_id],
            roi_settings=roi_settings,
            shoreline_settings=settings,
            transects_gdf=transects_gdf,
            shorelines_gdf=shoreline_gdf,
            roi_gdf=roi_gdf,
            epsg_code="epsg:4326",
            shoreline_extraction_area_gdf=shoreline_extraction_area_gdf,
        )

        # extract shorelines
        extracted_shorelines = extracted_shoreline.Extracted_Shoreline()
        extracted_shorelines = (
            extracted_shorelines.create_extracted_shorelines_from_session(
                model_info,
                roi_id,
                shoreline_gdf,
                roi_settings[roi_id],
                settings,
                session_path,
                new_session_path,
                shoreline_extraction_area=shoreline_extraction_area_gdf,
                apply_segmentation_filter=settings.get(
                    "apply_segmentation_filter", True
                ),
                **kwargs,
            )
        )

        good_bad_seg_csv = ""
        # If the segmentation filter is applied read the model scores
        if settings.get("apply_segmentation_filter", False):
            if os.path.exists(
                os.path.join(session_path, "segmentation_classification_results.csv")
            ):
                good_bad_seg_csv = os.path.join(
                    session_path, "segmentation_classification_results.csv"
                )
                # copy it to the new session path
                try:
                    shutil.copy(good_bad_seg_csv, new_session_path)
                except SameFileError:
                    pass  # we don't care if the file is the same
        # save extracted shorelines as geojson, detection jpgs, configs, model settings files to the session directory
        common.save_extracted_shorelines(extracted_shorelines, new_session_path)
        # save the classification scores to the extracted shorelines geojson files
        shorelines_lines_location = os.path.join(
            session_path, "extracted_shorelines_lines.geojson"
        )
        shorelines_points_location = os.path.join(
            session_path, "extracted_shorelines_points.geojson"
        )

        files = set([shorelines_lines_location, shorelines_points_location])
        add_classifer_scores_to_shorelines(good_bad_csv, good_bad_seg_csv, files=files)

        # common.save_extracted_shoreline_figures(extracted_shorelines, new_session_path)
        print(f"Saved extracted shorelines to {new_session_path}")

        # transects must be in the same CRS as the extracted shorelines otherwise intersections will all be NAN
        if not transects_gdf.empty:
            transects_gdf = transects_gdf.to_crs("EPSG:4326")

        # new method to compute intersections
        # Currently the method requires both the transects and extracted shorelines to be in the same CRS 4326
        extracted_shorelines_gdf_lines = extracted_shorelines.gdf.copy().to_crs(
            "EPSG:4326"
        )

        # Compute the transect timeseries by intersecting each transect with each extracted shoreline
        transect_timeseries_df = transect_timeseries(
            extracted_shorelines_gdf_lines, transects_gdf
        )
        # save two version of the transect timeseries, the transect settings and the transects as a dictionary
        save_transects(
            new_session_path,
            transect_timeseries_df,
            settings,
            ext="raw",
            good_bad_csv=good_bad_csv,
            good_bad_seg_csv=good_bad_seg_csv,
        )

    def postprocess_data(
        self,
        preprocessed_data: Dict[str, Any],
        session: sessions.Session,
        roi_directory: str,
    ) -> None:
        """
        Moves model outputs and copies config files to session directory.

        Args:
            preprocessed_data (Dict[str, Any]): Dictionary of inputs to the model.
            session (sessions.Session): Session object to track and save session.
            roi_directory (str): Directory containing downloaded data for a single ROI.

        Raises:
            Exception: If no model outputs were generated.
        """
        # get roi_ids
        session_path = session.path
        outputs_path = os.path.join(preprocessed_data["sample_direc"], "out")
        if not os.path.exists(outputs_path):
            logger.warning("No model outputs were generated")
            print("No model outputs were generated")
            raise Exception(
                f"No model outputs were generated. Check if {roi_directory} contained enough data to run the model or try raising the percentage of no data allowed."
            )
        logger.info(f"Moving from {outputs_path} files to {session_path}")

        # if configs do not exist then raise an error and do not save the session
        if not file_utilities.validate_config_files_exist(roi_directory):
            logger.warning(
                f"Config files config.json or config_gdf.geojson do not exist in roi directory {roi_directory}"
            )
            raise FileNotFoundError(
                f"Config files config.json or config_gdf.geojson do not exist in roi directory {roi_directory}"
            )
        # modify the config.json to only have the ROI ID that was used and save to session directory
        roi_id = file_utilities.extract_roi_id(roi_directory)
        common.save_new_config(
            os.path.join(roi_directory, "config.json"),
            roi_id,
            os.path.join(session_path, "config.json"),
        )
        # Copy over the config_gdf.geojson file
        config_gdf_path = os.path.join(roi_directory, "config_gdf.geojson")
        if os.path.exists(config_gdf_path):
            # Read in the GeoJSON file using geopandas
            gdf = gpd.read_file(config_gdf_path)

            # Project the GeoDataFrame to EPSG:4326
            gdf_4326 = gdf.to_crs("EPSG:4326")

            # Save the projected GeoDataFrame to a new GeoJSON file
            gdf_4326.to_file(
                os.path.join(session_path, "config_gdf.geojson"), driver="GeoJSON"
            )
        model_settings_path = os.path.join(session_path, "model_settings.json")
        file_utilities.write_to_json(model_settings_path, preprocessed_data)

        # copy files from out to session folder
        file_utilities.move_files(outputs_path, session_path, delete_src=True)
        session.save(session.path)

    def postprocess_data_without_session(
        self,
        preprocessed_data: dict,
        session: sessions.Session,
    ):
        """Moves the model outputs from
        as well copies the config files from the roi directory to the session directory

        Args:
            preprocessed_data (dict): dictionary of inputs to the model
            session (sessions.Session): session object that's used to keep track of session
            saves the session to the sessions directory

            typically starts with "ID_{roi_id}"
        """
        # get roi_ids
        session_path = session.path
        outputs_path = os.path.join(preprocessed_data["sample_direc"], "out")
        if not os.path.exists(outputs_path):
            logger.warning("No model outputs were generated")
            print("No model outputs were generated")
            raise Exception(
                f"No model outputs were generated. Check if {outputs_path} contained enough data to run the model or try raising the percentage of no data allowed."
            )
        logger.info(f"Moving from {outputs_path} files to {session_path}")

        model_settings_path = os.path.join(session_path, "model_settings.json")
        file_utilities.write_to_json(model_settings_path, preprocessed_data)

        # copy files from out to session folder
        file_utilities.move_files(outputs_path, session_path, delete_src=True)
        session.save(session.path)

    def get_weights_directory(self, model_implementation: str, model_id: str) -> str:
        """
        Retrieves the directory path where the model weights are stored.
        This method determines whether to use a local model path or to download the model
        from a remote source based on the settings provided. If the local model path is
        specified and exists, it will use that path. Otherwise, it will create a directory
        for the model and download the weights.
        Args:
            model_implementation (str): The implementation type of the model either 'BEST' or 'ENSEMBLE'
            model_id (str): The identifier for the model. This is the zenodo ID located at the end of the URL
        Returns:
            str: The directory path where the model weights are stored.
        Raises:
            FileNotFoundError: If the local model path is specified but does not exist.
        """

        USE_LOCAL_MODEL = self.settings.get("use_local_model", False)
        if USE_LOCAL_MODEL:
            LOCAL_MODEL_PATH = self.get_local_model_path()

        # check if a local model should be loaded or not
        if not USE_LOCAL_MODEL:
            # create the model directory & download the model
            weights_directory = self.get_model_directory(model_id)
            self.download_model(model_implementation, model_id, weights_directory)
        else:
            # load the model from the local model path
            weights_directory = LOCAL_MODEL_PATH

        return weights_directory

    def prepare_model(self, model_implementation: str, model_id: str):
        """
        Prepares the model for use by downloading the required files and loading the model.

        Args:
            model_implementation (str): The model implementation either 'BEST' or 'ENSEMBLE'
            model_id (str): The ID of the model.
        """
        # weights_directory is the directory that contains the model weights, the model card json files and the BEST_MODEL.txt file
        self.weights_directory = self.get_weights_directory(
            model_implementation, model_id
        )
        logger.info(f"self.weights_directory:{self.weights_directory}")

        weights_list = self.get_weights_list(model_implementation)

        # Load the model from the config files
        model, model_list, config_files, model_types = self.get_model(weights_list)
        logger.info(f"self.TARGET_SIZE: {self.TARGET_SIZE}")
        logger.info(f"self.N_DATA_BANDS: {self.N_DATA_BANDS}")
        logger.info(f"self.TARGET_SIZE: {self.TARGET_SIZE}")

        self.model_types = model_types
        self.model_list = model_list
        self.metadata_dict = self.get_metadatadict(
            weights_list, config_files, model_types
        )
        logger.info(f"self.metadatadict: {self.metadata_dict}")

    def get_metadatadict(
        self, weights_list: list, config_files: list, model_types: list
    ) -> dict:
        """
        Returns a dictionary containing metadata information.

        Args:
            weights_list (list): A list of model weights.
            config_files (list): A list of configuration files.
            model_types (list): A list of model types.

        Returns:
            dict: A dictionary containing the metadata information.
        """
        metadatadict = {}
        metadatadict["model_weights"] = weights_list
        metadatadict["config_files"] = config_files
        metadatadict["model_types"] = model_types
        return metadatadict

    def run_model_without_session(
        self,
        img_type: str,
        model_implementation: str,
        session_name: str,
        src_directory: str,
        model_name: str,
        use_GPU: str,
        use_otsu: bool,
        use_tta: bool,
        percent_no_data: float,
        coregistered: bool = False,
    ):
        """
        Runs the model for image segmentation.

        Args:
            img_type (str): The type of image.
            model_implementation (str): The implementation of the model.
            session_name (str): The name of the session.
            src_directory (str): The directory of RGB images.
            model_name (str): The name of the model.
            use_GPU (str): Whether to use GPU or not.
            use_otsu (bool): Whether to use Otsu thresholding or not.
            use_tta (bool): Whether to use test-time augmentation or not.
            percent_no_data (float): The percentage of no data allowed in the image.
            coregistered (bool, optional): Whether the images are coregistered or not. Defaults to False.

        Returns:
            None
        """
        logger.info(f"Selected directory of RGBs: {src_directory}")
        logger.info(f"session name: {session_name}")
        logger.info(f"model_name: {model_name}")
        logger.info(f"model_implementation: {model_implementation}")
        logger.info(f"use_GPU: {use_GPU}")
        logger.info(f"use_otsu: {use_otsu}")
        logger.info(f"use_tta: {use_tta}")

        print(f"Running model {model_name}")
        # print(f"self.settings: {self.settings}")
        self.prepare_model(model_implementation, model_name)

        # create a session
        session = sessions.Session()
        sessions_path = file_utilities.create_directory(
            core_utilities.get_base_dir(), "sessions"
        )
        session.path = file_utilities.create_directory(sessions_path, session_name)

        session.name = session_name
        model_dict = {
            "use_GPU": use_GPU,
            "sample_direc": "",
            "implementation": model_implementation,
            "model_type": model_name,
            "otsu": use_otsu,
            "tta": use_tta,
            "percent_no_data": percent_no_data,
        }

        print(f"Preprocessing the data at {src_directory}")

        model_dict = self.preprocess_data(
            src_directory, model_dict, img_type, functions=[apply_smooth_otsu_to_folder]
        )
        logger.info(f"model_dict after preprocessing: {model_dict}")

        self.compute_segmentation(model_dict, percent_no_data)
        self.postprocess_data_without_session(model_dict, session)
        print(f"\n Model results saved to {session.path}")

    def run_model(
        self,
        img_type: str,
        model_implementation: str,
        session_name: str,
        src_directory: str,
        model_name: str,
        use_GPU: str,
        use_otsu: bool,
        use_tta: bool,
        percent_no_data: float,
        coregistered: bool = False,
    ):
        """
        Runs the model for image segmentation.

        Args:
            img_type (str): The type of image.
            model_implementation (str): The implementation of the model.
            session_name (str): The name of the session.
            src_directory (str): The directory of RGB images.
            model_name (str): The name of the model.
            use_GPU (str): Whether to use GPU or not.
            use_otsu (bool): Whether to use Otsu thresholding or not.
            use_tta (bool): Whether to use test-time augmentation or not.
            percent_no_data (float): The percentage of no data allowed in the image.
            coregistered (bool, optional): Whether the images are coregistered or not. Defaults to False.

        Returns:
            None
        """
        logger.info(f"Selected directory of RGBs: {src_directory}")
        logger.info(f"session name: {session_name}")
        logger.info(f"model_name: {model_name}")
        logger.info(f"model_implementation: {model_implementation}")
        logger.info(f"use_GPU: {use_GPU}")
        logger.info(f"use_otsu: {use_otsu}")
        logger.info(f"use_tta: {use_tta}")

        print(f"Running model {model_name}")
        # print(f"self.settings: {self.settings}")
        self.prepare_model(model_implementation, model_name)

        # create a session
        session = sessions.Session()
        sessions_path = file_utilities.create_directory(
            core_utilities.get_base_dir(), "sessions"
        )
        session.path = file_utilities.create_directory(sessions_path, session_name)
        session.name = session_name
        local_model_path = self.get_local_model_path()
        model_dict = {
            "use_GPU": use_GPU,
            "sample_direc": "",
            "implementation": model_implementation,
            "model_type": model_name,
            "otsu": use_otsu,
            "tta": use_tta,
            "percent_no_data": percent_no_data,
            "use_local_model": self.settings.get("use_local_model", False),
            "local_model_path": local_model_path,
        }
        # @todo instead of requiring an ROI directory this could be passed in
        # get parent roi_directory from the selected imagery directory
        roi_directory = file_utilities.find_parent_directory(
            src_directory, "ID_", "data"
        )
        if not roi_directory:
            raise ValueError(
                f"The selected directory {src_directory} is not in a ROI directory. Please select a directory that is in a ROI directory that starts with 'ID_'"
            )

        if coregistered:
            roi_directory = os.path.join(roi_directory, "coregistered")

        print(f"Preprocessing the data at {roi_directory}")
        # DONT UNCOMMENT THE LINE BELOW: Logic to apply the smooth otsu filter to the RGB folder I didn't use it since I modified do_seg to apply the smooth otsu filter to the arrays instead
        # model_dict = self.preprocess_data(src_directory, model_dict, img_type,functions=[apply_smooth_otsu_to_folder])
        model_dict = self.preprocess_data(
            roi_directory, model_dict, img_type, functions=[]
        )
        logger.info(f"model_dict: {model_dict}")

        self.compute_segmentation(model_dict, percent_no_data)
        self.postprocess_data(model_dict, session, roi_directory)
        session.add_roi_ids([file_utilities.extract_roi_id(roi_directory)])
        print(f"\n Model results saved to {session.path}")

    def get_model_directory(self, model_id: str):
        # Create a directory to hold the downloaded models
        downloaded_models_path = common.get_downloaded_models_dir()
        model_directory = file_utilities.create_directory(
            downloaded_models_path, model_id
        )
        return model_directory

    def get_files_for_seg(
        self,
        sample_direc: str,
        avoid_patterns: List[str] = [],
        percent_no_data: float = 0.50,
    ) -> list:
        """
        Returns a list of files to be segmented.

        The function reads in the image filenames as either (`.npz`) OR (`.jpg`, or `.png`)
        and returns a sorted list of the file paths.

        Args:
        - sample_direc (str): The directory containing files to be segmented.
        - avoid_patterns (List[str], optional): A list of file names to be avoided.Don't include any file extensions. Default is [].
        - percent_no_data (float, optional): The percentage of no data pixels allowed in an image. Default is 0.50.
        Returns:
        - list: A list of full pathes files to be segmented.
        """
        logger.info(f"Searching directory for files: {sample_direc}")
        file_extensions = [".npz", ".jpg", ".png"]
        model_ready_files = get_sorted_files_with_extension(
            sample_direc, file_extensions
        )

        if not model_ready_files:
            raise FileNotFoundError(
                f"No files found in {sample_direc} with extensions {file_extensions}"
            )

        # filter out files whose filenames match any of the avoid_patterns
        model_ready_files = file_utilities.filter_files(
            model_ready_files, avoid_patterns
        )

        # Filter based on no-data pixel percentage
        initial_count = len(model_ready_files)
        # filter out files with no data pixels greater than percent_no_data
        model_ready_files = filter_no_data_pixels(model_ready_files, percent_no_data)
        filtered_count = initial_count - len(model_ready_files)

        print(
            f"{filtered_count}/{initial_count} files filtered out "
            f"due to > {percent_no_data:.0%} no-data pixels."
        )

        return model_ready_files

    def get_model_name(self) -> str:
        model_name = self.settings.get("model_type")
        if not model_name:
            raise ValueError(
                "Model type (aka the model name) must be specified in settings."
            )
        return model_name

    def get_local_model_path(self) -> str:
        """
        Returns the local model path if it exists, otherwise returns an empty string.

        Returns:
            str: The local model path or an empty string if not set.
        """
        model_path = self.settings.get("local_model_path", "")
        if model_path == "":
            return ""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"The model folder specified by 'local_model_path' does not exist: {model_path}"
            )
        return model_path

    def load_model_info(self, settings: dict) -> ModelInfo:
        """
        Load the model information based on settings.

        Args:
            settings (dict): Settings dictionary.

        Returns:
            ModelInfo: Loaded model information.

        Raises:
            FileNotFoundError: If a local model path is specified but does not exist.
        """
        # if a local model path was use then load the model info from the local path
        # Otherwise load the model info using the model_type (aka the model folder name) that was downloaded

        if settings.get("use_local_model", False):
            model_path = self.get_local_model_path()
            model_info = ModelInfo(model_directory=model_path)
        else:
            model_name = settings.get("model_type")
            model_info = ModelInfo(
                model_name=model_name
            )  # this will load the model info from the downloaded models folder

        model_info.load()
        return model_info

    def run_model_step(
        self,
        settings: dict,
        input_directory: str,
        session_name: str,
        coregistered: bool,
        progress: tqdm.tqdm,
    ) -> None:
        """
        Executes a single step of the model using the provided settings and parameters.
        This method retrieves model configuration from the `settings` dictionary and
        runs the model with the specified options. It also provides progress feedback
        using a tqdm progress bar.
        Args:
            settings (dict): Dictionary containing model configuration options.
                Expected keys include:
                    - "model_type": Name of the model to use (required).
                    - "img_type": Type of input images (required).
                    - "implementation": Model implementation variant (optional, default "BEST").
                    - "use_GPU": The GPU device to use. Defaults to "0" (optional).
                    - "otsu": Whether to use Otsu thresholding (optional).
                    - "tta": Whether to use test-time augmentation (optional).
                    - "percent_no_data": Allowed percentage of no-data pixels (optional).
            input_directory (str): Path to the directory containing input images.
            session_name (str): Name of the current session for output organization.
            coregistered (bool): Whether the input images are coregistered.
            progress: tqdm progress bar for tracking the execution progress.
        Raises:
            ValueError: If required settings ("model_type" or "img_type") are missing.
        Returns:
            None
        """
        model_name = self.get_model_name()

        img_type = settings.get("img_type")
        if img_type is None:
            raise ValueError("Input image type must be specified in settings.")

        model_implementation = settings.get("implementation", "BEST")

        progress.set_description(f"Running {model_name} model")
        self.run_model(
            img_type=img_type,
            model_implementation=model_implementation,
            session_name=session_name,
            src_directory=input_directory,
            model_name=model_name,
            use_GPU=settings.get("use_GPU", "0"),
            use_otsu=settings.get("otsu", False),
            use_tta=settings.get("tta", False),
            percent_no_data=settings.get("percent_no_data", 0.50),
            coregistered=coregistered,
        )
        progress.set_description("Model run complete. Extracting shorelines")
        progress.update(1)

    def compute_segmentation(
        self,
        preprocessed_data: dict,
        percent_no_data: float = 0.50,
    ):
        """
        Compute the segmentation for a given set of preprocessed data.

        Args:
            preprocessed_data (dict): A dictionary containing preprocessed data.
                This dictionary should contain the following keys:
                - sample_direc (str): The directory containing the sample images.
                - tta (bool): Whether to use test-time augmentation.
                - otsu (bool): Whether to use Otsu thresholding.

            percent_no_data (float, optional): The max ercentage of no data pixels allowed in the image. Defaults to 0.50.

        Returns:
            None
        """
        sample_direc = preprocessed_data["sample_direc"]
        use_tta = preprocessed_data["tta"]
        use_otsu = preprocessed_data["otsu"]
        # Create list of files of types .npz,.jpg, or .png to run model on
        files_to_segment = self.get_files_for_seg(
            sample_direc, avoid_patterns=[], percent_no_data=percent_no_data
        )
        logger.info(f"files_to_segment: {files_to_segment}")
        if self.model_types[0] != "segformer":
            ### mixed precision
            from tensorflow.keras import mixed_precision  # type: ignore

            mixed_precision.set_global_policy("mixed_float16")
        # Compute the segmentation for each of the files
        print(f"Found {len(files_to_segment)} files to run on model on")
        for file_to_seg in tqdm.auto.tqdm(files_to_segment, desc="Applying Model"):
            do_seg(
                file_to_seg,
                self.model_list,
                self.metadata_dict,
                self.model_types[0],
                sample_direc=sample_direc,
                NCLASSES=self.NCLASSES,
                N_DATA_BANDS=self.N_DATA_BANDS,
                TARGET_SIZE=self.TARGET_SIZE,
                TESTTIMEAUG=use_tta,
                WRITE_MODELMETADATA=False,
                OTSU_THRESHOLD=use_otsu,
                profile="meta",
                apply_smooth=True if "S1" in file_to_seg else False,
            )

    def get_model(self, weights_list: list):
        config_files = []
        logger.info(f"weights_list: {weights_list}")
        if weights_list == []:
            raise Exception("No Model Info Passed")
        for weights in weights_list:
            # "fullmodel" is for serving on zoo they are smaller and more portable between systems than traditional h5 files
            # gym makes a h5 file, then you use gym to make a "fullmodel" version then zoo can read "fullmodel" version
            weights = weights.strip()
            config_file = weights.replace(".h5", ".json").replace("weights", "config")
            if "fullmodel" in config_file:
                config_file = config_file.replace("_fullmodel", "")
            with open(config_file) as f:
                config = json.load(f)
            self.TARGET_SIZE = config.get("TARGET_SIZE")
            MODEL = config.get("MODEL")
            self.NCLASSES = config.get("NCLASSES")
            KERNEL = config.get("KERNEL")
            STRIDE = config.get("STRIDE")
            FILTERS = config.get("FILTERS")
            self.N_DATA_BANDS = config.get("N_DATA_BANDS")
            DROPOUT = config.get("DROPOUT")
            DROPOUT_CHANGE_PER_LAYER = config.get("DROPOUT_CHANGE_PER_LAYER")
            DROPOUT_TYPE = config.get("DROPOUT_TYPE")
            USE_DROPOUT_ON_UPSAMPLING = config.get("USE_DROPOUT_ON_UPSAMPLING")
            try:
                model = tf.keras.models.load_model(weights)
                #  nclasses=NCLASSES, may have to replace nclasses with NCLASSES
            except BaseException:
                if MODEL == "resunet":
                    model = custom_resunet(
                        (self.TARGET_SIZE[0], self.TARGET_SIZE[1], self.N_DATA_BANDS),
                        FILTERS,
                        nclasses=[
                            self.NCLASSES + 1 if self.NCLASSES == 1 else self.NCLASSES
                        ][0],
                        kernel_size=(KERNEL, KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,  # 0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                        dropout_type=DROPOUT_TYPE,  # "standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                    )
                elif MODEL == "unet":
                    model = custom_unet(
                        (self.TARGET_SIZE[0], self.TARGET_SIZE[1], self.N_DATA_BANDS),
                        FILTERS,
                        nclasses=[
                            self.NCLASSES + 1 if self.NCLASSES == 1 else self.NCLASSES
                        ][0],
                        kernel_size=(KERNEL, KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,  # 0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                        dropout_type=DROPOUT_TYPE,  # "standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                    )

                elif MODEL == "simple_resunet":
                    # num_filters = 8 # initial filters
                    model = simple_resunet(
                        (self.TARGET_SIZE[0], self.TARGET_SIZE[1], self.N_DATA_BANDS),
                        kernel=(2, 2),
                        num_classes=[
                            self.NCLASSES + 1 if self.NCLASSES == 1 else self.NCLASSES
                        ][0],
                        activation="relu",
                        use_batch_norm=True,
                        dropout=DROPOUT,  # 0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                        dropout_type=DROPOUT_TYPE,  # "standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                        filters=FILTERS,  # 8,
                        num_layers=4,
                        strides=(1, 1),
                    )
                # 346,564
                elif MODEL == "simple_unet":
                    model = simple_unet(
                        (self.TARGET_SIZE[0], self.TARGET_SIZE[1], self.N_DATA_BANDS),
                        kernel=(2, 2),
                        num_classes=[
                            self.NCLASSES + 1 if self.NCLASSES == 1 else self.NCLASSES
                        ][0],
                        activation="relu",
                        use_batch_norm=True,
                        dropout=DROPOUT,  # 0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                        dropout_type=DROPOUT_TYPE,  # "standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                        filters=FILTERS,  # 8,
                        num_layers=4,
                        strides=(1, 1),
                    )
                elif MODEL == "satunet":
                    model = simple_satunet(
                        (self.TARGET_SIZE[0], self.TARGET_SIZE[1], self.N_DATA_BANDS),
                        kernel=(2, 2),
                        num_classes=self.NCLASSES,  # [NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                        activation="relu",
                        use_batch_norm=True,
                        dropout=DROPOUT,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                        dropout_type=DROPOUT_TYPE,
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                        filters=FILTERS,
                        num_layers=4,
                        strides=(1, 1),
                    )
                elif MODEL == "segformer":
                    id2label = {}
                    for k in range(self.NCLASSES):
                        id2label[k] = str(k)
                    model = segformer(id2label, num_classes=self.NCLASSES)
                    model.compile(optimizer="adam")
                # 242,812
                else:
                    raise Exception(
                        f"An unknown model type {MODEL} was received. Please select a valid model.\n \
                        Model must be one of 'unet', 'resunet', 'segformer', or 'satunet'"
                    )

                # Load in the custom loss function from doodleverse_utils
                model.compile(
                    optimizer="adam", loss=dice_coef_loss(self.NCLASSES)
                )  # , metrics = [iou_multi(self.NCLASSESNCLASSES), dice_multi(self.NCLASSESNCLASSES)])

                # If model is a tuple (e.g., Segformer), extract the actual model before loading weights
                if isinstance(model, tuple):
                    model_obj = model[0]
                else:
                    model_obj = model
                model_obj.load_weights(weights)

            self.model_types.append(MODEL)
            self.model_list.append(model)
            config_files.append(config_file)
        return model, self.model_list, config_files, self.model_types

    def get_weights_list(self, model_choice: str = "ENSEMBLE") -> List[str]:
        """Returns a list of the model weights files (.h5) within the weights directory.

        Args:
            model_choice (str, optional): The type of model weights to return.
                Valid choices are 'ENSEMBLE' (default) to return all available
                weights files or 'BEST' to return only the best model weights file.

        Returns:
            list: A list of strings representing the file paths to the model weights
            files in the weights directory.

        Raises:
            FileNotFoundError: If the BEST_MODEL.txt file is not found in the weights directory.

        Example:
            trainer = ModelTrainer(weights_direc='/path/to/weights')
            weights_list = trainer.get_weights_list(model_choice='ENSEMBLE')
            print(weights_list)
            # Output: ['/path/to/weights/model1.h5', '/path/to/weights/model2.h5', ...]

            best_weights_list = trainer.get_weights_list(model_choice='BEST')
            print(best_weights_list)
            # Output: ['/path/to/weights/best_model.h5']
        """
        logger.info(f"{model_choice}")
        if model_choice == "ENSEMBLE":
            weights_list = glob(os.path.join(self.weights_directory, "*.h5"))
            logger.info(f"weights_list: {weights_list}")
            logger.info(f"{len(weights_list)} sets of model weights were found ")
            return weights_list
        elif model_choice == "BEST":
            # read model name (fullmodel.h5) from BEST_MODEL.txt
            with open(os.path.join(self.weights_directory, "BEST_MODEL.txt")) as f:
                model_name = f.readline()
            logger.info(f"model_name: {model_name}")
            # remove any leading or trailing whitespace and newline characters
            model_name = model_name.strip()
            weights_list = [os.path.join(self.weights_directory, model_name)]
            logger.info(f"weights_list: {weights_list}")
            logger.info(f"{len(weights_list)} sets of model weights were found ")
            return weights_list
        else:
            raise ValueError(
                f"Invalid model_choice: {model_choice}. Valid choices are 'ENSEMBLE' or 'BEST'."
            )

    def download_best(
        self, available_files: List[dict], model_path: Optional[str], model_id: str
    ):
        """
        Downloads the best model file and its corresponding JSON and classes.txt files from the given list of available files.

        Args:
            available_files (list): A list of files available to download.
            model_path (str): The local directory where the downloaded files will be stored.
            model_id (str): The ID of the model being downloaded.

        Raises:
            ValueError: If BEST_MODEL.txt file is not found in the given model_id.

        Returns:
            None
        """
        download_dict = {}
        # download best_model.txt and read the name of the best model
        best_model_json = next(
            (f for f in available_files if f["key"] == "BEST_MODEL.txt"), None
        )
        if best_model_json is None:
            raise ValueError(f"Cannot find BEST_MODEL.txt in {model_id}")
        # download best model file to check if it exists
        BEST_MODEL_txt_path = os.path.join(model_path, "BEST_MODEL.txt")
        logger.info(f"model_path for BEST_MODEL.txt: {BEST_MODEL_txt_path}")
        # if best BEST_MODEL.txt file not exist then download it
        if not os.path.isfile(BEST_MODEL_txt_path):
            common.download_url(
                best_model_json["links"]["self"],
                BEST_MODEL_txt_path,
                "Downloading best_model.txt",
            )

        with open(BEST_MODEL_txt_path, "r") as f:
            best_model_filename = f.read().strip()
        # get the json data of the best model _fullmodel.h5 file
        best_model_filename = best_model_filename.strip()
        best_json_filename = best_model_filename.replace("_fullmodel.h5", ".json")
        best_modelcard_filename = best_model_filename.replace(
            "_fullmodel.h5", "_modelcard.json"
        ).replace("_segformer", "")

        # download best model files(.h5, .json) file
        download_filenames = [
            best_json_filename,
            best_model_filename,
            best_modelcard_filename,
        ]
        logger.info(f"download_filenames: {download_filenames}")
        download_dict.update(
            get_files_to_download(
                available_files, download_filenames, model_id, model_path
            )
        )
        download_dict = check_if_files_exist(download_dict)
        # download the files that don't exist
        logger.info(f"URLs to download: {download_dict}")
        # if any files are not found locally download them asynchronous
        if download_dict != {}:
            download_status = download_url_dict(download_dict)
            if not download_status:
                raise Exception("Download failed")

    def download_ensemble(
        self, available_files: List[dict], model_path: str, model_id: str
    ):
        """
        Downloads all the model files and their corresponding JSON and classes.txt files from the given list of available files, for an ensemble model.

        Args:
            available_files (list): A list of files available to download.
            model_path (str): The local directory where the downloaded files will be stored.
            model_id (str): The ID of the model being downloaded.

        Raises:
            Exception: If no .h5 files or corresponding .json files are found in the given model_id.

        Returns:
            None
        """
        download_dict = {}
        # get json and models
        all_models_reponses = [f for f in available_files if f["key"].endswith(".h5")]
        all_model_names = [f["key"] for f in all_models_reponses]
        json_file_names = [
            model_name.replace("_fullmodel.h5", ".json")
            for model_name in all_model_names
        ]
        modelcard_file_names = [
            model_name.replace("_fullmodel.h5", "_modelcard.json").replace(
                "_segformer", ""
            )
            for model_name in all_model_names
        ]
        all_json_reponses = []

        # for each filename online check if there a .json file
        for available_file in available_files:
            if available_file["key"] in json_file_names + modelcard_file_names:
                all_json_reponses.append(available_file)
        if len(all_models_reponses) == 0:
            raise Exception(f"Cannot find any .h5 files at {model_id}")
        if len(all_json_reponses) == 0:
            raise Exception(
                f"Cannot find corresponding .json or .modelcard.json files for .h5 files at {model_id}"
            )

        logger.info(f"all_models_reponses : {all_models_reponses}")
        logger.info(f"all_json_reponses : {all_json_reponses}")
        for response in all_models_reponses + all_json_reponses:
            # get the link of the best model
            link = response["links"]["self"]
            filename = response["key"]
            filepath = os.path.join(model_path, filename)
            download_dict[filepath] = link

        download_dict.update(
            get_files_to_download(available_files, [], model_id, model_path)
        )
        download_dict = check_if_files_exist(download_dict)
        # download the files that don't exist
        logger.info(f"download_dict: {download_dict}")
        # if any files are not found locally download them asynchronous
        if download_dict != {}:
            download_status = download_url_dict(download_dict)
            if not download_status:
                raise Exception("Download failed")

    def download_model(
        self, model_choice: str, model_id: str, model_path: Optional[str] = None
    ) -> None:
        """downloads model specified by zenodo id in model_id.

        Downloads best model is model_choice = 'BEST' or all models in
        zenodo release if model_choice = 'ENSEMBLE'

        Args:
            model_choice (str): 'BEST' or 'ENSEMBLE'
            model_id (str): name of model followed by underscore zenodo_id ex.'orthoCT_RGB_2class_7574784'
            model_path (str): path to directory to save the downloaded files to
        """
        # Extract the Zenodo ID from the dataset ID
        zenodo_id = model_id.split("_")[-1]
        # get list of files available in zenodo release
        json_content = get_zenodo_release(zenodo_id)
        available_files = json_content["files"]

        # Download the best model if best or all models if ensemble
        if model_choice.upper() == "BEST":
            self.download_best(available_files, model_path, model_id)
        elif model_choice.upper() == "ENSEMBLE":
            self.download_ensemble(available_files, model_path, model_id)
