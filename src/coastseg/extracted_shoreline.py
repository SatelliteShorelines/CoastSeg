# Standard library imports
import fnmatch
import logging
import os
import json
import copy
from glob import glob
from typing import Optional, Union, List, Dict

# Internal dependencies imports
from coastseg import exceptions
from coastseg import common

# External dependencies imports
import dask
import dask.bag as db
from dask import compute
from dask.diagnostics import ProgressBar
import pandas as pd

from dask.diagnostics import ProgressBar
import geopandas as gpd
import numpy as np
from ipyleaflet import GeoJSON
from matplotlib.pyplot import get_cmap
from matplotlib.colors import rgb2hex
from tqdm.auto import tqdm
import pickle
import matplotlib.pyplot as plt
import skimage.measure as measure
from coastsat import SDS_shoreline
from coastsat import SDS_preprocess
from coastsat.SDS_download import get_metadata
from coastsat.SDS_shoreline import extract_shorelines
from coastsat.SDS_tools import (
    remove_duplicates,
    remove_inaccurate_georef,
    output_to_gdf,
    get_filepath,
    get_filenames,
    merge_output,
)

logger = logging.getLogger(__name__)

__all__ = ["Extracted_Shoreline"]


def combine_satellite_data(satellite_data: dict):
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
    merged_satellite_data = {}
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
        # For each key in the nested dictionary
        for key, value in sat_data.items():
            # Wrap non-list values in a list and concatenate to merged_satellite_data
            if not isinstance(value, list):
                merged_satellite_data[key] += [value]
            else:
                merged_satellite_data[key] += value
        # Add satellite name entries for each date
        if "dates" in sat_data.keys():
            merged_satellite_data["satname"] += [satname] * len(sat_data["dates"])
    # Sort chronologically
    idx_sorted = sorted(
        range(len(merged_satellite_data["dates"])),
        key=lambda i: merged_satellite_data["dates"][i],
    )

    for key in merged_satellite_data.keys():
        merged_satellite_data[key] = [merged_satellite_data[key][i] for i in idx_sorted]

    return merged_satellite_data


def process_satellite(
    satname: str, settings: dict, metadata: dict, session_path: str, class_indices: list
):
    collection = settings["inputs"]["landsat_collection"]
    default_min_length_sl = settings["min_length_sl"]
    # deep copy settings
    settings = copy.deepcopy(settings)
    filepath = get_filepath(settings["inputs"], satname)
    # get list of file associated with this satellite
    filenames = metadata[satname]["filenames"]
    # get the pixel size of the satellite in meters
    pixel_size = get_pixel_size_for_satellite(satname)
    # get the minimum beach area in number of pixels depending on the satellite
    settings["min_length_sl"] = get_min_shoreline_length(satname, default_min_length_sl)
    # loop through the images
    espg_list = []
    geoaccuracy_list = []
    timestamps = []
    tasks = []
    for index in tqdm(
        range(len(filenames)), desc="Mapping Shorelines", leave=True, position=0
    ):
        image_epsg = metadata[satname]["epsg"][index]
        # get image spatial reference system (epsg code) from metadata dict
        espg_list.append(metadata[satname]["epsg"][index])
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
            )
        )

    # run the tasks in parallel
    results = dask.compute(*tasks)
    # logger.info(f"results: {results}")

    output = {}
    for index, result in enumerate(results):
        if result is None:
            continue
        output.setdefault(satname, {})

        output[satname].setdefault("dates", []).append(timestamps[index])
        output[satname].setdefault("geoaccuracy", []).append(geoaccuracy_list[index])
        output[satname].setdefault("shorelines", []).append(result["shorelines"])
        output[satname].setdefault("cloud_cover", []).append(result["cloud_cover"])
        output[satname].setdefault("filename", []).append(filenames[index])
        output[satname].setdefault("idx", []).append(index)
        # print(data['MNDWI_threshold'][:2])
    return output


def process_satellite_image(
    filename,
    filepath,
    settings,
    satname,
    collection,
    image_epsg,
    pixel_size,
    session_path,
    class_indices,
):
    """_summary_

    Args:
        filename (_type_): _description_
        filepath (_type_): _description_
        settings (_type_): _description_
        satname (_type_): _description_
        collection (_type_): _description_
        image_epsg (_type_): _description_
        pixel_size (_type_): _description_
        session_path (_type_): _description_
        class_indices (_type_): _description_

    Returns:
        dict: Returns a dictionary containing the shorelines and cloud cover for the file passed in
        ex.
        {
            "shorelines": array([[ 432675. , 4003942.5],
                                [ 432675. , 4003957.5],
                                [ 432682.5, 4003965. ],
                                [ 432690. , 4003972.5],
                                [ 432690. , 4003987.5],])
            "cloud_cover": 0.0,
        }
    """
    # get image date
    date = filename[:19]
    # get image filename
    fn = get_filenames(filename, filepath, satname)
    # preprocess image (cloud mask + pansharpening/downsampling)
    # could apply dask delayed here
    (
        im_ms,
        georef,
        cloud_mask,
        im_extra,
        im_QA,
        im_nodata,
    ) = SDS_preprocess.preprocess_single(
        fn,
        satname,
        settings["cloud_mask_issue"],
        settings["pan_off"],
        collection,
    )
    # compute cloud_cover percentage (with no data pixels)
    cloud_cover_combined = np.divide(
        sum(sum(cloud_mask.astype(int))),
        (cloud_mask.shape[0] * cloud_mask.shape[1]),
    )
    if cloud_cover_combined > 0.99:  # if 99% of cloudy pixels in image skip
        return None
    # remove no data pixels from the cloud mask
    # (for example L7 bands of no data should not be accounted for)
    cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
    # compute updated cloud cover percentage (without no data pixels)
    cloud_cover = np.divide(
        sum(sum(cloud_mask_adv.astype(int))),
        (sum(sum((~im_nodata).astype(int)))),
    )
    # skip image if cloud cover is above user-defined threshold
    if cloud_cover > settings["cloud_thresh"]:
        return None
    # calculate a buffer around the reference shoreline (if any has been digitised)
    ref_shoreline_buffer = SDS_shoreline.create_shoreline_buffer(
        cloud_mask.shape, georef, image_epsg, pixel_size, settings
    )
    # read the model outputs from the npz file for this image
    npz_file = find_matching_npz(filename, session_path)
    logger.info(f"npz_file: {npz_file}")
    if npz_file is None:
        logger.warning(f"npz file not found for {filename}")
        return None

    # get the labels for water and land
    im_labels = load_image_labels(npz_file, class_indices=class_indices)
    if sum(im_labels[ref_shoreline_buffer]) < 50:
        logger.warning(
            f"{fn} Not enough sand pixels within the beach buffer to detect shoreline"
        )
        return None
    # get the shoreline from the image
    shoreline = process_image(
        date,
        fn,
        satname,
        image_epsg,
        settings,
        cloud_mask,
        cloud_mask_adv,
        im_nodata,
        georef,
        im_ms,
        im_labels,
    )
    if shoreline is None:
        logger.warning(f"\nShoreline not found for {fn}")
        return None
    # create dictionnary of output
    output = {
        "shorelines": shoreline,
        "cloud_cover": cloud_cover,
    }
    logger.info(f"output shorelines and cloud cover: {output}")
    return output


def get_model_card_classes(model_card_path: str) -> dict:
    """return the classes dictionary from the model card
        example classes dictionary {0: 'sand', 1: 'sediment', 2: 'whitewater', 3: 'water'}
    Args:
        model_card_path (str): path to model card

    Returns:
        dict: dictionary of classes in model card and their corresponding index
    """
    model_card_data = common.read_json_file(model_card_path, raise_error=True)
    model_card_classes = model_card_data["DATASET"]["CLASSES"]
    return model_card_classes


def get_indices_of_names(
    model_card_path: str,
    selected_class_names: List[str],
) -> List[int]:
    """
    Given the path to a model card and a list of selected class names, returns a list of indices of the selected classes
    in the model card. The model card should be a dictionary that maps class indices to class names.

    :param model_card_path: a string specifying the path to the model card.
    :param selected_class_names: a list of strings specifying the names of the selected classes.
    :return: a list of integers specifying the indices of the selected classes in the model card.
    """
    # example dictionary {0: 'sand', 1: 'sediment', 2: 'whitewater', 3: 'water'}
    model_card_classes = get_model_card_classes(model_card_path)
    print(model_card_classes)

    class_indices = []
    # get index of each class in class_mapping to match model card classes
    for index, class_name in model_card_classes.items():
        # see if the class name is in selected_class_names
        for selected_class_name in selected_class_names:
            if class_name == selected_class_name:
                class_indices.append(int(index))
                break
    # return list of indexes of selected_class_names that were found in model_card_classes
    return class_indices


def find_matching_npz(filename, directory):
    # Extract the timestamp and Landsat ID from the filename
    parts = filename.split("_")
    timestamp, landsat_id = parts[0], parts[1]
    print(landsat_id)

    # Construct a pattern to match the corresponding npz filename
    pattern = f"{timestamp}*{landsat_id}*.npz"

    # Search the directory for files that match the pattern
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, pattern):
            return os.path.join(directory, file)

    # If no matching file is found, return None
    return None


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


def load_image_labels(npz_file: str, class_indices: list = [2, 1, 0]) -> np.ndarray:
    """
    Load and process image labels from a .npz file.
    Pass in the indexes of the classes to merge. For instance, if you want to merge the water and white water classes, and
    the indexes of water is 0 and white water is 1, pass in [0, 1] as the class_indices parameter.

    Parameters:
    npz_file (str): The path to the .npz file containing the image labels.
    class_indices (list): The indexes of the classes to merge.

    Returns:
    np.ndarray: A 2D numpy array containing the image labels as 1 for the merged classes and 0 for all other classes.
    """
    if not os.path.isfile(npz_file) or not npz_file.endswith(".npz"):
        raise ValueError(f"{npz_file} is not a valid .npz file.")

    data = np.load(npz_file)
    # 1 for water, 0 for anything else (land, other, sand, etc.)
    im_labels = merge_classes(data["grey_label"], class_indices)

    return im_labels


def simplified_find_contours(im_labels: np.array) -> list[np.array]:
    """Find contours in a binary image using skimage.measure.find_contours and processes out contours that contain NaNs.
    Parameters:
    -----------
    im_labels: np.nd.array
        binary image with 0s and 1s
    Returns:
    -----------
    processed_contours: list oarray
        processed image contours (only the ones that do not contains NaNs)
    """
    # 0 or 1 labels means 0.5 is the threshold
    contours = measure.find_contours(im_labels, 0.5)
    # remove contour points that are NaNs (around clouds)
    processed_contours = SDS_shoreline.process_contours(contours)
    return processed_contours


def process_image(
    date,
    fn,
    satname,
    image_epsg,
    settings,
    cloud_mask,
    cloud_mask_adv,
    im_nodata,
    georef,
    im_ms,
    im_labels,
) -> np.array:
    try:
        contours = simplified_find_contours(im_labels)
    except Exception as e:
        logger.error(f"{e}\nCould not map shoreline for this image: {fn}")
        return None
    # process the water contours into a shoreline
    shoreline = SDS_shoreline.process_shoreline(
        contours, cloud_mask_adv, im_nodata, georef, image_epsg, settings
    )
    # if not settings["check_detection"]:
    #     plt.ioff()
    # SDS_shoreline.new_show_detection(
    #     im_ms,
    #     cloud_mask,
    #     im_labels,
    #     shoreline,
    #     image_epsg,
    #     georef,
    #     settings,
    #     date,
    #     satname,
    # )
    logger.info(f"shoreline: {shoreline}")
    return shoreline


def get_shorelines_from_images(
    index: int,
    file_info: list,
    filepath: str,
    satname: str,
    collection: str,
    pixel_size: int,
    settings: dict,
    metadata: dict,
    session_path: str,
    class_indices: list,
) -> dict:
    # get the shoreline from the image
    print(f"file_info: {file_info}")
    fn = get_filenames(file_info, filepath, satname)
    # preprocess image (cloud mask + pansharpening/downsampling)
    (
        im_ms,
        georef,
        cloud_mask,
        im_extra,
        im_QA,
        im_nodata,
    ) = SDS_preprocess.preprocess_single(
        fn,
        satname,
        settings["cloud_mask_issue"],
        settings["pan_off"],
        collection,
    )
    # get image spatial reference system (epsg code) from metadata dict
    image_epsg = metadata[satname]["epsg"][index]
    # compute cloud_cover percentage (with no data pixels)
    cloud_cover_combined = np.divide(
        sum(sum(cloud_mask.astype(int))),
        (cloud_mask.shape[0] * cloud_mask.shape[1]),
    )
    if cloud_cover_combined > 0.99:  # if 99% of cloudy pixels in image skip
        return None
    # remove no data pixels from the cloud mask
    # (for example L7 bands of no data should not be accounted for)
    cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
    # compute updated cloud cover percentage (without no data pixels)
    cloud_cover = np.divide(
        sum(sum(cloud_mask_adv.astype(int))),
        (sum(sum((~im_nodata).astype(int)))),
    )
    # skip image if cloud cover is above user-defined threshold
    if cloud_cover > settings["cloud_thresh"]:
        return None
    # calculate a buffer around the reference shoreline (if any has been digitised)
    ref_shoreline_buffer = SDS_shoreline.create_shoreline_buffer(
        cloud_mask.shape, georef, image_epsg, pixel_size, settings
    )
    # read the model outputs from the npz file for this image
    npz_file = find_matching_npz(file_info, session_path)
    logger.info(f"npz_file: {npz_file}")
    if npz_file is None:
        logger.warning(f"npz file not found for {file_info}")
        return None

    # get the labels for water and land
    im_labels = load_image_labels(npz_file, class_indices=class_indices)
    if sum(im_labels[ref_shoreline_buffer]) < 50:
        logger.warning(
            f"{fn} Not enough sand pixels within the beach buffer to detect shoreline"
        )
        return None
    # get the shoreline from the image
    date = file_info[:19]
    shoreline = process_image(
        date,
        fn,
        satname,
        image_epsg,
        settings,
        cloud_mask,
        cloud_mask_adv,
        im_nodata,
        georef,
        im_ms,
        im_labels,
    )
    if shoreline is None:
        logger.warning(f"\nShoreline not found for {fn}")
        return None
    # append to output variables
    output = dict()
    output_timestamp = metadata[satname]["dates"][index]
    output_shoreline = shoreline
    output_filename = file_info
    output_cloudcover = cloud_cover
    output_geoaccuracy = metadata[satname]["acc_georef"][index]
    output_idxkeep = index
    # create dictionnary of output
    output[satname] = {
        "dates": output_timestamp,
        "shorelines": output_shoreline,
        "filename": output_filename,
        "cloud_cover": output_cloudcover,
        "geoaccuracy": output_geoaccuracy,
        "idx": output_idxkeep,
    }
    return output


def extract_shorelines_dask_nested(
    session_path: str, metadata: dict, settings: dict, class_indices: list = None
) -> dict:
    print(f"extract shoreline settings loaded in: {settings}")
    sitename = settings["inputs"]["sitename"]
    filepath_data = settings["inputs"]["filepath"]
    # initialise output structure
    output = dict([])
    # create a subfolder to store the .jpg images showing the detection
    filepath_jpg = os.path.join(filepath_data, sitename, "jpg_files", "detection")
    if not os.path.exists(filepath_jpg):
        os.makedirs(filepath_jpg)

    # loop through satellite list
    tasks = []
    for satname in metadata.keys():
        # get images
        new_task = dask.delayed(process_satellite)(
            satname, settings, metadata, session_path, class_indices
        )
        tasks.append(new_task)

    with ProgressBar():
        tuple_of_dicts = dask.compute(*tasks)
    logger.info(f"dask tuple_of_dicts: {tuple_of_dicts}")

    # convert from a tuple of dicts to single dictionary
    result_dict = {}
    for d in tuple_of_dicts:
        result_dict.update(d)

    # change the format to have one list sorted by date with all the shorelines (easier to use)
    output = combine_satellite_data(result_dict)
    logger.info(f"final_output: {output}")

    filepath = os.path.join(filepath_data, sitename)

    with open(os.path.join(filepath, sitename + "_output.pkl"), "wb") as f:
        pickle.dump(output, f)

    return output


def extract_shorelines_dask_delayed(
    session_path: str, metadata: dict, settings: dict, class_indices: list = None
) -> dict:

    print(f"extract shoreline settings loaded in: {settings}")
    sitename = settings["inputs"]["sitename"]
    filepath_data = settings["inputs"]["filepath"]
    collection = settings["inputs"]["landsat_collection"]
    default_min_length_sl = settings["min_length_sl"]

    # initialise output structure
    output = dict([])
    # create a subfolder to store the .jpg images showing the detection
    filepath_jpg = os.path.join(filepath_data, sitename, "jpg_files", "detection")
    if not os.path.exists(filepath_jpg):
        os.makedirs(filepath_jpg)

    # loop through satellite list
    for satname in metadata.keys():
        # get images
        filepath = get_filepath(settings["inputs"], satname)
        filenames = metadata[satname]["filenames"]

        # initialise the output variables
        output_timestamp = []  # datetime at which the image was acquired (UTC time)
        output_shoreline = []  # vector of shoreline points
        output_filename = (
            []
        )  # filename of the images from which the shorelines where derived
        output_cloudcover = []  # cloud cover of the images
        output_geoaccuracy = []  # georeferencing accuracy of the images
        output_idxkeep = (
            []
        )  # index that were kept during the analysis (cloudy images are skipped)

        # get the pixel size of the satellite in meters
        pixel_size = get_pixel_size_for_satellite(satname)
        # get the minimum beach area in number of pixels depending on the satellite
        settings["min_length_sl"] = get_min_shoreline_length(
            satname, default_min_length_sl
        )

        # same code as before, up until the following line:
        # loop through the images
        tasks = []
        for i in tqdm(
            range(len(filenames)), desc="Mapping Shorelines", leave=True, position=0
        ):
            tasks.append(
                dask.delayed(get_shorelines_from_images)(
                    i,
                    filenames[i],
                    filepath,
                    satname,
                    collection,
                    pixel_size,
                    settings,
                    metadata,
                    session_path,
                    class_indices,
                )
            )
        # Compute the results using Dask
        with ProgressBar():
            results = dask.compute(*tasks)
        print(f"dask results: {results}")
        # @todo still need to merge all the outputs

    # same code as before, up until the following line:
    # change the format to have one list sorted by date with all the shorelines (easier to use)
    output = merge_output(results)

    filepath = os.path.join(filepath_data, sitename)

    with open(os.path.join(filepath, sitename + "_output.pkl"), "wb") as f:
        pickle.dump(output, f)

    return output


# def new_extract_shorelines_for_session_v2(
#     session_path: str, metadata: dict, settings: dict,class_indices: list = None
# ) -> dict:
#     """
#     Main function to extract shorelines from satellite images
#     original version: KV WRL 2018
#     Arguments:
#     -----------
#     metadata: dict
#         contains all the information about the satellite images that were downloaded
#     settings: dict with the following keys
#         'inputs': dict
#             input parameters (sitename, filepath, polygon, dates, sat_list)
#         'cloud_thresh': float
#             value between 0 and 1 indicating the maximum cloud fraction in
#             the cropped image that is accepted
#         'cloud_mask_issue': boolean
#             True if there is an issue with the cloud mask and sand pixels
#             are erroneously being masked on the image
#         'min_beach_area': int
#             minimum allowable object area (in metres^2) for the class 'sand',
#             the area is converted to number of connected pixels
#         'min_length_sl': int
#             minimum length (in metres) of shoreline contour to be valid
#         'sand_color': str
#             default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
#         'output_epsg': int
#             output spatial reference system as EPSG code
#         'check_detection': bool
#             if True, lets user manually accept/reject the mapped shorelines
#         'save_figure': bool
#             if True, saves a -jpg file for each mapped shoreline
#         'adjust_detection': bool
#             if True, allows user to manually adjust the detected shoreline
#         'pan_off': bool
#             if True, no pan-sharpening is performed on Landsat 7,8 and 9 imagery

#     Returns:
#     -----------
#     output: dict
#         contains the extracted shorelines and corresponding dates + metadata

#     """
#     # print(f"extract shoreline settings loaded in: {settings}")
#     sitename = settings["inputs"]["sitename"]
#     filepath_data = settings["inputs"]["filepath"]
#     collection = settings["inputs"]["landsat_collection"]
#     default_min_length_sl = settings["min_length_sl"]

#     # initialise output structure
#     output = dict([])
#     # create a subfolder to store the .jpg images showing the detection
#     filepath_jpg = os.path.join(filepath_data, sitename, "jpg_files", "detection")
#     if not os.path.exists(filepath_jpg):
#         os.makedirs(filepath_jpg)

#     # loop through satellite list
#     for satname in metadata.keys():
#         print(satname)
#         # get images
#         filepath = get_filepath(settings["inputs"], satname)
#         filenames = metadata[satname]["filenames"]

#         # get the pixel size of the satellite in meters
#         pixel_size = get_pixel_size_for_satellite(satname)
#         # get the minimum beach area in number of pixels depending on the satellite
#         settings["min_length_sl"] =get_min_shoreline_length(satname, default_min_length_sl)


#         # Using Dask Bag to represent data
#         bag = db.from_sequence(range(len(filenames)))

#         def process_image_idx(i:int):
#             # get image filename
#             fn = get_filenames(filenames[i], filepath, satname)
#             # preprocess image (cloud mask + pansharpening/downsampling)
#             (
#                 im_ms,
#                 georef,
#                 cloud_mask,
#                 im_extra,
#                 im_QA,
#                 im_nodata,
#             ) = SDS_preprocess.preprocess_single(
#                 fn,
#                 satname,
#                 settings["cloud_mask_issue"],
#                 settings["pan_off"],
#                 collection,
#             )
#             # get image spatial reference system (epsg code) from metadata dict
#             image_epsg = metadata[satname]["epsg"][i]

#             # compute cloud_cover percentage (with no data pixels)
#             cloud_cover_combined = np.divide(
#                 sum(sum(cloud_mask.astype(int))),
#                 (cloud_mask.shape[0] * cloud_mask.shape[1]),
#             )
#             if cloud_cover_combined > 0.99:  # if 99% of cloudy pixels in image skip
#                 return None
#             # remove no data pixels from the cloud mask
#             # (for example L7 bands of no data should not be accounted for)
#             cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
#             # compute updated cloud cover percentage (without no data pixels)
#             cloud_cover = np.divide(
#                 sum(sum(cloud_mask_adv.astype(int))),
#                 (sum(sum((~im_nodata).astype(int)))),
#             )
#             # skip image if cloud cover is above user-defined threshold
#             if cloud_cover > settings["cloud_thresh"]:
#                 return None

#             # calculate a buffer around the reference shoreline (if any has been digitised)
#             ref_shoreline_buffer = SDS_shoreline.create_shoreline_buffer(
#                 cloud_mask.shape, georef, image_epsg, pixel_size, settings
#             )
#             # read the model outputs from the npz file for this image
#             npz_file = find_matching_npz(filenames[i],session_path)
#             logger.info(f"npz_file: {npz_file}")
#             if npz_file is None:
#                 logger.warning(f"npz file not found for {filenames[i]}")
#                 return None
#             im_labels = load_image_labels(npz_file, class_indices=class_indices)

#             if (
#                 sum(im_labels[ref_shoreline_buffer, 0]) < 50
#             ):
#                 logger.warning(
#                     f"{fn} Not enough sand pixels within the beach buffer to detect shoreline"
#                 )
#                 return None
#             # get the shoreline from the image
#             shoreline = process_image(
#                 fn,
#                 satname,
#                 image_epsg,
#                 settings,
#                 cloud_mask,
#                 cloud_mask_adv,
#                 im_nodata,
#                 georef,
#                 im_ms,
#                 im_labels,
#             )
#             if shoreline is None:
#                 logger.warning(f"\nShoreline not found for {fn}")
#                 return None

#             # return a dictionary of output variables
#             return {
#                 "timestamp": metadata[satname]["dates"][i],
#                 "shoreline": shoreline,
#                 "filename": filenames[i],
#                 "cloud_cover": cloud_cover,
#                 "geoaccuracy": metadata[satname]["acc_georef"][i],
#                 "idx":i,
#             }

#         # Map the function to the bag
#         processed_bag = bag.map(process_image_idx)

#         print(f"processed_bag : {processed_bag}")


#         # Compute the results using Dask
#         with ProgressBar():
#             processed_results = processed_bag.compute()

#         # Filter out None values
#         processed_results = [result for result in processed_results if result is not None]

#         print(f"processed_results : {processed_results}")

#         # Create a DataFrame from the results
#         output_df = pd.DataFrame(processed_results)

#         print(f"output_df : {output_df}")

#         # create dictionnary of output
#         output[satname] = {
#             "dates": output_df["timestamp"].tolist(),
#             "shorelines": output_df["shoreline"].tolist(),
#             "filename": output_df["filename"].tolist(),
#             "cloud_cover": output_df["cloud_cover"].tolist(),
#             "geoaccuracy": output_df["geoaccuracy"].tolist(),
#             "idx": output_df["idx"].tolist(),
#         }

#     # change the format to have one list sorted by date with all the shorelines (easier to use)
#     output = merge_output(output)
#     # @todo replace this with save json
#     # save outputput structure as output.pkl
#     filepath = os.path.join(filepath_data, sitename)
#     with open(os.path.join(filepath, sitename + "_output.pkl"), "wb") as f:
#         pickle.dump(output, f)
#     return output


def new_extract_shorelines_for_session(
    session_path: str, metadata: dict, settings: dict, class_indices: list = None
) -> dict:
    """
    Main function to extract shorelines from satellite images
    original version: KV WRL 2018
    Arguments:
    -----------
    metadata: dict
        contains all the information about the satellite images that were downloaded
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'cloud_thresh': float
            value between 0 and 1 indicating the maximum cloud fraction in
            the cropped image that is accepted
        'cloud_mask_issue': boolean
            True if there is an issue with the cloud mask and sand pixels
            are erroneously being masked on the image
        'min_beach_area': int
            minimum allowable object area (in metres^2) for the class 'sand',
            the area is converted to number of connected pixels
        'min_length_sl': int
            minimum length (in metres) of shoreline contour to be valid
        'sand_color': str
            default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
        'output_epsg': int
            output spatial reference system as EPSG code
        'check_detection': bool
            if True, lets user manually accept/reject the mapped shorelines
        'save_figure': bool
            if True, saves a -jpg file for each mapped shoreline
        'adjust_detection': bool
            if True, allows user to manually adjust the detected shoreline
        'pan_off': bool
            if True, no pan-sharpening is performed on Landsat 7,8 and 9 imagery

    Returns:
    -----------
    output: dict
        contains the extracted shorelines and corresponding dates + metadata

    """
    print(f"extract shoreline settings loaded in: {settings}")
    sitename = settings["inputs"]["sitename"]
    filepath_data = settings["inputs"]["filepath"]
    collection = settings["inputs"]["landsat_collection"]
    default_min_length_sl = settings["min_length_sl"]

    # initialise output structure
    output = dict([])
    # create a subfolder to store the .jpg images showing the detection
    filepath_jpg = os.path.join(filepath_data, sitename, "jpg_files", "detection")
    if not os.path.exists(filepath_jpg):
        os.makedirs(filepath_jpg)

    # loop through satellite list
    for satname in metadata.keys():
        # get images
        filepath = get_filepath(settings["inputs"], satname)
        filenames = metadata[satname]["filenames"]

        # initialise the output variables
        output_timestamp = []  # datetime at which the image was acquired (UTC time)
        output_shoreline = []  # vector of shoreline points
        output_filename = (
            []
        )  # filename of the images from which the shorelines where derived
        output_cloudcover = []  # cloud cover of the images
        output_geoaccuracy = []  # georeferencing accuracy of the images
        output_idxkeep = (
            []
        )  # index that were kept during the analysis (cloudy images are skipped)

        # get the pixel size of the satellite in meters
        pixel_size = get_pixel_size_for_satellite(satname)
        # get the minimum beach area in number of pixels depending on the satellite
        settings["min_length_sl"] = get_min_shoreline_length(
            satname, default_min_length_sl
        )

        # loop through the images
        for i in tqdm(
            range(len(filenames)), desc="Mapping Shorelines", leave=True, position=0
        ):

            # print('\r%s:   %d%%' % (satname,int(((i+1)/len(filenames))*100)), end='')

            # get image filename
            fn = get_filenames(filenames[i], filepath, satname)
            # preprocess image (cloud mask + pansharpening/downsampling)
            (
                im_ms,
                georef,
                cloud_mask,
                im_extra,
                im_QA,
                im_nodata,
            ) = SDS_preprocess.preprocess_single(
                fn,
                satname,
                settings["cloud_mask_issue"],
                settings["pan_off"],
                collection,
            )
            # get image spatial reference system (epsg code) from metadata dict
            image_epsg = metadata[satname]["epsg"][i]

            # compute cloud_cover percentage (with no data pixels)
            cloud_cover_combined = np.divide(
                sum(sum(cloud_mask.astype(int))),
                (cloud_mask.shape[0] * cloud_mask.shape[1]),
            )
            if cloud_cover_combined > 0.99:  # if 99% of cloudy pixels in image skip
                continue
            # remove no data pixels from the cloud mask
            # (for example L7 bands of no data should not be accounted for)
            cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
            # compute updated cloud cover percentage (without no data pixels)
            cloud_cover = np.divide(
                sum(sum(cloud_mask_adv.astype(int))),
                (sum(sum((~im_nodata).astype(int)))),
            )
            # skip image if cloud cover is above user-defined threshold
            if cloud_cover > settings["cloud_thresh"]:
                continue

            # calculate a buffer around the reference shoreline (if any has been digitised)
            ref_shoreline_buffer = SDS_shoreline.create_shoreline_buffer(
                cloud_mask.shape, georef, image_epsg, pixel_size, settings
            )
            if sum(im_labels[ref_shoreline_buffer, 0]) < 50:
                logger.warning(
                    f"{fn} Not enough sand pixels within the beach buffer to detect shoreline"
                )
                continue

            # read the model outputs from the npz file for this image
            npz_file = find_matching_npz(filenames[i], session_path)
            logger.info(f"npz_file: {npz_file}")
            if npz_file is None:
                logger.warning(f"npz file not found for {filenames[i]}")
                continue

            # get the labels for water and land
            im_labels = load_image_labels(npz_file, class_indices=class_indices)
            # get the shoreline from the image
            shoreline = process_image(
                fn,
                satname,
                image_epsg,
                settings,
                cloud_mask,
                cloud_mask_adv,
                im_nodata,
                georef,
                im_ms,
                im_labels,
            )
            if shoreline is None:
                logger.warning(f"\nShoreline not found for {fn}")
                continue

            # append to output variables
            output_timestamp.append(metadata[satname]["dates"][i])
            output_shoreline.append(shoreline)
            output_filename.append(filenames[i])
            output_cloudcover.append(cloud_cover)
            output_geoaccuracy.append(metadata[satname]["acc_georef"][i])
            output_idxkeep.append(i)

        # create dictionnary of output
        output[satname] = {
            "dates": output_timestamp,
            "shorelines": output_shoreline,
            "filename": output_filename,
            "cloud_cover": output_cloudcover,
            "geoaccuracy": output_geoaccuracy,
            "idx": output_idxkeep,
        }

    # change the format to have one list sorted by date with all the shorelines (easier to use)
    output = merge_output(output)

    # @todo replace this with save json
    # save outputput structure as output.pkl
    filepath = os.path.join(filepath_data, sitename)

    with open(os.path.join(filepath, sitename + "_output.pkl"), "wb") as f:
        pickle.dump(output, f)

    return output


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


def get_pixel_size_for_satellite(satname: str) -> int:
    """Returns the pixel size of a given satellite.
    ["L5", "L7", "L8", "L9"] = 15 meters
    "S2" = 10 meters

    Args:
        satname (str): A string indicating the name of the satellite.

    Returns:
        int: The pixel size of the satellite in meters.

    Raises:
        None.
    """
    if satname in ["L5", "L7", "L8", "L9"]:
        pixel_size = 15
    elif satname == "S2":
        pixel_size = 10
    return pixel_size


# def new_extract_shorelines_for_session(
#     session_path: str, metadata: dict, settings: dict,class_indices: list = None
# ) -> dict:
#     """
#     Main function to extract shorelines from satellite images
#     original version: KV WRL 2018
#     Arguments:
#     -----------
#     metadata: dict
#         contains all the information about the satellite images that were downloaded
#     settings: dict with the following keys
#         'inputs': dict
#             input parameters (sitename, filepath, polygon, dates, sat_list)
#         'cloud_thresh': float
#             value between 0 and 1 indicating the maximum cloud fraction in
#             the cropped image that is accepted
#         'cloud_mask_issue': boolean
#             True if there is an issue with the cloud mask and sand pixels
#             are erroneously being masked on the image
#         'min_beach_area': int
#             minimum allowable object area (in metres^2) for the class 'sand',
#             the area is converted to number of connected pixels
#         'min_length_sl': int
#             minimum length (in metres) of shoreline contour to be valid
#         'sand_color': str
#             default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
#         'output_epsg': int
#             output spatial reference system as EPSG code
#         'check_detection': bool
#             if True, lets user manually accept/reject the mapped shorelines
#         'save_figure': bool
#             if True, saves a -jpg file for each mapped shoreline
#         'adjust_detection': bool
#             if True, allows user to manually adjust the detected shoreline
#         'pan_off': bool
#             if True, no pan-sharpening is performed on Landsat 7,8 and 9 imagery

#     Returns:
#     -----------
#     output: dict
#         contains the extracted shorelines and corresponding dates + metadata

#     """
#     print(f"extract shoreline settings loaded in: {settings}")
#     sitename = settings["inputs"]["sitename"]
#     filepath_data = settings["inputs"]["filepath"]
#     collection = settings["inputs"]["landsat_collection"]
#     default_min_length_sl = settings["min_length_sl"]

#     # initialise output structure
#     output = dict([])
#     # create a subfolder to store the .jpg images showing the detection
#     filepath_jpg = os.path.join(filepath_data, sitename, "jpg_files", "detection")
#     if not os.path.exists(filepath_jpg):
#         os.makedirs(filepath_jpg)

#     # loop through satellite list
#     for satname in metadata.keys():
#         # get images
#         filepath = get_filepath(settings["inputs"], satname)
#         filenames = metadata[satname]["filenames"]

#         # initialise the output variables
#         output_timestamp = []  # datetime at which the image was acquired (UTC time)
#         output_shoreline = []  # vector of shoreline points
#         output_filename = (
#             []
#         )  # filename of the images from which the shorelines where derived
#         output_cloudcover = []  # cloud cover of the images
#         output_geoaccuracy = []  # georeferencing accuracy of the images
#         output_idxkeep = (
#             []
#         )  # index that were kept during the analysis (cloudy images are skipped)
#         output_t_mndwi = []  # MNDWI threshold used to map the shoreline

#         if satname in ["L5", "L7", "L8", "L9"]:
#             pixel_size = 15
#         elif satname == "S2":
#             pixel_size = 10

#         # reduce min shoreline length for L7 because of the diagonal bands
#         if satname == "L7":
#             settings["min_length_sl"] = 200
#         else:
#             settings["min_length_sl"] = default_min_length_sl

#         # loop through the images
#         for i in tqdm(
#             range(len(filenames)), desc="Mapping Shorelines", leave=True, position=0
#         ):

#             # print('\r%s:   %d%%' % (satname,int(((i+1)/len(filenames))*100)), end='')

#             # get image filename
#             fn = get_filenames(filenames[i], filepath, satname)
#             # preprocess image (cloud mask + pansharpening/downsampling)
#             (
#                 im_ms,
#                 georef,
#                 cloud_mask,
#                 im_extra,
#                 im_QA,
#                 im_nodata,
#             ) = SDS_preprocess.preprocess_single(
#                 fn,
#                 satname,
#                 settings["cloud_mask_issue"],
#                 settings["pan_off"],
#                 collection,
#             )
#             # get image spatial reference system (epsg code) from metadata dict
#             image_epsg = metadata[satname]["epsg"][i]

#             # compute cloud_cover percentage (with no data pixels)
#             cloud_cover_combined = np.divide(
#                 sum(sum(cloud_mask.astype(int))),
#                 (cloud_mask.shape[0] * cloud_mask.shape[1]),
#             )
#             if cloud_cover_combined > 0.99:  # if 99% of cloudy pixels in image skip
#                 continue
#             # remove no data pixels from the cloud mask
#             # (for example L7 bands of no data should not be accounted for)
#             cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
#             # compute updated cloud cover percentage (without no data pixels)
#             cloud_cover = np.divide(
#                 sum(sum(cloud_mask_adv.astype(int))),
#                 (sum(sum((~im_nodata).astype(int)))),
#             )
#             # skip image if cloud cover is above user-defined threshold
#             if cloud_cover > settings["cloud_thresh"]:
#                 continue

#             # calculate a buffer around the reference shoreline (if any has been digitised)
#             im_ref_buffer = SDS_shoreline.create_shoreline_buffer(
#                 cloud_mask.shape, georef, image_epsg, pixel_size, settings
#             )

#             # read the model outputs from the npz file for this image
#             npz_file = find_matching_npz(filenames[i],session_path)
#             logger.info(f"npz_file: {npz_file}")
#             if npz_file is None:
#                 logger.warning(f"npz file not found for {filenames[i]}")
#                 continue

#             im_labels = load_image_labels(npz_file,class_indices=class_indices)
#             try:  # use try/except structure for long runs
#                 if (
#                     sum(im_labels[im_ref_buffer]) < 50
#                 ):  # minimum number of sand pixels
#                     print(
#                         f"{fn} Not enough sand pixels within the beach buffer to detect shoreline"
#                     )
#                     logger.info(
#                         f"{fn} Not enough sand pixels within the beach buffer to detect shoreline"
#                     )
#                     continue
#                 else:
#                     # use classification to refine threshold and extract the sand/water interface
#                     contours=simplified_find_contours(im_labels)
#                     logger.info(f"contours : {contours}")
#             except Exception as e:
#                 print(e)
#                 print("\nCould not map shoreline for this image: " + filenames[i])
#                 continue

#             # process the water contours into a shoreline
#             shoreline = SDS_shoreline.process_shoreline(
#                 contours, cloud_mask_adv, im_nodata, georef, image_epsg, settings
#             )

#             # visualise the mapped shorelines, there are two options:
#             # if settings['check_detection'] = True, shows the detection to the user for accept/reject
#             # if settings['save_figure'] = True, saves a figure for each mapped shoreline
#             if settings["check_detection"] or settings["save_figure"]:
#                 date = filenames[i][:19]
#                 if not settings["check_detection"]:
#                     plt.ioff()  # turning interactive plotting off
#                 skip_image = SDS_shoreline.new_show_detection(
#                     im_ms,
#                     cloud_mask,
#                     im_labels,
#                     shoreline,
#                     image_epsg,
#                     georef,
#                     settings,
#                     date,
#                     satname,
#                 )

#             # append to output variables
#             output_timestamp.append(metadata[satname]["dates"][i])
#             output_shoreline.append(shoreline)
#             output_filename.append(filenames[i])
#             output_cloudcover.append(cloud_cover)
#             output_geoaccuracy.append(metadata[satname]["acc_georef"][i])
#             output_idxkeep.append(i)

#         # create dictionnary of output
#         output[satname] = {
#             "dates": output_timestamp,
#             "shorelines": output_shoreline,
#             "filename": output_filename,
#             "cloud_cover": output_cloudcover,
#             "geoaccuracy": output_geoaccuracy,
#             "idx": output_idxkeep,
#         }

#     # change the format to have one list sorted by date with all the shorelines (easier to use)
#     output = merge_output(output)

#     # @todo replace this with save json
#     # save outputput structure as output.pkl
#     filepath = os.path.join(filepath_data, sitename)

#     with open(os.path.join(filepath, sitename + "_output.pkl"), "wb") as f:
#         pickle.dump(output, f)

#     return output


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

    for file_type, file_pattern in required_files.items():
        file_paths = glob(os.path.join(dir_path, file_pattern))
        if not file_paths:
            logger.warning(f"No {file_type} file could be loaded from {dir_path}")
            return None

        file_path = file_paths[0]  # Use the first file if there are multiple matches
        if file_type == "geojson":
            extracted_files[file_type] = common.read_gpd_file(file_path)
        else:
            extracted_files[file_type] = common.from_file(file_path)

    extracted_shorelines = Extracted_Shoreline()
    extracted_shorelines = extracted_shorelines.load_extracted_shorelines(
        extracted_files["dict"], extracted_files["settings"], extracted_files["geojson"]
    )

    logger.info(f"Loaded extracted shorelines from: {dir_path}")
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
        return f"Extracted Shoreline: ROI ID: {self.roi_id}\n geodataframe {self.gdf.head(5)}\nshoreline_settings{self.shoreline_settings}"

    def __repr__(self):
        return f"Extracted Shoreline:  ROI ID: {self.roi_id}\n geodataframe {self.gdf.head(5)}\nshoreline_settings{self.shoreline_settings}\ndictionary{self.dictionary}"

    def get_roi_id(self) -> Optional[str]:
        """
        Extracts the region of interest (ROI) ID from the shoreline settings.

        The method retrieves the sitename field from the shoreline settings inputs dictionary and extracts the
        ROI ID from it, if present. The sitename field is expected to be in the format "ID_XXXX_datetime03-22-23__07_29_15",
        where XXXX is the id of the ROI. If the sitename field is not present or is not in the
        expected format, the method returns None.

        shoreline_settings:
        {
            'inputs' {
                "sitename": 'ID_0_datetime03-22-23__07_29_15',
                }
        }

        Returns:
            The ROI ID as a string, or None if the sitename field is not present or is not in the expected format.
        """
        inputs = self.shoreline_settings.get("inputs", {})
        sitename = inputs.get("sitename", "")
        # checks if the ROI ID is present in the 'sitename' saved in the shoreline settings
        roi_id = sitename.split("_")[1] if sitename else None
        return roi_id

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

    def create_extracted_shorlines(
        self,
        roi_id: str = None,
        shoreline: gpd.GeoDataFrame = None,
        roi_settings: dict = None,
        settings: dict = None,
    ) -> "Extracted_Shoreline":
        """
        Extracts shorelines for a specified region of interest (ROI) and returns an Extracted_Shoreline class instance.

        Args:
        - self: The object instance.
        - roi_id (str): The ID of the region of interest for which shorelines need to be extracted.
        - shoreline (GeoDataFrame): A GeoDataFrame of shoreline features.
        - roi_settings (dict): A dictionary of region of interest settings.
        - settings (dict): A dictionary of extraction settings.

        Returns:
        - object: The Extracted_Shoreline class instance.
        """
        # validate input parameters are not empty and are of the correct type
        self._validate_input_params(roi_id, shoreline, roi_settings, settings)

        logger.info(f"Extracting shorelines for ROI id{roi_id}")
        self.dictionary = self.extract_shorelines(
            shoreline,
            roi_settings,
            settings,
        )

        if is_list_empty(self.dictionary["shorelines"]):
            logger.warning(f"No extracted shorelines for ROI {roi_id}")
            raise exceptions.No_Extracted_Shoreline(roi_id)

        map_crs = "EPSG:4326"
        # extracted shorelines have map crs so they can be displayed on the map
        self.gdf = self.create_geodataframe(
            self.shoreline_settings["output_epsg"], output_crs=map_crs
        )
        return self

    def create_extracted_shorlines_from_session(
        self,
        roi_id: str = None,
        shoreline: gpd.GeoDataFrame = None,
        roi_settings: dict = None,
        settings: dict = None,
        session_path: str = None,
    ) -> "Extracted_Shoreline":
        """
        Extracts shorelines for a specified region of interest (ROI) from a saved session and returns an Extracted_Shoreline class instance.

        Args:
        - self: The object instance.
        - roi_id (str): The ID of the region of interest for which shorelines need to be extracted.
        - shoreline (GeoDataFrame): A GeoDataFrame of shoreline features.
        - roi_settings (dict): A dictionary of region of interest settings.
        - settings (dict): A dictionary of extraction settings.
        - session_path (str): The path of the saved session from which the shoreline extraction needs to be resumed.

        Returns:
        - object: The Extracted_Shoreline class instance.
        """
        # validate input parameters are not empty and are of the correct type
        self._validate_input_params(roi_id, shoreline, roi_settings, settings)

        logger.info(f"Extracting shorelines for ROI id{roi_id}")

        # read model settings from session path
        model_settings_path = os.path.join(session_path, "model_settings.json")
        model_settings = common.read_json_file(model_settings_path, raise_error=True)
        # get model type from model settings
        model_type = model_settings.get("model_type", "")
        if model_type == "":
            raise ValueError(
                f"Model type cannot be empty.{model_settings_path} did not contain model_type key."
            )
        # read model card from downloaded model
        downloaded_models_path = os.path.join(
            os.getcwd(), "src", "coastseg", "downloaded_models", model_type
        )
        model_card_path = common.find_file_by_regex(
            downloaded_models_path, r".*modelcard\.json$"
        )
        # get the water index from the model card
        water_classes_indices = get_indices_of_names(
            model_card_path, ["water", "whitewater"]
        )

        self.dictionary = self.extract_shorelines(
            shoreline,
            roi_settings,
            settings,
            session_path=session_path,
            class_indices=water_classes_indices,
        )

        if is_list_empty(self.dictionary["shorelines"]):
            logger.warning(f"No extracted shorelines for ROI {roi_id}")
            raise exceptions.No_Extracted_Shoreline(roi_id)

        map_crs = "EPSG:4326"
        # extracted shorelines have map crs so they can be displayed on the map
        self.gdf = self.create_geodataframe(
            self.shoreline_settings["output_epsg"], output_crs=map_crs
        )
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
        shoreline_gdf: gpd.geodataframe,
        roi_settings: dict,
        settings: dict,
        session_path: str = None,
        class_indices: list = None,
    ) -> dict:
        """Returns a dictionary containing the extracted shorelines for roi specified by rois_gdf"""
        # project shorelines's crs from map's crs to output crs given in settings
        map_crs = 4326
        reference_shoreline = get_reference_shoreline(
            shoreline_gdf, settings["output_epsg"]
        )
        # Add reference shoreline to shoreline_settings
        self.shoreline_settings = self.create_shoreline_settings(
            settings, roi_settings, reference_shoreline
        )
        # gets metadata used to extract shorelines
        metadata = get_metadata(self.shoreline_settings["inputs"])
        logger.info(f"metadata: {metadata}")
        # extract shorelines from ROI
        if session_path is None:
            # extract shorelines with coastsat's models
            extracted_shorelines = extract_shorelines(metadata, self.shoreline_settings)
        elif session_path is not None:
            # extract shorelines with our models
            extracted_shorelines = extract_shorelines_dask_nested(
                session_path,
                metadata,
                self.shoreline_settings,
                class_indices=class_indices,
            )
            # extracted_shorelines = extract_shorelines_dask_delayed(
            #     session_path, metadata, self.shoreline_settings,class_indices=class_indices,
            # )
        logger.info(f"extracted_shoreline_dict: {extracted_shorelines}")
        # postprocessing by removing duplicates and removing in inaccurate georeferencing (set threshold to 10 m)
        extracted_shorelines = remove_duplicates(
            extracted_shorelines
        )  # removes duplicates (images taken on the same date by the same satellite)
        extracted_shorelines = remove_inaccurate_georef(
            extracted_shorelines, 10
        )  # remove inaccurate georeferencing (set threshold to 10 m)
        logger.info(
            f"after remove_inaccurate_georef : extracted_shoreline_dict: {extracted_shorelines}"
        )
        return extracted_shorelines

    def create_shoreline_settings(
        self,
        settings: dict,
        roi_settings: dict,
        reference_shoreline: dict,
    ) -> None:
        """sets self.shoreline_settings to dictionary containing settings, reference_shoreline
        and roi_settings

        shoreline_settings=
        {
            "reference_shoreline":reference_shoreline,
            "inputs": roi_settings,
            "adjust_detection": False,
            "check_detection": False,
            ...
            rest of items from settings
        }

        Args:
            settings (dict): map settings
            roi_settings (dict): settings of the roi. Must include 'dates'
            reference_shoreline (dict): reference shoreline
        """
        # deepcopy settings to shoreline_settings so it can be modified
        # shoreline_settings = copy.deepcopy(settings)
        shoreline_keys = [
            "cloud_thresh",
            "cloud_mask_issue",
            "min_beach_area",
            "min_length_sl",
            "output_epsg",
            "sand_color",
            "pan_off",
            "max_dist_ref",
            "dist_clouds",
        ]
        logger.info(f"settings used to create shoreline settings: {settings}")
        shoreline_settings = common.filter_dict_by_keys(settings, keys=shoreline_keys)
        logger.info(f"Loading shoreline_settings: {shoreline_settings}")
        # Add reference shoreline and shoreline buffer distance for this specific ROI
        shoreline_settings["reference_shoreline"] = reference_shoreline
        # disable adjusting shorelines manually in shoreline_settings
        shoreline_settings["adjust_detection"] = False
        # disable adjusting shorelines manually in shoreline_settings
        shoreline_settings["check_detection"] = False
        shoreline_settings["save_figure"] = True
        # copy roi_setting for this specific roi
        shoreline_settings["inputs"] = roi_settings
        logger.info(f"shoreline_settings: {shoreline_settings}")
        return shoreline_settings

    def create_geodataframe(
        self, input_crs: str, output_crs: str = None
    ) -> gpd.GeoDataFrame:
        """Creates a geodataframe with the crs specified by input_crs. Converts geodataframe crs
        to output_crs if provided.
        Args:
            input_crs (str ): coordinate reference system string. Format 'EPSG:4326'.
            output_crs (str, optional): coordinate reference system string. Defaults to None.
        Returns:
            gpd.GeoDataFrame: geodataframe with columns = ['geometery','date','satname','geoaccuracy','cloud_cover']
            converted to output_crs if provided otherwise geodataframe's crs will be
            input_crs
        """
        extract_shoreline_gdf = output_to_gdf(self.dictionary, "lines")
        extract_shoreline_gdf.crs = input_crs
        if output_crs is not None:
            extract_shoreline_gdf = extract_shoreline_gdf.to_crs(output_crs)
        return extract_shoreline_gdf

    def save_to_file(
        self,
        sitename: str,
        filepath: str,
    ):
        """save_to_file Save geodataframe to location specified by filepath into directory
        specified by sitename

        Args:
            sitename (str): directory of roi shoreline was extracted from
            filepath (str): full path to directory containing ROIs
        """
        savepath = os.path.join(filepath, sitename, Extracted_Shoreline.FILE_NAME)
        logger.info(
            f"Saving shoreline to file: {savepath}.\n Extracted Shoreline: {self.gdf}"
        )
        print(f"Saving shoreline to file: {savepath}")
        self.gdf.to_file(
            savepath,
            driver="GeoJSON",
            encoding="utf-8",
        )

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
                common.to_file(data, file_location)

    def style_layer(self, geojson: dict, layer_name: str, color: str) -> GeoJSON:
        """Return styled GeoJson object with layer name

        Args:
            geojson (dict): geojson dictionary to be styled
            layer_name(str): name of the GeoJSON layer
            color(str): hex code or name of color render shorelines

        Returns:
            "ipyleaflet.GeoJSON": shoreline as GeoJSON layer styled with color
        """
        assert geojson != {}, "ERROR.\n Empty geojson cannot be drawn onto  map"
        return GeoJSON(
            data=geojson,
            name=layer_name,
            style={
                "color": color,
                "opacity": 1,
                "weight": 3,
            },
        )

    def get_layer_name(self) -> list:
        """returns name of extracted shoreline layer"""
        layer_name = "extracted_shoreline"
        return layer_name

    def get_styled_layer(self, row_number: int = 0) -> GeoJSON:
        """
        Returns a single shoreline feature as a GeoJSON object with a specified style.

        Args:
        - row_number (int): The index of the shoreline feature to select from the GeoDataFrame.

        Returns:
        - GeoJSON: A single shoreline feature as a GeoJSON object with a specified style.
        """
        # load extracted shorelines onto map
        map_crs = 4326
        layers = []
        if self.gdf.empty:
            return layers
        # convert to map crs and turn in json dict
        projected_gdf = self.gdf.to_crs(map_crs)
        # select a single shoreline and convert it to json
        single_shoreline = projected_gdf.iloc[[row_number]]
        single_shoreline = common.stringify_datetime_columns(single_shoreline)
        logger.info(f"single_shoreline.columns: {single_shoreline.columns}")
        logger.info(f"single_shoreline: {single_shoreline}")
        # convert geodataframe to json
        features_json = json.loads(single_shoreline.to_json())
        logger.info(f"single_shoreline features_json: {features_json}")
        layer_name = self.get_layer_name()
        logger.info(f"layer_name: {layer_name}")
        logger.info(f"features_json['features']: {features_json['features']}")
        # create a single layer
        feature = features_json["features"][0]
        new_layer = self.style_layer(feature, layer_name, "red")
        logger.info(f"new_layer: {new_layer}")
        return new_layer


def get_reference_shoreline(
    shoreline_gdf: gpd.geodataframe, output_crs: str
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
    logger.info(f"reprojected_shorlines.crs: {reprojected_shorlines.crs}")
    logger.info(f"reprojected_shorlines: {reprojected_shorlines}")
    # convert shoreline_in_roi gdf to coastsat compatible format np.array([[lat,lon,0],[lat,lon,0]...])
    shorelines = make_coastsat_compatible(reprojected_shorlines)
    # shorelines = [([lat,lon],[lat,lon],[lat,lon]),([lat,lon],[lat,lon],[lat,lon])...]
    # Stack all the tuples into a single list of n rows X 2 columns
    shorelines = np.vstack(shorelines)
    # Add third column of 0s to represent mean sea level
    shorelines = np.insert(shorelines, 2, np.zeros(len(shorelines)), axis=1)
    return shorelines


def get_colors(length: int) -> list:
    # returns a list of color hex codes as long as length
    cmap = get_cmap("plasma", length)
    cmap_list = [rgb2hex(i) for i in cmap.colors]
    return cmap_list


def make_coastsat_compatible(feature: gpd.geodataframe) -> list:
    """Return the feature as an np.array in the form:
        [([lat,lon],[lat,lon],[lat,lon]),([lat,lon],[lat,lon],[lat,lon])...])
    Args:
        feature (gpd.geodataframe): clipped portion of shoreline within a roi
    Returns:
        list: shorelines in form:
            [([lat,lon],[lat,lon],[lat,lon]),([lat,lon],[lat,lon],[lat,lon])...])
    """
    features = []
    # Use explode to break multilinestrings in linestrings
    feature_exploded = feature.explode()
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
