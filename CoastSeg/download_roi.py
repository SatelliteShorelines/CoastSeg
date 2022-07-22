""""Module for downloading selected rois using Coastsat"""

from CoastSeg import SDS_tools, SDS_download, SDS_preprocess
import json
from tqdm import tqdm
import ee
import geojson
import os
import warnings
from .file_functions import generate_datestring
warnings.filterwarnings("ignore")


def read_json_file(filename: str):
    with open(filename, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)
    return data


def write_preprocess_settings_file(settings_file: str, settings: dict):
    """"Write the preprocess settings dictionary to json file"""
    with open(settings_file, 'w', encoding='utf-8') as output_file:
        json.dump(settings, output_file)


def download_imagery(
        selected_roi_geojson: dict,
        pre_process_settings: dict,
        dates: list,
        sat_list: list,
        collection: str) -> None:
    """
     Checks if the images exist with check_images_available(), downloads them with retrieve_images(), and
     transforms images to jpgs with save_jpg()

    Arguments:
    -----------
    selected_roi_geojson:dict
        A geojson dictionary containing all the ROIs selected by the user

    pre_process_settings:dict
        Dictionary containing the preprocessing settings used for quality control by CoastSat

    dates: list
        A list of length two that contains a valid start and end date

    collection : str
     whether to use LandSat Collection 1 (`C01`) or Collection 2 (`C02`).
     Note that after 2022/01/01, Landsat images are only available in Collection 2.
     Landsat 9 is therefore only available as Collection 2. So if the user has selected `C01`,
     images prior to 2022/01/01 will be downloaded from Collection 1,
     while images captured after that date will be automatically taken from `C02`.

    sat_list: list
        A list of strings containing the names of the satellite
    """
#     1. Check imagery available and check for ee credentials
    try:
        inputs_list = check_images_available_selected_ROI(
            selected_roi_geojson, dates, collection, sat_list)
        print("Images available: \n", inputs_list)
    except ee.EEException as exception:
        print(exception)
        handle_AuthenticationError()
        inputs_list = check_images_available_selected_ROI(
            selected_roi_geojson, dates, collection, sat_list)
    except Exception as general_exception:
        print(general_exception)
        if type(general_exception).__name__ == 'RefreshError':
            handle_AuthenticationError()
            inputs_list = check_images_available_selected_ROI(
            selected_roi_geojson, dates, collection, sat_list)
# Check if inputs for downloading imagery exist then download imagery
    assert inputs_list != [], "\n Error: No ROIs were selected. Please click a valid ROI on the map\n"
    for inputs in tqdm(inputs_list, desc="Downloading ROIs"):
        metadata = SDS_download.retrieve_images(inputs)
        # Add the inputs to the pre_process_settings
        pre_process_settings['inputs'] = inputs
        SDS_preprocess.save_jpg(metadata, pre_process_settings)


def read_geojson_file(geojson_file: str) -> dict:
    """Returns the geojson of the selected ROIs from the file specified by geojson_file"""
    with open(geojson_file) as f:
        data = geojson.load(f)
    return data


def get_selected_roi_geojson(selected_set: set(), roi_data: dict) -> dict:
    """
    Returns the geojson of the ROIs selected by the user
    Arguments:
    -----------
    selected_set:tuple
        ids of the ROIs selected by the user

    roi_data:dict
        A geojson dict containing all the rois currently on the map
    Returns:
    -----------
    geojson_polygons: dict
       geojson dictionary containing all the ROIs selected
    """
    # Check if selected_set is empty
    assert len(
        selected_set) != 0, "\n Please select at least one ROI from the map before continuing."
    # Create a dictionary for the selected ROIs and add the user's selected
    # ROIs to it
    selected_ROI = {}
    selected_ROI["features"] = [
        feature
        for feature in roi_data["features"]
        if feature["properties"]["id"] in selected_set
    ]
    return selected_ROI


def handle_AuthenticationError():
    ee.Authenticate()
    ee.Initialize()


def check_images_available_selected_ROI(
        selected_roi_geojson: dict,
        dates: list,
        collection: str,
        sat_list: list) -> list:
    """"

    Return a list of dictionaries containing all the input parameters such as polygon, dates, sat_list, sitename, and filepath. This list can be used
    to retrieve images using coastsat.

    Arguments:
    -----------
    selected_roi_geojson:  dict
        A geojson dictionary containing all the ROIs selected by the user.
    dates: list
        A list of length two that contains a valid start and end date
    sat_list: list
        A list of strings containing the names of the satellite
    collection : str
     whether to use LandSat Collection 1 (`C01`)
     or Collection 2 (`C02`). Note that after 2022/01/01, Landsat images are only available in Collection 2.
     Landsat 9 is therefore only available as Collection 2. So if the user has selected `C01`,
     images prior to 2022/01/01 will be downloaded from Collection 1,
     while images captured after that date will be automatically taken from `C02`.

   Returns:
    -----------
   inputs_list: list
        A list of dictionaries containing all the input parameters such as polygon, dates, sat_list, sitename, and filepath
    """
    inputs_list = []
    if selected_roi_geojson["features"] != []:
        date_str = generate_datestring()
        for counter, roi in enumerate(selected_roi_geojson["features"]):
            coastSatBBOX = roi["geometry"]["coordinates"]
            polygon = coastSatBBOX
            # it's recommended to convert the polygon to the smallest rectangle
            # (sides parallel to coordinate axes)
            polygon = SDS_tools.smallest_rectangle(polygon)
            # name of the site
            sitename = 'ID' + str(counter) + date_str
            # directory where the data will be stored
            filepath = os.path.join(os.getcwd(), 'data')
            # put all the inputs into a dictionnary
            inputs = {
                'polygon': polygon,
                'dates': dates,
                'sat_list': sat_list,
                'sitename': sitename,
                'filepath': filepath,
                'landsat_collection': collection}
            # before downloading the images, check how many images are
            # available for your inputs
            SDS_download.check_images_available(inputs)
            inputs_list.append(inputs)
    return inputs_list
