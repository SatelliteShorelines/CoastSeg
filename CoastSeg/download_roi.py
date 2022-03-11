""""Module for downloading selected rois using Coastsat"""
from CoastSeg import SDS_tools, SDS_download, SDS_preprocess
from CoastSeg import make_overlapping_roi
import json
from datetime import datetime
from tqdm.notebook import tqdm_notebook
import ee
import geojson
# import matplotlib.pyplot as plt
# import matplotlib
import os
import warnings
warnings.filterwarnings("ignore")
# # from matplotlib import gridspec
# matplotlib.use('Qt5Agg')
# plt.ion()


def download_imagery_with_metadata(
        inputs_file: str,
        pre_process_settings: dict):
    """
     Download imagery using CoastSat with the inputs in a given inputs file. Bypasses coastsat's retrieve_imagery.

    Arguments:
    -----------
    inputs_file:str
        A json file containing all the inputs from a previous download session

    pre_process_settings:dict
        Dictionary containing the preprocessing settings used for quality control by CoastSat
    """
    assert os.path.exists(
        inputs_file), "Path to inputs_file {inputs_file} did not exist"
    # Read the inputs dict from file and get the list of inputs
    inputs_dict = read_json_file(inputs_file)
    print(inputs_dict)
    inputs_list = inputs_dict['inputs_list']
    print(inputs_list)

    for inputs in tqdm_notebook(inputs_list,
                                desc="Downloading ROIs with metadata"):
        print("\ninputs: ", inputs, "\n")
        # Alternative method to get metadata if you already have the images
        # saved
        metadata = SDS_download.get_metadata(inputs)
        print("\nmetadata", metadata, "\n")
        # Add the inputs to the pre_process_settings
        pre_process_settings['inputs'] = inputs
        SDS_preprocess.save_jpg(metadata, pre_process_settings)


def read_json_file(filename: str):
    with open(filename, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)
    return data


def write_preprocess_settings_file(settings_file: str, settings: dict):
    """"Write the preprocess settings dictionary to json file"""
    with open(settings_file, 'w', encoding='utf-8') as output_file:
        json.dump(settings, output_file)


def write_inputs_file(inputs_file: str, inputs_list: list,):
    """ Write the inputs_list to a json file """
    dict_inputs = {"inputs_list": inputs_list}
    with open(inputs_file, 'w', encoding='utf-8') as output_file:
        json.dump(dict_inputs, output_file)


def download_imagery(
        selected_roi_geojson: dict,
        pre_process_settings: dict,
        dates: list,
        sat_list: list,
        inputs_filename="inputs.json") -> None:
    """
     Checks if the images exist with check_images_available(), downloads them with retrieve_images(), and
     transformes images to jpgs with save_jpg()

    Arguments:
    -----------
    selected_roi_geojson:dict
        A geojson dictionary containing all the ROIs selected by the user

    pre_process_settings:dict
        Dictionary containing the preprocessing settings used for quality control by CoastSat

    dates: list
        A list of length two that contains a valid start and end date
    sat_list: list
        A list of strings containing the names of the satellite
    """
#     1. Check imagery available and check for ee credentials
    try:
        inputs_list = check_images_available_selected_ROI(
            selected_roi_geojson, dates, sat_list)
        print(inputs_list)
    except ee.EEException as exception:
        print(exception)
        handle_AuthenticationError()
        inputs_list = check_images_available_selected_ROI(
            selected_roi_geojson, dates, sat_list)
# Check if inputs for downloading imagery exist then download imagery
    assert inputs_list != [], "\n Error: No ROIs were selected. Please click a valid ROI on the map\n"
    write_inputs_file(inputs_filename, inputs_list)
    for inputs in tqdm_notebook(inputs_list, desc="Downloading ROIs"):
        print("\ninputs: ", inputs, "\n")
        metadata = SDS_download.retrieve_images(inputs)
        # Alternative method to get metadata if you already have the images saved
        # metadata = SDS_download.get_metadata(inputs)
        print("\nmetadata", metadata, "\n")
        # Add the inputs to the pre_process_settings
        pre_process_settings['inputs'] = inputs
        # print(metadata)
        SDS_preprocess.save_jpg(metadata, pre_process_settings)


def save_roi(
        geojson_file: str,
        selected_roi_file: str,
        selected_roi_set: set) -> dict:
    """
     Returns the geojson of the selected ROIs from the file specified by selected_roi_file

    Arguments:
    -----------
    geojson_file: strdump
        The filename of the geojson file containing all the ROI

    selected_roi_file: str
        The filename of the geojson file containing all the ROI selected by the user
    selected_roi_set: set
        The set of the selected rois' ids
    Returns:
    -----------
    selected_ROI_geojson: dict
        geojson of the selected ROIs
    """
    # 1. Open the geojson file containing all the ROIs on the map
    geojson_data = read_geojson_file(geojson_file)
    # 2. Get the selected rois geojson from the map
    selected_ROI_geojson = get_selected_roi_geojson(
        selected_roi_set, geojson_data)
    # 3. Save the rois' geojson data to a file named selected_roi_file
    make_overlapping_roi.write_to_geojson_file(
        selected_roi_file, selected_ROI_geojson, perserve_id=True)
    return selected_ROI_geojson


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
        A tuple containing the ids of the ROIs selected by the user

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


def generate_datestring():
    """"
    Returns a string in the following format %Y-%m-%d__%H_hr_%M_min.
    EX: "ID02022-01-31__13_hr_19_min"
    """
    date = datetime.now()
    print(date)
    return date.strftime('%Y-%m-%d__%H_hr_%M_min')


def handle_AuthenticationError():
    ee.Authenticate()
    ee.Initialize()


def check_images_available_selected_ROI(
        selected_roi_geojson: dict,
        dates: list,
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

   Returns:
    -----------
   inputs_list: list
        A list of dictionaries containing all the input parameters such as polygon, dates, sat_list, sitename, and filepath
    """
    inputs_list = []
    if selected_roi_geojson["features"] != []:
        date_str = generate_datestring()
        for counter, ROI in enumerate(selected_roi_geojson["features"]):
            coastSatBBOX = ROI["geometry"]["coordinates"]
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
                'filepath': filepath}
            # before downloading the images, check how many images are
            # available for your inputs
            SDS_download.check_images_available(inputs)
            inputs_list.append(inputs)
    return inputs_list
