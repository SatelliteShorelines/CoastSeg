# standard python imports
import logging
import traceback
from typing import Union

# internal python imports
from coastseg.tkinter_window_creator import Tkinter_Window_Creator
from coastseg import exceptions
from coastseg import common
from coastseg.roi import ROI

# external python imports
from google.auth import exceptions as google_auth_exceptions
from tkinter import messagebox

logger = logging.getLogger(__name__)

NO_CONFIG_ROIS = (
    "No ROIs were selected. Cannot save ROIs to config until ROIs are selected."
)
NO_CONFIG_SETTINGS = "Settings must be loaded before configuration files can be made."
ROIS_NOT_DOWNLOADED = (
    "Not all ROI directories exist on your computer. Try downloading the ROIs again."
)
BBOX_NOT_FOUND = "Bounding Box not found on map"
EMPTY_SELECTED_ROIS = "Must select at least one ROI on the map"
SETTINGS_NOT_FOUND = "No settings found. Create settings before downloading"
SHORELINE_NOT_FOUND = "No Shoreline found. Please load a shoreline on the map first."
NO_INPUT_SETTINGS = (
    "No inputs settings found. Please click download ROIs first or upload configs"
)
NO_EXTRACTED_SHORELINES = "No shorelines have been extracted. Extract shorelines first."
NO_ROIS_WITH_EXTRACTED_SHORELINES = (
    "You must select an ROI and extract shorelines before you can compute transects"
)
NO_CROSS_DISTANCE_TRANSECTS = "No cross distances transects have been computed"

# Separate different exception checking and handling
# Checking decides the message
# Handling it sends message to user


def config_check_if_none(feature, feature_type: str = ""):
    if feature is None:
        message = (
            f"{feature_type} must be loaded before configuration files can be saved."
        )
        raise exceptions.Object_Not_Found(feature_type, message)


def check_file_not_found(path: str, filename: str, search_path: str):
    # if path is None raises FileNotFoundError
    if path is None:
        logger.error(f"{filename} file was not found at {search_path}")
        raise FileNotFoundError(f"{filename} file was not found at {search_path}")


def check_if_subset(subset: set, superset: set, superset_name: str, message: str = ""):
    if not subset.issubset(superset):
        logger.error(f"Missing keys {subset-superset} from {superset_name}\n{message}")
        raise ValueError(
            f"Missing keys {subset-superset} from {superset_name}\n{message}"
        )


def check_if_rois_downloaded(roi_settings: dict, roi_ids: list):
    if common.were_rois_downloaded(roi_settings, roi_ids) == False:
        logger.error(f"Not all rois were downloaded{roi_settings}")
        raise FileNotFoundError(ROIS_NOT_DOWNLOADED)


def can_feature_save_to_file(feature, feature_type: str = ""):
    if feature is None:
        logger.error(f"Feature {feature_type} did not exist. Cannot Save to File")
        raise ValueError(f"Feature {feature_type} did not exist. Cannot Save to File")


def check_empty_dict(feature, feature_type: str = ""):
    if feature == {}:
        if feature_type == "roi_settings":
            raise Exception(NO_INPUT_SETTINGS)
        if feature_type == "extracted_shorelines":
            raise Exception(NO_EXTRACTED_SHORELINES)
        if feature_type == "cross_distance_transects":
            raise Exception(NO_CROSS_DISTANCE_TRANSECTS)


def check_empty_layer(layer, feature_type: str = ""):
    if layer is None:
        if feature_type == ROI.LAYER_NAME:
            logger.error(f"No ROI layer found on map")
            raise Exception("No ROI layer found on map")
        if feature_type == ROI.SELECTED_LAYER_NAME:
            logger.error(f"No selected ROI layer found on map")
            raise Exception("No selected ROI layer found on map")
        logger.error(f"Cannot add an empty {feature_type} layer to the map.")
        raise Exception(f"Cannot add an empty {feature_type} layer to the map.")


def check_if_list_empty(items: list):
    if len(items) == 0:
        logger.error(f"{items}\n{NO_ROIS_WITH_EXTRACTED_SHORELINES}")
        raise Exception(NO_ROIS_WITH_EXTRACTED_SHORELINES)


def check_if_None(feature, feature_type: str = "", message: str = ""):
    if feature is None:
        if feature_type == "settings":
            message = SETTINGS_NOT_FOUND
        if feature_type == "shoreline":
            message = SHORELINE_NOT_FOUND
        logger.error(f"{feature_type} is None")
        raise exceptions.Object_Not_Found(feature_type, message)


def check_empty_roi_layer(layer):
    if layer is None:
        raise Exception(EMPTY_SELECTED_ROIS)


def check_selected_set(selected_set):
    if selected_set is None:
        raise Exception(EMPTY_SELECTED_ROIS)
    if len(selected_set) == 0:
        raise Exception(EMPTY_SELECTED_ROIS)


def check_if_gdf_empty(feature, feature_type: str, message: str = ""):
    if feature.empty == True:
        logger.error(f"{feature_type} {feature} is empty")
        raise exceptions.Object_Not_Found(feature_type, message)


def handle_exception(error):
    error_message = f"{error}\n\n" + traceback.format_exc()
    logger.error(f"{traceback.format_exc()}")
    if isinstance(error, exceptions.Object_Not_Found):
        error_message = str(error)
    logger.error(f"{error_message}")
    with Tkinter_Window_Creator():
        messagebox.showinfo("Error", error_message)


def handle_bbox_error(
    error_msg: Union[exceptions.BboxTooLargeError, exceptions.BboxTooSmallError]
):
    logger.error(f"Bounding Box Error{error_msg}")
    with Tkinter_Window_Creator():
        messagebox.showwarning("Bounding Box Error", f"{str(error_msg)}")
