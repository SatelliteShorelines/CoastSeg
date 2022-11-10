# standard python imports
import os
import logging
import traceback
import sys
from typing import Union

# internal python imports
from coastseg.tkinter_window_creator import Tkinter_Window_Creator
from coastseg import exceptions

# external python imports
from google.auth import exceptions as google_auth_exceptions
from tkinter import filedialog
from tkinter import messagebox

logger = logging.getLogger(__name__)

CONFIG_ROIS_NOT_FOUND = (
    "No ROIs were selected. Cannot save ROIs to config until ROIs are selected."
)
CONFIG_SETTINGS_NOT_FOUND = (
    "Settings must be loaded before configuration files can be made."
)
BBOX_NOT_FOUND = "Bounding Box not found on map"
EMPTY_SELECTED_ROIS = "Must select at least one ROI on the map"
SETTINGS_NOT_FOUND = "No settings found. Create settings before downloading"
SHORELINE_NOT_FOUND = "No Shoreline found. Please load a shoreline on the map first."
INPUT_SETTINGS_NOT_FOUND = (
    "No inputs settings found. Please click download ROIs first or upload configs"
)
EXTRACTED_SHORELINES_NOT_FOUND = (
    "No shorelines have been extracted. Extract shorelines first."
)
NO_ROIS_WITH_EXTRACTED_SHORELINES = (
    "You must select an ROI and extract shorelines before you can compute transects"
)

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


def can_feature_save_to_file(feature, feature_type: str = ""):
    if feature is None:
        logger.error(f"Feature {feature_type} did not exist. Cannot Save to File")
        raise ValueError(f"Feature {feature_type} did not exist. Cannot Save to File")


def check_empty_dict(feature, feature_type: str = ""):
    if feature == {}:
        if feature_type == "roi_settings":
            raise Exception(INPUT_SETTINGS_NOT_FOUND)
        if feature_type == "extracted_shorelines":
            raise Exception(EXTRACTED_SHORELINES_NOT_FOUND)


def check_empty_layer(layer, feature_type: str = ""):
    if layer is None:
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
