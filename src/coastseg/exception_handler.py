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

BBOX_NOT_FOUND = "Bounding Box not found on map"
EMPTY_SELECTED_ROIS = "Must select at least 1 ROI before you can save ROIs."

# Separate different exception checking and handling
# Checking decides the message
# Handling it sends message to user


def can_feature_save_to_file(feature, feature_type: str = ""):
    if feature is None:
        logger.error(f"Feature {feature_type} did not exist. Cannot Save to File")
        raise ValueError(f"Feature {feature_type} did not exist. Cannot Save to File")

def check_empty_layer(layer,feature_type: str = ""):
    if layer is None:
        logger.error(f"Cannot add an empty {feature_type} layer to the map.")
        raise Exception(f"Cannot add an empty {feature_type} layer to the map.")

def check_exception_None(feature, feature_type: str = "", message: str = ""):
    if feature is None:
        logger.error(f"{feature_type} is None")
        raise exceptions.Object_Not_Found(feature_type, message)


def check_selected_set(selected_set):
    if selected_set is None:
        raise Exception(EMPTY_SELECTED_ROIS)
    if len(selected_set) == 0:
        raise Exception(EMPTY_SELECTED_ROIS)


def check_exception_gdf_empty(feature, feature_type: str, message: str = ""):
    if feature.empty == True:
        logger.error(f"{feature_type} {feature} is empty")
        raise exceptions.Object_Not_Found(feature_type, message)


def handle_exception(error):
    error_message = f"{error}\n" + traceback.format_exc()
    logger.error(f"{traceback.format_exc()}")
    if isinstance(error, exceptions.Object_Not_Found):
        error_message = str(error)
    logger.error(f"{error_message}")
    with Tkinter_Window_Creator():
        messagebox.showinfo("Error", error_message)
