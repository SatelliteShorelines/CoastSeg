# standard python imports
import os
import logging
import traceback
import sys

# internal python imports
from coastseg.tkinter_window_creator import Tkinter_Window_Creator
from coastseg import exceptions

# external python imports
from google.auth import exceptions as google_auth_exceptions
from tkinter import filedialog
from tkinter import messagebox

logger = logging.getLogger(__name__)

BBOX_NOT_FOUND = "Bounding Box not found on map"

# Separate different exception checking and handling
# Checking decides the message
# Handling it sends message to user


def check_exception_None(feature, feature_type: str,message:str=""):
    if feature is None:
        logger.error(f"{feature_type} {feature} is None")
        raise exceptions.Object_Not_Found(feature_type,message)

def check_exception_gdf_empty(feature, feature_type: str,message:str=""):
    if feature.empty == True:
        logger.error(f"{feature_type} {feature} is empty")
        raise exceptions.Object_Not_Found(feature_type,message)
        # raise ValueError(f"{feature_type} is empty")

def handle_exception(error):
    error_message = traceback.format_exc()
    if isinstance(error, exceptions.Object_Not_Found):
        error_message = str(error)
    logger.error(f"Error Message{error}")
    logger.error(f"{traceback.format_exc()}")

    with Tkinter_Window_Creator():
        messagebox.showinfo("Error", error_message)
