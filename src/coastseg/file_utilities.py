import os
import re
import glob
import shutil
import json
import math
import logging
import random
import string
import datetime

# Specific classes/functions from modules
from typing import Callable, List, Optional, Union
from json import JSONEncoder

# Third-party imports
import geopandas as gpd
import geojson
import numpy as np

# Internal dependencies imports
from coastseg import exceptions

# Logger setup
logger = logging.getLogger(__name__)


def generate_datestring() -> str:
    """Returns a datetime string in the following format %m-%d-%y__%I_%M_%S
    EX: "ID_0__01-31-22_12_19_45"""
    date = datetime.datetime.now()
    return date.strftime("%m-%d-%y__%I_%M_%S")

def get_session_path(session_name: str, ROI_directory: str) -> str:
    session_path = os.path.join(os.getcwd(), "sessions", session_name)
    session_path = create_directory(session_path, ROI_directory)
    logger.info(f"session_path: {session_path}")
    return session_path

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

def write_to_json(filepath: str, settings: dict):
    """ "Write the  settings dictionary to json file"""
    to_file(settings, filepath)
    # with open(filepath, "w", encoding="utf-8") as output_file:
    #     json.dump(settings, output_file)


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


def find_directory_recursively(path: str = ".", name: str = "RGB") -> str:
    """
    Recursively search for a directory named "RGB" in the given path or its subdirectories.

    Args:
        path (str): The starting directory to search in. Defaults to current directory.

    Returns:
        str: The path of the first directory named "RGB" found, or None if not found.
    """
    dir_location = None
    if os.path.basename(path) == name:
        dir_location = path
    else:
        for dirpath, dirnames, filenames in os.walk(path):
            if name in dirnames:
                dir_location = os.path.join(dirpath, name)

    if not os.listdir(dir_location):
        raise Exception(f"{name} directory is empty.")

    if not dir_location:
        raise Exception(f"{name} directory could not be found")

    return dir_location


def find_file_recursively(path: str = ".", name: str = "RGB") -> str:
    """
    Recursively search for a file named "RGB" in the given path or its subdirectories.

    Args:
        path (str): The starting directory to search in. Defaults to current directory.

    Returns:
        str: The path of the first directory named "RGB" found, or None if not found.
    """
    file_location = None
    if os.path.basename(path) == name:
        file_location = path
    else:
        for dirpath, dirnames, filenames in os.walk(path):
            if name in filenames:
                file_location = os.path.join(dirpath, name)
                return file_location

    if not os.listdir(file_location):
        raise Exception(f"{name} directory is empty.")

    if not file_location:
        raise Exception(f"{name} directory could not be found")

    return file_location


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
        return gpd.read_file(filename)
    else:
        raise FileNotFoundError


def load_json_data_from_file(search_location: str, filename: str) -> dict:
    """
    Load JSON data from a file by searching for the file in the specified location.

    The function searches recursively in the provided search location for a file with
    the specified filename. Once the file is found, it loads the JSON data from the file
    and returns it as a dictionary.

    Args:
        search_location (str): Directory or path to search for the file.
        filename (str): Name of the file to load.

    Returns:
        dict: Data read from the JSON file as a dictionary.

    """
    file_path = find_file_recursively(search_location, filename)
    json_data = load_data_from_json(file_path)
    return json_data


def load_json_data_from_file(search_location: str, filename: str) -> dict:
    """
    Load JSON data from a file by searching for the file in the specified location.

    The function searches recursively in the provided search location for a file with
    the specified filename. Once the file is found, it loads the JSON data from the file
    and returns it as a dictionary.

    Args:
        search_location (str): Directory or path to search for the file.
        filename (str): Name of the file to load.

    Returns:
        dict: Data read from the JSON file as a dictionary.

    """
    file_path = find_file_recursively(search_location, filename)
    json_data = load_data_from_json(file_path)
    return json_data


def load_data_from_json(filepath: str) -> dict:
    """
    Reads data from a JSON file and returns it as a dictionary.

    The function reads the data from the specified JSON file using the provided filepath.
    It applies a custom object hook, `DecodeDateTime`, to decode the datetime and shoreline
    data if they exist in the dictionary.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: Data read from the JSON file as a dictionary.

    """

    def DecodeDateTime(readDict):
        """
        Helper function to decode datetime and shoreline data in the dictionary.

        Args:
            readDict (dict): Dictionary to decode.

        Returns:
            dict: Decoded dictionary.

        """
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
