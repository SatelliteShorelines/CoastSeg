from datetime import datetime
import os
import json
import logging
import glob
from typing import List, Optional, Tuple, Union
from collections import defaultdict

from coastseg.bbox import Bounding_Box
from coastseg import common
from coastseg import factory
from coastseg.shoreline import Shoreline
from coastseg.transects import Transects
from coastseg.roi import ROI
from coastseg import exceptions
from coastseg import extracted_shoreline
from coastseg import exception_handler
from coastseg.zoo_model import tidal_corrections
from coastseg.observable import Observable

from coastsat import (
    SDS_download,
)
import pandas as pd
import geopandas as gpd
from ipyleaflet import DrawControl, LayersControl, WidgetControl, GeoJSON
from leafmap import Map
from ipywidgets import Layout, HTML
from tqdm.auto import tqdm
from ipywidgets import HBox
from ipyleaflet import GeoJSON


logger = logging.getLogger(__name__)


class CoastSeg_Map:
    def __init__(self):
        # settings:  used to select data to download and preprocess settings
        self.settings = {}
        # create default values for settings
        self.set_settings()
        self.session_name = ""
        # Make the extracted shoreline layer observerable so that other widgets can update accordingly
        self.extracted_shoreline_layer = Observable(
            None, name="extracted_shoreline_layer"
        )
        # Make the numbrt extracted shorelines observerable so that other widgets can update accordingly
        self.number_extracted_shorelines = Observable(
            0, name="number_extracted_shorelines"
        )
        self.roi_ids_with_extracted_shorelines = Observable(
            [], name="roi_ids_with_shorelines"
        )

        # factory is used to create map objects
        self.factory = factory.Factory()

        # ids of the selected rois
        self.selected_set = set()

        # map objects
        self.rois = None
        self.transects = None
        self.shoreline = None
        self.bbox = None
        self.map = self.create_map()
        self.draw_control = self.create_DrawControl(DrawControl())
        self.draw_control.on_draw(self.handle_draw)
        self.map.add(self.draw_control)
        layer_control = LayersControl(position="topright")
        self.map.add(layer_control)

        # Warning and information boxes
        self.warning_box = HBox([])
        self.warning_widget = WidgetControl(widget=self.warning_box, position="topleft")
        self.map.add(self.warning_widget)
        self.roi_html = HTML("""""")
        self.roi_box = common.create_hover_box(title="ROI", feature_html=self.roi_html)
        self.roi_widget = WidgetControl(widget=self.roi_box, position="topright")
        self.map.add(self.roi_widget)
        self.feature_html = HTML("""""")
        self.hover_box = common.create_hover_box(
            title="Feature", feature_html=self.feature_html
        )
        self.hover_widget = WidgetControl(widget=self.hover_box, position="topright")
        self.map.add(self.hover_widget)

    def __str__(self):
        return f"CoastSeg: roi={self.rois}\n shoreline={self.shoreline}\n  transects={self.transects}\n bbox={self.bbox}"

    def __repr__(self):
        return f"CoastSeg(roi={self.rois}\n shoreline={self.shoreline}\n transects={self.transects}\n bbox={self.bbox} "

    def get_session_name(self):
        return self.session_name

    def set_session_name(self, name: str):
        self.session_name = name

    def create_map(self):
        """create an interactive map object using the map_settings
        Returns:
           ipyleaflet.Map: ipyleaflet interactive Map object
        """
        map_settings = {
            "center_point": (40.8630302395, -124.166267),
            "zoom": 13,
            "draw_control": False,
            "measure_control": False,
            "fullscreen_control": False,
            "attribution_control": True,
            "Layout": Layout(width="100%", height="100px"),
        }
        return Map(
            draw_control=map_settings["draw_control"],
            measure_control=map_settings["measure_control"],
            fullscreen_control=map_settings["fullscreen_control"],
            attribution_control=map_settings["attribution_control"],
            center=map_settings["center_point"],
            zoom=map_settings["zoom"],
            layout=map_settings["Layout"],
        )

    def compute_tidal_corrections(
        self, tides_location: str, beach_slope: float, reference_elevation: float
    ):
        session_name = self.get_session_name()
        session_path = os.path.join(os.getcwd(), "sessions", session_name)
        if not os.path.exists(session_path):
            raise FileNotFoundError(session_path)
        # get selected ROIs
        roi_ids = self.get_selected_roi_ids()
        exception_handler.check_selected_set(self.selected_set)
        tide_data = pd.read_csv(tides_location, parse_dates=["dates"])
        if "tides" not in tide_data.columns or "dates" not in tide_data.columns:
            logger.error(
                f"Invalid tides csv file {tides_location} provided must include columns : 'tides'and 'dates'"
            )
            raise Exception(
                f"Invalid tides csv file {tides_location} provided must include columns : 'tides'and 'dates'"
            )
        for roi_id in roi_ids:
            # create roi directory in session path
            ROI_directory = self.rois.roi_settings[roi_id]["sitename"]
            session_path = common.get_session_path(session_name, ROI_directory)
            # get extracted shoreline for each roi
            extracted_shoreline = self.rois.get_extracted_shoreline(roi_id)
            if extracted_shoreline is None:
                logger.info(f"No extracted shorelines for ROI: {roi_id}")
                print(f"No extracted shorelines for ROI: {roi_id}")
                continue
            # get extracted transect for each roi
            cross_shore_distance = self.rois.get_cross_shore_distances(roi_id)
            if cross_shore_distance is None:
                logger.info(f"No cross_shore_distance for ROI: {roi_id}")
                print(f"No cross_shore_distance for ROI: {roi_id}")
                continue
            tidal_corrections(
                roi_id,
                beach_slope,
                reference_elevation,
                extracted_shoreline.dictionary,
                cross_shore_distance,
                tide_data,
                session_path,
            )
            logger.info(f"{roi_id} was tidally corrected")
            print(f"\n{roi_id} was tidally corrected")
        logger.info(f"{roi_id} was tidally corrected")
        print("\ntidal corrections completed")

    def load_session_files(self, dir_path: str) -> None:
        """
        Load the configuration files from the given directory.

        The function looks for the following files in the directory:
        - config_gdf.geojson: contains the configuration settings for the project
        - transects_settings.json: contains the settings for the transects module
        - shoreline_settings.json: contains the settings for the shoreline module

        If the config_gdf.geojson file is not found, a message is printed to the console.

        Args:
            dir_path (str): The path to the directory containing the configuration files.

        Returns:
            None
        """
        if os.path.isdir(dir_path):
            # load the config files if they exist
            config_loaded = self.load_config_files(dir_path)
            # load in settings files
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                if not os.path.isfile(file_path):
                    continue
                if file_name == "shoreline_settings.json":
                    keys = [
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
                    ]
                    self.load_settings(file_path, keys)
                elif file_name == "transects_settings.json":
                    keys = [
                        "max_std",
                        "min_points",
                        "along_dist",
                        "max_range",
                        "min_chainage",
                        "multiple_inter",
                        "prc_multiple",
                    ]
                    self.load_settings(file_path, keys)
            if not config_loaded:
                logger.info(f"Not all config files not found at {dir_path}")

    def load_session_from_directory(self, dir_path: str) -> None:
        """
        Loads a session from a specified directory path.
        Loads config files, extracted shorelines, and transects & extracted shoreline intersections.

        Args:
            dir_path (str): The path of the directory to load the session from.

        Returns:
            None. The function updates the coastseg instance with ROIs, extracted shorelines, and transects
        """
        # load config files
        self.load_session_files(dir_path)
        # for every directory load extracted shorelines
        extracted_shorelines = extracted_shoreline.load_extracted_shoreline_from_files(
            dir_path
        )
        if extracted_shorelines is None:
            logger.warning(f"No extracted shorelines found in {dir_path}")
            return
        # get roi id from extracted shoreline
        roi_id = extracted_shorelines.get_roi_id()
        if roi_id is None:
            logger.warning(
                f"No roi id found extracted shorelines settings{extracted_shorelines.shoreline_settings}"
            )
            return

        # add extracted shoreline to ROI it was extracted from
        if self.rois is not None:
            self.rois.add_extracted_shoreline(extracted_shorelines, roi_id)
            # load extracted shoreline and transect intersections
            cross_distances = common.load_cross_distances_from_file(dir_path)
            # add extracted shoreline and transect intersections to ROI they were extracted from
            self.rois.add_cross_shore_distances(cross_distances, roi_id)

    def load_fresh_session(self, session_path: str) -> None:
        # remove all the old features from the map
        self.remove_all()
        self.load_session(session_path)

    def load_session(self, session_path: str) -> None:
        """
        Load a session from the given path.

        The function loads a session from the given path, which can contain one or more directories, each containing
        the files for a single ROI. For each subdirectory, the function calls `load_session_from_directory` to load
        the session files and objects on the map. If no subdirectories exist, the function calls `load_session_from_directory` with the
        session path.

        Args:
            session_path: The path to the session directory.

        Returns:
            None.
        """
        logger.info(f"Loading session: {session_path}")
        # load the session name
        session_name = os.path.basename(os.path.abspath(session_path))
        self.set_session_name(session_name)

        # load the session from the session path
        self.load_session_from_directory(session_path)
        # Use os.walk to efficiently traverse the directory tree
        for dirpath, dirnames, filenames in os.walk(session_path):
            # If there are subdirectories
            if dirpath != session_path:
                self.load_session_from_directory(dirpath)

        # Check if any ROIs are loaded
        if self.rois is None:
            logger.warning("No ROIs found. Please load ROIs.")
            return
        # Get the ids of the ROIs with extracted shorelines
        ids_with_extracted_shorelines = self.rois.get_ids_with_extracted_shorelines()
        # update observable list of ROI ids with extracted shorelines
        self.roi_ids_with_extracted_shorelines.set(ids_with_extracted_shorelines)

    def load_settings(
        self,
        filepath: str = "",
        keys: set = (
            "sat_list",
            "dates",
            "sand_color",
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
            "max_std",
            "min_points",
            "along_dist",
            "max_range",
            "min_chainage",
            "multiple_inter",
            "prc_multiple",
        ),
        load_nested_settings:bool=True,
    ):
        """
        Loads settings from a JSON file and applies them to the object.

        Args:
            filepath (str, optional): The filepath to the JSON file containing the settings. Defaults to an empty string.
            load_nested_setting (bool, optional): Load settings from a nest subdictionary 'settings' or not.
            keys (list or set, optional): A list of keys specifying which settings to load from the JSON file. If empty, no settings are loaded. Defaults to a set with the following
            "sat_list",
                                                        "dates",
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
                                                        "max_std",
                                                        "min_points",
                                                        "along_dist",
                                                        "max_range",
                                                        "min_chainage",
                                                        "multiple_inter",
                                                        "prc_multiple".

        Returns:
            None

        """
        # Convert keys to a list if a set is passed
        if isinstance(keys, set):
            keys = list(keys)

        new_settings = common.read_json_file(filepath, raise_error=False)
        logger.info(f"all of new settings read from file : {filepath} \n {new_settings}")
        
        nested_settings = new_settings.get('settings',{})
        logger.info(f"all of new nested settings read from file : {filepath} \n {nested_settings }")
        

        if new_settings is None:
            new_settings = {}

        # Load only settings with provided keys
        if keys:
            new_settings = {k: new_settings[k] for k in keys if k in new_settings}
            if nested_settings:
                nested_settings  = {k: nested_settings[k] for k in keys if k in nested_settings}

        if new_settings != {}:
            self.set_settings(**new_settings)
        if nested_settings != {} and load_nested_settings:
            self.set_settings(**nested_settings)
            logger.info(
                f"Loaded new_settings from {filepath}:\n new self.settings {self.settings}"
            )

    def load_gdf_config(self, filepath: str) -> None:
        """Load features from geodataframe located in geojson file at filepath onto map.

        Features in config file should contain a column named "type" which contains one of the
        following possible feature types: "roi", "shoreline", "transect", "bbox".

        Args:
            filepath (str): full path to config_gdf.geojson
        """
        gdf = common.read_gpd_file(filepath)
        gdf = common.stringify_datetime_columns(gdf)

        # each possible type of feature and the columns that should be loaded
        feature_types = {
            "bbox": ["geometry"],
            "roi": ["id", "geometry"],
            "transect": ["id", "slope", "geometry"],
            "shoreline": ["geometry"],
        }

        for feature_name, columns in feature_types.items():
            feature_gdf = self._extract_feature_gdf(gdf, feature_name, columns)
            if feature_name == "roi":
                exception_handler.check_if_gdf_empty(
                    feature_gdf, "ROIs", "Cannot load empty ROIs onto map"
                )
                if self.rois is None:
                    self.rois = ROI(rois_gdf=feature_gdf)
                    self.load_feature_on_map(
                        feature_name, gdf=feature_gdf, zoom_to_bounds=True
                    )
                elif self.rois is not None:
                    # add the new roi to the existing rois
                    self.rois = self.rois.add_geodataframe(feature_gdf)
                    # load the new rois onto the map
                    self.add_feature_on_map(self.rois, feature_name)
            else:
                # load shorelines, transects, or bbox features onto the map
                self.load_feature_on_map(feature_name, gdf=feature_gdf)
        del gdf

    def _extract_feature_gdf(
        self, gdf: gpd.GeoDataFrame, feature_type: str, columns: List[str]
    ) -> gpd.GeoDataFrame:
        """
        Extracts a GeoDataFrame of features of a given type and specified columns from a larger GeoDataFrame.

        Args:
            gdf (gpd.GeoDataFrame): The GeoDataFrame containing the features to extract.
            feature_type (str): The type of feature to extract.
            columns (List[str]): A list of column names to extract from the GeoDataFrame.

        Returns:
            gpd.GeoDataFrame: A new GeoDataFrame containing only the features of the specified type and columns.

        Raises:
            None
        """
        # select only the columns that are in the gdf
        keep_columns = [col for col in columns if col in gdf.columns]
        # select only the features that are of the correct type and have the correct columns
        feature_gdf = gdf[gdf["type"] == feature_type][keep_columns]
        return feature_gdf

    def download_imagery(self) -> None:
        """
        Downloads all images for the selected ROIs  from Landsat 5, Landsat 7, Landsat 8 and Sentinel-2  covering the area of interest and acquired between the specified dates.
        The downloaded imagery for each ROI is stored in a directory that follows the convention
        ID_{ROI}_datetime{current date}__{time}' ex.ID_0_datetime04-11-23__10_20_48. The files are saved as jpgs in a subdirectory
        'jpg_files' in a subdirectory 'preprocessed' which contains subdirectories for RGB, NIR, and SWIR jpg imagery. The downloaded .TIF images are organised in subfolders, divided
        by satellite mission. The bands are also subdivided by pixel resolution.

        Raises:
            Exception: raised if settings is missing
            Exception: raised if 'dates','sat_list', and 'landsat_collection' are not in settings
            Exception: raised if no ROIs have been selected
        """

        self.validate_download_imagery_inputs()

        # selected_layer contains the selected ROIs
        selected_layer = self.map.find_layer(ROI.SELECTED_LAYER_NAME)
        logger.info(f"selected_layer: {selected_layer}")

        # Get the file path where the downloaded imagery will be saved
        file_path = os.path.abspath(os.path.join(os.getcwd(), "data"))
        date_str = common.generate_datestring()

        settings = self.get_settings()
        # Create a list of download settings for each ROI
        roi_settings = common.create_roi_settings(
            settings, selected_layer.data, file_path, date_str
        )

        # Save the ROI settings
        self.rois.set_roi_settings(roi_settings)

        # create a list of settings for each ROI
        inputs_list = list(roi_settings.values())
        logger.info(f"inputs_list {inputs_list}")

        # 2. For each ROI use download settings to download imagery and save to jpg
        print("Download in progress")
        # for each ROI use the ROI settings to download imagery and save to jpg
        for inputs_for_roi in tqdm(inputs_list, desc="Downloading ROIs"):
            SDS_download.retrieve_images(
                inputs_for_roi,
                cloud_threshold=settings.get("cloud_thresh"),
                cloud_mask_issue=settings.get("cloud_mask_issue"),
                save_jpg=True,
            )
        # 3.save settings used to download rois and the objects on map to config files
        self.save_config()
        logger.info("Done downloading")

    def load_json_config(self, filepath: str, data_path: str) -> None:
        """
        Loads a .json configuration file specified by the user.
        It replaces the coastseg_map.settings with the settings from the config file,
        and replaces the roi_settings for each ROI with the contents of the json_data.
        Finally, it saves the input dictionaries for all ROIs.

        Args:
            self (object): CoastsegMap instance
            filepath (str): The filepath to the json config file
            datapath (str): Full path to the coastseg data directory

        Returns:
            None

        Raises:
            FileNotFoundError: If the config file is not found
            MissingDirectoriesError: If one or more directories specified in the config file are missing

        """
        logger.info(f"filepath: {filepath}")
        exception_handler.check_if_None(self.rois)

        json_data = common.read_json_file(filepath, raise_error=True)
        json_data = json_data or {}

        # Replace coastseg_map.settings with settings from config file
        self.set_settings(**json_data.get("settings", {}))

        # Replace roi_settings for each ROI with contents of json_data
        roi_settings = {}
        missing_directories = []
        fields_of_interest = [
            "dates",
            "sitename",
            "polygon",
            "roi_id",
            "sat_list",
            "sitename",
            "landsat_collection",
            "filepath",
        ]

        logger.info(f"json_data: {json_data}\n")

        for roi_id in json_data.get("roi_ids", []):
            logger.info(f"roi_id: {roi_id}")
            # get the fields of interest from an roi with a matching id
            roi_data = common.extract_roi_data(json_data, roi_id, fields_of_interest)

            sitename = roi_data.get("sitename", "")
            roi_path = os.path.join(data_path, sitename)
            roi_data["filepath"] = data_path

            if not os.path.exists(roi_path):
                missing_directories.append(sitename)

            roi_settings[str(roi_id)] = roi_data

        # if any directories are missing tell the user list of missing directories
        exception_handler.check_if_dirs_missing(missing_directories)

        # Save input dictionaries for all ROIs
        self.rois.roi_settings = roi_settings
        logger.info(f"roi_settings: {roi_settings}")

    def load_config_files(self, dir_path: str) -> None:
        """Loads the configuration files from the specified directory
            Loads config_gdf.geojson first, then config.json.

        - config.json relies on config_gdf.geojson to load the rois on the map
        Args:
            dir_path (str): path to directory containing config files
        Raises:
            Exception: raised if config files are missing
        """
        # check if config files exist
        config_geojson_path = os.path.join(dir_path, "config_gdf.geojson")
        config_json_path = os.path.join(dir_path, "config.json")
        logger.info(
            f"config_gdf.geojson: {config_geojson_path} \n config.json location: {config_json_path}"
        )
        # cannot load config.json file if config_gdf.geojson file is missing
        if not os.path.exists(config_geojson_path):
            logger.warning(f"config_gdf.geojson file missing at {config_geojson_path}")
            return False

        # load the config files
        # ensure coastseg\data location exists
        data_path = common.create_directory(os.getcwd(), "data")
        # load general settings from config.json file
        self.load_gdf_config(config_geojson_path)
        # do not attempt to load config.json file if it is missing
        if not os.path.exists(config_json_path):
            logger.warning(f"config.json file missing at {config_json_path}")
            raise Exception(f"config.json file missing at {config_json_path}")
        self.load_json_config(config_json_path, data_path)
        # return true if both config files exist
        return True

    def save_config(self, filepath: str = None) -> None:
        """saves the configuration settings of the map into two files
            config.json and config_gdf.geojson
            Saves the inputs such as dates, landsat_collection, satellite list, and ROIs
            Saves the settings such as preprocess settings
        Args:
            file_path (str, optional): path to directory to save config files. Defaults to None.
        Raises:
            Exception: raised if self.settings is missing
            ValueError: raised if any of "dates", "sat_list", "landsat_collection" is missing from self.settings
            Exception: raised if self.rois is missing
            Exception: raised if selected_layer is missing
        """
        settings = self.get_settings()
        # if no rois exist on the map do not allow configs to be saved
        exception_handler.config_check_if_none(self.rois, "ROIs")

        # selected_layer must contain selected ROI
        selected_layer = self.map.find_layer(ROI.SELECTED_LAYER_NAME)
        exception_handler.check_empty_roi_layer(selected_layer)
        logger.info(f"self.rois.roi_settings: {self.rois.roi_settings}")

        if not self.rois.roi_settings:
            filepath = filepath or os.path.abspath(os.getcwd())
            roi_settings = common.create_roi_settings(
                settings, selected_layer.data, filepath
            )
            self.rois.set_roi_settings(roi_settings)

        # create dictionary to be saved to config.json
        roi_ids = self.get_selected_roi_ids()
        selected_roi_settings = {
            roi_id: self.rois.roi_settings[roi_id] for roi_id in roi_ids
        }
        config_json = common.create_json_config(selected_roi_settings, settings)

        shorelines_gdf = self.shoreline.gdf if self.shoreline else None
        transects_gdf = self.transects.gdf if self.transects else None
        bbox_gdf = self.bbox.gdf if self.bbox else None
        selected_rois = self.get_selected_rois(roi_ids)

        # save all selected rois, shorelines, transects and bbox to config geodataframe
        logger.info(f"selected_rois: {selected_rois}")
        if selected_rois is not None:
            if not selected_rois.empty:
                epsg_code = selected_rois.crs
        config_gdf = common.create_config_gdf(
            selected_rois,
            shorelines_gdf=shorelines_gdf,
            transects_gdf=transects_gdf,
            bbox_gdf=bbox_gdf,
            epsg_code=epsg_code,
        )
        logger.info(f"config_gdf: {config_gdf} ")
        is_downloaded = common.were_rois_downloaded(self.rois.roi_settings, roi_ids)

        if filepath is not None:
            # if a filepath is provided then save the config.json and config_gdf.geojson immediately
            common.config_to_file(config_json, filepath)
            common.config_to_file(config_gdf, filepath)
        elif filepath is None:
            # data has been downloaded before so inputs have keys 'filepath' and 'sitename'
            if is_downloaded == True:
                # write config_json file to each directory where a roi was saved
                roi_ids = config_json["roi_ids"]
                for roi_id in roi_ids:
                    sitename = str(config_json[roi_id]["sitename"])
                    filepath = os.path.abspath(
                        os.path.join(config_json[roi_id]["filepath"], sitename)
                    )
                    # save to config.json
                    common.config_to_file(config_json, filepath)
                    # save to config_gdf.geojson
                    common.config_to_file(config_gdf, filepath)
                print("Saved config files for each ROI")
            elif is_downloaded == False:
                # if data is not downloaded save to coastseg directory
                filepath = os.path.abspath(os.getcwd())
                # save to config.json
                common.config_to_file(config_json, filepath)
                # save to config_gdf.geojson
                common.config_to_file(config_gdf, filepath)
                print("Saved config files for each ROI")

    def set_settings(self, **kwargs):
        """
        Saves the settings for downloading data by updating the `self.settings` dictionary with the provided key-value pairs.
        If any of the keys are missing, they will be set to their default value as specified in `default_settings`.

        Example: set_settings(sat_list=sat_list, dates=dates,**more_settings)

        Args:
        **kwargs: Keyword arguments representing the key-value pairs to be added to or updated in `self.settings`.

        Returns:
        None
        """
        logger.info(f"New Settings: {kwargs}")
        # Check if any of the keys are missing
        # if any keys are missing set the default value
        self.default_settings = {
            "landsat_collection": "C02",
            "dates": ["2017-12-01", "2018-01-01"],
            "sat_list": ["L8"],
            "cloud_thresh": 0.5,  # threshold on maximum cloud cover
            "dist_clouds": 300,  # ditance around clouds where shoreline can't be mapped
            "output_epsg": 4326,  # epsg code of spatial reference system desired for the output
            # quality control:
            # if True, shows each shoreline detection to the user for validation
            "check_detection": False,
            # if True, allows user to adjust the position of each shoreline by changing the threshold
            "adjust_detection": False,
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
        }
        self.settings.update(kwargs)
        if "dates" in kwargs.keys():
            updated_dates = []
            self.settings["dates"] = kwargs['dates']
            for date_str in kwargs['dates']:
                try:
                    dt = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
                except ValueError:
                    dt = datetime.strptime(date_str, '%Y-%m-%d')
                updated_dates.append(dt.strftime('%Y-%m-%d'))
            self.settings["dates"] = updated_dates
        
        for key, value in self.default_settings.items():
            self.settings.setdefault(key, value)

        logger.info(f"Settings: {self.settings}")

    def get_settings(self):
        """
        Retrieves the current settings.

        Returns:
            dict: A dictionary containing the current settings.

        Raises:
            Exception: If no settings are found. Click save settings or load a config file.

        """
        SETTINGS_NOT_FOUND = (
            "No settings found. Click save settings or load a config file."
        )
        logger.info(f"self.settings: {self.settings}")
        if self.settings is None or self.settings == {}:
            raise Exception(SETTINGS_NOT_FOUND)
        return self.settings

    def update_transects_html(self, feature: dict, **kwargs):
        """
        Modifies the HTML when a transect is hovered over.

        Args:
            feature (dict): The transect feature.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        """
        properties = feature["properties"]
        transect_id = properties.get("id", "unknown")
        slope = properties.get("slope", "unknown")

        self.feature_html.value = (
            "<div style='max-width: 230px; max-height: 200px; overflow-x: auto; overflow-y: auto'>"
            "<b>Transect</b>"
            f"<p>Id: {transect_id}</p>"
            f"<p>Slope: {slope}</p>"
        )

    def update_extracted_shoreline_html(self, feature, **kwargs):
        """
        Modifies the HTML content when an extracted shoreline is hovered over.

        Args:
            feature (dict): The extracted shoreline feature.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        """
        # Modifies html when extracted shoreline is hovered over
        properties = defaultdict(lambda: "unknown", feature["properties"])
        self.feature_html.value = """
        <div style='max-width: 230px; max-height: 200px; overflow-x: auto; overflow-y: auto'>
        <b>Extracted Shoreline</b>
        <p>Date: {}</p>
        <p>Geoaccuracy: {}</p>
        <p>Cloud Cover: {}</p>
        <p>Satellite Name: {}</p>
        """.format(
            properties["date"],
            properties["geoaccuracy"],
            properties["cloud_cover"],
            properties["satname"],
        )

    def update_roi_html(self, feature, **kwargs):
        # Modifies html when roi is hovered over
        values = defaultdict(lambda: "unknown", feature["properties"])
        # convert roi area m^2 to km^2
        roi_area = common.get_area(feature["geometry"]) * 10**-6
        roi_area = round(roi_area, 5)
        self.roi_html.value = """ 
        <div style='max-width: 230px; max-height: 200px; overflow-x: auto; overflow-y: auto'>
        <b>ROI</b>
        <p>Id: {}</p>
        <p>Area(km²): {}</p>
        """.format(
            values["id"], roi_area
        )

    def update_shoreline_html(self, feature, **kwargs):
        # Modifies html when shoreline is hovered over
        values = defaultdict(lambda: "unknown", feature["properties"])
        self.feature_html.value = """
        <b>Shoreline</b>
        <p>Mean Sig Waveheight: {}</p>
        <p>Tidal Range: {}</p>
        <p>Erodibility: {}</p>
        <p>River: {}</p>
        <p>Sinuosity: {}</p>
        <p>Slope: {}</p>
        <p>Turbid: {}</p>
        <p>CSU_ID: {}</p>
        """.format(
            values["MEAN_SIG_WAVEHEIGHT"],
            values["TIDAL_RANGE"],
            values["ERODIBILITY"],
            values["river_label"],
            values["sinuosity_label"],
            values["slope_label"],
            values["turbid_label"],
            values["CSU_ID"],
        )

    def get_all_roi_ids(self) -> List[str]:
        """
        Return a list of all ROI IDs.

        Args:
            None.

        Returns:
            A list of all ROI IDs.

        Raises:
            None.
        """
        if self.rois is None:
            return []
        if self.rois.gdf.empty:
            return []
        return self.rois.gdf["id"].tolist()

    def load_extracted_shoreline_files(self) -> None:
        exception_handler.config_check_if_none(self.rois, "ROIs")
        # if no rois are selected throw an error
        # exception_handler.check_selected_set(self.selected_set)
        roi_ids = self.get_selected_roi_ids()
        if roi_ids == []:
            roi_ids = self.get_all_roi_ids()
            if roi_ids == []:
                raise Exception("No ROIs found. Please load ROIs.")
            roi_ids = roi_ids[0]
        logger.info(f"roi_ids: {roi_ids}")
        logger.info(f"self.rois.roi_settings: {self.rois.roi_settings}")
        # set of roi ids that didn't have missing shorelines
        rois_no_extracted_shorelines = set()
        # for each ROI that has extracted shorelines load onto map
        for roi_id in roi_ids:
            filepath = self.rois.roi_settings[roi_id]["filepath"]
            sitename = self.rois.roi_settings[roi_id]["sitename"]
            roi_path = os.path.join(filepath, sitename)
            glob_str = os.path.abspath(roi_path + os.sep + "*shoreline*")
            extracted_sl_gdf = None
            shoreline_settings = None
            extracted_shoreline_dict = None
            for file in glob.glob(glob_str):
                if file.endswith(".geojson"):
                    # load geodataframe
                    extracted_sl_gdf = common.read_gpd_file(file)
                if file.endswith(".json"):
                    if "settings" in os.path.basename(file):
                        shoreline_settings = common.load_data_from_json(file)
                    if "dict" in os.path.basename(file):
                        extracted_shoreline_dict = common.load_data_from_json(file)

            logger.info(f"ROI {roi_id} extracted_sl_gdf: {extracted_sl_gdf}")
            logger.info(f"ROI {roi_id} shoreline_settings: {shoreline_settings}")
            logger.info(
                f"ROI {roi_id} extracted_shoreline_dict: {extracted_shoreline_dict}"
            )
            # error handling for none
            if (
                extracted_sl_gdf is None
                or extracted_sl_gdf is None
                or extracted_sl_gdf is None
            ):
                logger.info(
                    f"ROI {roi_id} didn't have extracted shoreline files to load"
                )
                rois_no_extracted_shorelines.add(roi_id)
                # set this roi's entry in extracted shorelines dict to None because there was no shoreline extracted
                self.rois.add_extracted_shoreline(None, roi_id)
                continue
            else:
                extracted_shorelines = extracted_shoreline.Extracted_Shoreline()
                extracted_shorelines = extracted_shorelines.load_extracted_shorelines(
                    extracted_shoreline_dict, shoreline_settings, extracted_sl_gdf
                )
                self.rois.add_extracted_shoreline(extracted_shorelines, roi_id)
                logger.info(
                    f"ROI {roi_id} successfully loaded extracted shorelines: {self.rois.get_extracted_shoreline(roi_id).dictionary}"
                )

        if len(rois_no_extracted_shorelines) > 0:
            print(
                f"The following ROIs didn't have extracted shoreline files to load: {rois_no_extracted_shorelines}\n"
            )

        rois_with_shorelines = set(roi_ids) - rois_no_extracted_shorelines
        if len(rois_with_shorelines) > 0:
            print(f"Loaded Extracted Shorelines for ROIs {rois_with_shorelines}")
        elif len(rois_with_shorelines) == 0:
            raise Exception("No extracted shorelines could be loaded")

    def extract_shoreline_for_roi(
        self,
        roi_id: str,
        rois_gdf: gpd.GeoDataFrame,
        shoreline_gdf: gpd.GeoDataFrame,
        settings: dict,
    ) -> Optional[extracted_shoreline.Extracted_Shoreline]:
        """
        Extracts the shoreline for a given ROI and returns the extracted shoreline object.

        Parameters:
        - roi_id (str): the ID of the ROI to extract the shoreline from
        - rois_gdf (geopandas.GeoDataFrame): the GeoDataFrame containing all the ROIs
        - shoreline_gdf (geopandas.GeoDataFrame): the GeoDataFrame containing the shoreline
        - settings (dict): the settings to use for the shoreline extraction


        Returns:
        - extracted_shoreline.Extracted_Shoreline or None: the extracted shoreline object for the ROI, or None if an error occurs

        Raises:
        - No exceptions are raised by this function.
        """
        try:
            logger.info(f"Extracting shorelines from ROI with the id: {roi_id}")
            roi_settings = self.rois.roi_settings[roi_id]
            single_roi = common.extract_roi_by_id(rois_gdf, roi_id)
            # Clip shoreline to specific roi
            shoreline_in_roi = gpd.clip(shoreline_gdf, single_roi)
            # extract shorelines from ROI
            extracted_shorelines = extracted_shoreline.Extracted_Shoreline()
            extracted_shorelines = extracted_shorelines.create_extracted_shorlines(
                roi_id,
                shoreline_in_roi,
                roi_settings,
                settings,
            )
            logger.info(f"extracted_shoreline_dict[{roi_id}]: {extracted_shorelines}")
            return extracted_shorelines
        except exceptions.Id_Not_Found as id_error:
            logger.warning(f"exceptions.Id_Not_Found {id_error}")
            print(f"ROI with id {roi_id} was not found. \n Skipping to next ROI")
        except exceptions.No_Extracted_Shoreline as no_shoreline:
            logger.warning(f"{roi_id}: {no_shoreline}")
            print(f"{roi_id}: {no_shoreline}")
        except Exception as e:
            logger.warning(
                f"Exception occurred while extracting shoreline for ROI {roi_id}: {e}"
            )
            print(
                f"An error occurred while extracting shoreline for ROI {roi_id}. \n Skipping to next ROI"
            )
        return None

    def update_settings(self):
        """Updates settings with the most accurate epsg code based on lat and lon if output epsg
        was 4326 or 4327.
        """
        settings = self.get_settings()
        new_espg = common.get_most_accurate_epsg(
            settings.get("output_epsg", 4326), self.bbox.gdf
        )
        self.set_settings(output_epsg=new_espg)

    def validate_transect_inputs(self, settings):
        # ROIs,settings, roi-settings cannot be None or empty
        exception_handler.check_if_empty_string(self.get_session_name(), "session name")
        # ROIs, transects, and extracted shorelines must exist
        exception_handler.check_if_None(self.rois, "ROIs")
        exception_handler.check_if_None(self.transects, "transects")
        exception_handler.check_empty_dict(
            self.rois.get_all_extracted_shorelines(), "extracted_shorelines"
        )
        # settings must contain key 'along_dist'
        exception_handler.check_if_subset(
            set(["along_dist"]), set(list(settings.keys())), "settings"
        )
        # if no rois are selected throw an error
        exception_handler.check_selected_set(self.selected_set)

        # ids of ROIs that have had their shorelines extracted
        extracted_shoreline_ids = set(
            list(self.rois.get_all_extracted_shorelines().keys())
        )
        # Get ROI ids that are selected on map and have had their shorelines extracted
        roi_ids = list(extracted_shoreline_ids & self.selected_set)
        # if none of the selected ROIs on the map have had their shorelines extracted throw an error
        exception_handler.check_if_list_empty(roi_ids)

    def validate_extract_shoreline_inputs(self):
        # ROIs,settings, roi-settings cannot be None or empty
        settings = self.get_settings()
        exception_handler.check_if_empty_string(self.get_session_name(), "session name")
        # ROIs, transects,shorelines and a bounding box must exist
        exception_handler.check_if_None(self.rois, "ROIs")
        exception_handler.check_if_None(self.shoreline, "shoreline")
        exception_handler.check_if_None(self.transects, "transects")
        exception_handler.check_if_None(self.bbox, "bounding box")
        # ROI settings must not be empty
        exception_handler.check_empty_dict(self.rois.roi_settings, "roi_settings")

        # settings must contain keys in subset
        superset = set(list(settings.keys()))
        exception_handler.check_if_subset(
            set(["dates", "sat_list", "landsat_collection"]), superset, "settings"
        )

        # if no rois are selected throw an error
        exception_handler.check_selected_set(self.selected_set)

        # roi_settings must contain roi ids in selected set
        superset = set(list(self.rois.roi_settings.keys()))
        error_message = "To extract shorelines you must first select ROIs and have the data downloaded."
        exception_handler.check_if_subset(
            self.selected_set, superset, "roi_settings", error_message
        )
        # raise error if selected rois were not downloaded
        exception_handler.check_if_rois_downloaded(
            self.rois.roi_settings, self.get_selected_roi_ids()
        )

    def validate_download_imagery_inputs(self):
        # settings cannot be None
        settings = self.get_settings()
        # Ensure the required keys are present in the settings
        required_settings_keys = set(["dates", "sat_list", "landsat_collection"])
        superset = set(list(settings.keys()))
        exception_handler.check_if_subset(required_settings_keys, superset, "settings")

        # selected_layer must contain selected ROI
        selected_layer = self.map.find_layer(ROI.SELECTED_LAYER_NAME)
        exception_handler.check_empty_layer(selected_layer, ROI.SELECTED_LAYER_NAME)
        exception_handler.check_empty_roi_layer(selected_layer)

    def get_roi_ids_with_extracted_shorelines(self, is_selected: bool = True) -> list:
        # ids of ROIs that have had their shorelines extracted
        roi_ids = set(self.rois.get_ids_with_extracted_shorelines())
        logger.info(f"extracted_shoreline_ids:{roi_ids}")
        # Get ROI ids that are selected on map and have had their shorelines extracted
        if is_selected:
            roi_ids = list(roi_ids & self.selected_set)
        return roi_ids

    def extract_all_shorelines(self) -> None:
        """Use this function when the user interactively downloads rois
        Iterates through all the ROIS downloaded by the user as indicated by the roi_settings generated by
        download_imagery() and extracts a shoreline for each of them
        """
        self.validate_extract_shoreline_inputs()
        roi_ids = self.get_selected_roi_ids()
        logger.info(f"roi_ids to extract shorelines from: {roi_ids}")

        # update the settings with the most accurate epsg
        self.update_settings()
        # update configs with new output_epsg
        self.save_config()

        # get selected ROIs on map and extract shoreline for each of them
        for roi_id in tqdm(roi_ids, desc="Extracting Shorelines"):
            print(f"Extracting shorelines from ROI with the id:{roi_id}")
            extracted_shorelines = self.extract_shoreline_for_roi(
                roi_id, self.rois.gdf, self.shoreline.gdf, self.get_settings()
            )
            self.rois.add_extracted_shoreline(extracted_shorelines, roi_id)

        # save the ROI IDs that had extracted shoreline to observable variable roi_ids_with_extracted_shorelines
        ids_with_extracted_shorelines = self.get_roi_ids_with_extracted_shorelines(
            is_selected=False
        )
        self.roi_ids_with_extracted_shorelines.set(ids_with_extracted_shorelines)

        self.save_session(roi_ids, save_transects=False)

        # Get ROI ids that are selected on map and have had their shorelines extracted
        roi_ids = self.get_roi_ids_with_extracted_shorelines(is_selected=True)
        self.compute_transects(self.transects.gdf, self.get_settings(), roi_ids)
        # load extracted shorelines to map
        self.load_extracted_shorelines_to_map()

    def get_selected_rois(self, roi_ids: list) -> gpd.GeoDataFrame:
        """Returns a geodataframe of all rois whose ids are in given list
        roi_ids.

        Args:
            roi_ids (list[str]): ids of ROIs

        Returns:
            gpd.GeoDataFrame:  geodataframe of all rois selected by the roi_ids
        """
        selected_rois_gdf = self.rois.gdf[self.rois.gdf["id"].isin(roi_ids)]
        return selected_rois_gdf

    def get_cross_distance(
        self,
        roi_id: str,
        transects_in_roi_gdf: gpd.GeoDataFrame,
        settings: dict,
        output_epsg: int,
    ) -> Tuple[float, Optional[str]]:
        """
        Compute the cross shore distance of transects and extracted shorelines for a given ROI.

        Parameters:
        -----------
        roi_id : str
            The ID of the ROI to compute the cross shore distance for.
        transects_in_roi_gdf : gpd.GeoDataFrame
            All the transects in the ROI. Must contain the columns ["id", "geometry"]
        settings : dict
            A dictionary of settings to be used in the computation.
        output_epsg : int
            The EPSG code of the output projection.

        Returns:
        --------
        Tuple[float, Optional[str]]
            The computed cross shore distance, or 0 if there was an issue in the computation.
            The reason for failure, or '' if the computation was successful.
        """
        failure_reason = ""
        cross_distance = 0

        # Get extracted shorelines object for the currently selected ROI
        roi_extracted_shoreline = self.rois.get_extracted_shoreline(roi_id)

        transects_in_roi_gdf = transects_in_roi_gdf.loc[:, ["id", "geometry"]]

        if roi_extracted_shoreline is None:
            failure_reason = "No extracted shorelines were found"

        elif transects_in_roi_gdf.empty:
            failure_reason = "No transects intersect"

        else:
            extracted_shoreline_x_transect = transects_in_roi_gdf[
                transects_in_roi_gdf.intersects(roi_extracted_shoreline.gdf.unary_union)
            ]

            if extracted_shoreline_x_transect.empty:
                failure_reason = "No extracted shorelines intersected transects"
            else:
                # Convert transects_in_roi_gdf to output_crs from settings
                transects_in_roi_gdf = transects_in_roi_gdf.to_crs(output_epsg)

                # Compute cross shore distance of transects and extracted shorelines
                extracted_shorelines_dict = roi_extracted_shoreline.dictionary
                cross_distance = extracted_shoreline.compute_transects_from_roi(
                    extracted_shorelines_dict,
                    transects_in_roi_gdf,
                    settings,
                )
                if cross_distance == 0:
                    failure_reason = "Cross distance computation failed"

        return cross_distance, failure_reason

    def save_timeseries_csv(self, session_path: str, roi_id: str, rois: ROI) -> None:
        """Saves cross distances of transects and
        extracted shorelines in ROI to csv file within each ROI's directory.
        If no shorelines were extracted for an ROI then nothing is saved
        Args:
            roi_ids (list): list of roi ids
            rois (ROI): ROI instance containing keys:
                'extracted_shorelines': extracted shoreline from roi
                'cross_distance_transects': cross distance of transects and extracted shoreline from roi
        """
        roi_extracted_shorelines = rois.get_extracted_shoreline(roi_id)
        # if roi does not have extracted shoreline skip it
        if roi_extracted_shorelines is None:
            logger.info(f"No extracted shorelines for roi: {roi_id}")
            return
        # get extracted_shorelines from extracted shoreline object in rois
        extracted_shorelines = roi_extracted_shorelines.dictionary
        # if no shorelines were extracted then skip
        if extracted_shorelines == {}:
            logger.info(f"No extracted shorelines for roi: {roi_id}")
            return
        cross_distance_transects = rois.get_cross_shore_distances(roi_id)
        logger.info(f"ROI: {roi_id} extracted_shorelines : {extracted_shorelines}")
        # if no cross distance was 0 then skip
        if cross_distance_transects == 0:
            print(
                f"ROI: {roi_id} cross distance is 0 will not have time-series of shoreline change along transects "
            )
            logger.info(f"ROI: {roi_id} cross distance is 0")
            return
        # saves all transects in a single directory
        filepath = common.save_transect_intersections(
            session_path, extracted_shorelines, cross_distance_transects
        )
        print(
            f"ROI: {roi_id} Time-series of the shoreline change along the transects saved as:{filepath}"
        )

    def compute_transects(
        self, transects_gdf: gpd.GeoDataFrame, settings: dict, roi_ids: list[str]
    ) -> dict:
        """Returns a dict of cross distances for each roi's transects
        Args:
            selected_rois (dict): rois selected by the user. Must contain the following fields:
                {'features': [
                    'id': (str) roi_id
                    'geometry':{
                        'type':Polygon
                        'coordinates': list of coordinates that make up polygon
                    }
                ],...}
            extracted_shorelines (dict): dictionary with roi_id keys that identify roi associates with shorelines
            {
                roi_id:{
                    dates: [datetime.datetime,datetime.datetime], ...
                    shorelines: [array(),array()]    }
            }

            settings (dict): settings used for CoastSat. Must have the following fields:
               'output_epsg': int
                    output spatial reference system as EPSG code
                'along_dist': int
                    alongshore distance considered caluclate the intersection

        Returns:
            dict: cross_distances_rois with format:
            { roi_id :  dict
                time-series of cross-shore distance along each of the transects. Not tidally corrected. }
        """
        self.validate_transect_inputs(settings)
        # user selected output projection
        output_epsg = "epsg:" + str(settings["output_epsg"])
        # for each ROI save cross distances for each transect that intersects each extracted shoreline
        for roi_id in tqdm(roi_ids, desc="Computing Cross Distance Transects"):
            # get transects that intersect with ROI
            single_roi = common.extract_roi_by_id(self.rois.gdf, roi_id)
            # save cross distances by ROI id
            transects_in_roi_gdf = transects_gdf[
                transects_gdf.intersects(single_roi.unary_union)
            ]
            cross_distance, failure_reason = self.get_cross_distance(
                str(roi_id), transects_in_roi_gdf, settings, output_epsg
            )
            if cross_distance == 0:
                logger.warning(f"{failure_reason} for ROI {roi_id}")
                print(f"{failure_reason} for ROI {roi_id}")
            self.rois.add_cross_shore_distances(cross_distance, roi_id)

        self.save_session(roi_ids)

    def save_session(self, roi_ids: list[str], save_transects: bool = True):
        # Save extracted shoreline info to session directory
        session_name = self.get_session_name()
        for roi_id in roi_ids:
            ROI_directory = self.rois.roi_settings[roi_id]["sitename"]
            # create session directory
            session_path = common.get_session_path(session_name, ROI_directory)
            # save source data
            self.save_config(session_path)
            # save extracted shorelines
            extracted_shoreline = self.rois.get_extracted_shoreline(roi_id)
            logger.info(f"Extracted shorelines for ROI {roi_id}: {extracted_shoreline}")
            if extracted_shoreline is None:
                logger.info(f"No extracted shorelines for ROI: {roi_id}")
                continue
            # move extracted shoreline figures to session directory
            common.save_extracted_shoreline_figures(extracted_shoreline, session_path)
            common.save_extracted_shorelines(extracted_shoreline, session_path)

            # save transects to session folder
            if save_transects:
                # Saves the cross distances of the transects & extracted shorelines to csv file within each ROI's directory
                self.save_timeseries_csv(session_path, roi_id, self.rois)
                self.save_csv_per_transect_for_roi(session_path, roi_id, self.rois)

                save_path = os.path.join(session_path, "transects_cross_distances.json")
                cross_shore_distance = self.rois.get_cross_shore_distances(roi_id)
                common.to_file(cross_shore_distance, save_path)

                # save transect settings to file
                transect_settings = common.get_transect_settings(self.get_settings())
                transect_settings_path = os.path.join(
                    session_path, "transects_settings.json"
                )
                common.to_file(transect_settings, transect_settings_path)

        print(f"Saved to session at {session_name}")

    def save_csv_per_transect_for_roi(
        self, session_path: str, roi_id: list, rois: ROI
    ) -> None:
        """Saves cross distances of transects and
        extracted shorelines in ROI to csv file within each ROI's directory.
        If no shorelines were extracted for an ROI then nothing is saved
        Args:
            roi_ids (list): list of roi ids
            rois (ROI): ROI instance containing keys:
                'extracted_shorelines': extracted shoreline from roi
                'cross_distance_transects': cross distance of transects and extracted shoreline from roi
        """
        # get extracted shorelines for this roi id
        roi_extracted_shorelines = rois.get_extracted_shoreline(roi_id)
        # if roi does not have extracted shoreline skip it
        if roi_extracted_shorelines is None:
            return
        # get extracted_shorelines from extracted shoreline object in rois
        extracted_shorelines_dict = roi_extracted_shorelines.dictionary
        cross_distance_transects = rois.get_cross_shore_distances(roi_id)
        logger.info(f"ROI: {roi_id} extracted_shorelines : {extracted_shorelines_dict}")
        # if no cross distance was 0 then skip
        if cross_distance_transects == 0:
            return
        # if no shorelines were extracted then skip
        if extracted_shorelines_dict == {}:
            return
        # for each transect id in cross_distance_transects make a new csv file
        common.create_csv_per_transect(
            roi_id, session_path, cross_distance_transects, extracted_shorelines_dict
        )

    def save_csv_per_transect(self, roi_ids: list, rois: ROI) -> None:
        """Saves cross distances of transects and
        extracted shorelines in ROI to csv file within each ROI's directory.
        If no shorelines were extracted for an ROI then nothing is saved
        Args:
            roi_ids (list): list of roi ids
            rois (ROI): ROI instance containing keys:
                'extracted_shorelines': extracted shoreline from roi
                'roi_settings': must have keys 'filepath' and 'sitename'
                'cross_distance_transects': cross distance of transects and extracted shoreline from roi
        """
        # set of roi ids that have add their transects successfully computed
        rois_computed_transects = set()
        for roi_id in tqdm(roi_ids, desc="Saving csv for each transect for ROIs"):
            roi_extracted_shorelines = rois.get_extracted_shoreline(roi_id)
            # if roi does not have extracted shoreline skip it
            if roi_extracted_shorelines is None:
                logger.info(f"ROI: {roi_id} had no extracted shorelines ")
                continue

            # get extracted_shorelines from extracted shoreline object in rois
            extracted_shorelines_dict = roi_extracted_shorelines.dictionary
            cross_distance_transects = rois.get_cross_shore_distances(roi_id)
            logger.info(
                f"ROI: {roi_id} extracted_shorelines : {extracted_shorelines_dict}"
            )
            # if no cross distance was 0 then skip
            if cross_distance_transects == 0:
                logger.info(f"ROI: {roi_id} cross distance is 0")
                continue
            # if no shorelines were extracted then skip
            if extracted_shorelines_dict == {}:
                logger.info(f"ROI: {roi_id} had no extracted shorelines ")
                continue

            # for each transect id in cross_distance_transects make a new csv file
            for key in cross_distance_transects.keys():
                df = pd.DataFrame()
                out_dict = dict([])
                # copy shoreline intersects for each transect
                out_dict[key] = cross_distance_transects[key]
                logger.info(
                    f"out dict roi_ids columns : {[roi_id for _ in range(len(extracted_shorelines_dict['dates']))]}"
                )
                out_dict["roi_id"] = [
                    roi_id for _ in range(len(extracted_shorelines_dict["dates"]))
                ]
                out_dict["dates"] = extracted_shorelines_dict["dates"]
                out_dict["satname"] = extracted_shorelines_dict["satname"]
                logger.info(f"out_dict : {out_dict}")
                df = pd.DataFrame(out_dict)
                df.index = df["dates"]
                df.pop("dates")

                # Save extracted shoreline info to session directory
                session_name = self.get_session_name()
                session_path = os.path.join(os.getcwd(), "sessions", session_name)
                ROI_directory = rois.roi_settings[roi_id]["sitename"]
                session_path = common.create_directory(session_path, ROI_directory)
                logger.info(f"session_path: {session_path}")
                # save source data
                self.save_config(session_path)
                # save to csv file session path

                fn = os.path.join(session_path, "%s_timeseries_raw.csv" % key)
                logger.info(f"Save time series to {fn}")
                if os.path.exists(fn):
                    logger.info(f"Overwriting:{fn}")
                    os.remove(fn)
                df.to_csv(fn, sep=",")
                logger.info(
                    f"ROI: {roi_id} time-series of shoreline change along transects"
                )
                logger.info(
                    f"Time-series of the shoreline change along the transects saved as:{fn}"
                )
                rois_computed_transects.add(roi_id)
        print(f"Computed transects for the following ROIs: {rois_computed_transects}")

    def save_cross_distance_to_file(self, roi_ids: list, rois: ROI) -> None:
        """Saves cross distances of transects and
        extracted shorelines in ROI to csv file within each ROI's directory.
        If no shorelines were extracted for an ROI then nothing is saved
        Args:
            roi_ids (list): list of roi ids
            rois (ROI): ROI instance containing keys:
                'extracted_shorelines': extracted shoreline from roi
                'roi_settings': must have keys 'filepath' and 'sitename'
                'cross_distance_transects': cross distance of transects and extracted shoreline from roi
        """
        for roi_id in tqdm(roi_ids, desc="Saving ROI cross distance transects"):
            roi_extracted_shorelines = rois.get_extracted_shoreline(roi_id)
            # if roi does not have extracted shoreline skip it
            if roi_extracted_shorelines is None:
                print(
                    f"ROI: {roi_id} had no extracted shorelines and therfore has no time-series of shoreline change along transects"
                )
                logger.info(
                    f"ROI: {roi_id} had no extracted shorelines.ROI: {roi_id} will not have time-series of shoreline change along transects."
                )
                continue
            # get extracted_shorelines from extracted shoreline object in rois
            extracted_shorelines = roi_extracted_shorelines.dictionary
            # if no shorelines were extracted then skip
            if extracted_shorelines == {}:
                print(
                    f"ROI: {roi_id} had no extracted shorelines and will not have time-series of shoreline change along transects "
                )
                logger.info(f"ROI: {roi_id} had no extracted shorelines ")
                continue

            cross_distance_transects = rois.get_cross_shore_distances(roi_id)
            logger.info(f"ROI: {roi_id} extracted_shorelines : {extracted_shorelines}")
            # if no cross distance was 0 then skip
            if cross_distance_transects == 0:
                print(
                    f"ROI: {roi_id} cross distance is 0 will not have time-series of shoreline change along transects "
                )
                logger.info(f"ROI: {roi_id} cross distance is 0")
                continue

            cross_distance_df = common.get_cross_distance_df(
                extracted_shorelines, cross_distance_transects
            )
            logger.info(f"ROI: {roi_id} cross_distance_df : {cross_distance_df}")

            # Save extracted shoreline info to session directory
            session_name = self.get_session_name()
            ROI_directory = rois.roi_settings[roi_id]["sitename"]
            session_path = os.path.join(os.getcwd(), "sessions", session_name)
            session_path = common.create_directory(session_path, ROI_directory)
            logger.info(f"session_path: {session_path}")
            # save source data
            self.save_config(session_path)

            filepath = os.path.join(session_path, "transect_time_series.csv")
            if os.path.exists(filepath):
                print(f"Overwriting:{filepath}")
                os.remove(filepath)
            cross_distance_df.to_csv(filepath, sep=",")
            print(f"ROI: {roi_id} time-series of shoreline change along transects")
            print(
                f"Time-series of the shoreline change along the transects saved as:{filepath}"
            )

    def remove_all(self):
        """Remove the bbox, shoreline, all rois from the map"""
        self.remove_bbox()
        self.remove_shoreline()
        self.remove_transects()
        self.remove_all_rois()
        self.remove_layer_by_name("geodataframe")
        self.remove_extracted_shorelines()

    def remove_extracted_shorelines(self):
        """Removes extracted shorelines from the map and removes extracted shorelines from ROIs"""
        # empty extracted shorelines dictionary
        if self.rois is not None:
            self.rois.remove_extracted_shorelines(remove_all=True)
        # remove extracted shoreline vectors from the map
        self.remove_extracted_shoreline_layers()

    def remove_extracted_shoreline_layers(self):
        if self.extracted_shoreline_layer.get() is not None:
            self.map.remove_layer(self.extracted_shoreline_layer.get())
            self.extracted_shoreline_layer.set(None)

    def remove_bbox(self):
        """Remove all the bounding boxes from the map"""
        if self.bbox is not None:
            del self.bbox
            self.bbox = None
        self.draw_control.clear()
        existing_layer = self.map.find_layer(Bounding_Box.LAYER_NAME)
        if existing_layer is not None:
            self.map.remove_layer(existing_layer)
        self.bbox = None

    def remove_layer_by_name(self, layer_name: str):
        existing_layer = self.map.find_layer(layer_name)
        if existing_layer is not None:
            self.map.remove(existing_layer)
        logger.info(f"Removed layer {layer_name}")

    def remove_shoreline(self):
        del self.shoreline
        self.remove_layer_by_name(Shoreline.LAYER_NAME)
        self.shoreline = None

    def remove_transects(self):
        del self.transects
        self.transects = None
        self.remove_layer_by_name(Transects.LAYER_NAME)

    def replace_layer_by_name(
        self, layer_name: str, new_layer: GeoJSON, on_hover=None, on_click=None
    ) -> None:
        """Replaces layer with layer_name with new_layer on map. Adds on_hover and on_click callable functions
        as handlers for hover and click events on new_layer
        Args:
            layer_name (str): name of layer to replace
            new_layer (GeoJSON): ipyleaflet GeoJSON layer to add to map
            on_hover (callable, optional): Callback function that will be called on hover event on a feature, this function
            should take the event and the feature as inputs. Defaults to None.
            on_click (callable, optional): Callback function that will be called on click event on a feature, this function
            should take the event and the feature as inputs. Defaults to None.
        """
        if new_layer is None:
            return
        logger.info(
            f"layer_name {layer_name} \non_hover {on_hover}\n on_click {on_click}"
        )
        self.remove_layer_by_name(layer_name)
        exception_handler.check_empty_layer(new_layer, layer_name)
        # when feature is hovered over on_hover function is called
        if on_hover is not None:
            new_layer.on_hover(on_hover)
        if on_click is not None:
            # when feature is clicked on on_click function is called
            new_layer.on_click(on_click)
        self.map.add_layer(new_layer)
        logger.info(f"Add layer to map: {layer_name}")

    def remove_all_rois(self) -> None:
        """Removes all the unselected rois from the map"""
        logger.info("Removing all ROIs from map")
        # Remove the selected and unselected rois
        self.remove_layer_by_name(ROI.SELECTED_LAYER_NAME)
        self.remove_layer_by_name(ROI.LAYER_NAME)
        # clear all the ids from the selected set
        self.selected_set = set()
        del self.rois
        self.rois = None

    def create_DrawControl(self, draw_control: "ipyleaflet.leaflet.DrawControl"):
        """Modifies given draw control so that only rectangles can be drawn

        Args:
            draw_control (ipyleaflet.leaflet.DrawControl): draw control to modify

        Returns:
            ipyleaflet.leaflet.DrawControl: modified draw control with only ability to draw rectangles
        """
        draw_control.polyline = {}
        draw_control.circlemarker = {}
        draw_control.polygon = {}
        draw_control.rectangle = {
            "shapeOptions": {
                "fillColor": "green",
                "color": "green",
                "fillOpacity": 0.1,
                "Opacity": 0.1,
            },
            "drawError": {"color": "#dd253b", "message": "Ops!"},
            "allowIntersection": False,
            "transform": True,
        }
        return draw_control

    def handle_draw(
        self, target: "ipyleaflet.leaflet.DrawControl", action: str, geo_json: dict
    ):
        """Adds or removes the bounding box  when drawn/deleted from map
        Args:
            target (ipyleaflet.leaflet.DrawControl): draw control used
            action (str): name of the most recent action ex. 'created', 'deleted'
            geo_json (dict): geojson dictionary
        """
        if (
            self.draw_control.last_action == "created"
            and self.draw_control.last_draw["geometry"]["type"] == "Polygon"
        ):
            # validate the bbox size
            geometry = self.draw_control.last_draw["geometry"]
            bbox_area = common.get_area(geometry)
            try:
                Bounding_Box.check_bbox_size(bbox_area)
            except exceptions.BboxTooLargeError as bbox_too_big:
                self.remove_bbox()
                exception_handler.handle_bbox_error(bbox_too_big, self.warning_box)
            except exceptions.BboxTooSmallError as bbox_too_small:
                self.remove_bbox()
                exception_handler.handle_bbox_error(bbox_too_small, self.warning_box)
            else:
                # if no exceptions occur create new bbox, remove old bbox, and load new bbox
                logger.info(f"Made it with bbox area: {bbox_area}")
                self.load_feature_on_map("bbox")

        if self.draw_control.last_action == "deleted":
            self.remove_bbox()

    def get_selected_roi_ids(self) -> list:
        """Gets the ids of the selected rois

        Returns:
            list: list of ids of selected rois
        """
        return list(self.selected_set)

    def load_extracted_shoreline_by_id(self, selected_id: str, row_number: int = 0):
        """
        Loads extracted shorelines onto a map for a single region of interest specified by its ID.

        Args:
            selected_id (str): The ID of the region of interest to plot extracted shorelines for.
            row_number (int, optional): The row number of the region of interest to plot. Defaults to 0.
        """
        # remove any existing extracted shorelines
        self.remove_extracted_shoreline_layers()
        # get the extracted shorelines for the selected roi

        if self.rois is not None:
            extracted_shorelines = self.rois.get_extracted_shoreline(selected_id)
            logger.info(
                f"ROI ID { selected_id} extracted shorelines {extracted_shorelines}"
            )
            # if extracted shorelines exist, load them onto map
            if extracted_shorelines is not None:
                logger.info(f"Extracted shorelines found for ROI {selected_id}")
                self.load_extracted_shorelines_on_map(extracted_shorelines, row_number)
            # if extracted shorelines do not exist, set the status to no extracted shorelines found
            if extracted_shorelines is None:
                logger.info(f"No extracted shorelines found for ROI {selected_id}")

    def load_extracted_shorelines_to_map(self, row_number: int = 0) -> None:
        """Loads stylized extracted shorelines onto the map for a single selected region of interest (ROI).

        Args:
            row_number (int, optional): The row number of the ROI to load the extracted shorelines for. Defaults to 0.

        Raises:
            Exception: If no ROIs are found or the selected ROI has no extracted shorelines.

        Returns:
            None: This function does not return anything, but rather loads the extracted shorelines onto the map.
        """

        logger.info(f"row_number: {row_number}")

        # Remove any existing extracted shorelines
        self.remove_extracted_shoreline_layers()

        # Check if any ROIs are loaded
        if self.rois is None:
            logger.warning("No ROIs found. Please load ROIs.")
            raise Exception("No ROIs found. Please load ROIs.")

        # Get the extracted shorelines for all ROIs
        ids_with_extracted_shorelines = self.rois.get_ids_with_extracted_shorelines()
        self.roi_ids_with_extracted_shorelines.set(ids_with_extracted_shorelines)

        # Get the available ROI IDs
        available_ids = self.get_all_roi_ids()

        if not available_ids:
            logger.warning("No ROIs found. Please load ROIs.")
            raise Exception("No ROIs found. Please load ROIs.")

        # Find ROI IDs with extracted shorelines
        roi_ids_with_extracted_shorelines = set(available_ids).intersection(
            ids_with_extracted_shorelines
        )

        if not roi_ids_with_extracted_shorelines:
            logger.warning("No ROIs found with extracted shorelines.")
            return

        # Load extracted shorelines for the first ROI ID with extracted shorelines
        for selected_id in roi_ids_with_extracted_shorelines:
            extracted_shorelines = self.rois.get_extracted_shoreline(selected_id)
            logger.info(
                f"ROI ID {selected_id} extracted shorelines {extracted_shorelines}"
            )

            if extracted_shorelines is not None:
                logger.info(f"Extracted shorelines found for ROI {selected_id}")
                self.load_extracted_shorelines_on_map(extracted_shorelines, row_number)
                break

    def load_extracted_shorelines_on_map(
        self,
        extracted_shoreline: extracted_shoreline.Extracted_Shoreline,
        row_number: int = 0,
    ):
        """
        Loads a stylized extracted shoreline layer onto a map for a single region of interest.

        Args:
            extracted_shoreline (Extracted_Shoreline): An instance of the Extracted_Shoreline class containing the extracted shoreline data.
            row_number (int, optional): The row number of the region of interest to plot. Defaults to 0.
        """
        # Loads stylized extracted shorelines onto map for single roi
        logger.info(f"row_number: {row_number}")
        new_layer = extracted_shoreline.get_styled_layer(row_number)
        layer_name = extracted_shoreline.get_layer_name()
        logger.info(
            f"Extracted shoreline layer: {new_layer}\n"
            f"Layer name: {layer_name}\n"
            f"Extracted shoreline layers: {new_layer}\n"
        )
        # new_layer.on_hover(self.update_extracted_shoreline_html)
        self.map.add_layer(new_layer)
        # update the extracted shoreline layer and number of shorelines available
        self.extracted_shoreline_layer.set(new_layer)
        self.number_extracted_shorelines.set(len(extracted_shoreline.gdf) - 1)

    def load_feature_on_map(
        self, feature_name: str, file: str = "", gdf: gpd.GeoDataFrame = None, **kwargs
    ) -> None:
        """Loads feature of type feature_name onto the map either from a file or from a geodataframe given by gdf

        if feature_name given is not one "shoreline","transects","bbox", or "rois" throw exception

        Args:
            feature_name (str): name of feature must be one of the following
            "shoreline","transects","bbox","rois"
            file (str, optional): geojson file containing feature. Defaults to "".
            gdf (gpd.GeoDataFrame, optional): geodataframe containing feature geometry. Defaults to None.
        """
        # if file is passed read gdf from file
        if file:
            gdf = common.read_gpd_file(file)
        # convert gdf to the proper format
        if gdf is not None:
            if gdf.empty:
                logger.info("No {feature_name} was loaded on map")
                return
            # if a z axis exists remove it
            gdf = common.remove_z_axis(gdf)
            logger.info(f"gdf after z-axis removed: {gdf}")

        # create the feature
        new_feature = self.factory.make_feature(self, feature_name, gdf, **kwargs)
        if new_feature is None:
            return
        logger.info(f"new_feature: {new_feature}")
        logger.info(f"gdf: {gdf}")
        # load the features onto the map
        self.add_feature_on_map(new_feature, feature_name, **kwargs)

    def add_feature_on_map(
        self,
        new_feature,
        feature_name: str,
        layer_name: str = "",
        zoom_to_bounds=False,
        **kwargs,
    ) -> None:
        """
        Adds a feature to the map as well as the feature's on_click and on_hover handlers.

        Args:
        - new_feature: The feature to be added to the map.
        - feature_name (str): A string representing the name of the feature.
        - layer_name (str): A string representing the name of the layer to which the feature should be added. Default value is an empty string.

        Returns:
        - None
        """
        logger.info(
            f"feature_name: {feature_name.lower()}\n layer_name: {layer_name}\n new_feature: {new_feature}"
        )
        # get on hover and on click handlers for feature
        on_hover = self.get_on_hover_handler(feature_name)
        on_click = self.get_on_click_handler(feature_name)
        # if layer name is not given use the layer name of the feature
        if not layer_name and hasattr(new_feature, "LAYER_NAME"):
            layer_name = new_feature.LAYER_NAME
        # if the feature has a geodataframe zoom the map to the bounds of the feature
        if zoom_to_bounds and hasattr(new_feature, "gdf"):
            bounds = new_feature.gdf.total_bounds
            self.map.zoom_to_bounds(bounds)
        self.load_on_map(new_feature, layer_name, on_hover, on_click)

    def get_on_click_handler(self, feature_name: str) -> callable:
        """
        Returns a callable function that handles mouse click events for a given feature.

        Args:
        - feature_name (str): A string representing the name of the feature for which the click handler needs to be returned.

        Returns:
        - callable: A callable function that handles mouse click events for a given feature.
        """
        on_click = None
        if "roi" in feature_name.lower():
            on_click = self.geojson_onclick_handler
        return on_click

    def get_on_hover_handler(self, feature_name: str) -> callable:
        """
        Returns a callable function that handles mouse hover events for a given feature.

        Args:
        - feature_name (str): A string representing the name of the feature for which the hover handler needs to be returned.

        Returns:
        - callable: A callable function that handles mouse hover events for a given feature.
        """
        on_hover = None
        feature_name_lower = feature_name.lower()
        if "shoreline" in feature_name_lower:
            on_hover = self.update_shoreline_html
        elif "transect" in feature_name_lower:
            on_hover = self.update_transects_html
        elif "roi" in feature_name_lower:
            on_hover = self.update_roi_html
        return on_hover

    def load_on_map(
        self, feature, layer_name: str, on_hover=None, on_click=None
    ) -> None:
        """Loads feature on map as a new layer

        Replaces current feature layer on map with given feature

        Raises:
            Exception: raised if feature layer is empty
        """
        # style and add the feature to the map
        new_layer = self.create_layer(feature, layer_name)
        # Replace old feature layer with new feature layer
        self.replace_layer_by_name(
            layer_name, new_layer, on_hover=on_hover, on_click=on_click
        )

    def create_layer(self, feature, layer_name: str):
        if feature.gdf.empty:
            logger.warning("Cannot add an empty geodataframe layer to the map.")
            print("Cannot add an empty geodataframe layer to the map.")
            return None
        layer_geojson = json.loads(feature.gdf.to_json())
        # convert layer to GeoJson and style it accordingly
        styled_layer = feature.style_layer(layer_geojson, layer_name)
        return styled_layer

    def geojson_onclick_handler(
        self, event: str = None, id: "NoneType" = None, properties: dict = None, **args
    ):
        """On click handler for when unselected geojson is clicked.

        Adds geojson's id to selected_set. Replaces current selected layer with a new one that includes
        recently clicked geojson.

        Args:
            event (str, optional): event fired ('click'). Defaults to None.
            id (NoneType, optional):  Defaults to None.
            properties (dict, optional): geojson dict for clicked geojson. Defaults to None.
        """
        if properties is None:
            return
        logger.info(f"properties : {properties}")
        logger.info(f"ROI_id : {properties['id']}")
        logger.info(type(event))
        logger.info(type(id))
        logger.info(id)
        # Add id of clicked ROI to selected_set
        ROI_id = str(properties["id"])
        self.selected_set.add(ROI_id)
        logger.info(f"Added ID to selected set: {self.selected_set}")
        # remove old selected layer
        self.remove_layer_by_name(ROI.SELECTED_LAYER_NAME)

        selected_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(self.selected_set),
            name=ROI.SELECTED_LAYER_NAME,
            hover_style={"fillColor": "blue", "fillOpacity": 0.1, "color": "aqua"},
        )
        logger.info(f"selected_layer: {selected_layer}")
        self.replace_layer_by_name(
            ROI.SELECTED_LAYER_NAME,
            selected_layer,
            on_click=self.selected_onclick_handler,
            on_hover=self.update_roi_html,
        )

    def selected_onclick_handler(
        self, event: str = None, id: "NoneType" = None, properties: dict = None, **args
    ):
        """On click handler for selected geojson layer.

        Removes clicked layer's cid from the selected_set and replaces the select layer with a new one with
        the clicked layer removed from select_layer.

        Args:
            event (str, optional): event fired ('click'). Defaults to None.
            id (NoneType, optional):  Defaults to None.
            properties (dict, optional): geojson dict for clicked selected geojson. Defaults to None.
        """
        if properties is None:
            return
        # Remove the current layers cid from selected set
        logger.info(f"selected_onclick_handler: properties : {properties}")
        logger.info(f"selected_onclick_handler: ROI_id to remove : {properties['id']}")
        cid = properties["id"]
        self.selected_set.remove(cid)
        logger.info(f"selected set after ID removal: {self.selected_set}")
        self.remove_layer_by_name(ROI.SELECTED_LAYER_NAME)
        # Recreate selected layers without layer that was removed

        selected_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(self.selected_set),
            name=ROI.SELECTED_LAYER_NAME,
            hover_style={"fillColor": "blue", "fillOpacity": 0.1, "color": "aqua"},
        )
        self.replace_layer_by_name(
            ROI.SELECTED_LAYER_NAME,
            selected_layer,
            on_click=self.selected_onclick_handler,
            on_hover=self.update_roi_html,
        )

    def save_feature_to_file(
        self,
        feature: Union[Bounding_Box, Shoreline, Transects, ROI],
        feature_type: str = "",
    ):
        """
        Save a geographical feature (Bounding_Box, Shoreline, Transects, ROI) as a GeoJSON file.

        Args:
            feature (Union[Bounding_Box, Shoreline, Transects, ROI]): The geographical feature to save.
            feature_type (str, optional): The type of the feature, e.g. Bounding_Box, Shoreline, Transects, or ROI.
                                        Default value is an empty string.
        """
        exception_handler.can_feature_save_to_file(feature, feature_type)
        if isinstance(feature, ROI):
            # raise exception if no rois were selected
            exception_handler.check_selected_set(self.selected_set)
            feature.gdf[feature.gdf["id"].isin(self.selected_set)].to_file(
                feature.filename, driver="GeoJSON"
            )
        else:
            logger.info(f"Saving feature type( {feature}) to file")
            if hasattr(feature, "gdf"):
                feature.gdf.to_file(feature.filename, driver="GeoJSON")
                print(f"Save {feature.LAYER_NAME} to {feature.filename}")
                logger.info(f"Save {feature.LAYER_NAME} to {feature.filename}")
            else:
                print(f"Empty {feature.LAYER_NAME} cannot be saved to file")

    def convert_selected_set_to_geojson(self, selected_set: set) -> dict:
        """Returns a geojson dict containing a FeatureCollection for all the geojson objects in the
        selected_set
        Args:
            selected_set (set): ids of selected geojson
        Returns:
           dict: geojson dict containing FeatureCollection for all geojson objects in selected_set
        """
        # create a new geojson dictionary to hold selected ROIs
        selected_rois = {"type": "FeatureCollection", "features": []}
        roi_layer = self.map.find_layer(ROI.LAYER_NAME)
        # if ROI layer does not exist throw an error
        if roi_layer is not None:
            exception_handler.check_empty_layer(roi_layer, "ROI")
        # Copy only selected ROIs with id in selected_set
        selected_rois["features"] = [
            feature
            for feature in roi_layer.data["features"]
            if feature["properties"]["id"] in selected_set
        ]
        # Each selected ROI will be blue and unselected rois will appear black
        for feature in selected_rois["features"]:
            feature["properties"]["style"] = {
                "color": "blue",
                "weight": 2,
                "fillColor": "blue",
                "fillOpacity": 0.1,
            }
        return selected_rois
