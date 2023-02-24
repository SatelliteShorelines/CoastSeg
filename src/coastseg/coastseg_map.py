import os
import json
import logging
import copy
import glob
from typing import Optional, Union
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

from coastsat import (
    SDS_download,
    SDS_transects,
    SDS_preprocess,
)
import pandas as pd
import geopandas as gpd
from ipyleaflet import DrawControl, LayersControl, WidgetControl, GeoJSON
from leafmap import Map
from ipywidgets import Layout, HTML
from tqdm.auto import tqdm
from ipywidgets import HBox


logger = logging.getLogger(__name__)


class CoastSeg_Map:
    def __init__(self):
        # settings:  used to select data to download and preprocess settings
        self.settings = {}
        # create default values for settings
        self.set_settings()
        self.session_name = ""

        # factory is used to create map objects
        self.factory = factory.Factory()

        # Selections
        self.selected_set = set()  # ids of the selected rois
        self.extracted_shoreline_layers = (
            []
        )  # names of extracted shorelines vectors on the map
        self.rois = None

        # map objects
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

    def load_session(self, session_path: str) -> None:
        session_name = os.path.basename(os.path.abspath(session_path))
        self.set_session_name(session_name)
        for count, dir_name in enumerate(os.listdir(session_path)):
            dir_path = os.path.join(session_path, dir_name)
            # only load in settings from first ROI since all settings SHOULD be same for a session
            if os.path.isdir(dir_path):
                if count == 0:
                    for file_name in os.listdir(dir_path):
                        file_path = os.path.join(dir_path, file_name)
                        if os.path.isfile(file_path):
                            if file_name == "config_gdf.geojson":
                                self.load_configs(file_path)
                            elif file_name == "transects_settings.json":
                                self.load_settings(file_path)
                                print(f"Loaded transect settings")
                            elif file_name == "shoreline_settings.json":
                                self.load_settings(file_path)
                                print(f"Loaded shoreline settings")

                for file_name in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file_name)
                    if os.path.isfile(file_path):
                        if file_name == "transects_settings.json":
                            self.load_settings(file_path)
                            print(f"Loaded transect settings")
                        elif file_name == "shoreline_settings.json":
                            self.load_settings(file_path)
                            print(f"Loaded shoreline settings")

                # for every directory load extracted shorelines and transect files
                extracted_shorelines = (
                    extracted_shoreline.load_extracted_shoreline_from_files(dir_path)
                )
                # get roi id from directory name
                roi_id = os.path.basename(dir_path).split("_")[1]
                self.rois.add_extracted_shoreline(extracted_shorelines, roi_id)
                # load cross distances from file
                cross_distances = common.load_cross_distances_from_file(dir_path)
                self.rois.add_cross_shore_distances(cross_distances, roi_id)

    def load_settings(self, filepath: str = "", new_settings: dict = {}):
        """
        Load settings from a JSON file. Only loads settings for transects and shorelines.
        If no JSON file is provided then load then load from new_settings.

        Args:
            filepath: The path to the JSON file containing the new settings.
            new_settings: A dictionary of new settings to apply.

        Returns:
            None.
        """
        if os.path.exists(filepath):
            new_settings = common.read_json_file(filepath)

        # load in transect settings if found
        transect_keys = [
            "max_std",
            "min_points",
            "along_dist",
            "max_range",
            "min_chainage",
            "multiple_inter",
            "prc_multiple",
        ]
        transect_settings = common.filter_dict_by_keys(new_settings, keys=transect_keys)
        if transect_settings != {}:
            logger.info(f"Loading transect_settings: {transect_settings}")
            self.set_settings(**transect_settings)
            logger.info(f"Loaded transect_settings from {filepath}")

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
        shoreline_settings = common.filter_dict_by_keys(
            new_settings, keys=shoreline_keys
        )
        shoreline_settings["save_figure"] = True
        shoreline_settings["check_detection"] = False
        shoreline_settings["adjust_detection"] = False

        if shoreline_settings != {}:
            logger.info(f"Loading shoreline_settings: {shoreline_settings}")
            self.set_settings(**shoreline_settings)
            logger.info(f"Loaded shoreline_settings from {filepath}")

    def load_configs(self, filepath: str) -> None:
        """Loads features from geojson config file onto map and loads
        config.json file into settings. The config.json should be located into
        the same directory as the config.geojson file
        Args:
            filepath (str): full path to config.geojson file
        """
        # load geodataframe from config and load features onto map
        self.load_gdf_config(filepath)
        # path to directory to search for config_gdf.json file
        search_path = os.path.dirname(os.path.realpath(filepath))
        # create path to config.json file in search_path directory
        config_file = common.find_config_json(search_path)
        config_path = os.path.join(search_path, config_file)
        logger.info(f"Loaded json config file from {config_path}")
        # load settings from config.json file
        # ensure coastseg\data location exists
        data_path = common.create_directory(os.getcwd(), "data")
        self.load_json_config(config_path, data_path)

    def load_gdf_config(self, filepath: str) -> None:
        """Load features from geodataframe located in geojson file
        located at filepath onto map.

        features in config file should contain a column named "type"
        which contains the name of one of the following possible feature types
        "roi","shoreline","transect","bbox".
        Args:
            filepath (str): full path to config.geojson
        """
        logger.info(f"Loading {filepath}")
        gdf = common.read_gpd_file(filepath)
        # Extract ROIs from gdf and create new dataframe
        roi_gdf = gdf[gdf["type"] == "roi"].copy()
        exception_handler.check_if_gdf_empty(
            roi_gdf,
            "ROIs from config",
            "No ROIs were present in the config file: {filepath}",
        )
        # drop all columns except id and geometry
        columns_to_drop = list(set(roi_gdf.columns) - set(["id", "geometry"]))
        logger.info(f"Dropping columns from ROI: {columns_to_drop}")
        roi_gdf.drop(columns_to_drop, axis=1, inplace=True)
        logger.info(f"roi_gdf: {roi_gdf}")
        # Extract the shoreline from the gdf
        shoreline_gdf = gdf[gdf["type"] == "shoreline"].copy()
        # drop columns id and type
        if "slope" in shoreline_gdf.columns:
            columns_to_drop = ["id", "type", "slope"]
        else:
            columns_to_drop = ["id", "type"]
        logger.info(f"Dropping columns from shoreline: {columns_to_drop}")
        shoreline_gdf.drop(columns_to_drop, axis=1, inplace=True)
        logger.info(f"shoreline_gdf: {shoreline_gdf}")
        # Extract the transects from the gdf
        transect_gdf = gdf[gdf["type"] == "transect"].copy()
        columns_to_drop = list(
            set(transect_gdf.columns) - set(["id", "slope", "geometry"])
        )
        logger.info(f"Dropping columns from transects: {columns_to_drop}")
        transect_gdf.drop(columns_to_drop, axis=1, inplace=True)
        logger.info(f"transect_gdf: {transect_gdf}")
        # Extract bbox from gdf
        bbox_gdf = gdf[gdf["type"] == "bbox"].copy()
        columns_to_drop = list(set(bbox_gdf.columns) - set(["geometry"]))
        logger.info(f"Dropping columns from bbox: {columns_to_drop}")
        bbox_gdf.drop(columns_to_drop, axis=1, inplace=True)
        logger.info(f"bbox_gdf: {bbox_gdf}")
        # delete the original gdf read in from config geojson file
        del gdf
        if bbox_gdf.empty:
            self.bbox = None
            logger.info("No Bounding Box was loaded on map")
        else:
            self.load_feature_on_map("bbox", gdf=bbox_gdf)
        # make sure there are rois in the config file
        exception_handler.check_if_gdf_empty(
            roi_gdf, "ROIs", "Cannot load empty ROIs onto map"
        )
        # Create ROI object from roi_gdf
        self.rois = ROI(rois_gdf=roi_gdf)
        self.load_feature_on_map("rois", gdf=roi_gdf)
        # Create Shoreline object from shoreline_gdf
        if shoreline_gdf.empty:
            self.shoreline = None
            logger.info("No shoreline was loaded on map")
            print("No shoreline was loaded on map")
        else:
            self.load_feature_on_map("shoreline", gdf=shoreline_gdf)

        # Create Transect object from transect_gdf
        if transect_gdf.empty:
            self.transects = None
            logger.info("No transects were loaded on map")
            print("No transects were loaded on map")
        else:
            self.load_feature_on_map("transects", gdf=transect_gdf)

    def download_imagery(self) -> None:
        """downloads selected rois as jpgs

        Creates a directory for each ROI which contains all the downloaded imagery and
        the metadata files.

        Raises:
            Exception: raised if settings is missing
            Exception: raised if 'dates','sat_list', and 'landsat_collection' are not in settings
            Exception: raised if no ROIs have been selected
        """
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

        logger.info(f"selected_layer: {selected_layer}")

        # Get the file path where the downloaded imagery will be saved
        file_path = os.path.abspath(os.path.join(os.getcwd(), "data"))
        date_str = common.generate_datestring()

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
        print("Download in process")
        # make a deep copy so settings doesn't get modified by the temp copy
        tmp_settings = copy.deepcopy(settings)

        # for each ROI use the ROI settings to download imagery and save to jpg
        for inputs_for_roi in tqdm(inputs_list, desc="Downloading ROIs"):
            metadata = SDS_download.retrieve_images(inputs_for_roi)
            tmp_settings["inputs"] = inputs_for_roi
            logger.info(f"inputs: {inputs_for_roi}")
            logger.info(f"Saving to jpg. Metadata: {metadata}")
            SDS_preprocess.save_jpg(metadata, tmp_settings)
        # tmp settings is no longer needed
        del tmp_settings
        # 3.save settings used to download rois and the objects on map to config files
        self.save_config()
        logger.info("Done downloading")

    def load_json_config(self, filepath: str, data_path: str) -> None:
        """
        Loads a .json configuration file specified by the user.
        It replaces the coastseg_map.settings with the settings from the config file,
        and replaces the roi_settings for each ROI with the contents of the json_data.
        Finally, it saves the input dictionaries for all ROIs.
        Parameters:
        self (object): CoastsegMap instance
        filepath (str): The filepath to the json config file
        datapath(str): full path to the coastseg data directory
        Returns:
            None
        """
        exception_handler.check_if_None(self.rois)
        json_data = common.read_json_file(filepath)
        # replace coastseg_map.settings with settings from config file
        self.set_settings(**json_data["settings"])
        logger.info(f"Loaded settings from file: {self.get_settings()}")

        # replace roi_settings for each ROI with contents of json_data
        roi_settings = {}
        # flag raised if a single directory is missing
        missing_directories = []
        for roi_id in json_data["roi_ids"]:
            # if sitename is empty means user has not downloaded ROI data
            if json_data[roi_id]["sitename"] != "":
                sitename = json_data[roi_id]["sitename"]
                roi_path = os.path.join(data_path, sitename)
                json_data[roi_id]["filepath"] = data_path

                if not os.path.exists(roi_path):
                    missing_directories.append(sitename)
            # copy setting from json file to roi
            roi_settings[str(roi_id)] = json_data[roi_id]

        # if any directories are missing tell the user list of missing directories
        exception_handler.check_if_dirs_missing(missing_directories)
        # Save input dictionaries for all ROIs
        self.rois.roi_settings = roi_settings
        logger.info(f"roi_settings: {roi_settings}")

    def save_config(self, filepath: str = None) -> None:
        """saves the configuration settings of the map into two files
            config.json and config.geojson
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
        # settings must contain keys in subset
        subset = set(["dates", "sat_list", "landsat_collection"])
        superset = set(list(settings.keys()))
        exception_handler.check_if_subset(subset, superset, "settings")
        exception_handler.config_check_if_none(self.rois, "ROIs")
        # selected_layer must contain selected ROI
        selected_layer = self.map.find_layer(ROI.SELECTED_LAYER_NAME)
        exception_handler.check_empty_roi_layer(selected_layer)
        logger.info(f"self.rois.roi_settings: {self.rois.roi_settings}")
        if self.rois.roi_settings == {}:
            if filepath is None:
                filepath = os.path.abspath(os.getcwd())
            roi_settings = common.create_roi_settings(
                settings, selected_layer.data, filepath
            )
            # Save download settings dictionary to instance of ROI
            self.rois.set_roi_settings(roi_settings)
        # create dictionary to be saved to config.json
        roi_ids = list(self.selected_set)
        selected_roi_settings = {}
        for roi_id in roi_ids:
            selected_roi_settings[roi_id] = self.rois.roi_settings[roi_id]

        config_json = common.create_json_config(selected_roi_settings, settings)
        shorelines_gdf = None
        transects_gdf = None
        bbox_gdf = None
        if self.shoreline is not None:
            shorelines_gdf = self.shoreline.gdf
        if self.transects is not None:
            transects_gdf = self.transects.gdf
        if self.bbox is not None:
            bbox_gdf = self.bbox.gdf
        # save all selected rois, shorelines, transects and bbox to config geodataframe
        selected_rois = self.get_selected_rois(roi_ids)
        logger.info(f"selected_rois: {selected_rois} ")
        config_gdf = common.create_config_gdf(
            selected_rois, shorelines_gdf, transects_gdf, bbox_gdf
        )
        logger.info(f"config_gdf: {config_gdf} ")
        is_downloaded = common.were_rois_downloaded(self.rois.roi_settings, roi_ids)
        if filepath is not None:
            # save to config.json
            common.config_to_file(config_json, filepath)
            # save to config_gdf.geojson
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
        # Check if any of the keys are missing
        # if any keys are missing set the default value
        default_settings = {
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
            # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
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
        if kwargs:
            self.settings.update({key: value for key, value in kwargs.items()})
        self.settings.update(
            {
                key: default_settings[key]
                for key in default_settings
                if key not in self.settings
            }
        )
        logger.info(f"Settings: {self.settings}")

    def get_settings(self):
        SETTINGS_NOT_FOUND = (
            "No settings found. Click save settings or load a config file."
        )
        logger.info(f"self.settings: {self.settings}")
        if self.settings is None or self.settings == {}:
            raise Exception(SETTINGS_NOT_FOUND)
        return self.settings

    def update_transects_html(self, feature, **kwargs):
        # Modifies html when transect is hovered over
        values = defaultdict(lambda: "unknown", feature["properties"])
        self.feature_html.value = """ 
        <b>Transect</b>
        <p>Id: {}</p>
        <p>Slope: {}</p>
        """.format(
            values["id"],
            values["slope"],
        )

    def update_extracted_shoreline_html(self, feature, **kwargs):
        # Modifies html when extracted shoreline is hovered over
        values = defaultdict(lambda: "unknown", feature["properties"])
        self.feature_html.value = """
        <b>Extracted Shoreline</b>
        <p>Date: {}</p>
        <p>Geoaccuracy: {}</p>
        <p>Cloud Cover: {}</p>
        <p>Satellite Name: {}</p>
        """.format(
            values["date"],
            values["geoaccuracy"],
            values["cloud_cover"],
            values["satname"],
        )

    def update_roi_html(self, feature, **kwargs):
        # Modifies html when roi is hovered over
        values = defaultdict(lambda: "unknown", feature["properties"])
        # convert roi area m^2 to km^2
        roi_area = common.get_area(feature["geometry"]) * 10**-6
        roi_area = round(roi_area, 5)
        self.roi_html.value = """ 
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

    def load_extracted_shoreline_files(self) -> None:
        exception_handler.config_check_if_none(self.rois, "ROIs")
        # if no rois are selected throw an error
        exception_handler.check_selected_set(self.selected_set)
        roi_ids = list(self.selected_set)
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
                        shoreline_settings = common.from_file(file)
                    if "dict" in os.path.basename(file):
                        extracted_shoreline_dict = common.from_file(file)

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

    def get_most_accurate_epsg(self, settings: dict, bbox: gpd.GeoDataFrame) -> int:
        """Returns most accurate epsg code based on lat and lon if output epsg
        was 4326 or 4327

        Args:
            settings (dict): settings for map must contain output_epsg
            bbox (gpd.GeoDataFrame): geodataframe for bounding box on map

        Returns:
            int: epsg code that is most accurate or unchanged if crs not 4326 or 4327
        """
        output_epsg = settings["output_epsg"]
        logger.info(f"Old output_epsg:{output_epsg}")
        # coastsat cannot use 4326 to extract shorelines so modify output_epsg
        if output_epsg == 4326 or output_epsg == 4327:
            geometry = bbox.iloc[0]["geometry"]
            output_epsg = common.get_epsg_from_geometry(geometry)
            logger.info(f"New output_epsg:{output_epsg}")
        return output_epsg

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

    def extract_all_shorelines(self) -> None:
        """Use this function when the user interactively downloads rois
        Iterates through all the ROIS downloaded by the user as indicated by the roi_settings generated by
        download_imagery() and extracts a shoreline for each of them
        """
        # ROIs,settings, roi-settings cannot be None or empty
        settings = self.get_settings()
        exception_handler.check_if_empty_string(self.get_session_name(), "session name")
        exception_handler.check_if_None(self.rois, "ROIs")
        exception_handler.check_if_None(self.shoreline, "shoreline")
        exception_handler.check_if_None(self.transects, "transects")
        exception_handler.check_empty_dict(self.rois.roi_settings, "roi_settings")
        # settings must contain keys in subset
        subset = set(["dates", "sat_list", "landsat_collection"])
        superset = set(list(settings.keys()))
        exception_handler.check_if_subset(subset, superset, "settings")
        # roi_settings must contain roi ids in selected set
        subset = self.selected_set
        superset = set(list(self.rois.roi_settings.keys()))
        error_message = "To extract shorelines you must first select ROIs and have the data downloaded."
        exception_handler.check_if_subset(
            subset, superset, "roi_settings", error_message
        )
        # if no rois are selected throw an error
        exception_handler.check_selected_set(self.selected_set)
        roi_ids = list(self.selected_set)
        logger.info(f"roi_ids: {roi_ids}")
        # ensure selected rois have been downloaded
        exception_handler.check_if_rois_downloaded(self.rois.roi_settings, roi_ids)
        # ensure a bounding box exists on the map
        exception_handler.check_if_None(self.bbox, "bounding box")

        # if output_epsg is 4326 or 4327 change output_epsg to most accurate crs
        new_espg = self.get_most_accurate_epsg(settings, self.bbox.gdf)
        # update settings with the new output_epsg
        self.set_settings(output_epsg=new_espg)
        # update configs with new output_epsg
        self.save_config()

        # get selected ROIs on map and extract shoreline for each of them
        for roi_id in tqdm(roi_ids, desc="Extracting Shorelines"):
            print(f"Extracting shorelines from ROI with the id:{roi_id}")
            extracted_shorelines = self.extract_shoreline_for_roi(
                roi_id, self.rois.gdf, self.shoreline.gdf, self.get_settings()
            )
            self.rois.add_extracted_shoreline(extracted_shorelines, roi_id)

        self.save_session(roi_ids, save_transects=False)
        # for each ROI that has extracted shorelines load onto map
        # self.load_extracted_shorelines_to_map(roi_ids)

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

    def compute_transects_from_roi(
        self,
        roi_id: str,
        extracted_shorelines: dict,
        transects_gdf: gpd.GeoDataFrame,
        settings: dict,
    ) -> dict:
        """Computes the intersection between the 2D shorelines and the shore-normal.
            transects. It returns time-series of cross-shore distance along each transect.
        Args:
            roi_id(str): id of roi that transects intersect
            extracted_shorelines (dict): contains the extracted shorelines and corresponding metadata
            transects_gdf (gpd.GeoDataFrame): transects in ROI with crs= output_crs in settings
            settings (dict): settings dict with keys
                        'along_dist': int
                            alongshore distance considered calculate the intersection
        Returns:
            dict:  time-series of cross-shore distance along each of the transects.
                   Not tidally corrected.
        """
        # create dict of numpy arrays of transect start and end points
        transects = common.get_transect_points_dict(transects_gdf)
        logger.info(
            f"ROI{roi_id} extracted_shorelines for transects: {extracted_shorelines}"
        )
        logger.info(f"transects: {transects}")
        # cross_distance: along-shore distance over which to consider shoreline points to compute median intersection (robust to outliers)
        cross_distance = SDS_transects.compute_intersection_QC(
            extracted_shorelines, transects, settings
        )
        return cross_distance

    def get_cross_distance(self, roi_id: str, settings: dict, output_epsg: str):
        """
        Compute the cross shore distance of transects and extracted shorelines for a given ROI.

        Parameters:
        -----------
        roi_id : str
            The ID of the ROI to compute the cross shore distance for.
        settings : dict
            A dictionary of settings to be used in the computation.
        output_epsg : str
            The EPSG code of the output projection.

        Returns:
        --------
        float
            The computed cross shore distance, or 0 if there was an issue in the computation.

        Raises:
        -------
        exceptions.No_Extracted_Shoreline
            Raised if no shorelines were extracted for the ROI with roi_id

        Usage:
        ------
        cross_distance = get_cross_distance(roi_id, settings, output_epsg)
        """
        # get extracted shorelines object for the currently selected ROI
        roi_extracted_shoreline = self.rois.get_extracted_shoreline(roi_id)
        # get transects that intersect with ROI
        single_roi = common.extract_roi_by_id(self.rois.gdf, roi_id)
        transects_in_roi_gdf = self.transects.gdf[
            self.transects.gdf.intersects(single_roi.unary_union)
        ]
        transects_in_roi_gdf = transects_in_roi_gdf[["id", "geometry"]]

        failure_reason = None
        if roi_extracted_shoreline is None:
            cross_distance = 0
            failure_reason = "No extracted shorelines were found"
        elif transects_in_roi_gdf.empty:
            cross_distance = 0
            failure_reason = "No transects intersect"
        else:
            extracted_shoreline_x_transect = transects_in_roi_gdf[
                transects_in_roi_gdf.intersects(roi_extracted_shoreline.gdf.unary_union)
            ]
            if extracted_shoreline_x_transect.empty:
                cross_distance = 0
                failure_reason = "No extracted shorelines intersected transects"
            else:
                # convert transects_in_roi_gdf to output_crs from settings
                transects_in_roi_gdf = transects_in_roi_gdf.to_crs(output_epsg)
                # compute cross shore distance of transects and extracted shorelines
                extracted_shorelines_dict = roi_extracted_shoreline.dictionary
                cross_distance = self.compute_transects_from_roi(
                    roi_id,
                    extracted_shorelines_dict,
                    transects_in_roi_gdf,
                    settings,
                )

        if cross_distance == 0:
            logger.warning(f"{failure_reason} for ROI {roi_id}")
            print(f"{failure_reason} for ROI {roi_id}")

        return cross_distance

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
            return
        # get extracted_shorelines from extracted shoreline object in rois
        extracted_shorelines = roi_extracted_shorelines.dictionary
        # if no shorelines were extracted then skip
        if extracted_shorelines == {}:
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
        cross_distance_df = common.get_cross_distance_df(
            extracted_shorelines, cross_distance_transects
        )
        logger.info(f"ROI: {roi_id} cross_distance_df : {cross_distance_df}")

        filepath = os.path.join(session_path, "transect_time_series.csv")
        if os.path.exists(filepath):
            print(f"Overwriting:{filepath}")
            os.remove(filepath)
        cross_distance_df.to_csv(filepath, sep=",")
        print(
            f"ROI: {roi_id} Time-series of the shoreline change along the transects saved as:{filepath}"
        )

    def compute_transects(self) -> dict:
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
        settings = self.get_settings()
        exception_handler.check_if_empty_string(self.get_session_name(), "session name")
        exception_handler.check_if_None(self.rois, "ROIs")
        exception_handler.check_if_None(self.transects, "transects")
        exception_handler.check_empty_dict(
            self.rois.get_all_extracted_shorelines(), "extracted_shorelines"
        )
        exception_handler.check_if_subset(
            set(["along_dist"]), set(list(settings.keys())), "settings"
        )
        # ids of ROIs that have had their shorelines extracted
        extracted_shoreline_ids = set(
            list(self.rois.get_all_extracted_shorelines().keys())
        )
        logger.info(f"extracted_shoreline_ids:{extracted_shoreline_ids}")

        # Get ROI ids that are selected on map and have had their shorelines extracted
        roi_ids = list(extracted_shoreline_ids & self.selected_set)
        # if none of the selected ROIs on the map have had their shorelines extracted throw an error
        exception_handler.check_if_list_empty(roi_ids)

        # user selected output projection
        output_epsg = "epsg:" + str(settings["output_epsg"])
        # for each ROI save cross distances for each transect that intersects each extracted shoreline
        for roi_id in tqdm(roi_ids, desc="Computing Cross Distance Transects"):
            # save cross distances by ROI id
            cross_distance = self.get_cross_distance(str(roi_id), settings, output_epsg)
            self.rois.add_cross_shore_distances(cross_distance, roi_id)

        self.save_session(roi_ids)

    def save_session(self, roi_ids: list[str], save_transects: bool = True):
        # Save extracted shoreline info to session directory
        session_name = self.get_session_name()
        for roi_id in roi_ids:
            ROI_directory = self.rois.roi_settings[roi_id]["sitename"]
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
            data_path = extracted_shoreline.shoreline_settings["inputs"]["filepath"]
            sitename = extracted_shoreline.shoreline_settings["inputs"]["sitename"]
            extracted_shoreline_figure_path = os.path.join(
                data_path, sitename, "jpg_files", "detection"
            )
            logger.info(
                f"extracted_shoreline_figure_path: {extracted_shoreline_figure_path}"
            )

            if os.path.exists(extracted_shoreline_figure_path):
                dst_path = os.path.join(session_path, "jpg_files", "detection")
                logger.info(f"dst_path : {dst_path }")
                common.move_files(
                    extracted_shoreline_figure_path, dst_path, delete_src=True
                )

            extracted_shoreline.to_file(
                session_path, "extracted_shorelines.geojson", extracted_shoreline.gdf
            )
            extracted_shoreline.to_file(
                session_path,
                "shoreline_settings.json",
                extracted_shoreline.shoreline_settings,
            )
            extracted_shoreline.to_file(
                session_path,
                "extracted_shorelines_dict.json",
                extracted_shoreline.dictionary,
            )

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
        # set of roi ids that have add their transects successfully computed
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

            # save to csv file session path
            fn = os.path.join(session_path, "%s_timeseries_raw.csv" % key)
            logger.info(f"Save time series to {fn}")
            if os.path.exists(fn):
                logger.info(f"Overwriting:{fn}")
                os.remove(fn)
            df.to_csv(fn, sep=",")
            logger.info(
                f"ROI: {roi_id}Time-series of the shoreline change along the transects saved as:{fn}"
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
        if self.extracted_shoreline_layers != []:
            for layername in self.extracted_shoreline_layers:
                self.remove_layer_by_name(layername)
            self.extracted_shoreline_layers = []

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

    def load_extracted_shorelines_to_map(self, roi_ids: list) -> None:
        # for each ROI that has extracted shorelines load onto map
        for roi_id in roi_ids:
            roi_extract_shoreline = self.rois.get_extracted_shoreline(roi_id)
            logger.info(roi_extract_shoreline)
            if roi_extract_shoreline is not None:
                self.load_extracted_shorelines_on_map(roi_extract_shoreline)

    def load_extracted_shorelines_on_map(self, extracted_shoreline):
        # Loads stylized extracted shorelines onto map for single roi
        layers = extracted_shoreline.get_styled_layers()
        layer_names = extracted_shoreline.get_layer_names()
        self.extracted_shoreline_layers = [
            *self.extracted_shoreline_layers,
            *layer_names,
        ]
        logger.info(f"{layers}")
        for new_layer in layers:
            new_layer.on_hover(self.update_extracted_shoreline_html)
            self.map.add_layer(new_layer)

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
        if file != "":
            gdf = common.read_gpd_file(file)
        # convert gdf to the proper format
        if gdf is not None:
            logger.info("gdf is not None")
            if (
                "shoreline" in feature_name.lower()
                or "transect" in feature_name.lower()
            ):
                logger.info(feature_name.lower())
                # if 'id' column is not present and 'name' column is replace 'name' with 'id'
                # id neither exist create a new column named 'id' with row index
                if "ID" in gdf.columns:
                    logger.info(f"ID in gdf.columns: {gdf.columns}")
                    gdf.rename(columns={"ID": "id"}, inplace=True)
                common.replace_column(gdf, new_name="id", replace_col="name")
                logger.info(f"new gdf: {gdf}")

            # if a z axis exists remove it
            gdf = common.remove_z_axis(gdf)
            logger.info(f"gdf after z-axis removed: {gdf}")
            bounds = gdf.total_bounds
            self.map.zoom_to_bounds(bounds)

        new_feature = self.factory.make_feature(self, feature_name, gdf, **kwargs)
        logger.info(f"new_feature: {new_feature}")
        logger.info(f"new_feature.gdf: {new_feature.gdf}")
        logger.info(f" gdf: {gdf}")
        logger.info(f"feature_name: {feature_name.lower()}")
        on_hover = None
        on_click = None
        if "shoreline" in feature_name.lower():
            on_hover = self.update_shoreline_html
        if "transects" in feature_name.lower():
            on_hover = self.update_transects_html
        if "rois" in feature_name.lower():
            on_hover = self.update_roi_html
            on_click = self.geojson_onclick_handler
        # bounding box does not have any hover/click handlers
        # load new feature on map
        layer_name = new_feature.LAYER_NAME
        self.load_on_map(new_feature, layer_name, on_hover, on_click)

    def load_on_map(
        self, feature, layer_name: str, on_hover=None, on_click=None
    ) -> None:
        """Loads feature on map

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

    def generate_ROIS_fishnet(
        self, large_len: float = 7500, small_len: float = 5000
    ) -> ROI:
        """Generates series of overlapping ROIS along shoreline on map using fishnet method"""
        exception_handler.check_if_None(self.bbox, "bounding box")
        logger.info(f"bbox for ROIs: {self.bbox.gdf}")
        # If no shoreline exists on map then load one in
        if self.shoreline is None:
            self.load_feature_on_map("shoreline")
        logger.info(f"self.shoreline used for ROIs:{self.shoreline}")
        # create rois within the bbox that intersect shorelines
        rois = ROI(
            self.bbox.gdf,
            self.shoreline.gdf,
            square_len_lg=large_len,
            square_len_sm=small_len,
        )
        return rois

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
        exception_handler.can_feature_save_to_file(feature, feature_type)
        if isinstance(feature, ROI):
            # raise exception if no rois were selected
            exception_handler.check_selected_set(self.selected_set)
            feature.gdf[feature.gdf["id"].isin(self.selected_set)].to_file(
                feature.filename, driver="GeoJSON"
            )
        else:
            logger.info(f"Saving feature type( {feature}) to file")
            feature.gdf.to_file(feature.filename, driver="GeoJSON")
        print(f"Save {feature.LAYER_NAME} to {feature.filename}")
        logger.info(f"Save {feature.LAYER_NAME} to {feature.filename}")

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
