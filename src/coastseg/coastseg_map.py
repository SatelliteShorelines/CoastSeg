import os
import json
import logging
import copy

from typing import Union
from coastseg.bbox import Bounding_Box
from coastseg import common
from coastseg.shoreline import Shoreline
from coastseg.transects import Transects
from coastseg.roi import ROI
from coastseg import exceptions
from coastseg import map_UI
from coastseg import extracted_shoreline
from coastseg import exception_handler

import geopandas as gpd
import numpy as np
import matplotlib

from coastsat import (
    SDS_tools,
    SDS_download,
    SDS_tools,
    SDS_transects,
    SDS_shoreline,
    SDS_preprocess,
)
from ipyleaflet import DrawControl, LayersControl, WidgetControl, GeoJSON
from leafmap import Map
from ipywidgets import Layout, HTML, Accordion
from tqdm.auto import tqdm
from pyproj import Proj, transform

logger = logging.getLogger(__name__)


class CoastSeg_Map:
    def __init__(self, map_settings: dict = None):
        # settings:  used to select data to download and preprocess settings
        self.settings = {}
        # selected_set set(str): ids of the selected rois
        self.selected_set = set()
        # self.extracted_shoreline_layers : names of extracted shorelines vectors on the map
        self.extracted_shoreline_layers = []
        # ROI_layer : layer containing all rois
        self.ROI_layer = None
        # rois : ROI(Region of Interest)
        self.rois = None
        # selected_ROI_layer :  layer containing all selected rois
        self.selected_ROI_layer = None
        # self.transect : transect object containing transects loaded on map
        self.transects = None
        # self.shoreline : shoreline object containing shoreline loaded on map
        self.shoreline = None
        # Bbox saved by the user
        self.bbox = None
        # preprocess_settings : dictionary of settings used by coastsat to download imagery
        self.preprocess_settings = {}
        # create map if map_settings not provided else use default settings
        self.map = self.create_map(map_settings)
        # create controls and add to map
        self.draw_control = self.create_DrawControl(DrawControl())
        self.draw_control.on_draw(self.handle_draw)
        self.map.add(self.draw_control)
        layer_control = LayersControl(position="topright")
        self.map.add(layer_control)
        hover_shoreline_control = self.create_shoreline_widget()
        self.map.add(hover_shoreline_control)

    def create_map(self, map_settings: dict):
        """create an interactive map object using the map_settings

        Args:
            map_settings (dict): settings to control how map is created

        Returns:
           ipyleaflet.Map: ipyleaflet interactive Map object
        """
        if not map_settings:
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

    def create_shoreline_widget(self):
        """creates a accordion style widget controller to hold shoreline data.

        Returns:
           ipyleaflet.WidgetControl: an widget control for an accordion widget
        """
        html = HTML("Hover over shoreline")
        html.layout.margin = "0px 20px 20px 20px"
        self.shoreline_accordion = Accordion(
            children=[html], titles=("Shoreline Data",)
        )
        self.shoreline_accordion.set_title(0, "Shoreline Data")

        return WidgetControl(widget=self.shoreline_accordion, position="topright")

    def load_configs(self, filepath: str):
        self.load_gdf_config(filepath)
        # remove geojson file from path
        dir_path = os.path.dirname(os.path.realpath(filepath))
        # get name of config.json file in dir_path
        config_file = common.find_config_json(dir_path)
        if config_file is None:
            logger.error(f"config.json file was not found at {dir_path}")
            raise Exception(f"config.json file was not found at {dir_path}")
        logger.info(f"config_file: {config_file}")
        config_path = os.path.join(dir_path, config_file)
        logger.info(f"Loaded json config file from {config_path}")
        self.load_json_config(config_path)

    def load_gdf_config(self, filepath: str):
        print(f"Loaded geojson{filepath}")
        gdf = common.read_gpd_file(filepath)
        # Extract the ROIs from the gdf and create new dataframe
        roi_gdf = gdf[gdf["type"] == "roi"].copy()
        if roi_gdf.empty:
            raise Exception(f"No ROIs were present in the config file: {filepath}")
        # drop all columns except id and geometry
        columns_to_drop = list(set(roi_gdf.columns) - set(["id", "geometry"]))
        logger.info(f"Dropping columns from ROI: {columns_to_drop}")
        roi_gdf.drop(columns_to_drop, axis=1, inplace=True)
        logger.info(f"roi_gdf: {roi_gdf}")

        # Extract the shoreline from the gdf
        shoreline_gdf = gdf[gdf["type"] == "shoreline"].copy()
        # drop columns id and type
        if "slope" in roi_gdf.columns:
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

        # Extract the bbox from the gdf
        bbox_gdf = gdf[gdf["type"] == "bbox"].copy()
        columns_to_drop = list(set(bbox_gdf.columns) - set(["geometry"]))
        logger.info(f"Dropping columns from bbox: {columns_to_drop}")
        bbox_gdf.drop(columns_to_drop, axis=1, inplace=True)
        logger.info(f"bbox_gdf: {bbox_gdf}")

        # delete the original gdf read in from config geojson file
        del gdf

        if bbox_gdf.empty:
            self.bbox = None
            logger.info("No Bounding Box was loaded on the map")
            print("No Bounding Box was loaded on the map")
        else:
            bbox = Bounding_Box(rectangle=bbox_gdf)
            logger.info(f"bbox: {bbox}")
            self.bbox = bbox
            logger.info(f"self.bbox: {self.bbox}")
            self.load_bbox_on_map()
            logger.info("Bounding Box was loaded on the map")
            print("Bounding Box was loaded on the map")

        # Create ROI object from roi_gdf
        if roi_gdf.empty:
            raise Exception("Cannot load empty ROIs onto the map")
        roi = ROI(rois_gdf=roi_gdf)
        self.rois = roi
        self.load_rois_on_map()

        if shoreline_gdf.empty:
            self.shoreline = None
            logger.info("No shoreline was loaded on the map")
            print("No shoreline was loaded on the map")
        else:
            # Create Shoreline object from shoreline_gdf
            shoreline = Shoreline(shoreline=shoreline_gdf)
            self.shoreline = shoreline
            logger.info("Shoreline was loaded on the map")
            print("Shoreline was loaded on the map")
            self.load_shoreline_on_map()
        # Create Transect object from transect_gdf
        if transect_gdf.empty:
            self.transects = None
            logger.info("No transects were loaded on the map")
            print("No transects were loaded on the map")
        else:
            transect = Transects(transects=transect_gdf)
            self.transects = transect
            logger.info("Transects were loaded on the map")
            print("Transects were loaded on the map")
            self.load_transects_on_map()

    def download_imagery(self) -> None:
        """download_imagery  downloads selected rois as jpgs

        Raises:
            Exception: raised if settings is missing
            Exception: raised if 'dates','sat_list', and 'landsat_collection' are not in settings
            Exception: raised if no ROIs have been selected
        """
        if self.settings is None:
            logger.error("No settings found.")
            raise Exception("No settings found. Create settings before downloading")
        if not set(["dates", "sat_list", "landsat_collection"]).issubset(
            set(self.settings.keys())
        ):
            logger.error(
                f"Missing keys from settings: {set(['dates','sat_list','landsat_collection'])-set(self.settings.keys())}"
            )
            raise Exception(
                f"Missing keys from settings: {set(['dates','sat_list','landsat_collection'])-set(self.settings.keys())}"
            )
        if self.selected_ROI_layer is None:
            logger.error("No ROIs were selected.")
            raise Exception(
                "No ROIs were selected. Make sure to click save roi before downloading."
            )

        logger.info(f"self.settings: {self.settings}")
        logger.info(f"self.selected_ROI_layer: {self.selected_ROI_layer}")

        # 1. Create a list of download settings for each ROI.
        # filepath: path to directory where downloaded imagery will be saved. Defaults to data directory in CoastSeg
        filepath = os.path.abspath(os.path.join(os.getcwd(), "data"))
        date_str = common.generate_datestring()
        roi_settings = common.create_roi_settings(
            self.settings, self.selected_ROI_layer.data, filepath, date_str
        )
        # Save download settings dictionary to instance of ROI
        self.rois.set_roi_settings(roi_settings)
        inputs_list = list(roi_settings.values())
        logger.info(f"inputs_list {inputs_list}")

        # 2. For each ROI used download settings to download imagery and save to jpg
        print("Download in process")
        # make a deep copy so settings doesn't get modified by the temp copy
        tmp_settings = copy.deepcopy(self.settings)
        # for each ROI use inputs dictionary to download imagery and save to jpg
        for inputs_for_roi in tqdm(inputs_list, desc="Downloading ROIs"):
            metadata = SDS_download.retrieve_images(inputs_for_roi)
            print(f"inputs: {inputs_for_roi}")
            logger.info(f"inputs: {inputs_for_roi}")
            tmp_settings["inputs"] = inputs_for_roi
            logger.info(f"Saving to jpg. Metadata: {metadata}")
            SDS_preprocess.save_jpg(metadata, tmp_settings)

        # tmp settings is no longer needed
        del tmp_settings

        # 3.save settings used to download rois and the objects on map to config files
        self.save_config()
        logger.info("Done downloading")

    def load_json_config(self, filepath: str) -> None:
        if self.rois is None:
            raise Exception("Must load ROIs onto the map first")

        json_data = common.read_json_file(filepath)
        # replace coastseg_map.settings with settings from config file
        self.settings = json_data["settings"]
        logger.info(f"Loaded settings from file: {self.settings}")
        # replace roi_settings for each ROI with contents of json_data
        roi_settings = {}
        for roi_id in json_data["roi_ids"]:
            roi_settings[str(roi_id)] = json_data[roi_id]
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
            Exception: raised if any of "dates", "sat_list", "landsat_collection" is missing from self.settings
            Exception: raised if self.rois is missing
            Exception: raised if self.selected_ROI_layer is missing
        """
        if self.settings is None:
            logger.error(
                "Settings must be loaded before configuration files can be made."
            )
            raise Exception(
                "Settings must be loaded before configuration files can be made."
            )
        if not set(["dates", "sat_list", "landsat_collection"]).issubset(
            set(self.settings.keys())
        ):
            logger.error(
                f"Missing keys from settings: {set(['dates','sat_list','landsat_collection'])-set(self.settings.keys())}"
            )
            raise Exception(
                f"Missing keys from settings: {set(['dates','sat_list','landsat_collection'])-set(self.settings.keys())}"
            )
        if self.rois is None:
            logger.error(
                "ROIs must be loaded onto the map before configuration files can be made."
            )
            raise Exception(
                "ROIs must be loaded onto the map before configuration files can be made."
            )
        if self.selected_ROI_layer is None:
            logger.error(
                "No ROIs were selected. Cannot save ROIs to config until ROIs are selected."
            )
            raise Exception(
                "No ROIs were selected. Cannot save ROIs to config until ROIs are selected."
            )

        logger.info(f"self.rois.roi_settings: {self.rois.roi_settings}")
        if self.rois.roi_settings == {}:
            if filepath is None:
                filepath = os.path.abspath(os.getcwd())
            roi_settings = common.create_roi_settings(
                self.settings, self.selected_ROI_layer.data, filepath
            )
            # Save download settings dictionary to instance of ROI
            self.rois.set_roi_settings(roi_settings)

        # create dictionary to be saved to config.json
        config_json = common.create_json_config(self.rois.roi_settings, self.settings)
        logger.info(f"config_json: {config_json} ")

        # get currently selected rois selected
        roi_ids = config_json["roi_ids"]
        selected_rois = self.get_selected_rois(roi_ids)
        logger.info(f"selected_rois: {selected_rois} ")

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

    def save_settings(self, **kwargs):
        """Saves the settings for downloading data in a dictionary
        Pass in data in the form of
        save_settings(sat_list=sat_list, landsat_collection='C01',dates=dates,**preprocess_settings)
        *You must use the names sat_list, landsat_collection, and dates
        """
        for key, value in kwargs.items():
            self.settings[key] = value

    def update_shoreline_html(self, feature, **kwargs):
        # Modifies main html body of Shoreline Data Accordion
        self.shoreline_accordion.children[
            0
        ].value = """
        <p>Mean Sig Waveheight: {}</p>
        <p>Tidal Range: {}</p>
        <p>Erodibility: {}</p>
        <p>River: {}</p>
        <p>Sinuosity: {}</p>
        <p>Slope: {}</p>
        <p>Turbid: {}</p>
        <p>CSU_ID: {}</p>
        """.format(
            feature["properties"]["MEAN_SIG_WAVEHEIGHT"],
            feature["properties"]["TIDAL_RANGE"],
            feature["properties"]["ERODIBILITY"],
            feature["properties"]["river_label"],
            feature["properties"]["sinuosity_label"],
            feature["properties"]["slope_label"],
            feature["properties"]["turbid_label"],
            feature["properties"]["CSU_ID"],
        )

    def load_extracted_shorelines_to_map(self, roi_ids: list) -> None:
        # for each ROI that has extracted shorelines load onto map
        for roi_id in roi_ids:
            roi_extract_shoreline = self.rois.extracted_shorelines[roi_id]
            logger.info(roi_extract_shoreline)
            if roi_extract_shoreline is not None:
                self.load_extracted_shorelines_on_map(roi_extract_shoreline)

    def save_extracted_shorelines_to_file(self, roi_ids: list) -> None:
        # Saves extracted_shorelines to ROI's directory to file 'extracted_shorelines.geojson'
        for roi_id in roi_ids:
            roi_extract_shoreline = self.rois.extracted_shorelines[roi_id]
            if roi_extract_shoreline is not None:
                sitename = self.rois.roi_settings[roi_id]["sitename"]
                filepath = self.rois.roi_settings[roi_id]["filepath"]
                roi_extract_shoreline.save_to_file(sitename, filepath)

    def extract_all_shorelines(self) -> None:
        """Use this function when the user interactively downloads rois
        Iterates through all the ROIS downloaded by the user as indicated by the roi_settings generated by
        download_imagery() and extracts a shoreline for each of them
        """
        if self.rois is None:
            raise Exception("No Rois on map")
        if self.shoreline is None:
            raise Exception(
                "No Shoreline found. Please load a shoreline on the map first."
            )
        if self.rois.roi_settings == {}:
            logger.error(
                f"No inputs settings found. Please click download ROIs first. self.rois: {self.rois}"
            )
            raise Exception(
                "No inputs settings found. Please click download ROIs first or upload configs"
            )
        if self.settings is None:
            logger.error("No settings found. Please load settings")
            raise Exception("No settings found. Please load settings")
        if not set(["dates", "sat_list", "landsat_collection"]).issubset(
            set(self.settings.keys())
        ):
            logger.error(
                f"Missing keys from self.settings: {set(['dates','sat_list','landsat_collection'])-set(self.settings.keys())}"
            )
            raise Exception(
                f"Missing keys from self.settings: {set(['dates','sat_list','landsat_collection'])-set(self.settings.keys())}"
            )
        if not self.selected_set.issubset(set(self.rois.roi_settings.keys())):
            logger.error(
                f"self.selected_set: {self.selected_set} was not a subset of self.rois.roi_settings: {set(self.rois.roi_settings.keys())}"
            )
            raise Exception(
                f"To extract shorelines you must first select ROIs and have the data downloaded."
            )

        # get selected ROIs on map and extract shoreline for each of them
        roi_ids = list(self.selected_set)
        # if none are selected throw an error
        if len(roi_ids) == 0:
            raise Exception(f"Please select ROIs on the map first.")
        if common.were_rois_downloaded(self.rois.roi_settings, roi_ids) == False:
            logger.error(f" not all rois were downloaded{self.rois.roi_settings}")
            raise Exception(
                f"Some ROIs filepaths did not exist on your computer. Try downloading the rois to overwrite the filepath ."
            )

        logger.info(f" roi_ids: {roi_ids}")
        # extracted_shoreline_dict: holds extracted shorelines for each ROI
        extracted_shoreline_dict = {}
        for roi_id in tqdm(roi_ids, desc="Extracting Shorelines"):
            try:
                print(f"Extracting shorelines from ROI with the id:{roi_id}")
                roi_settings = self.rois.roi_settings[roi_id]
                single_roi = common.extract_roi_by_id(self.rois.gdf, roi_id)
                # Clip shoreline to specific roi
                shoreline_in_roi = gpd.clip(self.shoreline.gdf, single_roi)
                # extract shorelines from ROI
                extracted_shorelines = extracted_shoreline.Extracted_Shoreline(
                    roi_id, shoreline_in_roi, roi_settings, self.settings
                )
                logger.info(
                    f"extracted_shoreline_dict[{roi_id}]: {extracted_shorelines}"
                )
            except exceptions.Id_Not_Found as id_error:
                logger.warning(f"exceptions.Id_Not_Found {id_error}")
                print(f"exceptions.Id_Not_Found:{id_error}. \n Skipping to next ROI")
            except exceptions.No_Extracted_Shoreline as no_shoreline:
                extracted_shoreline_dict[roi_id] = None
                logger.warning(f"{no_shoreline}")
                print(no_shoreline)
            else:
                extracted_shoreline_dict[roi_id] = extracted_shorelines

        # Save all the extracted_shorelines to ROI
        self.rois.update_extracted_shorelines(extracted_shoreline_dict)
        logger.info(
            f"extract_all_shorelines : self.rois.extracted_shorelines {self.rois.extracted_shorelines}"
        )
        # Saves extracted_shorelines to ROI's directory to file 'extracted_shorelines.geojson'
        self.save_extracted_shorelines_to_file(roi_ids)
        # for each ROI that has extracted shorelines load onto map
        self.load_extracted_shorelines_to_map(roi_ids)

    def get_selected_rois(self, roi_ids: list) -> gpd.GeoDataFrame:
        """Returns a geodataframe of all rois selected by roi_ids

        Args:
            roi_ids (list[str]): ids of ROIs

        Returns:
            gpd.GeoDataFrame:  geodataframe of all rois selected by the roi_ids
        """
        selected_rois_gdf = self.rois.gdf[self.rois.gdf["id"].isin(roi_ids)]
        return selected_rois_gdf

    def compute_transects_from_roi(
        self,
        extracted_shorelines: list,
        transects_gdf: gpd.GeoDataFrame,
        settings: dict,
    ) -> dict:
        """Computes the intersection between the 2D shorelines and the shore-normal.
            transects. It returns time-series of cross-shore distance along each transect.
        Args:
            extracted_shorelines (list): contains the extracted shorelines and corresponding metadata
            transects_gdf (gpd.GeoDataFrame): transects in ROI with crs= output_crs in settings
            settings (dict): settings dict with keys
                        'along_dist': int
                            alongshore distance considered calculate the intersection
        Returns:
            dict:  time-series of cross-shore distance along each of the transects.
                   Not tidally corrected.
        """
        transects_coords = common.make_coastsat_compatible(transects_gdf)
        logger.info(f"transects_coords:{transects_coords}")
        # create dict of numpy arrays of transect start and end points
        # {'NA<index>': array([[start point],[end point]]),...}
        transects = common.get_transect_points_dict(transects_coords)
        logger.info(f"transects: {transects}")
        # cross_distance: along-shore distance over which to consider shoreline points to compute median intersection (robust to outliers)
        cross_distance = SDS_transects.compute_intersection(
            extracted_shorelines, transects, settings
        )
        print(f"transects cross_distance:{cross_distance}")
        return cross_distance

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
        cross_distance_transects = {}
        if self.rois is None:
            logger.error("No ROIs have been loaded")
            raise Exception("No ROIs have been loaded")
        if self.transects is None:
            logger.error("No transects were loaded onto the map.")
            raise Exception("No transects were loaded onto the map.")
        if self.rois.extracted_shorelines == {}:
            logger.error("No shorelines have been extracted. Extract shorelines first.")
            raise Exception(
                "No shorelines have been extracted. Extract shorelines first."
            )
        if self.settings is None:
            logger.error("No settings have been loaded")
            raise Exception("No settings have been loaded")
        if "along_dist" not in self.settings.keys():
            raise Exception("Missing key 'along_dist' in settings")

        settings = self.settings
        extracted_shoreline_ids = set(self.rois.extracted_shorelines.keys())
        logger.info(
            f"self.rois.extracted_shorelines.keys(): {list(self.rois.extracted_shorelines.keys())}"
        )
        # only get roi ids that are currently selected on map and have had their shorelines extracted
        roi_ids = list(extracted_shoreline_ids & self.selected_set)
        if len(roi_ids) == 0:
            logger.error(
                f"self.selected_set: {self.selected_set} was not a subset of extracted_shoreline_ids: {set(extracted_shoreline_ids)}"
            )
            raise Exception(
                f"You must select an ROI and extract it shorelines before you can compute transects"
            )

        # user selected output projection
        output_epsg = "epsg:" + str(settings["output_epsg"])

        # Save cross distances for each set of transects intersecting each extracted shoreline for each ROI
        cross_distance_transects = {}
        for roi_id in tqdm(roi_ids, desc="Computing Cross Distance Transects"):
            failure_msg = ""
            cross_distance = None
            try:
                # if no extracted shoreline exist for ROI's id return cross distance = 0
                roi_extracted_shoreline = self.rois.extracted_shorelines[str(roi_id)]
                if roi_extracted_shoreline is None:
                    cross_distance = 0
                    failure_msg = f"No shorelines were extracted for {roi_id}"
                elif roi_extracted_shoreline is not None:
                    single_roi = common.extract_roi_by_id(self.rois.gdf, roi_id)
                    # get extracted shorelines array
                    extracted_shorelines = roi_extracted_shoreline.extracted_shorelines
                    transects_in_roi_gdf = gpd.sjoin(
                        left_df=self.transects.gdf,
                        right_df=single_roi,
                        how="inner",
                        predicate="intersects",
                    )
                    columns_to_drop = list(
                        set(transects_in_roi_gdf.columns) - set(["id_left", "geometry"])
                    )
                    transects_in_roi_gdf.drop(columns_to_drop, axis=1, inplace=True)
                    transects_in_roi_gdf.rename(columns={"id_left": "id"}, inplace=True)
                    if transects_in_roi_gdf.empty:
                        cross_distance = 0
                        failure_msg = f"No transects intersected {roi_id}"
                    # Check if any extracted shorelines in ROI intersect with transect
                    extracted_shoreline_x_transect = gpd.sjoin(
                        left_df=transects_in_roi_gdf,
                        right_df=roi_extracted_shoreline.gdf,
                        how="inner",
                        predicate="intersects",
                    )
                    if extracted_shoreline_x_transect.empty:
                        cross_distance = 0
                        failure_msg = f"No extracted shorelines intersected transects for {roi_id}"
                    del extracted_shoreline_x_transect
                    # convert transects_in_roi_gdf to output_crs from settings
                    transects_in_roi_gdf = transects_in_roi_gdf.to_crs(output_epsg)

                    # if shoreline and transects in ROI and extracted_shorelines intersect transect
                    # compute cross distances of transects and extracted shorelines
                    if cross_distance is None:
                        cross_distance = self.compute_transects_from_roi(
                            extracted_shorelines, transects_in_roi_gdf, settings
                        )
                if cross_distance == 0:
                    logger.warning(failure_msg)
                    print(failure_msg)
                # save cross distances by ROI id
                cross_distance_transects[roi_id] = cross_distance
                logger.info(
                    f"\ncross_distance_transects[{roi_id}]: {cross_distance_transects[roi_id]}"
                )
                self.rois.save_transects_to_json(roi_id, cross_distance)
            except exceptions.No_Extracted_Shoreline:
                logger.warning(
                    f"ROI id:{roi_id} has no extracted shoreline. No transects computed"
                )
                print(
                    f"ROI id:{roi_id} has no extracted shoreline. No transects computed"
                )

        # save cross distances for all transects to ROIs
        self.rois.cross_distance_transects = cross_distance_transects

    def save_transects_to_csv(self) -> None:
        """save_cross_distance_df Saves the cross distances of the transects and
        extracted shorelines in ROI to csv file within each ROI's directory.
        If no shorelines were extracted for an ROI then nothing is saved
        Raises:
            Exception: No ROIs have been loaded
            Exception: No transects were loaded
            Exception: No shorelines have been extracted
            Exception: No roi settings have been loaded
            Exception: No rois selected that have had shorelines extracted
        """
        if self.rois is None:
            logger.error("No ROIs have been loaded")
            raise Exception("No ROIs have been loaded")
        if self.transects is None:
            logger.error("No transects were loaded onto the map.")
            raise Exception("No transects were loaded onto the map.")
        if self.rois.extracted_shorelines == {}:
            logger.error("No shorelines have been extracted. Extract shorelines first.")
            raise Exception(
                "No shorelines have been extracted. Extract shorelines first."
            )
        if self.rois.roi_settings == {}:
            logger.error("No roi settings have been loaded")
            raise Exception("No roi settings have been loaded")
        if self.rois.cross_distance_transects == {}:
            logger.error("No cross distances transects have been computed")
            raise Exception("No roi settings transects have been computed")

        # only get roi ids that are currently selected on map and have had their shorelines extracted
        extracted_shoreline_ids = set(self.rois.extracted_shorelines.keys())
        roi_ids = list(extracted_shoreline_ids & self.selected_set)
        if len(roi_ids) == 0:
            logger.error(
                f"{self.selected_set} was not a subset of extracted_shoreline_ids: {extracted_shoreline_ids}"
            )
            raise Exception(
                f"You must select an ROI and extract it shorelines before you can compute transects"
            )
        # save cross distances for transects and extracted shorelines to csv file
        # each csv file is saved to ROI directory
        self.save_cross_distance_df(roi_ids, self.rois)

    def save_cross_distance_df(self, roi_ids: list, rois: ROI) -> None:
        """save_cross_distance_df Saves the cross distances of the transects and
        extracted shorelines in ROI to csv file within each ROI's directory.
        If no shorelines were extracted for an ROI then nothing is saved
        Args:
            roi_ids (list): list of roi ids
            rois (ROI): ROI instance containing keys:
                'extracted_shorelines': extracted shoreline from roi
                'roi_settings': must have keys 'filepath' and 'sitename'
                'cross_distance_transects': cross distance of transects and extracted shoreline from roi
        """
        for roi_id in roi_ids:
            extracted_shorelines = rois.extracted_shorelines[
                roi_id
            ].extracted_shorelines
            cross_distance_transects = rois.cross_distance_transects[roi_id]
            # if no shorelines were extracted then skip
            if extracted_shoreline is None:
                continue
            cross_distance_df = common.get_cross_distance_df(
                extracted_shorelines, cross_distance_transects
            )
            filepath = rois.roi_settings[roi_id]["filepath"]
            sitename = rois.roi_settings[roi_id]["sitename"]
            fn = os.path.join(filepath, sitename, "transect_time_series.csv")
            if os.path.exists(fn):
                print(f"Overwriting:{fn}")
                os.remove(fn)
            cross_distance_df.to_csv(fn, sep=",")
            print(
                f"Time-series of the shoreline change along the transects saved as:{fn}"
            )

    def load_transects_on_map(self) -> None:
        """Loads transects within bounding box on map"""
        # if no transects on map then create new transects
        try:
            if self.transects is None:
                exception_handler.check_exception_None(
                    self.bbox,
                    "bounding box",
                    "Cannot load transects on map because bbox does not exist",
                )
                exception_handler.check_exception_gdf_empty(
                    self.bbox.gdf,
                    "bounding box",
                    "Cannot load transects on map because bbox does not exist",
                )
                # if bounding box exists load transects within it
                transects = Transects(self.bbox.gdf)
                exception_handler.check_exception_gdf_empty(
                    transects.gdf,
                    "transects",
                    "Transects Not Found in this region. Draw a new bounding box",
                )
                # Save transects to coastseg_map
                self.transects = transects

            layer_name = "transects"
            # Replace old transect layer with new transect layer
            self.remove_layer_by_name(layer_name)
            # style and add the transect to the map
            new_layer = self.create_layer(self.transects, layer_name)
            if new_layer is None:
                print("Cannot add an empty transects layer to the map.")
            else:
                self.map.add_layer(new_layer)
                print("Loaded transects")
                logger.info(f"Added layer to map: {new_layer}")
        except Exception as err:
            exception_handler.handle_exception(err)

    def remove_all(self):
        """Remove the bbox, shoreline, all rois from the map"""
        self.remove_bbox()
        self.remove_shoreline()
        self.remove_transects()
        self.remove_all_rois()
        self.remove_shoreline_html()
        self.remove_layer_by_name("geodataframe")
        self.remove_extracted_shorelines()

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

    def remove_extracted_shorelines(self):
        """Remove all the extracted shorelines layers from map"""
        for layer in self.extracted_shoreline_layers:
            self.remove_layer_by_name(layer)
        self.extracted_shoreline_layers = []

    def remove_layer_by_name(self, layer_name):
        existing_layer = self.map.find_layer(layer_name)
        if existing_layer is not None:
            self.map.remove_layer(existing_layer)
        logger.info(f"Removed layer {layer_name}")

    def remove_shoreline_html(self):
        """Clear the shoreline html accoridon"""
        self.shoreline_accordion.children[0].value = "Hover over the shoreline."

    def remove_shoreline(self):
        del self.shoreline
        self.remove_layer_by_name(Shoreline.LAYER_NAME)
        self.shoreline = None

    def remove_transects(self):
        del self.transects
        self.transects = None
        self.remove_layer_by_name(Transects.LAYER_NAME)

    def remove_all_rois(self, delete_rois: bool = True) -> None:
        """Removes all the unselected rois from the map
        delete_rois: bool
             controls whether rois are deleted from coastseg_map class instance
        """
        if delete_rois == True:
            del self.rois
            self.rois = None

        # Remove the selected rois
        self.selected_ROI_layer = None
        del self.ROI_layer
        self.ROI_layer = None
        # remove both roi layers from map
        existing_layer = self.map.find_layer("Selected ROIs")
        if existing_layer is not None:
            self.map.remove_layer(existing_layer)

        existing_layer = self.map.find_layer(ROI.LAYER_NAME)
        if existing_layer is not None:
            # Remove the layer from the map
            self.map.remove_layer(existing_layer)
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
        self.action = action
        self.geo_json = geo_json
        self.target = target
        if (
            self.draw_control.last_action == "created"
            and self.draw_control.last_draw["geometry"]["type"] == "Polygon"
        ):
            # validate the bbox size
            bbox_area = common.get_area(self.draw_control.last_draw["geometry"])
            try:
                Bounding_Box.check_bbox_size(bbox_area)
            except exceptions.BboxTooLargeError as bbox_too_big:
                map_UI.handle_bbox_error(str(bbox_too_big))
                self.remove_bbox()
            except exceptions.BboxTooSmallError as bbox_too_small:
                map_UI.handle_bbox_error(str(bbox_too_small))
                self.remove_bbox()
            else:
                # if a bbox already exists delete its layer from map
                if self.bbox is not None:
                    temp_bbox = Bounding_Box(self.draw_control.last_draw["geometry"])
                    self.remove_bbox()
                    self.bbox = temp_bbox
                    new_layer = self.create_layer(self.bbox, self.bbox.LAYER_NAME)
                    if new_layer is None:
                        print("Cannot add an empty bbox layer to the map.")
                    else:
                        self.map.add_layer(new_layer)
                else:
                    # Save new bbox to coastseg_map
                    self.bbox = Bounding_Box(self.draw_control.last_draw["geometry"])
        if self.draw_control.last_action == "deleted":
            self.remove_bbox()

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
            self.map.add_layer(new_layer)

    def load_gdf_on_map(self, layer_name: str, file=None):
        # assumes the geodataframe has a crs
        self.remove_layer_by_name(layer_name)
        if file is not None:
            gdf = gpd.read_file(file)
            new_gdf = gdf.to_crs("EPSG:4326")
            new_gdf.drop(
                new_gdf.columns.difference(["geometry"]), "columns", inplace=True
            )
            layer_geojson = json.loads(new_gdf.to_json())
            new_layer = GeoJSON(
                data=layer_geojson,
                name=layer_name,
                style={
                    "color": "#9933ff",
                    "fill_color": "#9933ff",
                    "opacity": 1,
                    "fillOpacity": 0.5,
                    "weight": 2,
                },
                hover_style={"color": "white", "fillOpacity": 0.7},
            )
            self.map.add_layer(new_layer)
            logger.info(f"Loaded geodataframe from file:\n{file}")

    def load_bbox_on_map(self, file=None):
        # remove old bbox if it exists
        self.draw_control.clear()
        existing_layer = self.map.find_layer(Bounding_Box.LAYER_NAME)
        if existing_layer is not None:
            self.map.remove_layer(existing_layer)
        logger.info(f"self.bbox: {self.bbox}")
        if file is not None:
            gdf = gpd.read_file(file)
            self.bbox = Bounding_Box(gdf)
            new_layer = self.create_layer(self.bbox, self.bbox.LAYER_NAME)
            if new_layer is None:
                print("Cannot add an empty bbox layer to the map.")
            else:
                self.map.add_layer(new_layer)
                print(f"Loaded the bbox from the file :\n{file}")
                logger.info(f"Loaded the bbox from the file :\n{file}")
        elif self.bbox is not None:
            if self.bbox.gdf.empty:
                logger.warning("self.bbox.gdf.empty")
                raise exceptions.Object_Not_Found("bbox")
            new_layer = self.create_layer(self.bbox, self.bbox.LAYER_NAME)
            if new_layer is None:
                logger.warning("Cannot add an empty bbox layer to the map.")
                print("Cannot add an empty bbox layer to the map.")
            else:
                self.map.add_layer(new_layer)
                logger.info("Loaded the bbox on map from self.bbox")

    def load_shoreline_on_map(self) -> None:
        """Adds shoreline within the drawn bbox onto the map"""
        # if no shorelines have been loaded onto the map before create a new shoreline
        if self.shoreline is None:
            if self.bbox is None:
                raise exceptions.Object_Not_Found("bounding box")
            elif self.bbox.gdf.empty:
                raise exceptions.Object_Not_Found("bounding box")
            # if a bounding box exists create a shoreline within it
            shoreline = Shoreline(self.bbox.gdf)
            if shoreline.gdf.empty:
                raise exceptions.Object_Not_Found("shorelines")
            # Save shoreline to coastseg_map
            self.shoreline = shoreline

        layer_name = "shoreline"
        # Replace old shoreline layer with new shoreline layer
        self.remove_layer_by_name(layer_name)
        # style and add the shoreline to the map
        new_layer = self.create_layer(self.shoreline, layer_name)
        if new_layer is None:
            print("Cannot add an empty shoreline layer to the map.")
        else:
            # add on_hover handler to update shoreline widget when user hovers over shoreline
            new_layer.on_hover(self.update_shoreline_html)
            self.map.add_layer(new_layer)
            logger.info(f"Add layer to map: {new_layer}")

    def create_layer(self, feature, layer_name: str):
        if feature.gdf.empty:
            logger.warning("Cannot add an empty geodataframe layer to the map.")
            print("Cannot add an empty geodataframe layer to the map.")
            return None
        layer_geojson = json.loads(feature.gdf.to_json())
        # convert layer to GeoJson and style it accordingly
        styled_layer = feature.style_layer(layer_geojson, layer_name)
        return styled_layer

    def load_rois_on_map(self, large_len=7500, small_len=5000, file: str = None):
        # Remove old ROI_layers
        if self.rois is not None or file is not None:
            # removes the old rois from the map
            if self.rois is not None:
                self.remove_all_rois(delete_rois=False)
            # read rois from geojson file and save them to the map
            if file is not None:
                self.remove_all_rois()
                logger.info(f"Loading ROIs from file {file}")
                gdf = gpd.read_file(file)
                self.rois = ROI(rois_gdf=gdf)

            logger.info(f"ROIs: {self.rois}")
            # Create new ROI layer
            self.ROI_layer = self.create_layer(self.rois, ROI.LAYER_NAME)
            if self.ROI_layer is None:
                logger.error("Cannot add an empty ROI layer to the map.")
                raise Exception("Cannot add an empty ROI layer to the map.")

            logger.info(f"ROI_layer: {self.ROI_layer}")
            # add on click handler to add the ROI to the selected geojson layer when its clicked
            self.ROI_layer.on_click(self.geojson_onclick_handler)
            self.map.add_layer(self.ROI_layer)
        else:
            # if no rois exist on the map then generate ROIs within the bounding box
            self.remove_all_rois()
            logger.info(f"No file provided. Generating ROIs")
            self.generate_ROIS_fishnet(large_len, small_len)

    def generate_ROIS_fishnet(self, large_len: float = 7500, small_len: float = 5000):
        """Generates series of overlapping ROIS along shoreline on map using fishnet method"""
        if self.bbox is None:
            raise exceptions.Object_Not_Found("bounding box")
        logger.info(f"bbox for ROIs: {self.bbox.gdf}")
        # If no shoreline exists on map then load one in
        if self.shoreline is None:
            self.load_shoreline_on_map()
        logger.info(f"self.shoreline used for ROIs:{ self.shoreline}")
        # create rois within the bbox that intersect shorelines
        self.rois = ROI(
            self.bbox.gdf,
            self.shoreline.gdf,
            square_len_lg=large_len,
            square_len_sm=small_len,
        )
        # create roi layer to add to map
        self.ROI_layer = self.create_layer(self.rois, ROI.LAYER_NAME)
        if self.ROI_layer is None:
            logger.error("Cannot add an empty ROI layer to the map.")
            raise Exception("Cannot add an empty ROI layer to the map.")
        # new rois have been generated so they have no downloaded data associated with them yet
        self.ROI_layer.on_click(self.geojson_onclick_handler)
        self.map.add_layer(self.ROI_layer)

    def geojson_onclick_handler(
        self, event: str = None, id: "NoneType" = None, properties: dict = None, **args
    ):
        """On click handler for when unselected geojson is clicked.

        Adds the geojson's id to the selected_set. Replaces current selected layer with a new one that includes
        the recently clicked geojson.

        Args:
            event (str, optional): event fired ('click'). Defaults to None.
            id (NoneType, optional):  Defaults to None.
            properties (dict, optional): geojson dict for clicked geojson. Defaults to None.
        """
        if properties is None:
            return
        logger.info(f"geojson_onclick_handler: properties : {properties}")
        logger.info(f"geojson_onclick_handler: ROI_id : {properties['id']}")

        # Add the id of the clicked ROI to selected_set
        ROI_id = str(properties["id"])
        self.selected_set.add(ROI_id)
        logger.info(f"Added ID to selected set: {self.selected_set}")
        if self.selected_ROI_layer is not None:
            self.map.remove_layer(self.selected_ROI_layer)

        self.selected_ROI_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(self.selected_set),
            name="Selected ROIs",
            hover_style={"fillColor": "blue", "fillOpacity": 0.1, "color": "aqua"},
        )
        logger.info(f"selected_ROI_layer: {self.selected_ROI_layer}")
        self.selected_ROI_layer.on_click(self.selected_onclick_handler)
        self.map.add_layer(self.selected_ROI_layer)

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
        if self.selected_ROI_layer is not None:
            self.map.remove_layer(self.selected_ROI_layer)
        # Recreate the selected layers without the layer that was removed
        self.selected_ROI_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(self.selected_set),
            name="Selected ROIs",
            hover_style={"fillColor": "blue", "fillOpacity": 0.1, "color": "aqua"},
        )
        # Recreate the onclick handler for the selected layers
        self.selected_ROI_layer.on_click(self.selected_onclick_handler)
        # Add selected layer to the map
        self.map.add_layer(self.selected_ROI_layer)


    def save_feature_to_file(
        self, feature: Union[Bounding_Box, Shoreline, Transects, ROI],
        feature_type:str=""
    ):
        try:
            exception_handler.can_feature_save_to_file(feature,feature_type)
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
        except Exception as err:
            exception_handler.handle_exception(err)
            
            
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
        # Copy only selected ROIs based on the ids in selected_set from ROI_geojson
        selected_rois["features"] = [
            feature
            for feature in self.ROI_layer.data["features"]
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
