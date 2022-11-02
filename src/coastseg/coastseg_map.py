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

import geopandas as gpd
import numpy as np

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
        logger.info(f"load_configs:: config_file: {config_file}")
        config_path = os.path.join(dir_path, config_file)
        logger.info(f"load_configs:: Loaded json config file from {config_path}")
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
            logger.info(f"load_gdf_config:: bbox : {bbox}")
            self.bbox = bbox
            logger.info(f"load_gdf_config:: self.bbox : {self.bbox}")
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
        logger.info(f"load_json_config::Loaded settings from file: {self.settings}")
        # replace roi_settings for each ROI with contents of json_data
        roi_settings = {}
        for roi_id in json_data["roi_ids"]:
            roi_settings[str(roi_id)] = json_data[roi_id]
        # Save input dictionaries for all ROIs
        self.rois.roi_settings = roi_settings
        logger.info(f"load_json_config:: roi_settings: {roi_settings}")

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

        logger.info(f"save_config::self.rois.roi_settings: {self.rois.roi_settings}")
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
        logger.info(f"save_config :: config_json: {config_json} ")

        # get currently selected rois selected
        roi_ids = config_json["roi_ids"]
        selected_rois = self.get_selected_rois(roi_ids)
        logger.info(f"save_config :: selected_rois: {selected_rois} ")

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
        logger.info(f"saved gdf config :: config_gdf: {config_gdf} ")

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
                f"extract_all_shorelines::No inputs settings found. Please click download ROIs first. self.rois: {self.rois}"
            )
            raise Exception(
                "No inputs settings found. Please click download ROIs first or upload configs"
            )
        if self.settings is None:
            logger.error(
                "extract_all_shorelines::No settings found. Please load settings"
            )
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

        settings = self.settings
        # get selected ROIs on map and extract shoreline for each of them
        roi_ids = list(self.selected_set)
        if common.were_rois_downloaded(self.rois.roi_settings, roi_ids) == False:
            logger.error(
                f"extract_all_shorelines:: not all rois were downloaded{self.rois.roi_settings}"
            )
            raise Exception(
                f"Some ROIs filepaths did not exist on your computer. Try downloading the rois to overwrite the filepath ."
            )

        logger.info(f"extract_all_shorelines:: roi_ids: {roi_ids}")
        print(f"Extracting shorelines from ROIs: {roi_ids}")
        selected_rois_gdf = self.get_selected_rois(roi_ids)
        # if none of the ids in roi_settings are in the ROI geodataframe raise an Exception
        if selected_rois_gdf.empty:
            logger.error("extract_all_shorelines::No ROIs selected")
            raise Exception("No ROIs selected")

        # extracted_shoreline_dict: holds all extracted shoreline geodataframes associated with each ROI
        extracted_shoreline_dict = {}
        for id in tqdm(roi_ids, desc="Extracting Shorelines"):
            try:
                print(f"Extracting shorelines from ROI with the id:{id}")
                inputs = self.rois.roi_settings[id]
                # raise exception if no files were download to extract shorelines from
                common.does_filepath_exist(inputs)
                extracted_shoreline = self.extract_shorelines_from_roi(
                    selected_rois_gdf, self.shoreline.gdf, inputs, settings, id=id
                )
                logger.info(
                    f"extract_all_shorelines : ROI: {id} extracted_shoreline: {extracted_shoreline}"
                )
                extracted_shoreline_dict[id] = extracted_shoreline
                logger.info(
                    f"extract_all_shorelines : extracted_shoreline_dict[{id}]: {extracted_shoreline_dict[id]}"
                )
                # load extracted shorelines onto map
                if len(extracted_shoreline_dict[id]["shorelines"]) > 0:
                    shorelines = extracted_shoreline_dict[id]["shorelines"]
                    logger.info(f"shorelines: {shorelines}")
                    # if any of shorelines np.arrays are not empty load them onto map
                    if any(list(map(lambda x: len(x) > 0, shorelines))):
                        print(
                            f"Successfully extracted shoreline for ROI: {id}. Loading it onto the map"
                        )
                        extracted_sl_filename = f"shorelines_{id}.geojson"
                        sitename = inputs["sitename"]
                        filepath = inputs["filepath"]
                        extracted_sl_filepath = os.path.join(
                            filepath, sitename, extracted_sl_filename
                        )
                        self.load_gdf_on_map(
                            extracted_sl_filename, extracted_sl_filepath
                        )
            except exceptions.Id_Not_Found as id_error:
                logger.warning(f"exceptions.Id_Not_Found {id_error}")
                print(f"exceptions.Id_Not_Found:{id_error}. \n Skipping to next ROI")

        # Save all the extracted_shorelines to ROI
        self.rois.update_extracted_shorelines(extracted_shoreline_dict)
        logger.info(
            f"extract_all_shorelines : self.rois.extracted_shorelines {self.rois.extracted_shorelines}"
        )

    def get_selected_rois(self, roi_ids: list) -> gpd.GeoDataFrame:
        """Returns a geodataframe of all rois selected by roi_ids

        Args:
            roi_ids (list[str]): ids of ROIs

        Returns:
            gpd.GeoDataFrame:  geodataframe of all rois selected by the roi_ids
        """
        selected_rois_gdf = self.rois.gdf[self.rois.gdf["id"].isin(roi_ids)]
        return selected_rois_gdf

    def extract_shorelines_from_roi(
        self,
        rois_gdf: gpd.GeoDataFrame,
        shorelines_gdf: gpd.geodataframe,
        inputs: dict,
        settings: dict,
        id: int = None,
    ) -> dict:
        """Returns a dictionary containing the extracted shorelines for roi specified by rois_gdf"""
        single_roi = common.extract_roi_by_id(rois_gdf, id)

        # Clip shoreline to specific roi
        shoreline_in_roi = gpd.clip(shorelines_gdf, single_roi)
        # if no shorelines exist within the roi return an empty dictionary
        if shoreline_in_roi.empty:
            logger.warn(f"No shorelines could be clipped to ROI: {id}")
            return {}
        logger.info(f"clipped shorelines{shoreline_in_roi}")
        # convert shoreline_in_roi gdf to coastsat compatible format np.array([[lat,lon,0],[lat,lon,0]...])
        shorelines = common.make_coastsat_compatible(shoreline_in_roi)

        logger.info(f"coastsat shorelines{shorelines}")
        # project shorelines's espg from map's espg to output espg given in settings
        map_espg = 4326
        s_proj = common.convert_espg(map_espg, settings["output_epsg"], shorelines)
        logger.info(f"extract_shorelines_from_roi::s_proj: {s_proj}")
        # deepcopy settings to shoreline_settings so it can be modified
        shoreline_settings = copy.deepcopy(settings)
        # Add reference shoreline and shoreline buffer distance for this specific ROI
        shoreline_settings["reference_shoreline"] = s_proj

        # DO NOT have user adjust shorelines manually
        shoreline_settings["adjust_detection"] = False
        # DO NOT have user check for valid shorelines
        shoreline_settings["check_detection"] = False

        # copy inputs for this specific roi
        shoreline_settings["inputs"] = inputs
        logger.info(f"shoreline_settings: {shoreline_settings}")

        # get the metadata used to extract the shoreline
        metadata = SDS_download.get_metadata(inputs)
        logger.info(f"extract_shorelines_from_roi::metadata: {metadata}")
        extracted_shoreline_dict = SDS_shoreline.extract_shorelines(
            metadata, shoreline_settings
        )
        logger.info(f"extracted_shoreline_dict: {extracted_shoreline_dict}")

        # postprocessing by removing duplicates and removing in inaccurate georeferencing (set threshold to 10 m)
        extracted_shoreline_dict = SDS_tools.remove_duplicates(
            extracted_shoreline_dict
        )  # removes duplicates (images taken on the same date by the same satellite)
        # logger.info(f"after remove_duplicates : extracted_shoreline_dict: {extracted_shoreline_dict}")
        extracted_shoreline_dict = SDS_tools.remove_inaccurate_georef(
            extracted_shoreline_dict, 10
        )  # remove inaccurate georeferencing (set threshold to 10 m)
        logger.info(
            f"after remove_inaccurate_georef : extracted_shoreline_dict: {extracted_shoreline_dict}"
        )

        logger.info(f"extracted_shoreline_dict: {extracted_shoreline_dict}")
        geomtype = "lines"  # choose 'points' or 'lines' for the layer geometry

        extract_shoreline_gdf = SDS_tools.output_to_gdf(
            extracted_shoreline_dict, geomtype
        )
        logger.info(f"extract_shoreline_gdf: {extract_shoreline_gdf}")
        # if extracted shorelines is None then return an empty geodataframe
        if extract_shoreline_gdf is None:
            logger.warn(f"No shorelines could be extracted for for ROI {id}")
            print(f"No shorelines could be extracted for for ROI {id}")
        else:
            extract_shoreline_gdf.crs = "epsg:" + str(settings["output_epsg"])
            # save GEOJSON layer to file
            sitename = inputs["sitename"]
            filepath = inputs["filepath"]
            logger.info(
                f"Saving shoreline to file{filepath}. \n Extracted Shoreline: {extract_shoreline_gdf}"
            )
            extract_shoreline_gdf.to_file(
                os.path.join(filepath, sitename, f"shorelines_{id}.geojson"),
                driver="GeoJSON",
                encoding="utf-8",
            )

        logger.info(f"Returning extracted shoreline {extracted_shoreline_dict}")
        return extracted_shoreline_dict

    def get_intersecting_transects(
        self, rois_gdf: gpd.geodataframe, transect_data: gpd.geodataframe, id: str
    ) -> gpd.geodataframe:
        """Returns a transects that intersect with the roi with id provided
        Args:
            rois_gdf (gpd.geodataframe): rois with geometery, ids and more
            transect_data (gpd.geodataframe): transects geomemtry
            id (str): id of roi

        Returns:
            gpd.geodataframe: _description_
        """
        poly_roi = common.convert_gdf_to_polygon(rois_gdf, id)
        transect_mask = transect_data.intersects(poly_roi, align=False)
        return transect_data[transect_mask]

    def compute_transects_from_roi(
        self, roi_id: int, inProj: Proj, outProj: Proj, settings: dict
    ) -> dict:
        cross_distance = 0
        try:
            single_roi = common.extract_roi_by_id(self.rois.gdf, roi_id)

            # if no extracted shoreline exist for ROI's id return cross distance = 0
            extracted_shoreline = self.rois.extracted_shorelines[str(roi_id)]
            if extracted_shoreline == {}:
                return 0

            # geodataframe of transects intersecting this ROI
            transect_in_roi = self.get_intersecting_transects(
                single_roi, self.transects.gdf, roi_id
            )
            logger.info(
                f"compute_transects_from_roi() :transect_in_roi: {transect_in_roi}"
            )
            # @todo Find which transects intersect extracted shorelines

            # Do not compute cross distances of transects if no shoreline exists
            if not common.is_shoreline_present(
                self.rois.extracted_shorelines, str(roi_id)
            ):
                raise exceptions.No_Extracted_Shoreline(
                    str(roi_id), "No extracted shoreline at this roi"
                )
            else:
                print("\nExtracted shoreline present at ROI: ", roi_id)
                logger.info(
                    f"compute_transects_from_roi() :Shoreline present at ROI: {roi_id}"
                )
                # convert transects to lan,lon tuples
                transects_coords = []
                for k in transect_in_roi["geometry"].keys():
                    transects_coords.append(
                        tuple(np.array(transect_in_roi["geometry"][k]).tolist())
                    )
                logger.info(
                    f"compute_transects_from_roi() : transects_coords: {transects_coords}"
                )

                # convert to dict of numpy arrays of start and end points
                transects = {}
                for counter, i in enumerate(transects_coords):
                    x0, y0 = transform(inProj, outProj, i[0][0], i[0][1])
                    x1, y1 = transform(inProj, outProj, i[1][0], i[1][1])
                    transects["NA" + str(counter)] = np.array([[x0, y0], [x1, y1]])
                logger.info(f"compute_transects_from_roi():: transects: {transects}")

                # cross_distance:  along-shore distance over which to consider shoreline points to compute median intersection (robust to outliers)
                cross_distance = SDS_transects.compute_intersection(
                    extracted_shoreline, transects, settings
                )
                print(f"\n\ntransects cross_distance:{cross_distance}")
        except Exception as err:
            logger.exception(f"Error from Compute Transects: {err}")
        return cross_distance

    def compute_transects(self) -> dict:
        """Returns a dict of cross distances for each roi's transects

        Args:
            selected_rois (dict): rois selected by the user. Must contain the following fields:
                {'features': [
                    'id': (str) roi_id
                    ''geometry':{
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
            logger.error("compute_transects:: No ROIs have been loaded")
            raise Exception("No ROIs have been loaded")
        if self.transects is None:
            logger.error("compute_transects:: No transects were loaded onto the map.")
            raise Exception("No transects were loaded onto the map.")
        if self.rois.extracted_shorelines == {}:
            logger.error(
                "compute_transects:: No shorelines have been extracted. Extract shorelines first."
            )
            raise Exception(
                "No shorelines have been extracted. Extract shorelines first."
            )
        if self.settings is None:
            logger.error("compute_transects:: No settings have been loaded")
            raise Exception("No settings have been loaded")
        if "along_dist" not in self.settings.keys():
            raise Exception("Missing key 'along_dist' in settings")

        settings = self.settings
        extracted_shoreline_ids = set(self.rois.extracted_shorelines.keys())
        logger.info(
            f"compute_transects:: self.rois.extracted_shorelines.keys(): {list(self.rois.extracted_shorelines.keys())}"
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

        # Input projection is the map project and user selected output projection
        inProj = Proj(init="epsg:4326")
        outProj = Proj(init="epsg:" + str(settings["output_epsg"]))

        # Save cross distances for each set of transects intersecting each extracted shoreline for each ROI
        cross_distance_transects = {}
        for roi_id in tqdm(roi_ids, desc="Computing Cross Distance Transects"):
            cross_distance = 0
            try:
                cross_distance = self.compute_transects_from_roi(
                    roi_id, inProj, outProj, settings
                )
                if cross_distance == 0:
                    print(f"No transects existed for ROI {roi_id}")
                cross_distance_transects[roi_id] = cross_distance
                # print(f"\ncross_distance_transects[{roi_id}]: {cross_distance_transects[roi_id]}")
                logger.info(
                    f"\ncompute_transects:: cross_distance_transects[{roi_id}]: {cross_distance_transects[roi_id]}"
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

    def load_transects_on_map(self) -> None:
        """Adds transects within the drawn bbox onto the map"""
        # if no transects have been loaded onto the map before create a new transects
        if self.transects is None:
            if self.bbox is None:
                logger.error(f"load_transects_on_map:: self.bbox is None")
                raise Exception(
                    "Cannot load transects on map because bbox does not exist"
                )
            elif self.bbox.gdf.empty:
                logger.error(f"load_transects_on_map:: self.bbox is empty")
                raise Exception("Cannot load transects on map because bbox is empty")
            # if a bounding box exists create a shoreline within it
            transects = Transects(self.bbox.gdf)
            # Save transect to coastseg_map
            self.transects = transects
            if transects.gdf.empty:
                raise exceptions.Object_Not_Found(
                    "Transects Not Found in this region. Draw a new bounding box"
                )

        layer_name = "transects"
        # Replace old transect layer with new transect layer
        self.remove_layer_by_name(layer_name)
        # style and add the transect to the map
        new_layer = self.create_layer(self.transects, layer_name)
        if new_layer is None:
            print("Cannot add an empty transects layer to the map.")
        else:

            self.map.add_layer(new_layer)
            logger.info(f"Added layer to map: {new_layer}")

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
            Bounding_Box.check_bbox_size(bbox_area)
            # if a bbox already exists delete it
            self.bbox = None
            # Save new bbox to coastseg_map
            self.bbox = Bounding_Box(self.draw_control.last_draw["geometry"])
        if self.draw_control.last_action == "deleted":
            self.remove_bbox()

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
            logger.info(
                f"load_gdf_on_map::Loaded the geodataframe from the file :\n{file}"
            )
            self.extracted_shoreline_layers.append(layer_name)

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
                logger.info(
                    f"load_bbox_on_map:: Loaded the bbox from the file :\n{file}"
                )
        elif self.bbox is not None:
            if self.bbox.gdf.empty:
                logger.warning("load_bbox_on_map:: self.bbox.gdf.empty")
                raise exceptions.Object_Not_Found("bbox")
            new_layer = self.create_layer(self.bbox, self.bbox.LAYER_NAME)
            if new_layer is None:
                logger.warning("Cannot add an empty bbox layer to the map.")
                print("Cannot add an empty bbox layer to the map.")
            else:
                self.map.add_layer(new_layer)
                logger.info("load_bbox_on_map:: Loaded the bbox on map from self.bbox")

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
        logger.info(f"type(event): {type(event)}")
        logger.info(f"event: {event}")
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

    def check_selected_set(self):
        if self.selected_set is None:
            raise Exception(
                "Must select at least 1 ROI first before you can save ROIs."
            )
        if len(self.selected_set) == 0:
            raise Exception(
                "Must select at least 1 ROI first before you can save ROIs."
            )

    def save_feature_to_file(
        self, feature: Union[Bounding_Box, Shoreline, Transects, ROI]
    ):
        if feature is None:
            raise exceptions.Object_Not_Found(feature.LAYER_NAME)
        elif isinstance(feature, ROI):
            # check if any ROIs were selected by making sure the selected set isn't empty
            self.check_selected_set()
            # @ todo replace this with get selected rois
            logger.info(
                f"feature: {feature.gdf[feature.gdf['id'].isin(self.selected_set)]}"
            )
            feature.gdf[feature.gdf["id"].isin(self.selected_set)].to_file(
                feature.filename, driver="GeoJSON"
            )
        else:
            logger.info(f"type( {feature})")
            feature.gdf.to_file(feature.filename, driver="GeoJSON")
        print(f"Save {feature.LAYER_NAME} to {feature.filename}")
        logger.info(f"Save {feature.LAYER_NAME} to {feature.filename}")

    def convert_selected_set_to_geojson(self, selected_set: set) -> dict:
        """Returns a geojson dict containing a FeatureCollection for all the geojson objects in the
        selected_set
        Args:
            selected_set (set): ids of the selected geojson

        Returns:
           dict: geojson dict containing FeatureCollection for all geojson objects in selected_set
        """
        # create a new geojson dictionary to hold the selected ROIs
        selected_rois = {"type": "FeatureCollection", "features": []}
        # Copy only the selected ROIs based on the ids in selected_set from ROI_geojson
        selected_rois["features"] = [
            feature
            for feature in self.ROI_layer.data["features"]
            if feature["properties"]["id"] in selected_set
        ]
        # Modify change the style of each ROI in the new selected
        # selected rois will appear blue and unselected rois will appear grey
        for feature in selected_rois["features"]:
            feature["properties"]["style"] = {
                "color": "blue",
                "weight": 2,
                "fillColor": "blue",
                "fillOpacity": 0.1,
            }
        return selected_rois
