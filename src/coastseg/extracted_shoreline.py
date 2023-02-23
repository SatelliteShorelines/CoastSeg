# Standard library imports
import logging
import os
import json
import copy
from glob import glob
from typing import Union


# Internal dependencies imports
from coastseg import exceptions
from coastseg import common

# External dependencies imports
import geopandas as gpd
import numpy as np
from ipyleaflet import GeoJSON
from matplotlib.pyplot import get_cmap
from matplotlib.colors import rgb2hex


from coastsat.SDS_tools import (
    remove_duplicates,
    remove_inaccurate_georef,
    output_to_gdf,
)

from coastsat.SDS_download import get_metadata
from coastsat.SDS_shoreline import extract_shorelines


logger = logging.getLogger(__name__)


def load_extracted_shoreline_from_files(dir_path):

    extracted_sl_gdf = None
    shoreline_settings = None
    extracted_shoreline_dict = None

    shoreline_glob = os.path.join(dir_path, "*shoreline*")
    for file in glob(shoreline_glob):
        if file.endswith(".geojson"):
            extracted_sl_gdf = common.read_gpd_file(file)
        if file.endswith(".json"):
            if "settings" in os.path.basename(file):
                shoreline_settings = common.from_file(file)
            if "dict" in os.path.basename(file):
                extracted_shoreline_dict = common.from_file(file)

    if extracted_sl_gdf is None or extracted_sl_gdf is None or extracted_sl_gdf is None:
        logger.warning(f"No extracted shorelnes could be loaded from {dir_path}")
        return None
    # load extracted shorelines from file
    extracted_shorelines = Extracted_Shoreline()
    extracted_shorelines = extracted_shorelines.load_extracted_shorelines(
        extracted_shoreline_dict, shoreline_settings, extracted_sl_gdf
    )
    logger.info(f"Loaded extracted shorelines from: {dir_path}")
    return extracted_shorelines


class Extracted_Shoreline:
    """Extracted_Shoreline: contains the extracted shorelines within a Region of Interest (ROI)"""

    LAYER_NAME = "extracted_shoreline"
    FILE_NAME = "extracted_shorelines.geojson"

    def __init__(
        self,
    ):
        # gdf: geodataframe containing extracted shoreline for ROI_id
        self.gdf = gpd.GeoDataFrame()
        # Use roi id to identify which ROI extracted shorelines derive from
        self.roi_id = ""
        # dictionary : dictionary of extracted shorelines
        # contains keys 'dates', 'shorelines', 'filename', 'cloud_cover', 'geoaccuracy', 'idx', 'MNDWI_threshold', 'satname'
        self.dictionary = {}
        # shoreline_settings: dictionary of settings used to extract shoreline
        self.shoreline_settings = {}

    def load_extracted_shorelines(
        self,
        extracted_shoreline_dict: dict = None,
        shoreline_settings: dict = None,
        extracted_shorelines_gdf: gpd.GeoDataFrame = None,
    ):

        if not isinstance(extracted_shoreline_dict, dict):
            raise ValueError(
                f"extracted_shoreline_dict must be dict. not {type(extracted_shoreline_dict)}"
            )
        if extracted_shoreline_dict == {}:
            raise ValueError("extracted_shoreline_dict cannot be empty.")

        if extracted_shorelines_gdf is not None:
            if not isinstance(extracted_shorelines_gdf, gpd.GeoDataFrame):
                raise ValueError(
                    f"extracted_shorelines_gdf must be valid geodataframe. not {type(extracted_shorelines_gdf)}"
                )
            if extracted_shorelines_gdf.empty:
                raise ValueError("extracted_shorelines_gdf cannot be empty.")
            self.gdf = extracted_shorelines_gdf

        if not isinstance(shoreline_settings, dict):
            raise ValueError(
                f"shoreline_settings must be dict. not {type(shoreline_settings)}"
            )
        if shoreline_settings == {}:
            raise ValueError("shoreline_settings cannot be empty.")

        # dictionary : dictionary of extracted shorelines
        self.dictionary = extracted_shoreline_dict
        # shoreline_settings: dictionary of settings used to extract shoreline
        self.shoreline_settings = shoreline_settings
        # Use roi id to identify which ROI extracted shorelines derive from
        self.roi_id = shoreline_settings["inputs"]["roi_id"]
        return self

    def create_extracted_shorlines(
        self,
        roi_id: str = None,
        shoreline: gpd.GeoDataFrame = None,
        roi_settings: dict = None,
        settings: dict = None,
    ):
        if not isinstance(roi_id, str):
            raise ValueError(f"ROI id must be string. not {type(roi_id)}")

        if not isinstance(shoreline, gpd.GeoDataFrame):
            raise ValueError(
                f"shoreline must be valid geodataframe. not {type(shoreline)}"
            )
        if shoreline.empty:
            raise ValueError("shoreline cannot be empty.")

        if not isinstance(roi_settings, dict):
            raise ValueError(f"roi_settings must be dict. not {type(roi_settings)}")
        if roi_settings == {}:
            raise ValueError("roi_settings cannot be empty.")

        if not isinstance(settings, dict):
            raise ValueError(f"settings must be dict. not {type(settings)}")
        if settings == {}:
            raise ValueError("settings cannot be empty.")

        logger.info(f"Extracting shorelines for ROI id{roi_id}")
        self.dictionary = self.extract_shorelines(
            shoreline,
            roi_settings,
            settings,
        )

        if is_list_empty(self.dictionary["shorelines"]):
            logger.warning(f"No extracted shorelines for ROI {roi_id}")
            raise exceptions.No_Extracted_Shoreline(roi_id)

        map_crs = "EPSG:4326"
        # extracted shorelines have map crs so they can be displayed on the map
        self.gdf = self.create_geodataframe(
            self.shoreline_settings["output_epsg"], output_crs=map_crs
        )
        return self

    def extract_shorelines(
        self,
        shoreline_gdf: gpd.geodataframe,
        roi_settings: dict,
        settings: dict,
    ) -> dict:
        """Returns a dictionary containing the extracted shorelines for roi specified by rois_gdf"""
        # project shorelines's crs from map's crs to output crs given in settings
        map_crs = 4326
        reference_shoreline = get_reference_shoreline(
            shoreline_gdf, settings["output_epsg"]
        )
        # Add reference shoreline to shoreline_settings
        self.shoreline_settings = self.create_shoreline_settings(
            settings, roi_settings, reference_shoreline
        )
        # gets metadata used to extract shorelines
        metadata = get_metadata(self.shoreline_settings["inputs"])
        logger.info(f"metadata: {metadata}")
        # extract shorelines from ROI
        extracted_shorelines = extract_shorelines(metadata, self.shoreline_settings)
        logger.info(f"extracted_shoreline_dict: {extracted_shorelines}")
        # postprocessing by removing duplicates and removing in inaccurate georeferencing (set threshold to 10 m)
        extracted_shorelines = remove_duplicates(
            extracted_shorelines
        )  # removes duplicates (images taken on the same date by the same satellite)
        extracted_shorelines = remove_inaccurate_georef(
            extracted_shorelines, 10
        )  # remove inaccurate georeferencing (set threshold to 10 m)
        logger.info(
            f"after remove_inaccurate_georef : extracted_shoreline_dict: {extracted_shorelines}"
        )
        return extracted_shorelines

    def create_shoreline_settings(
        self,
        settings: dict,
        roi_settings: dict,
        reference_shoreline: dict,
    ) -> None:
        """sets self.shoreline_settings to dictionary containing settings, reference_shoreline
        and roi_settings

        shoreline_settings=
        {
            "reference_shoreline":reference_shoreline,
            "inputs": roi_settings,
            "adjust_detection": False,
            "check_detection": False,
            ...
            rest of items from settings
        }

        Args:
            settings (dict): map settings
            roi_settings (dict): settings of the roi. Must include 'dates'
            reference_shoreline (dict): reference shoreline
        """
        # deepcopy settings to shoreline_settings so it can be modified
        # shoreline_settings = copy.deepcopy(settings)
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
        logger.info(f"settings used to create shoreline settings: {settings}")
        shoreline_settings = common.filter_dict_by_keys(settings, keys=shoreline_keys)
        logger.info(f"Loading shoreline_settings: {shoreline_settings}")
        # Add reference shoreline and shoreline buffer distance for this specific ROI
        shoreline_settings["reference_shoreline"] = reference_shoreline
        # disable adjusting shorelines manually in shoreline_settings
        shoreline_settings["adjust_detection"] = False
        # disable adjusting shorelines manually in shoreline_settings
        shoreline_settings["check_detection"] = False
        shoreline_settings["save_figure"] = True
        # copy roi_setting for this specific roi
        shoreline_settings["inputs"] = roi_settings
        logger.info(f"shoreline_settings: {shoreline_settings}")
        return shoreline_settings

    def create_geodataframe(
        self, input_crs: str, output_crs: str = None
    ) -> gpd.GeoDataFrame:
        """Creates a geodataframe with the crs specified by input_crs. Converts geodataframe crs
        to output_crs if provided.
        Args:
            input_crs (str ): coordinate reference system string. Format 'EPSG:4326'.
            output_crs (str, optional): coordinate reference system string. Defaults to None.
        Returns:
            gpd.GeoDataFrame: geodataframe with columns = ['geometery','date','satname','geoaccuracy','cloud_cover']
            converted to output_crs if provided otherwise geodataframe's crs will be
            input_crs
        """
        extract_shoreline_gdf = output_to_gdf(self.dictionary, "lines")
        extract_shoreline_gdf.crs = input_crs
        if output_crs is not None:
            extract_shoreline_gdf = extract_shoreline_gdf.to_crs(output_crs)
        return extract_shoreline_gdf

    def save_to_file(
        self,
        sitename: str,
        filepath: str,
    ):
        """save_to_file Save geodataframe to location specified by filepath into directory
        specified by sitename

        Args:
            sitename (str): directory of roi shoreline was extracted from
            filepath (str): full path to directory containing ROIs
        """
        savepath = os.path.join(filepath, sitename, Extracted_Shoreline.FILE_NAME)
        logger.info(
            f"Saving shoreline to file: {savepath}.\n Extracted Shoreline: {self.gdf}"
        )
        print(f"Saving shoreline to file: {savepath}")
        self.gdf.to_file(
            savepath,
            driver="GeoJSON",
            encoding="utf-8",
        )

    def to_file(
        self, filepath: str, filename: str, data: Union[gpd.GeoDataFrame, dict]
    ):
        """Save geopandas dataframe to file, or save data to file with common.to_file().

        Args:
            filepath (str): The directory where the file should be saved.
            filename (str): The name of the file to be saved.
            data (Any): The data to be saved to file.

        Raises:
            ValueError: Raised when data is not a geopandas dataframe and cannot be saved with common.to_file().

        """
        file_location = os.path.abspath(os.path.join(filepath, filename))

        if isinstance(data, gpd.GeoDataFrame):
            data.to_file(
                file_location,
                driver="GeoJSON",
                encoding="utf-8",
            )
        elif isinstance(data, dict):
            if data != {}:
                common.to_file(data, file_location)

    def style_layer(
        self, geojson: dict, layer_name: str, color: str
    ) -> "ipyleaflet.GeoJSON":
        """Return styled GeoJson object with layer name

        Args:
            geojson (dict): geojson dictionary to be styled
            layer_name(str): name of the GeoJSON layer
            color(str): hex code or name of color render shorelines

        Returns:
            "ipyleaflet.GeoJSON": shoreline as GeoJSON layer styled with yellow dashes
        """
        assert geojson != {}, "ERROR.\n Empty geojson cannot be drawn onto  map"
        return GeoJSON(
            data=geojson,
            name=layer_name,
            style={
                "color": color,
                "opacity": 1,
                "weight": 4,
            },
            hover_style={"color": "red", "dashArray": "4", "fillOpacity": 0.7},
        )

    def get_layer_names(self) -> list:
        """returns a list of strings of format: 'ID_<roi_id>_<date>'
        ex.'ID21_2018-12-31 19:03:17'
        """
        roi_id = self.roi_id
        layer_names = [
            "ID" + roi_id + "_" + date_str for date_str in self.gdf["date"].to_list()
        ]
        return layer_names

    def get_styled_layers(self) -> list:
        # load extracted shorelines onto map
        map_crs = 4326
        # convert to map crs and turn in json dict
        features_json = json.loads(self.gdf.to_crs(map_crs).to_json())
        layer_colors = get_colors(len(features_json["features"]))
        layer_names = self.get_layer_names()
        layers = []
        for idx, feature in enumerate(features_json["features"]):
            geojson_layer = self.style_layer(
                feature, layer_names[idx], layer_colors[idx]
            )
            layers.append(geojson_layer)
        return layers


def get_reference_shoreline(
    shoreline_gdf: gpd.geodataframe, output_crs: str
) -> np.ndarray:
    # project shorelines's espg from map's espg to output espg given in settings
    reprojected_shorlines = shoreline_gdf.to_crs(output_crs)
    logger.info(f"reprojected_shorlines.crs: {reprojected_shorlines.crs}")
    logger.info(f"reprojected_shorlines: {reprojected_shorlines}")
    # convert shoreline_in_roi gdf to coastsat compatible format np.array([[lat,lon,0],[lat,lon,0]...])
    shorelines = make_coastsat_compatible(reprojected_shorlines)
    # shorelines = [([lat,lon],[lat,lon],[lat,lon]),([lat,lon],[lat,lon],[lat,lon])...]
    # Stack all the tuples into a single list of n rows X 2 columns
    shorelines = np.vstack(shorelines)
    # Add third column of 0s to represent mean sea level
    shorelines = np.insert(shorelines, 2, np.zeros(len(shorelines)), axis=1)
    return shorelines


def get_colors(length: int) -> list:
    # returns a list of color hex codes as long as length
    cmap = get_cmap("plasma", length)
    cmap_list = [rgb2hex(i) for i in cmap.colors]
    return cmap_list


def make_coastsat_compatible(feature: gpd.geodataframe) -> list:
    """Return the feature as an np.array in the form:
        [([lat,lon],[lat,lon],[lat,lon]),([lat,lon],[lat,lon],[lat,lon])...])
    Args:
        feature (gpd.geodataframe): clipped portion of shoreline within a roi
    Returns:
        list: shorelines in form:
            [([lat,lon],[lat,lon],[lat,lon]),([lat,lon],[lat,lon],[lat,lon])...])
    """
    features = []
    # Use explode to break multilinestrings in linestrings
    feature_exploded = feature.explode()
    # For each linestring portion of feature convert to lat,lon tuples
    lat_lng = feature_exploded.apply(
        lambda row: tuple(np.array(row.geometry.coords).tolist()), axis=1
    )
    features = list(lat_lng)
    return features


def is_list_empty(main_list: list) -> bool:
    all_empty = True
    for np_array in main_list:
        if len(np_array) != 0:
            all_empty = False
    return all_empty
