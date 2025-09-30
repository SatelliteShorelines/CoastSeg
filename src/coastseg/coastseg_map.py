import glob
import json
import logging
import math

# Standard library imports
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Collection, Dict, Iterable, List, Optional, Tuple, Union

# Third-party imports
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import traitlets
from coastsat import SDS_download
from coastsat.SDS_download import get_metadata
from ipyleaflet import DrawControl, GeoJSON, LayersControl, WidgetControl
from ipywidgets import HTML, HBox, Layout
from leafmap import Map
from shapely.geometry import shape
from tqdm.auto import tqdm

# Internal/Local imports: modules
from coastseg import (
    common,
    core_utilities,
    exception_handler,
    exceptions,
    extracted_shoreline,
    factory,
    file_utilities,
    geodata_processing,
    tide_correction,
)

# Internal/Local imports: specific classes/functions
from coastseg.bbox import Bounding_Box
from coastseg.downloads import count_images_in_ee_collection
from coastseg.feature import Feature
from coastseg.roi import ROI
from coastseg.shoreline import Shoreline
from coastseg.shoreline_extraction_area import Shoreline_Extraction_Area
from coastseg.transects import Transects

from . import __version__

logger = logging.getLogger(__name__)

SELECTED_LAYER_NAME = "Selected Shorelines"

__all__ = ["IDContainer", "ExtractShorelinesContainer", "CoastSeg_Map"]


def normalize_path(path: Optional[Union[str, Path]]) -> Optional[str]:
    """
    Normalizes optional path to string.

    Args:
        path: Input path as str, Path, or None.

    Returns:
        Normalized path as str or None.
    """
    if path is None:
        return None
    return str(path)


class IDContainer(traitlets.HasTraits):
    ids = traitlets.List(trait=traitlets.Unicode())


class ExtractShorelinesContainer(traitlets.HasTraits):
    """A container class for managing shorelines extraction.

    This class provides a container for managing shorelines extraction.
    It holds lists of shorelines that can be loaded, shorelines that will be thrown away, and
    ROI (Region of Interest) IDs that have extracted shorelines.

    Args:
        traitlets (type): The traitlets module for defining traits.

    Attributes:
        load_list (List[str]): A list of shorelines that can be loaded.
        trash_list (List[str]): A list of shorelines that will be thrown away.
        roi_ids_list (List[str]): A list of ROI IDs that have extracted shorelines.
    """

    # list of shorelines that can be loaded
    load_list = traitlets.List(trait=traitlets.Unicode())
    # list of shorelines that will be thrown away
    trash_list = traitlets.List(trait=traitlets.Unicode())
    # list of roi ids that have extracted shorelines
    roi_ids_list = traitlets.List(trait=traitlets.Unicode())

    def __init__(
        self, load_list_widget=None, trash_list_widget=None, roi_list_widget=None
    ):
        super().__init__()
        if load_list_widget:
            self.link_load_list(load_list_widget)
        if trash_list_widget:
            self.link_trash_list(trash_list_widget)
        # Link the widgets and the traits
        if roi_list_widget:
            self.link_roi_list(roi_list_widget)

    def link_load_list(self, widget):
        if hasattr(widget, "options"):
            traitlets.dlink((self, "load_list"), (widget, "options"))

    def link_trash_list(self, widget):
        if hasattr(widget, "options"):
            traitlets.dlink((self, "trash_list"), (widget, "options"))

    def link_roi_list(self, widget):
        if hasattr(widget, "options"):
            traitlets.dlink((self, "roi_ids_list"), (widget, "options"))

    def clear(self):
        self.load_list = []
        self.trash_list = []
        self.roi_ids_list = []


def find_shorelines_directory(path: str, roi_id: str) -> Optional[str]:
    """
    Finds directory with extracted shorelines GeoJSON.

    Args:
        path: Path to search.
        roi_id: ROI ID to match.

    Returns:
        Path to directory or None.
    """
    # List the contents of the specified path
    contents = os.listdir(path)

    # Check for extracted shorelines geojson file in the specified path
    extracted_shorelines_file = [
        file
        for file in contents
        if "extracted_shorelines" in file and file.endswith(".geojson")
    ]
    if extracted_shorelines_file:
        return path

    # If the file is not found, check for a directory with the ROI ID
    roi_directory = [
        directory
        for directory in contents
        if os.path.isdir(os.path.join(path, directory)) and roi_id in directory
    ]
    if roi_directory:
        roi_path = os.path.join(path, roi_directory[0])
        roi_contents = os.listdir(roi_path)
        extracted_shorelines_file = [
            file
            for file in roi_contents
            if "extracted_shorelines" in file and file.endswith(".geojson")
        ]
        if extracted_shorelines_file:
            return roi_path

    return None


def delete_extracted_shorelines_files(session_path: str, selected_items: List) -> None:
    """
    Deletes selected shorelines from session files.

    Args:
        session_path: Path to session directory.
        selected_items: List of shoreline items to delete. In the format "satname_dates", e.g., "S2_2023-12-09".

    Returns:
        None

    Removes the extracted shorelines from the following files:
        - extracted_shorelines_lines.geojson
        - extracted_shorelines_points.geojson
        - extracted_shorelines_dict.json
        - transect_time_series.csv (generated by older versions of CoastSeg; new name is 'raw_transect_time_series.csv')
        - raw_transect_time_series.csv
        - transect_time_series_tidally_corrected.csv (generated by older versions of CoastSeg; new name is 'tidally_corrected_transect_time_series_merged.csv')
        - tidally_corrected_transect_time_series_merged.csv
        As well as all the csv files matching the following patterns:
        - _timeseries_raw.csv
        - _timeseries_tidally_corrected.csv
    """
    # Extract dates and satellite names from the selected items
    dates_list, sat_list = common.extract_dates_and_sats(selected_items)

    # delete the extracted shorelines from all extracted shoreline geojson files
    filenames = [
        "extracted_shorelines_lines.geojson",
        "extracted_shorelines_points.geojson",
        "raw_transect_time_series_vectors.geojson",
        "tidally_corrected_transect_time_series_vectors.geojson",
        "raw_transect_time_series_points.geojson",
        "tidally_corrected_transect_time_series_points.geojson",
    ]
    filepaths = [
        os.path.join(session_path, filename)
        for filename in filenames
        if os.path.isfile(os.path.join(session_path, filename))
    ]

    # the date column must be a dateime formatted in "%Y-%m-%d %H:%M:%S" without a timezone
    formatted_dates = [date.replace(tzinfo=None) for date in dates_list]
    # edit the shoreline geojson files and remove the selected shorelines that having matching date and satname
    geodata_processing.edit_geojson_files(
        filepaths, common.remove_matching_rows, date=formatted_dates, satname=sat_list
    )
    # delete the extracted shorelines from the extracted_shorelines_dict.json file
    filename = "extracted_shorelines_dict.json"
    # update extracted_shorelines_dict.json and transects_cross_distances.json
    common.update_extracted_shorelines_dict_transects_dict(
        session_path, filename, dates_list, sat_list
    )
    # delete the extracted shorelines from the transect_time_series.csv files
    filenames = [
        "transect_time_series.csv",  # old name for  raw_transect_time_series.csv
        "raw_transect_time_series.csv",  # time series as matrix of dates x transects
        "transect_time_series_merged.csv",  # old name for  raw_transect_time_series_merged.csv
        "raw_transect_time_series_merged.csv",  # timeseries with columns dates, transect_id, x,y, shore_x, shore_y,cross_distance, along_distance
        "transect_time_series_tidally_corrected.csv",  # old name for tidally_corrected_transect_time_series_merged.csv
        "tidally_corrected_transect_time_series_merged.csv",  # tidally corrected timeseries with columns dates, transect_id, x,y, shore_x, shore_y,cross_distance, along_distance
        "tidally_corrected_transect_time_series.csv",  # tidally corrected time series as matrix of dates x transects
    ]
    filepaths = [
        os.path.join(session_path, filename)
        for filename in filenames
        if os.path.isfile(os.path.join(session_path, filename))
    ]
    common.update_transect_time_series(filepaths, dates_list)
    # delete the extracted shorelines from the jpg detection files
    jpg_path = os.path.join(session_path, "jpg_files", "detection")
    if os.path.exists(jpg_path) and os.path.isdir(jpg_path):
        common.delete_jpg_files(dates_list, sat_list, jpg_path)


class CoastSeg_Map:
    def __init__(self, create_map: bool = True):
        # Basic settings and configurations
        self.settings = {}
        self.map = None
        self.draw_control = None
        self.warning_box = None
        self.roi_html = None
        self.roi_box = None
        self.roi_widget = None
        self.feature_html = None
        self.hover_box = None
        # Assume that the user is not drawing a reference buffer
        self.drawing_shoreline_extraction_area = False
        # Optional: a user drawn polygon only keep extracted shorelines within this polygon
        self.shoreline_extraction_area = None

        self.set_settings()
        self.session_name = ""

        # Factory for creating map objects
        self.factory = factory.Factory()

        # Observables
        self.id_container = IDContainer(ids=[])
        self.extract_shorelines_container = ExtractShorelinesContainer()

        # Map objects and configurations
        self.rois = None
        self.transects = None
        self.shoreline = None
        self.bbox = None
        self.selected_set = set()
        self.selected_shorelines_set = set()
        if create_map:
            self._init_map_components()
            # Warning and information boxes that appear on top of the map
            self._init_info_boxes()

    def print_features(self) -> None:
        """
        Prints count of loaded features.

        Returns:
            None
        """
        # print the number of each kind of feature
        if self.rois is not None:
            print(f"Number of ROIs: {len(self.rois.gdf)}")
            print(f"ROI IDs: {self.rois.gdf.id}")
        if self.shoreline is not None:
            print(f"Number of shorelines: {len(self.shoreline.gdf)}")
        if self.transects is not None:
            print(f"Number of transects: {len(self.transects.gdf)}")
        if self.bbox is not None:
            print(f"Number of bounding boxes: {len(self.bbox.gdf)}")
        if self.shoreline_extraction_area is not None:
            print(
                f"Number of shoreline extraction areas: {len(self.shoreline_extraction_area.gdf)}"
            )
        if (
            self.rois is None
            and self.shoreline is None
            and self.transects is None
            and self.bbox is None
            and self.shoreline_extraction_area is None
        ):
            print("No features loaded")

    def get_map(self) -> Map:
        """
        Returns map object, creating if needed.

        Returns:
            ipyleaflet Map object.
        """
        if self.map is None:
            self.map = self.create_map()
        return self.map

    def _init_map_components(self) -> None:
        """
        Initializes map-related attributes and settings.

        Returns:
            None
        """
        map_instance = self.get_map()
        self.draw_control = self.create_DrawControl(DrawControl())
        self.draw_control.on_draw(self.handle_draw)
        map_instance.add(self.draw_control)
        map_instance.add(LayersControl(position="topright"))

    def _init_info_boxes(self) -> None:
        """
        Initializes info and warning boxes for map.

        Sets up the warning, ROI, and hover boxes:
        - Warning box: shows warning messages
        - ROI box: displays region of interest details
        - Hover box: shows selected feature info

        Returns:
            None
        """
        if not self.map:
            return
        self.warning_box = HBox([], layout=Layout(height="242px"))
        self.warning_widget = WidgetControl(widget=self.warning_box, position="topleft")
        self.map.add(self.warning_widget)

        self.roi_html = HTML("""""")
        self.roi_box = common.create_hover_box(
            title="ROI", feature_html=self.roi_html, default_msg="Hover over a ROI"
        )
        self.roi_widget = WidgetControl(widget=self.roi_box, position="topright")
        self.map.add(self.roi_widget)

        self.feature_html = HTML("""""")
        self.hover_box = common.create_hover_box(
            title="Feature", feature_html=self.feature_html
        )
        self.hover_widget = WidgetControl(widget=self.hover_box, position="topright")
        self.map.add(self.hover_widget)

    def __str__(self) -> str:
        return f"CoastSeg: roi={self.rois}\n shoreline={self.shoreline}\n  transects={self.transects}\n bbox={self.bbox}"

    def __repr__(self) -> str:
        return f"CoastSeg(roi={self.rois}\n shoreline={self.shoreline}\n transects={self.transects}\n bbox={self.bbox} "

    def get_session_name(self) -> str:
        """
        Returns the current session name.

        Returns:
            str: The session name.
        """
        return self.session_name

    def set_session_name(self, name: str) -> None:
        """
        Sets the session name.

        Args:
            name (str): The session name to set.
        """
        self.session_name = name

    def load_extracted_shoreline_layer(
        self, gdf, layer_name: str, colormap: str
    ) -> None:
        """
        Loads an extracted shoreline layer onto the map with a specified colormap.

        Args:
            gdf: The GeoDataFrame containing shoreline data.
            layer_name (str): The name of the layer.
            colormap (str): The colormap to use for styling.

        Returns:
            None
        """
        map_crs = "epsg:4326"
        # create a layer with the extracted shorelines selected
        points_gdf = extracted_shoreline.convert_linestrings_to_multipoints(gdf)
        projected_gdf = points_gdf.to_crs(map_crs)
        # convert date column to datetime
        projected_gdf["date"] = pd.to_datetime(projected_gdf["date"])
        # Sort the GeoDataFrame based on the 'date' column
        projected_gdf = projected_gdf.sort_values(by="date")
        # normalize the dates to 0-1 scale so that we can use them to get colors from the colormap
        unique_dates = projected_gdf["date"].unique()
        if len(unique_dates) == 1:
            # If there's only one date, set delta to 0.25
            date_to_delta = {unique_dates[0]: 0.25}
        else:
            date_to_delta = {
                date: (date - unique_dates.min())
                / (unique_dates.max() - unique_dates.min())
                for date in unique_dates
            }
        # get the colors from the colormap based on the assigned delta values
        # Map the 'date' column to normalized delta values
        projected_gdf["delta"] = projected_gdf["date"].map(date_to_delta)
        # Get colors from colormap based on assigned delta values
        colors = plt.get_cmap(colormap)(projected_gdf["delta"].to_numpy())
        # Convert RGBA colors to Hex
        colors_hex = [
            "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
            for r, g, b, a in colors
        ]
        # add the colors to the GeoDataFrame
        projected_gdf["color"] = colors_hex

        projected_gdf = common.stringify_datetime_columns(projected_gdf)
        # Convert GeoDataFrame to GeoJSON
        features_json = json.loads(projected_gdf.to_json())

        # Add 'id' field to features in GeoJSON
        for feature, index in zip(
            features_json["features"], range(len(features_json["features"]))
        ):
            feature["id"] = str(index)

        # define a style callback function that takes a feature and returns a style dictionary
        def style_callback(feature):
            # find the color for the current feature based on its id
            color = projected_gdf.iloc[int(feature["id"])]["color"]
            return {"color": color, "weight": 5, "fillColor": color, "fillOpacity": 0.5}

        # create an ipyleaflet GeoJSON layer with the extracted shorelines selected
        new_layer = GeoJSON(
            data=features_json,
            name=layer_name,
            style_callback=style_callback,
            point_style={
                "radius": 1,
                "opacity": 1,
            },
            hover_style={"color": "red"},
        )
        # this will add the new layer on map and update the widget on the side with the extracted shoreline information
        self.replace_layer_by_name(
            layer_name,
            new_layer,
            on_hover=self.update_extracted_shoreline_html,
            on_click=None,
        )

    def delete_selected_shorelines(
        self,
        layer_name: str,
        selected_id: str,
        selected_shorelines: Optional[List] = None,
    ) -> None:
        """
        Deletes selected shorelines from the map and removes the corresponding layer.

        Args:
            layer_name (str): The name of the layer to remove.
            selected_id (str): The ID of the selected shoreline.
            selected_shorelines (Optional[List]): List of selected shorelines to delete.

        Returns:
            None
        """
        if selected_shorelines and selected_id:
            if self.map:
                self.map.default_style = {"cursor": "wait"}
            session_name = self.get_session_name()
            session_path = file_utilities.get_session_location(
                session_name=session_name, raise_error=True
            )
            # get the path to the session directory that contains the extracted shoreline files
            session_path = find_shorelines_directory(session_path, selected_id)
            # remove the extracted shorelines from the extracted shorelines object stored in the ROI
            dates, satellites = common.extract_dates_and_sats(selected_shorelines)
            if self.rois:
                self.rois.remove_selected_shorelines(selected_id, dates, satellites)
            # remove the extracted shorelines from the files in the session location
            if (
                session_path is not None
                and os.path.exists(session_path)
                and os.path.isdir(session_path)
            ):
                delete_extracted_shorelines_files(
                    session_path, list(selected_shorelines)
                )
            # this will remove the selected shorelines from the files
        self.remove_layer_by_name(layer_name)
        if self.map:
            self.map.default_style = {"cursor": "default"}

    def load_selected_shorelines_on_map(
        self,
        selected_id: str,
        selected_shorelines: List,
        layer_name: str,
        colormap: str,
    ) -> None:
        """
        Loads selected shorelines for a given ROI ID onto the map.

        Args:
            selected_id (str): The ROI ID.
            selected_shorelines (List): List of selected shoreline items.
            layer_name (str): The name of the layer.
            colormap (str): The colormap to use for styling.

        Returns:
            None
        """

        def get_selected_shorelines(
            gdf, selected_items: list[str]
        ) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
            """
            Filter the GeoDataFrame based on selected items.

            Args:
                gdf (gpd.GeoDataFrame): The input GeoDataFrame.
                selected_items (list[str]): A list of selected items in the format "satname_dates".

            Returns:
                Union[gpd.GeoDataFrame, pd.DataFrame]: The filtered GeoDataFrame containing the selected shorelines.
            """
            # check if the dates column is in the format 2023-12-09 18:40:32
            # If its not then convert the dates to datetime
            # To ensure the gdf is in the expected format
            gdf["date"] = pd.to_datetime(gdf["date"])
            # strftime converts datetime to string and formats it to the expected format
            gdf["date"] = gdf["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
            # then convert it back to datetime object
            gdf["date"] = pd.to_datetime(gdf["date"])
            # Filtering criteria
            frames = []  # List to collect filtered frames
            # Loop through each dictionary in dates_tuple
            for criteria in list(selected_items):
                satname, dates = criteria.split("_")
                # if "date" column string already then don't convert to datetime
                if gdf["date"].dtype == "object":
                    filtered = gdf[(gdf["date"] == dates) & (gdf["satname"] == satname)]
                else:
                    filtered = gdf[
                        (gdf["date"] == datetime.strptime(dates, "%Y-%m-%d %H:%M:%S"))
                        & (gdf["satname"] == satname)
                    ]
                frames.append(filtered)

            # Concatenate the frames to get the final result
            filtered_gdf = gpd.GeoDataFrame(columns=["geometry"])
            filtered_gdf.crs = "epsg:4326"
            if frames:
                filtered_gdf = pd.concat(frames)
            return filtered_gdf

        # load the extracted shorelines for the selected ROI ID
        if self.rois is not None:
            extracted_shorelines = self.rois.get_extracted_shoreline(selected_id)
            if extracted_shoreline is None:
                logger.warning(
                    f"No extracted shorelines found for ROI ID {selected_id}. Selected ID was type({type(selected_id)})."
                )
                return
            # get the GeoDataFrame for the extracted shorelines
            if hasattr(extracted_shorelines, "gdf"):
                selected_gdf = get_selected_shorelines(
                    extracted_shorelines.gdf,  # type: ignore
                    selected_shorelines,  # type: ignore
                )
                if not selected_gdf.empty:
                    self.load_extracted_shoreline_layer(
                        selected_gdf, layer_name, colormap
                    )

    def update_extracted_shorelines_display(
        self,
        selected_id: str,
    ) -> None:
        """
        Updates the display of extracted shorelines for a selected ROI ID.
        It removes old layers, updates the load and trash lists, and loads new extracted shorelines onto the map.

        Args:
            selected_id (str): The ROI ID.

        Returns:
            None
        """
        # remove the old layers
        self.remove_extracted_shoreline_layers()
        # update the load_list and trash_list
        extracted_shorelines = self.update_loadable_shorelines(selected_id)
        self.extract_shorelines_container.trash_list = []
        # load the new extracted shorelines onto the map
        self.load_extracted_shorelines_on_map(extracted_shorelines, 0)

    def create_map(self) -> Map:
        """
        Creates an interactive map object using the map settings.

        Returns:
            Map: ipyleaflet interactive Map object.
        """
        map_settings = {
            "center_point": (36.8470, -121.8024),
            "zoom": 7,
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
            world_copy_jump=True,
        )

    def compute_tidal_corrections(
        self,
        roi_ids: Collection,
        beach_slope: Union[str, float] = 0.02,
        reference_elevation: float = 0,
        model: str = "FES2022",
        tides_file: str = "",
        use_progress_bar: bool = True,
    ) -> None:
        """
        Computes tidal corrections for the specified region of interest (ROI) IDs.

        Args:
            roi_ids (Collection): A collection of ROI IDs for which tidal corrections need to be computed.
            beach_slope (str or float): The slope of the beach or file containing the slopes of the beach
                This can be either a file containing the slopes of the beach or a float value.
                Available file formats for the beach slope:
                    - A CSV file containing the beach slope data with columns 'dates' and 'slope'.
                    - A CSV file containing the beach slope data with columns 'dates', 'transect_id', and 'slope'.
                    - A CSV file containing the beach slope data with columns 'dates', 'latitude', 'longitude', and 'slope'.
                    - A CSV file containing the transect ids as the columns and the dates as the row indices.

            reference_elevation (float, optional): The reference elevation in meters relative to MSL (Mean Sea Level). Defaults to 0.
             model (str, optional): The tidal model to use. To not use tide model set to "". Defaults to "FES2022".
                - Other option is "FES2014". The tide model must be installed prior to use.
                - If model = "" then the tide_file must be provided.

            tides_file (str, optional): Path to the CSV file containing tide data. Defaults to an empty string.
                - Acceptable tide formats:
                    - CSV file containing tide data with columns 'dates' and 'tide'.
                    - CSV file containing tide data with columns 'dates', 'transect_id', and 'tide'.
                    - CSV file containing tide data with columns 'dates', 'latitude', 'longitude', and 'tide'.
                    - CSV file containing the transect ids are the columns and the dates as the row indices
            use_progress_bar (bool, optional): If True, display a progress bar. Defaults to True.
        Returns:
            None
        """
        logger.info(
            f"Computing tides for ROIs {roi_ids} beach_slope: {beach_slope} reference_elevation: {reference_elevation}"
        )

        session_name = self.get_session_name()
        # if True then only the intersection point ON the transect are kept. If False all intersection points are kept.
        only_keep_points_on_transects = self.get_settings().get(
            "drop_intersection_pts", False
        )
        try:
            tide_correction.correct_all_tides(
                roi_ids,
                session_name,
                reference_elevation,
                beach_slope,
                use_progress_bar=use_progress_bar,
                only_keep_points_on_transects=only_keep_points_on_transects,
                model=model,
                tides_file=tides_file,
            )
        except Exception as e:
            if self.map is not None:
                exception_handler.handle_exception(
                    e,
                    self.warning_box,
                    title="Tide Model Error",
                    msg=str(e),
                )
            else:
                raise Exception(f"""Tide Model Error:\n {e}""")
        else:
            print("\nTidal corrections completed")

    def load_metadata(
        self, settings: Optional[dict] = None, ids: Collection = set([])
    ) -> Dict:
        """
        Loads metadata for a site using provided settings or for multiple ROIs using their IDs.
        This also creates a metadata file for each ROI in the data directory.
        Note that coastsat's `get_metadata` is used to actually perform the metadata loading.

        Args:
            settings (dict, optional): Site-specific settings, including 'sitename' and 'filepath_data'. Defaults to empty dict.
            ids (Collection, optional): Collection of ROI IDs to load metadata for. Defaults to empty set.

        Returns:
            Dict: Dictionary containing the loaded metadata.

        Raises:
            FileNotFoundError: If the specified directory does not exist.
            Exception: If neither settings nor ids are provided.

        Example:
            >>> load_metadata(settings={'sitename': 'site1', 'filepath_data': '/path/to/data'})
            >>> load_metadata(ids={1, 2, 3})
        """
        if settings and isinstance(settings, dict):
            return get_metadata(settings)
        elif (
            ids
        ):  # Alternatively loop through each ROI ID and load metadata from that ROI's settings
            for roi_id in ids:
                # if the ROI directory did not exist then print a warning and proceed
                try:
                    logger.info(
                        f"Loading metadata using {self.rois.roi_settings[str(roi_id)]}"  # type: ignore
                    )
                    metadata = get_metadata(self.rois.roi_settings[str(roi_id)])  # type: ignore
                    logger.info(f"Metadata for ROI ID {str(roi_id)}:{metadata}")
                    return metadata
                except FileNotFoundError as e:
                    logger.error(f"Metadata not loaded for ROI ID {str(roi_id)} {e}")
                    print(f"Metadata not loaded for ROI ID {str(roi_id)} {e}")
            return {}
        else:
            raise Exception("Must provide settings or list of IDs to load metadata.")

    def load_session_files(self, dir_path: str, data_path: str = "") -> None:
        """
        Loads configuration files from the specified directory.

        Processes core config files (config_gdf.geojson, config.json) and module-specific
        settings files (shoreline_settings.json, transects_settings.json). Also loads
        metadata for any ROIs found in the configuration.

        Args:
            dir_path (str): Path to the directory containing configuration files.
            data_path (str, optional): Full path to the coastseg data directory where
                downloaded data is saved. If empty, defaults to base_dir/data.

        Returns:
            None
        """
        if os.path.isdir(dir_path):
            # ensure coastseg\data location exists
            data_path = self.ensure_data_directory(data_path)
            config_geojson_path = os.path.join(dir_path, "config_gdf.geojson")
            config_json_path = os.path.join(dir_path, "config.json")
            # load the config files if they exist
            config_loaded = self.load_config_files(
                data_path, config_geojson_path, config_json_path
            )
            # create metadata files for each ROI loaded in using coastsat's get_metadata()
            if self.rois and getattr(self.rois, "roi_settings"):
                self.load_metadata(ids=list(self.rois.roi_settings.keys()))
            else:
                logger.warning("No ROIs were able to have their metadata loaded.")
            # load in setting from shoreline_settings.json and transects_settings.json
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
                    settings = common.load_settings(file_path, keys)
                    self.set_settings(**settings)
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
                    settings = common.load_settings(file_path, keys)
                    self.set_settings(**settings)
            if not config_loaded:
                logger.info(f"Not all config files not found at {dir_path}")

    def load_session_from_directory(self, dir_path: str, data_path: str = "") -> None:
        """
        Loads a session from a specified directory path.
        Loads config files, extracted shorelines, and transects & extracted shoreline intersections.

        Args:
            dir_path (str): The path of the directory to load the session from.
            data_path (str): Full path to the coastseg data directory where downloaded data is saved.

        Returns:
            None
        """
        self.load_session_files(dir_path, data_path)
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
            self.rois.add_cross_shore_distances(cross_distances, roi_id)  # type: ignore

    def load_data_from_directory(self, dir_path: str, data_path: str = "") -> None:
        """
        Clears the previous session and loads data from the specified directory.
        This will load the config_gdf.geojson, config.json, shoreline_settings.json, and transects_settings.json files
        Also loads the metadata for each ROI in the config_gdf.geojson file if it exists.

        Args:
            dir_path (str): The path to the directory containing the session data.
            data_path (str, optional): The path to the specific data file within the directory. Defaults to an empty string.

        Raises:
            FileNotFoundError: If the specified directory does not exist.

        Returns:
            None
        """
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Session path {dir_path} does not exist")
        # remove all the old features from the map
        self.remove_all()
        # only load data from the directory
        self.load_session_files(dir_path, data_path)

    def load_fresh_session(self, session_path: str) -> None:
        """
        Load a fresh session by removing all the old features from the map and loading a new session.

        Args:
            session_path (str): The path to the session directory

        Returns:
            None
        """
        if not os.path.exists(session_path):
            raise FileNotFoundError(f"Session path {session_path} does not exist")
        # remove all the old features from the map
        self.remove_all()
        self.load_session(session_path)

    def load_session(self, session_path: str, data_path: str = "") -> None:
        """
        Load a session from the given path.

        The function loads a session from the given path, which can contain one or more directories, each containing
        the files for a single ROI. For each subdirectory, the function calls `load_session_from_directory` to load
        the session files and objects on the map. If no subdirectories exist, the function calls `load_session_from_directory` with the
        session path.

        Args:
            session_path: The path to the session directory.
            data_path (str): Full path to the coastseg data directory where downloaded data is saved
        Returns:
            None.
        """

        def get_parent_session_name(session_path: str) -> str:
            split_array = session_path.split(os.sep)
            # get the index of the sessions directory which contains all the sessions
            if "data" in split_array:
                return os.path.basename(session_path)
            elif "sessions" in split_array:
                parent_index = split_array.index("sessions")
            else:
                return ""
            # get the parent session name aka not a sub directory for a specific ROI
            parent_session_name = split_array[parent_index + 1]
            if not (
                os.path.exists(os.sep.join(split_array[: parent_index + 1]))
                and os.path.isdir(os.sep.join(split_array[: parent_index + 1]))
            ):
                return ""
            return parent_session_name

        data_path = self.ensure_data_directory(data_path)
        session_path = os.path.abspath(session_path)
        # load the session name

        session_name = get_parent_session_name(session_path)
        if session_name == "":
            raise Exception(
                f"A session can only be loaded from the /sessions or /data directory. The {session_path} directory is not a valid session directory."
            )

        logger.info(f"session_name: {session_name} session_path: {session_path}")
        self.set_session_name(session_name)
        logger.info(f"Loading session from session directory: {session_path}")

        # load the session from the parent directory and subdirectories within session path
        directories_to_load = [
            dirpath for dirpath, dirnames, filenames in os.walk(session_path)
        ]
        for directory in directories_to_load:
            self.load_session_from_directory(directory, data_path)

        # update the list of roi's ids who have extracted shorelines
        ids_with_extracted_shorelines = self.update_roi_ids_with_shorelines()
        # update the loadable shorelines on the map
        if self.map is not None:
            for roi_id in ids_with_extracted_shorelines:
                self.update_extracted_shorelines_display(roi_id)
        logger.info(
            f"Available roi_ids from extracted shorelines: {ids_with_extracted_shorelines}"
        )

        # Raise a warning if any of the ROIs were missing in in the data directory
        missing_directories = self.get_missing_directories()
        exception_handler.check_if_dirs_missing(missing_directories, data_path)

    def load_gdf_config(self, filepath: str) -> None:
        """
        Loads features from a GeoDataFrame configuration file onto the map.

        Args:
            filepath (str): Full path to config_gdf.geojson.

        Returns:
            None
        """
        gdf = geodata_processing.read_gpd_file(filepath)
        gdf = common.stringify_datetime_columns(gdf)

        # each possible type of feature and the columns that should be loaded
        feature_types = {
            "bbox": ["geometry"],
            "roi": ["id", "geometry"],
            "transect": list(Transects.COLUMNS_TO_KEEP),
            "shoreline": ["geometry", "id"],
            "shoreline_extraction_area": ["geometry"],
        }

        feature_names = {
            "bbox": ["bbox"],
            "roi": ["roi"],
            "transect": ["transect", "transects"],
            "shoreline": [
                "shoreline",
                "shorelines",
                "reference_shoreline",
                "reference_shorelines",
                "reference shoreline",
                "reference shorelines",
            ],
            "shoreline_extraction_area": ["shoreline_extraction_area"],
        }

        # attempt to load each feature type onto the map from the config_gdf.geojson file
        for feature_name, columns in feature_types.items():
            # collect all the features of the same type into a single geodataframe
            feature_gdf = self.collect_features(
                gdf, feature_names[feature_name], columns  # type: ignore
            )  # type: ignore

            if feature_gdf.empty:
                continue

            # load the features onto the map
            if feature_name == "roi":
                self.integrate_rois(feature_gdf)  # Need special handling for ROIs
            else:  # load other features normally
                self.load_feature_on_map(
                    feature_name, gdf=feature_gdf, zoom_to_bounds=True
                )

    def collect_features(
        self, gdf: gpd.GeoDataFrame, names: list[str], columns: list[str]
    ) -> gpd.GeoDataFrame:
        """Collect features from gdf matching any of the given names.

        Args:
            gdf (gpd.GeoDataFrame): The GeoDataFrame containing the features to extract.
            names (list[str]): List of feature type names to match.
            columns (list[str]): List of column names to extract from the GeoDataFrame.

        Returns:
            gpd.GeoDataFrame: A new GeoDataFrame containing only the features of the specified types and columns.

        Example:
            >>> gdf = gpd.read_file("config_gdf.geojson")
            >>> feature_gdf = collect_features(gdf, ["shoreline", "reference shoreline"], ["geometry", "id"])
            >>> print(feature_gdf)
            feature_gdf will contain all features of type "shoreline" or "reference shoreline" with only the "geometry" and "id" columns.
        """
        feature_gdfs = [self._extract_feature_gdf(gdf, name, columns) for name in names]
        non_empty = [f for f in feature_gdfs if not f.empty]
        return (
            pd.concat(non_empty, ignore_index=True) if non_empty else gpd.GeoDataFrame()
        )  # type: ignore

    def integrate_rois(self, feature_gdf: gpd.GeoDataFrame) -> None:
        """
        Integrates ROI features into the map, either creating new ROI objects or merging with existing ones.

        This method handles the special logic required for ROI integration. If no ROIs currently exist
        on the map, it creates a new ROI object and loads the features onto the map with zoom-to-bounds
        functionality. If ROIs already exist, it merges the new ROI features with the existing ones
        and updates the map display accordingly.

        Args:
            feature_gdf (gpd.GeoDataFrame): A GeoDataFrame containing ROI features to integrate.
                Must contain 'id' and 'geometry' columns at minimum.

        Returns:
            None

        Raises:
            Exception: If the provided GeoDataFrame is empty (via exception_handler.check_if_gdf_empty).

        Example:
            >>> # Load new ROIs onto an empty map
            >>> roi_gdf = gpd.read_file("new_rois.geojson")
            >>> self.integrate_rois(roi_gdf)

            >>> # Add additional ROIs to existing ones
            >>> additional_rois = gpd.read_file("more_rois.geojson")
            >>> self.integrate_rois(additional_rois)
        """
        exception_handler.check_if_gdf_empty(
            feature_gdf, "ROIs", "Cannot load empty ROIs onto map"
        )

        if (
            self.rois is None
        ):  # if no ROIs exist on the map, create a new ROI object and load the ROIs onto the map
            self.rois = ROI(rois_gdf=feature_gdf)  # type: ignore
            self.load_feature_on_map("roi", gdf=feature_gdf, zoom_to_bounds=True)  # type: ignore
        else:  # If ROIs already exist on the map, add the new ROIs to the existing ROIs
            self.rois = self.rois.add_geodataframe(feature_gdf)  # type: ignore
            self.add_feature_on_map(self.rois, "roi")

    def _extract_feature_gdf(
        self,
        gdf: Union[gpd.GeoDataFrame, pd.DataFrame],
        feature_type: Union[int, str, List[str]],
        columns: List[str],
    ) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
        """
        Extracts a GeoDataFrame of features of a given type and specified columns from a larger GeoDataFrame.

        Args:
            gdf (Union[gpd.GeoDataFrame, pd.DataFrame]): The GeoDataFrame containing the features to extract.
            feature_type (Union[int, str]): The type of feature to extract. Typically one of the following 'shoreline','rois','transects','bbox'
        Feature_type can also be list of strings such as ['shoreline','shorelines', 'reference shoreline'] to match the same kind of feature with muliple names.
            columns (List[str]): A list of column names to extract from the GeoDataFrame.

        Returns:
            Union[gpd.GeoDataFrame, pd.DataFrame]: A new GeoDataFrame containing only the features of the specified type and columns.

        Raises:
            ValueError: Raised when feature_type or any of the columns specified do not exist in the GeoDataFrame.
        """
        # Check if feature_type exists in the GeoDataFrame
        if "type" not in gdf.columns:
            raise ValueError(
                "Column 'type' does not exist in the GeoDataFrame. Incorrect config_gdf.geojson loaded"
            )

        # select only the columns that are in the gdf
        keep_columns = [col for col in columns if col in gdf.columns]

        # If no columns from columns list exist in the GeoDataFrame, raise an error
        if not keep_columns:
            raise ValueError(
                f"None of the columns {columns} exist in the GeoDataFrame."
            )

        # select only the features that are of the correct type and have the correct columns
        feature_gdf = common.extract_feature_from_geodataframe(gdf, feature_type)
        feature_gdf = feature_gdf[keep_columns]
        return feature_gdf

    def preview_available_images(self, selected_ids: Optional[set] = None) -> None:
        """
        Prints the number of available satellite images for selected ROIs and satellites.

        Args:
            selected_ids (set, optional): Set of selected ROI IDs. Defaults to None.

        Raises:
            Exception: If ROIs are missing, empty, or no ROI is selected.

        Example:
            >>> preview_available_images({1}) # Preview available images for ROI ID 1
            ROI ID: 1
                L5: 10 images
                L7: 8 images
                L8: 12 images
                L9: 5 images
                S2: 11 images
        """
        # Get the months list from the settings or use the default list
        months_list = self.settings.get(
            "months_list", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        )

        # check that ROIs exist and one has been clicked
        exception_handler.check_if_None(self.rois, "ROI")
        exception_handler.check_if_gdf_empty(self.rois.gdf, "ROI")

        if selected_ids is None:
            selected_ids = self.get_selected_ids()

        exception_handler.check_selected_set(selected_ids)

        # get the start and end date to check available images
        start_date, end_date = self.settings["dates"]
        # for each selected ID return the images available for each site
        for roi_id in tqdm(selected_ids, desc="Processing", leave=False):
            polygon = common.get_roi_polygon(self.rois.gdf, roi_id)
            if polygon is None:
                raise Exception(f"ROI ID {roi_id} not found in the ROIs GeoDataFrame")
            if polygon:
                # only get the imagery in tier 1
                images_count = count_images_in_ee_collection(
                    polygon,
                    start_date,
                    end_date,
                    satellites=set(self.settings["sat_list"]),
                    max_cloud_cover=95,
                    tiers=[1],
                    months_list=months_list,
                    min_roi_coverage=self.settings.get("min_roi_coverage", 0.0),
                )
                satellite_messages = [f"\nROI ID: {roi_id}"]
                for sat in self.settings["sat_list"]:
                    satellite_messages.append(f"{sat}: {images_count[sat]} images")

                print("\n".join(satellite_messages))

    def get_selected_ids(
        self, selected_ids: Optional[Union[set, str, list]] = None
    ) -> set:
        """
        Returns selected ROI IDs, either from the provided input or the current selection of ROIs on the map.
        This also updates the current selection on the map if new IDs are provided.

        Side Effects:
            Updates the self.selected_set attribute with the new selection if selected_ids is provided.

        Args:
            selected_ids (str, list, set, optional): IDs to select. If None, returns current selection.

        Returns:
            set: Selected ROI IDs.

        Raises:
            ValueError: If selected_ids is not str, list, set, or None.

        Example:
            >>> self.get_selected_ids("roi_1")
            {'roi_1'}
            >>> self.get_selected_ids(["roi_1", "roi_2"])
            {'roi_1', 'roi_2'}
            >>> self.get_selected_ids()  # Returns current selection
            {'roi_3', 'roi_4'}
        """
        # Return current selection if no IDs provided
        if selected_ids is None:
            return set(getattr(self, "selected_set", set()))

        # Normalize input to set
        if isinstance(selected_ids, str):
            normalized_ids = {selected_ids}
        elif isinstance(selected_ids, (list, set)):
            normalized_ids = set(selected_ids)
        else:
            raise ValueError(
                f"selected_ids must be str, list, set, or None, got {type(selected_ids).__name__}"
            )

        # Update and return the selection
        self.selected_set = normalized_ids
        return normalized_ids

    def get_missing_directories(self, new_settings: Optional[dict] = None) -> dict:
        """
        Identifies ROI directories that are missing from the data directory.

        This method checks which ROI directories are missing based on the current ROI settings.
        If new settings are provided, it updates the ROI settings with the new global settings
        (such as updated date ranges or satellite lists) before checking for missing directories.

        Args:
            new_settings (Optional[dict]): New global settings to update ROI settings with.
                May include keys like 'dates', 'sat_list', etc. Defaults to None.

        Returns:
            dict: A dictionary mapping ROI IDs to their missing directory information.
                Empty dict if no ROIs exist or no directories are missing.

        Example:
            >>> missing = self.get_missing_directories()
            >>> print(missing)
            {'roi_1': {'sitename': 'site1', 'filepath': '/path/to/missing/dir'}}

            >>> # Update settings and check again
            >>> new_settings = {'dates': ['2023-01-01', '2023-12-31']}
            >>> missing = self.get_missing_directories(new_settings)
        """
        # get the ROIs and check if they have settings
        if self.rois is None:
            return {}

        roi_settings = self.rois.get_roi_settings()
        # the user might have updated the date range or satellite list to extract shorelines from so update the ROI settings
        if new_settings:
            roi_settings = common.update_roi_settings_with_global_settings(
                roi_settings, new_settings
            )
        self.rois.set_roi_settings(roi_settings)
        logger.info(f"Checking roi_settings for missing ROIs: {roi_settings}")
        # check if any of the ROIs were missing in in the data directory
        missing_directories = common.get_missing_roi_dirs(roi_settings)
        logger.info(f"missing_directories: {missing_directories}")
        return missing_directories

    def make_roi_settings(
        self,
        rois: gpd.GeoDataFrame = gpd.GeoDataFrame(),
        settings: Optional[dict] = None,
        selected_ids: Optional[Iterable] = None,
        file_path: Optional[str] = None,
    ) -> dict:
        """
        Generates settings for downloading imagery for selected ROIs.

        Args:
            rois (gpd.GeoDataFrame): GeoDataFrame containing ROIs.
            settings (dict, optional): Additional settings for imagery download. Defaults to None.
            selected_ids (Iterable, optional): ROI IDs to include. Defaults to None.
            file_path (str, optional): Directory to save imagery. Defaults to CoastSeg data directory.

        Returns:
            dict: ROI-specific settings for imagery download.

        Example:
            >>> make_roi_settings(rois, settings, selected_ids={1,2}, file_path="/tmp/data")
            {
                '1': {
                    'id': '1',
                    'geometry': {...},
                    'file_path': '/tmp/data/ID_1_datetime06-05-23__04_16_45',
                    ...
                },
                '2': {
                    'id': '2',
                    'geometry': {...},
                    'file_path': '/tmp/data/ID_2_datetime06-05-23__04_16_45',
                    ...
                }
            }
        """
        # Ensure we have settings
        if settings is None:
            settings = {}

        # Get the location where the downloaded imagery will be saved
        norm_path = normalize_path(file_path)
        if not norm_path:
            norm_path = os.path.abspath(
                os.path.join(os.path.abspath(core_utilities.get_base_dir()), "data")
            )

        # used to uniquely identify the folder where the imagery will be saved
        # example  ID_12_datetime06-05-23__04_16_45
        date_str = file_utilities.generate_datestring()
        # get only the ROIs whose IDs are in the selected_ids
        filtered_gdf = rois[
            rois["id"].isin(selected_ids if selected_ids is not None else [])
        ]
        geojson_str = filtered_gdf.to_json()
        geojson_dict = json.loads(geojson_str)
        # Create a list of download settings for each ROI
        roi_settings = common.create_roi_settings(
            settings, geojson_dict, norm_path, date_str
        )
        return roi_settings

    def resolve_settings(self, settings: Optional[dict]) -> dict:
        """
        Resolves effective settings by merging provided overrides with current settings.

        Args:
            settings (dict | None): Incoming overrides.

        Returns:
            dict: Effective settings in object state.
        """
        if settings is None:
            return self.get_settings()
        self.set_settings(**settings)
        return self.get_settings()

    def ensure_rois(self, rois: Optional[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        """
        Ensure ROIs are present (load if provided).

        Args:
            rois (gpd.GeoDataFrame | None): Optional new ROI GeoDataFrame.

        Returns:
            gpd.GeoDataFrame: Active ROI GeoDataFrame.

        Raises:
            Exception: If no ROIs available.
        """
        if rois is not None:
            self.rois = ROI(rois_gdf=rois)
        if not hasattr(self, "rois") or self.rois is None:
            raise Exception("No ROIs loaded. Cannot proceed.")
        return self.rois.gdf

    def prepare_roi_settings(
        self,
        roi_gdf: gpd.GeoDataFrame,
        settings: dict,
        selected_ids: Collection,
        file_path: Optional[str],
    ) -> dict:
        """
        Merge existing ROI settings with new selections or create fresh ones.

        Args:
            roi_gdf (gpd.GeoDataFrame): Full ROI GeoDataFrame.
            settings (dict): Global settings.
            selected_ids (Collection): Selected ROI IDs.
            file_path (str | None): Base download directory.

        Returns:
            dict: Per-ROI settings mapping.
        """
        existing_settings = self.rois.get_roi_settings(selected_ids)  # type: ignore
        if existing_settings:
            missing = set(selected_ids) - set(existing_settings.keys())
            if missing:
                new_part = self.make_roi_settings(
                    rois=roi_gdf,
                    settings=settings,
                    selected_ids=missing,
                    file_path=file_path,
                )
                existing_settings.update(new_part)
            # Sync global changes (e.g. dates/sat_list)
            existing_settings = common.update_roi_settings_with_global_settings(
                existing_settings, settings
            )
            return existing_settings

        # None existed; create all
        return self.make_roi_settings(
            rois=roi_gdf,
            settings=settings,
            selected_ids=selected_ids,
            file_path=file_path,
        )

    def retrieve_images_for_rois(
        self,
        inputs_list: List[dict],
        settings: dict,
    ) -> None:
        """
        Retrieve imagery for each ROI input block.

        Args:
            inputs_list (List[dict]): Per-ROI input configs.
            settings (dict): Effective settings.


        Returns:
            None
        """
        cloud_thresh = settings.get("download_cloud_thresh", 0.80)
        months_list = settings.get(
            "months_list", list(range(1, 13))
        )  # Default to all months 1-12
        max_cloud_no_data = settings.get(
            "percent_no_data", 0.80
        )  # No more than 80% of the ROI can be covered by clouds or no data
        min_roi_cov = settings.get(
            "min_roi_coverage", 0.50
        )  # At least 50% of the ROI must be covered by the image for it to be downloaded
        apply_cloud_mask = settings.get(
            "apply_cloud_mask", True
        )  # Whether to apply a cloud mask to the images
        cloud_mask_issue = settings.get(
            "cloud_mask_issue", False
        )  # CoastSat specific setting for cloud masking

        for roi_inputs in tqdm(inputs_list, desc="Downloading ROIs"):
            SDS_download.retrieve_images(
                roi_inputs,
                cloud_threshold=cloud_thresh,
                cloud_mask_issue=cloud_mask_issue,
                save_jpg=True,
                apply_cloud_mask=apply_cloud_mask,
                months_list=months_list,
                max_cloud_no_data_cover=max_cloud_no_data,
                min_roi_coverage=min_roi_cov,
            )

    def apply_post_download_filters(self, settings: dict, roi_settings: dict) -> None:
        """
        Apply optional post-download filters (e.g., image size filter).

        Args:
            settings (dict): Effective settings.
            roi_settings (dict): Dictionary of ROI settings.

        Returns:
            None
        """
        if settings.get("image_size_filter", True):
            common.filter_images_by_roi(roi_settings)

    def ensure_data_directory(self, data_path: str) -> str:
        """
        Ensures the data directory exists and returns its path.

        Args:
            data_path (str): Provided data path or empty string.

        Returns:
            str: Validated data directory path.
        """
        if data_path:
            return data_path

        base_path = os.path.abspath(core_utilities.get_base_dir())
        return file_utilities.create_directory(base_path, "data")

    def load_single_settings_file(
        self, file_path: str, file_name: str, keys: List[str]
    ) -> None:
        """
        Loads settings from a single JSON file.

        Args:
            file_path (str): Full path to the settings file.
            file_name (str): Name of the settings file (for logging).
            keys (List[str]): Expected setting keys to extract.

        Returns:
            None

        Raises:
            Exception: Re-raises any exceptions from common.load_settings for debugging.

        Example:
            >>> self.load_single_settings_file(
            ...     "/path/shoreline_settings.json",
            ...     "shoreline_settings.json",
            ...     ["cloud_thresh", "min_beach_area"]
            ... )
        """
        try:
            settings = common.load_settings(file_path, keys)
            self.set_settings(**settings)
            logger.info(f"Loaded settings from {file_name}: {list(settings.keys())}")
        except Exception as e:
            logger.error(f"Failed to load settings from {file_name}: {e}")

    def download_imagery(
        self,
        rois: Optional[gpd.GeoDataFrame] = None,
        settings: Optional[dict] = None,
        selected_ids: Optional[set] = None,
        file_path: Optional[str] = None,
    ) -> None:
        """
        Downloads all images for the selected ROIs from Landsat 5, Landsat 7, Landsat 8 and Sentinel-2 covering the area of interest and acquired between the specified dates.

        The downloaded imagery for each ROI is stored in a directory that follows the convention:
            ID_{ROI}_datetime{current date}__{time}' ex.ID_0_datetime04-11-23__10_20_48. The files are saved as jpgs in a subdirectory
            'jpg_files' in a subdirectory 'preprocessed' which contains subdirectories for RGB, NIR, and SWIR jpg imagery. The downloaded .TIF images are organised in subfolders, divided
            by satellite mission. The bands are also subdivided by pixel resolution.

        Args:
            rois (gpd.GeoDataFrame, optional): The ROIs GeoDataFrame.
            settings (dict, optional): Settings for downloading imagery. If None, uses current settings.
            selected_ids (Optional[set]): Set of selected ROI IDs.
            file_path (Optional[str]): Path to save the downloaded imagery.

        Raises:
            Exception: raised if settings is missing
            Exception: raised if 'dates','sat_list', and 'landsat_collection' are not in settings
            Exception: raised if no ROIs have been selected
        """
        # Update settings with any new settings provided
        updated_settings = self.resolve_settings(settings)
        norm_path = normalize_path(file_path)
        selected_ids = self.get_selected_ids(selected_ids)
        roi_gdf = self.ensure_rois(rois)
        self.validate_download_imagery_inputs(
            updated_settings,
            selected_ids=selected_ids,
            roi_gdf=roi_gdf,
        )

        # Create or update the roi_settings for each selected ROI
        roi_settings = self.prepare_roi_settings(
            roi_gdf=roi_gdf,
            settings=updated_settings,
            selected_ids=selected_ids,
            file_path=norm_path,
        )

        self.rois.set_roi_settings(roi_settings)  # type: ignore Validation is done by ensure_rois

        # create a list of settings for each ROI, which will be used to download the imagery
        inputs_list = list(roi_settings.values())
        logger.info(f"inputs_list {inputs_list}")

        # Save settings used to download rois and the objects on map to config files
        self.save_config()

        # For each ROI use download settings to download imagery and save to jpg
        print("Download in progress")
        self.retrieve_images_for_rois(inputs_list, updated_settings)

        self.apply_post_download_filters(updated_settings, roi_settings)

        logger.info("Done downloading")

    def load_json_config(
        self,
        filepath: str,
    ) -> dict:
        """
        Loads a JSON configuration file and updates the object's settings accordingly.
        Args:
            filepath (str): The path to the JSON configuration file.
        Returns:
            dict: The contents of the loaded JSON configuration file.
        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file content is not valid JSON or required keys are missing.

        """
        logger.info(f"Loading json config from filepath: {filepath}")
        exception_handler.check_if_None(self.rois, "rois")

        json_data = file_utilities.read_json_file(filepath, raise_error=True)
        json_data = json_data or {}
        # Replace coastseg_map.settings with settings from config file
        settings = common.load_settings(
            new_settings=json_data,
        )
        self.set_settings(**settings)
        return json_data

    def load_config_files(
        self, data_path: str, config_geojson_path: str, config_json_path: str
    ) -> bool:
        """
        Loads the configuration files from the specified directory.
        Loads config_gdf.geojson first, then config.json. This is because config.json relies on config_gdf.geojson to load the rois on the map.

        Args:
            data_path (str): Path to the directory where downloaded data will be saved.
            config_geojson_path (str): Path to the config_gdf.geojson file.
            config_json_path (str): Path to the config.json file.

        Returns:
            bool: True if both config files exist, False otherwise.

        Raises:
            Exception: Raised if config files are missing.
        """
        # if config_gdf.geojson does not exist, then this might be the wrong directory
        if not os.path.exists(config_geojson_path):
            logger.warning(f"config_gdf.geojson file missing at {config_geojson_path}")
            return False

        # config.json contains all the settings for the map, shorelines and transects it must exist
        if not os.path.exists(config_json_path):
            raise Exception(f"config.json file missing at {config_json_path}")

        # load the config files
        # load general settings from config.json file
        self.load_gdf_config(config_geojson_path)
        json_data = self.load_json_config(config_json_path)
        # creates a dictionary mapping ROI IDs to their extracted settings from json_data
        roi_settings = common.process_roi_settings(json_data, data_path)
        # Make sure each ROI has the specific settings for its save location, its ID, coordinates etc.
        if hasattr(self, "rois") and self.rois is not None:
            self.rois.update_roi_settings(roi_settings)
        logger.info(f"roi_settings: {roi_settings} loaded from {config_json_path}")

        # update the config.json files with the filepath of the data directory on this computer
        common.update_downloaded_configs(roi_settings)
        # return true if both config files exist
        return True

    def save_config(
        self,
        filepath: Optional[str] = None,
        selected_only: bool = True,
        roi_ids: Optional[list] = None,
    ) -> None:
        """
        Saves the configuration settings of the map into two files: config.json and config_gdf.geojson.

        Args:
            filepath (Optional[str]): Path to directory to save config files. Defaults to None.
            selected_only (bool, optional): If True, only the selected ROIs will be saved. Defaults to True.
            roi_ids (Optional[list], optional): A list of ROI IDs to save. Defaults to None.

        Returns:
            None

        Raises:
            Exception: If self.settings is missing.
            ValueError: If any of "dates", "sat_list", "landsat_collection" is missing from self.settings.
            Exception: If self.rois is missing.
            Exception: If selected_layer is missing.
        """
        settings = self.get_settings()

        # if no rois exist on the map do not allow configs to be saved
        exception_handler.config_check_if_none(self.rois, "ROIs")

        # Only get the selected ROIs if selected_only is True
        if not roi_ids:
            roi_ids = self.get_roi_ids(is_selected=selected_only)

        if isinstance(roi_ids, str):
            roi_ids = [roi_ids]

        # if the rois do not have any settings then save the currently loaded settings to the ROIs
        if not self.rois.get_roi_settings():  # type: ignore Validation is done by config_check_if_none
            filtered_gdf = self.rois.gdf[self.rois.gdf["id"].isin(roi_ids)]  # type: ignore Validation is done by config_check_if_none
            geojson_str = filtered_gdf.to_json()
            geojson_dict = json.loads(geojson_str)
            base_path = os.path.abspath(core_utilities.get_base_dir())
            filepath_data = filepath or os.path.abspath(os.path.join(base_path, "data"))
            roi_settings = common.create_roi_settings(
                settings,
                geojson_dict,
                filepath_data,
            )
            self.rois.set_roi_settings(roi_settings)  # type: ignore Validation is done by config_check_if_none

        # create dictionary of settings for each ROI to be saved to config.json
        selected_roi_settings = {
            roi_id: self.rois.roi_settings[roi_id] for roi_id in roi_ids
        }
        # combine the settings for each ROI with the rest of the currently loaded settings
        config_json = common.create_json_config(selected_roi_settings, settings)

        shorelines_gdf = (
            getattr(self.shoreline, "gdf", None) if self.shoreline else None
        )
        transects_gdf = getattr(self.transects, "gdf", None) if self.transects else None
        bbox_gdf = getattr(self.bbox, "gdf", None) if self.bbox else None
        ref_polygon_gdf = (
            getattr(self.shoreline_extraction_area, "gdf", None)
            if self.shoreline_extraction_area
            else None
        )

        # get the GeoDataFrame containing all the selected rois
        selected_rois = self.rois.gdf[self.rois.gdf["id"].isin(roi_ids)]
        logger.info(f"selected_rois: {selected_rois}")

        # save all selected rois, shorelines, transects and bbox to config GeoDataFrame
        if selected_rois is not None:
            if not selected_rois.empty:
                epsg_code = selected_rois.crs
        # config should always be in epsg 4326
        epsg_code = "4326"
        config_gdf = common.create_config_gdf(
            selected_rois,
            shorelines_gdf=shorelines_gdf,
            transects_gdf=transects_gdf,
            bbox_gdf=bbox_gdf,
            epsg_code=epsg_code,
            shoreline_extraction_area_gdf=ref_polygon_gdf,
        )
        logger.info(f"config_gdf: {config_gdf} ")

        def save_config_files(config_json, config_gdf, path):
            """Helper function to save config files."""
            file_utilities.config_to_file(config_json, path)
            file_utilities.config_to_file(config_gdf, path)

        if filepath:
            # save the config.json and config_gdf.geojson immediately to the filepath directory
            save_config_files(config_json, config_gdf, filepath)
            print(f"Saved config files to {filepath}")
        else:
            is_downloaded = common.were_rois_downloaded(
                self.rois.get_roi_settings(), roi_ids
            )
            # if  data has been downloaded before then inputs have keys 'filepath' and 'sitename'
            if is_downloaded:
                # write config_json file to each directory where a roi was saved
                roi_ids = config_json.get("roi_ids", [])
                for roi_id in roi_ids:
                    sitename = str(config_json[roi_id]["sitename"])
                    filepath = os.path.abspath(
                        os.path.join(config_json[roi_id]["filepath"], sitename)
                    )
                    save_config_files(config_json, config_gdf, filepath)
                print("Saved config files for each ROI")
            else:
                # if data is not downloaded save to coastseg directory
                filepath = os.path.abspath(core_utilities.get_base_dir())
                save_config_files(config_json, config_gdf, filepath)
                print(f"Saved config files for each ROI to {filepath}")

    def set_settings(self, **kwargs) -> dict:
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
            "months_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "sat_list": ["L8"],
            "download_cloud_thresh": 0.8,
            "min_roi_coverage": 0.5,
            "cloud_thresh": 0.8,
            "percent_no_data": 0.8,
            "dist_clouds": 300,
            "output_epsg": 4326,
            "check_detection": False,
            "adjust_detection": False,
            "save_figure": True,
            "min_beach_area": 4500,
            "min_length_sl": 100,
            "cloud_mask_issue": False,
            "sand_color": "default",
            "pan_off": "False",
            "max_dist_ref": 25,
            "along_dist": 25,
            "min_points": 3,
            "max_std": 15,
            "max_range": 30,
            "min_chainage": -100,
            "multiple_inter": "auto",
            "prc_multiple": 0.1,
            "apply_cloud_mask": True,
            "image_size_filter": True,
            "drop_intersection_pts": False,
            "coastseg_version": __version__,
        }

        # Function to parse dates with flexibility for different formats
        def parse_date(date_str):
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
                except ValueError:
                    continue
            raise ValueError(f"Date format for {date_str} not recognized.")

        # Update the settings with the new key-value pairs
        self.settings.update(kwargs)

        # Special handling for 'dates'
        if "dates" in kwargs:
            self.settings["dates"] = [parse_date(d) for d in kwargs["dates"]]

        for key, value in self.default_settings.items():
            self.settings.setdefault(key, value)

        logger.info(f"Set Settings: {self.settings}")
        return self.settings.copy()

    def get_settings(self) -> dict:
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
        if self.settings is None or self.settings == {}:
            raise Exception(SETTINGS_NOT_FOUND)
        return self.settings

    def _is_unknown_or_invalid(self, value) -> bool:
        """Check if a value is unknown, None, NaN, or empty string."""
        if isinstance(value, str) and value in ("unknown", ""):
            return True
        if value is None:
            return True
        try:
            return math.isnan(float(value))
        except (ValueError, TypeError):
            return False

    def _format_numeric_value(self, value, precision: int = 3) -> str:
        """Format numeric values with proper precision or return 'unknown'."""
        if self._is_unknown_or_invalid(value):
            return "unknown"
        try:
            return str(round(float(value), precision))
        except (ValueError, TypeError):
            return "unknown"

    def update_transects_html(self, feature: dict, **kwargs) -> None:
        """
        Updates the HTML display when a transect is hovered over.

        Args:
            feature (dict): The transect feature.
            **kwargs: Additional keyword arguments.
        """
        props = feature.get("properties", {})

        transect_id = props.get("id", "unknown")
        slope = props.get("slope", "unknown")
        distance = self._format_numeric_value(props.get("distance"), 3)
        feature_x = self._format_numeric_value(props.get("feature_x"), 6)
        feature_y = self._format_numeric_value(props.get("feature_y"), 6)
        nearest_x = self._format_numeric_value(props.get("nearest_x"), 6)
        nearest_y = self._format_numeric_value(props.get("nearest_y"), 6)

        if self.feature_html:
            self.feature_html.value = (
                "<div style='max-width: 230px; max-height: 200px; overflow-x: auto; overflow-y: auto'>"
                "<b>Transect</b>"
                f"<p>Id: {transect_id}</p>"
                f"<p>Slope: {slope}</p>"
                f"<p>Distance btw slope and transect: {distance}</p>"
                f"<p>Transect (x,y): ({feature_x},{feature_y})</p>"
                f"<p>Nearest Slope (x,y): ({nearest_x},{nearest_y})</p>"
                "</div>"
            )

    def update_extracted_shoreline_html(self, feature, **kwargs) -> None:
        """
        Updates the HTML display when an extracted shoreline is hovered over.

        Args:
            feature (dict): The extracted shoreline feature.
            **kwargs: Additional keyword arguments.
        """
        props = feature.get("properties", {})

        date = props.get("date", "unknown")
        cloud_cover = props.get("cloud_cover", "unknown")
        satname = props.get("satname", "unknown")
        geoaccuracy = props.get("geoaccuracy", "unknown")

        if self.feature_html:
            self.feature_html.value = (
                "<div style='max-width: 230px; max-height: 200px; overflow-x: auto; overflow-y: auto'>"
                "<b>Extracted Shoreline</b>"
                f"<p>Date: {date}</p>"
                f"<p>Geoaccuracy: {geoaccuracy}</p>"
                f"<p>Cloud Cover: {cloud_cover}</p>"
                f"<p>Satellite Name: {satname}</p>"
                "</div>"
            )

    def update_roi_html(self, feature, **kwargs) -> None:
        """
        Updates the HTML display for ROI feature information.

        Args:
            feature: The feature object.
            **kwargs: Additional keyword arguments.
        """
        props = feature.get("properties", {})

        if self.roi_html:
            self.roi_html.value = (
                "<div style='max-width: 230px; max-height: 200px; overflow-x: auto; overflow-y: auto'>"
                "<b>ROI</b>"
                f"<p>Id: {props.get('id', 'unknown')}</p>"
                "</div>"
            )

    def update_shoreline_html(self, feature, **kwargs) -> None:
        """
        Updates the HTML display when a shoreline is hovered over.

        Args:
            feature (dict): The shoreline feature.
            **kwargs: Additional keyword arguments.
        """
        props = feature.get("properties", {})

        shoreline_id = props.get("id", "unknown")
        mean_sig_waveheight = props.get("MEAN_SIG_WAVEHEIGHT", "unknown")
        tidal_range = props.get("TIDAL_RANGE", "unknown")
        erodibility = props.get("ERODIBILITY", "unknown")
        river_label = props.get("river_label", "unknown")
        sinuosity_label = props.get("sinuosity_label", "unknown")
        slope_label = props.get("slope_label", "unknown")
        turbid_label = props.get("turbid_label", "unknown")
        csu_id = props.get("CSU_ID", "unknown")

        if self.feature_html:
            self.feature_html.value = (
                "<div style='max-width: 230px; max-height: 200px; overflow-x: auto; overflow-y: auto'>"
                "<b>Shoreline</b>"
                f"<p>ID: {shoreline_id}</p>"
                f"<p>Mean Sig Waveheight: {mean_sig_waveheight}</p>"
                f"<p>Tidal Range: {tidal_range}</p>"
                f"<p>Erodibility: {erodibility}</p>"
                f"<p>River: {river_label}</p>"
                f"<p>Sinuosity: {sinuosity_label}</p>"
                f"<p>Slope: {slope_label}</p>"
                f"<p>Turbid: {turbid_label}</p>"
                f"<p>CSU_ID: {csu_id}</p>"
                "</div>"
            )

    def get_all_roi_ids(self) -> List[str]:
        """
        Return a list of all ROI IDs.

        Returns:
            List[str]: List of all ROI IDs.
        """
        if self.rois is None or not hasattr(self, "rois"):
            return []
        if not hasattr(self.rois, "gdf"):
            return []
        if self.rois.gdf.empty:
            return []
        if "id" not in self.rois.gdf.columns:
            return []
        return self.rois.gdf["id"].tolist()

    def get_any_available_roi_id(self) -> List[str]:
        """
        Returns a list containing any available ROI ID.

        Returns:
            List[str]: List with any available ROI ID.
        """
        roi_ids = self.get_roi_ids(is_selected=True)
        if roi_ids == []:
            roi_ids = self.get_all_roi_ids()
            if roi_ids == []:
                return roi_ids
            roi_ids = [roi_ids[0]]
        return roi_ids

    def load_extracted_shoreline_files(self) -> None:
        """
        Loads extracted shoreline files for each ROI and adds them to the map.

        Side Effects:
            Updates the `self.rois` object with loaded extracted shorelines.

        Raises:
            Exception: If no extracted shorelines could be loaded.
        """
        exception_handler.config_check_if_none(self.rois, "ROIs")
        # load extracted shorelines for either a selected ROI or the first ROI if no ROI is selected
        roi_ids = self.get_any_available_roi_id()
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
                    # load GeoDataFrame
                    extracted_sl_gdf = geodata_processing.read_gpd_file(file)
                if file.endswith(".json"):
                    if "settings" in os.path.basename(file):
                        shoreline_settings = file_utilities.load_data_from_json(file)
                    if "dict" in os.path.basename(file):
                        extracted_shoreline_dict = file_utilities.load_data_from_json(
                            file
                        )
            # If any of the extracted shoreline files are missing, skip to next ROI
            if extracted_sl_gdf is None:
                logger.info(
                    f"ROI {roi_id} didn't have extracted shoreline files to load"
                )
                rois_no_extracted_shorelines.add(roi_id)
                # set this roi's entry in extracted shorelines dict to None because there was no shoreline extracted
                self.rois.add_extracted_shoreline(None, roi_id)  # type: ignore
                continue
            else:
                extracted_shorelines = extracted_shoreline.Extracted_Shoreline()
                extracted_shorelines = extracted_shorelines.load_extracted_shorelines(
                    extracted_shoreline_dict,
                    shoreline_settings,
                    extracted_sl_gdf,  # type: ignore
                )
                self.rois.add_extracted_shoreline(extracted_shorelines, roi_id)  # type: ignore
                logger.info(
                    f"ROI {roi_id} successfully loaded extracted shorelines: {self.rois.get_extracted_shoreline(roi_id)}"  # type: ignore
                )
        if len(rois_no_extracted_shorelines) > 0:
            logger.warning(
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
        session_path: Optional[str] = None,
        shoreline_extraction_area: Optional[gpd.GeoDataFrame] = None,
    ) -> Optional[extracted_shoreline.Extracted_Shoreline]:
        """
        Extracts shoreline for a given ROI.

        Args:
            roi_id (str): The ROI ID to extract the shoreline from
            rois_gdf (gpd.GeoDataFrame): The ROIs GeoDataFrame containing all ROIs.
            shoreline_gdf (gpd.GeoDataFrame): The shoreline GeoDataFrame.
            settings (dict): Settings for shoreline extraction.
            session_path (Optional[str]): Path to the session directory.
            shoreline_extraction_area (gpd.GeoDataFrame, optional): Extraction area.
        Returns:
            Optional[Extracted_Shoreline]: The extracted shoreline object for the ROI or None if an error occurs.
        """
        try:
            logger.info(f"Extracting shorelines from ROI with the id: {roi_id}")
            roi_settings = self.rois.get_roi_settings(roi_id)  # type: ignore
            single_roi = ROI.extract_roi_by_id(rois_gdf, roi_id)
            # Clip shoreline to specific roi
            shoreline_in_roi = gpd.clip(shoreline_gdf, single_roi)
            # extract shorelines from ROI
            extracted_shorelines = extracted_shoreline.Extracted_Shoreline()
            extracted_shorelines = extracted_shorelines.create_extracted_shorelines(
                roi_id,
                shoreline_in_roi,
                roi_settings,
                settings,
                output_directory=session_path,
                shoreline_extraction_area=shoreline_extraction_area,
            )
            logger.info(f"extracted_shoreline_dict[{roi_id}]: {extracted_shorelines}")
            return extracted_shorelines
        except exceptions.Id_Not_Found as id_error:
            logger.warning(
                f"exceptions.Id_Not_Found {id_error} {traceback.format_exc()}"
            )
            print(f"ROI with id {roi_id} was not found. \n Skipping to next ROI")
        except exceptions.No_Extracted_Shoreline as no_shoreline:
            logger.warning(f"{roi_id}: {no_shoreline} {traceback.format_exc()}")
            print(f"{roi_id}: {no_shoreline}")
        except Exception as e:
            logger.warning(
                f"An error occurred while extracting shoreline for ROI {roi_id}: {e} \n {traceback.format_exc()}"
            )
            print(
                f"An error occurred while extracting shoreline for ROI {roi_id}. \n Skipping to next ROI \n {e} \n {traceback.format_exc()}"
            )

        return None

    def update_settings_with_accurate_epsg(self, gdf: "gpd.GeoDataFrame") -> dict:
        """
        Updates the settings with the UTM EPSG code based on the provided GeoDataFrame.
        This is necessary because for shoreline extraction and transect computation, the data needs to be in a projected coordinate system (like UTM) rather than a geographic coordinate system (like WGS84).

        Args:
            gdf (gpd.GeoDataFrame): The GeoDataFrame to extract EPSG from.

        Returns:
            dict: The updated settings dictionary.
        Raises:
            Exception: If the GeoDataFrame does not have a CRS attribute.
        """
        settings = self.get_settings()
        if hasattr(gdf, "crs"):
            new_espg = common.get_most_accurate_epsg(
                settings.get("output_epsg", 4326), gdf
            )
            self.set_settings(output_epsg=new_espg)
        else:
            raise Exception("The GeoDataFrame does not have a crs attribute")
        return self.get_settings()

    def validate_transect_inputs(
        self, settings: dict, roi_ids: Optional[list] = None
    ) -> None:
        """
        Validates the inputs for transect computation.

        Args:
            settings (dict): Settings for transect computation.
            roi_ids (Optional[list]): List of ROI IDs.

        Returns:
            None

        Raises:
            ValueError: If any of the following conditions are met:
                - The session name is an empty string.
                - The ROIs, transects, or extracted shorelines are None.
                - The extracted shorelines dictionary is empty.
                - The 'along_dist' key is missing from the settings dictionary.
                - The roi_ids list is empty.
        """
        # ROIs, settings, roi-settings cannot be None or empty
        exception_handler.check_if_empty_string(self.get_session_name(), "session name")
        # ROIs, transects, and extracted shorelines must exist
        exception_handler.check_if_None(self.rois, "ROIs")
        exception_handler.check_if_None(self.transects, "transects")
        exception_handler.check_empty_dict(
            self.rois.get_all_extracted_shorelines(),
            "extracted_shorelines",  # type: ignore
        )
        # settings must contain key 'along_dist'
        exception_handler.check_if_subset(
            set(["along_dist"]), set(list(settings.keys())), "settings"
        )
        exception_handler.check_if_empty(roi_ids)

    def validate_extract_shoreline_inputs(
        self, roi_ids: Optional[list] = None, settings: Optional[dict] = None
    ) -> None:
        """
        Validates the inputs required for extracting shorelines.

        Ensures that the required settings are present and that the selected layer contains the selected ROI.
        It also validates that the ROIs have been downloaded before by checking for the presence roi_settings and ensures
        that the directories for the ROIs exist.

        Args:
            roi_ids (list, optional): List of ROI IDs. Defaults to None.
            settings (dict, optional): Dictionary of settings. Defaults to None.

        Raises:
            Exception: If any of the required inputs are missing or invalid.

        Returns:
            None
        """
        # ROIs,settings, roi-settings cannot be None or empty
        if not settings:
            settings = self.get_settings()
        if not roi_ids:
            # if no rois are selected throw an error
            exception_handler.check_selected_set(self.selected_set)

        exception_handler.check_if_empty_string(self.get_session_name(), "session name")
        # ROIs, transects,shorelines and a bounding box must exist
        exception_handler.validate_feature(self.rois, "roi")
        exception_handler.validate_feature(self.shoreline, "shoreline")
        exception_handler.validate_feature(self.transects, "transects")
        # ROI settings must not be empty
        if hasattr(self.rois, "roi_settings"):
            exception_handler.check_empty_dict(self.rois.roi_settings, "roi_settings")  # type: ignore
        else:
            raise Exception(
                "None of the ROIs have been downloaded on this machine or the location where they were downloaded has been moved. Please download the ROIs again."
            )

        # settings must contain keys "dates", "sat_list", "landsat_collection"
        superset = set(list(settings.keys()))
        exception_handler.check_if_subset(
            set(["dates", "sat_list", "landsat_collection"]), superset, "settings"
        )

        # roi_settings must contain roi ids in selected set
        superset = set(list(self.rois.roi_settings.keys()))  # type: ignore
        error_message = "To extract shorelines you must first select ROIs and have the data downloaded."
        exception_handler.check_if_subset(
            self.selected_set, superset, "roi_settings", error_message
        )
        # get only the rois with missing directories that are selected on the map
        roi_ids = self.get_roi_ids(is_selected=True)
        # check if any of the ROIs are missing their downloaded data directory
        missing_directories = common.get_missing_roi_dirs(
            self.rois.get_roi_settings(),
            roi_ids,  # type: ignore
        )
        # raise an warning if any of the selected ROIs were not downloaded
        exception_handler.check_if_dirs_missing(missing_directories)

    def validate_download_imagery_inputs(
        self,
        settings: Optional[dict] = None,
        selected_ids: Optional[set] = None,
        roi_gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> None:
        """
        Validates the inputs required for downloading imagery.

        This method checks if the necessary settings are present and if the selected layer contains the selected ROI.

        Args:
            settings (Optional[dict]): Settings for downloading imagery.
            selected_ids (Optional[set]): Set of selected ROI IDs.
            roi_gdf (gpd.GeoDataFrame, optional): The ROIs GeoDataFrame.

        Raises:
            SubsetError: If the required settings keys are not present in the settings.
            EmptyLayerError: If the selected layer is empty.
            EmptyROILayerError: If the selected layer does not contain any ROI.
        """
        if settings is None:
            raise Exception("Settings are missing")
        # Ensure the required keys are present in the settings
        required_settings_keys = set(["dates", "sat_list", "landsat_collection"])
        superset = set(list(settings.keys()))
        exception_handler.check_if_subset(required_settings_keys, superset, "settings")
        dates = settings.get("dates", [])
        if dates == []:
            raise Exception(
                'No dates provided to download imagery. Please provide a start date and end date in the format "YYYY-MM-DD". Example  ["2017-12-01", "2018-01-01"]'
            )
        dates = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
        if dates[1] <= dates[0]:
            raise Exception(
                "Verify that your dates are in the correct chronological order"
            )
        if settings.get("sat_list", []) == []:
            raise Exception("No satellite list provided to download imagery")
        if settings.get("landsat_collection", "") == "":
            raise Exception("No landsat collection provided to download imagery")
        if settings.get("landsat_collection", "") == "CO2":
            raise Exception(
                "Error CO2 is not a valid collection. Did you mean to use the landsat collection 'C02'?"
            )

        if not selected_ids:
            raise Exception(
                "No ROIs have been selected. Please enter the IDs of the ROIs you want to download imagery for"
            )

        if roi_gdf is None:
            raise Exception("No ROIs provided to download imagery")
        if roi_gdf.empty:
            raise Exception("No ROIs provided to download imagery")
        filtered_gdf = roi_gdf[roi_gdf["id"].isin(selected_ids)]
        if filtered_gdf.empty:
            raise Exception(
                "None of the selected ids were ids in ROIs. Please enter the IDs of the ROIs you want to download imagery for"
            )

    def get_roi_ids(
        self, is_selected: bool = False, has_shorelines: bool = False
    ) -> list:
        """
        Returns a list of ROI IDs based on selection and shoreline availability.

        Args:
            is_selected (bool, optional): Whether to consider only the selected ROIs on the map. Defaults to True.
            has_shorelines (bool, optional): Whether to consider only the ROIs that have extracted shorelines. Defaults to False.

        Returns:
            list: The IDs of the ROIs that meet the specified criteria.
        """
        if self.rois is None:
            return []
        roi_ids = self.get_all_roi_ids()
        if has_shorelines:
            roi_ids = set(self.rois.get_ids_with_extracted_shorelines())  # type: ignore
        if is_selected:
            roi_ids = set(roi_ids) & self.selected_set
        return list(roi_ids)

    def extract_all_shorelines(self, roi_ids: Optional[list] = None) -> None:
        """
        Extracts shorelines for all selected regions of interest (ROIs).

        This method performs the following steps:
        1. Validates the inputs for shoreline extraction.
        2. Retrieves the IDs of the selected ROIs.
        3. Updates the settings with the most accurate EPSG.
        4. Saves the updated configurations.
        5. Extracts shorelines for each selected ROI.
        6. Saves the ROI IDs that had extracted shorelines.
        7. Saves a session for each ROI.
        8. Computes transects for selected ROIs with extracted shorelines.
        9. Loads extracted shorelines to the map.
        10. Updates the available ROI IDs with extracted shorelines.
        Note: This method assumes that the necessary data structures and attributes are already initialized.

        Args:
            roi_ids (Optional[list]): List of ROI IDs. If none are provided, the method will use the selected ROIs on the map.

        Returns:
            None
        """
        if isinstance(roi_ids, str):
            roi_ids = [roi_ids]
        # 1. validate the inputs for shoreline extraction exist: ROIs, transects,shorelines and a downloaded data for each ROI
        self.validate_extract_shoreline_inputs(roi_ids)

        # if ROI ids are not provided then get the selected ROI ids from the map
        if not roi_ids:
            roi_ids = self.get_roi_ids(is_selected=True)

        # get the ROIs and check if they have settings
        missing_directories = self.get_missing_directories()

        # remove the ROI IDs that are missing directories from the list of ROI IDs
        if len(missing_directories) > 0:
            original_roi_ids = roi_ids
            roi_ids = list(set(roi_ids) - set(missing_directories.keys()))
            if len(roi_ids) == 0:
                raise Exception(
                    f"None of the selected ROIs {original_roi_ids} have been downloaded. Please download the ROIs again or load their data into the data directory."
                )

        logger.info(f"roi_ids to extract shorelines from: {roi_ids}")
        # 2. update the settings with the most accurate epsg
        if self.bbox:
            self.update_settings_with_accurate_epsg(self.bbox.gdf)
        else:
            # pick the first ROI ID and use it to update the settings with the most accurate epsg
            roi_id = roi_ids[0]
            single_roi = ROI.extract_roi_by_id(self.rois.gdf, roi_id)
            self.update_settings_with_accurate_epsg(single_roi)

        shoreline_extraction_area_gdf = (
            getattr(self.shoreline_extraction_area, "gdf", None)
            if self.shoreline_extraction_area
            else None
        )

        # 3. get selected ROIs on map and extract shoreline for each of them
        for roi_id in tqdm(roi_ids, desc="Extracting Shorelines"):
            # Create the session for the selected ROIs
            session_path = self.create_session(
                self.get_session_name(), roi_id, save_config=True
            )
            print(f"Extracting shorelines from ROI with the id:{roi_id}")
            extracted_shorelines = self.extract_shoreline_for_roi(
                roi_id,
                self.rois.gdf,
                self.shoreline.gdf,
                self.get_settings(),
                session_path,
                shoreline_extraction_area_gdf,
            )
            self.rois.add_extracted_shoreline(extracted_shorelines, roi_id)

            # update the extracted shorelines on the map if the map is available
            if extracted_shorelines is not None and self.map is not None:
                self.update_extracted_shorelines_display(roi_id)

        # 4. save the ROI IDs that had extracted shoreline to observable variable roi_ids_with_extracted_shorelines
        ids_with_extracted_shorelines = self.get_roi_ids(
            is_selected=False, has_shorelines=True
        )
        # update the available ROI IDs and this will update the extracted shorelines on the map
        if ids_with_extracted_shorelines is None:
            self.id_container.ids = []
        elif not isinstance(ids_with_extracted_shorelines, list):
            self.id_container.ids = list(ids_with_extracted_shorelines)
        else:
            self.id_container.ids = ids_with_extracted_shorelines

        # 4. save a session for each ROI under one session name
        self.save_session(roi_ids, save_transects=False)

        # 5. Get ROI ids with retrieved shorelines and compute the shoreline transect intersections
        roi_ids_with_extracted_shorelines = self.get_roi_ids(has_shorelines=True)
        # get the transects for the selected ROIs with extracted shorelines
        selected_roi_ids = list(set(roi_ids) & set(roi_ids_with_extracted_shorelines))
        if hasattr(self.transects, "gdf"):
            self.compute_transects(
                self.transects.gdf,
                self.get_settings(),
                selected_roi_ids,  # type: ignore
            )

        # update the available ROI IDs and this will update the extracted shorelines on the map
        ids_with_extracted_shorelines = self.update_roi_ids_with_shorelines()
        logger.info(
            f"Available roi_ids from extracted shorelines: {ids_with_extracted_shorelines}"
        )

    def get_cross_distance(
        self,
        roi_id: str,
        transects_in_roi_gdf: gpd.GeoDataFrame,
        settings: dict,
        output_epsg: Union[int, str],
    ) -> Tuple[float, Optional[str]]:
        """
        Computes the cross shore distance of transects and extracted shorelines for a given ROI.

        Args:
            roi_id (str): The ROI ID.
            transects_in_roi_gdf (gpd.GeoDataFrame): Transects GeoDataFrame for the ROI.
            settings (dict): Settings for computation.
            output_epsg (Union[int, str]): Output EPSG code.
        Returns:
        --------
        Tuple[float, Optional[str]]
            The computed cross shore distance, or 0 if there was an issue in the computation.
            The reason for failure, or '' if the computation was successful.
        """
        failure_reason = ""
        cross_distance = 0

        transects_in_roi_gdf = transects_in_roi_gdf.loc[:, ["id", "geometry"]]

        if transects_in_roi_gdf.empty:
            failure_reason = f"No transects intersect for the ROI {roi_id}"
            return cross_distance, failure_reason

        # Get extracted shorelines object for the currently selected ROI
        roi_extracted_shoreline = self.rois.get_extracted_shoreline(roi_id)  # type: ignore

        if roi_extracted_shoreline is None:
            failure_reason = f"No extracted shorelines were found for the ROI {roi_id}"
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

        return cross_distance, failure_reason  # type: ignore

    def compute_transects_per_roi(
        self,
        roi_gdf: gpd.GeoDataFrame,
        transects_gdf: gpd.GeoDataFrame,
        settings: dict,
        roi_id: str,
        output_epsg: Union[int, str],
    ) -> float:
        """
        Computes transects for a specific ROI.

        Args:
            roi_gdf (gpd.GeoDataFrame): ROI GeoDataFrame.
            transects_gdf (gpd.GeoDataFrame): Transects GeoDataFrame.
            settings (dict): Settings for computation.
            roi_id (str): The ROI ID.
            output_epsg (Union[int, str]): Output EPSG code.

        Returns:
            float: The computed cross distance for the ROI.
        """
        # get transects that intersect with ROI
        single_roi = ROI.extract_roi_by_id(roi_gdf, roi_id)
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
        return cross_distance

    def compute_transects(
        self, transects_gdf: gpd.GeoDataFrame, settings: dict, roi_ids: list[str]
    ) -> None:
        """
        Computes transects for a list of ROI IDs.

        Args:
            transects_gdf (gpd.GeoDataFrame): Transects GeoDataFrame.
            settings (dict): settings used for CoastSat. Must have the following fields:
               'output_epsg': int
                    output spatial reference system as EPSG code
                'along_dist': int
                    alongshore distance considered to calculate the intersection
            roi_ids (list[str]): List of ROI IDs.

        Returns:
            dict: cross_distances_rois with format:
                { roi_id :  dict
                    time-series of cross-shore distance along each of the transects. Not tidally corrected. }
        """
        self.validate_transect_inputs(settings, roi_ids)
        # user selected output projection
        output_epsg = "epsg:" + str(settings["output_epsg"])
        # for each ROI save cross distances for each transect that intersects each extracted shoreline
        for roi_id in tqdm(roi_ids, desc="Computing Cross Distance Transects"):
            cross_distance = self.compute_transects_per_roi(
                self.rois.gdf, transects_gdf, settings, roi_id, output_epsg
            )
            self.rois.add_cross_shore_distances(cross_distance, roi_id)
            # save all the files that use the cross distance (aka the timeseries of shoreline intersections along transects)
            session_path = self.create_session(
                self.get_session_name(), roi_id, save_config=False
            )
            self.save_transect_timeseries(
                session_path, self.rois.get_extracted_shoreline(roi_id), roi_id
            )

    def create_session(
        self, session_name: str, roi_id: Optional[str] = None, save_config: bool = False
    ) -> str:
        """
        Creates a new session with the given name and optional ROI ID.

        Args:
            session_name (str): The name of the session.
            roi_id (Optional[str]): The ROI ID.
            save_config (bool, optional): Whether to save the configuration. Defaults to False.

        Returns:
            str: The path to the created session.
        """
        # name of the directory where the extracted shorelines will be saved under the session name
        ROI_directory = self.rois.roi_settings[roi_id]["sitename"]  # type: ignore
        session_path = file_utilities.create_session_path(session_name, ROI_directory)
        if save_config:
            self.save_config(session_path)
        return session_path

    def save_session(self, roi_ids: list[str], save_transects: bool = True) -> None:
        """
        Saves the current session for the specified ROI IDs. Ensures that the session directory exists and saves the configuration and extracted shorelines.

        Args:
            roi_ids (list[str]): List of ROI IDs.
            save_transects (bool): Whether to save transects. Defaults to True.

        Returns:
            None
        """
        if self.rois is None:
            raise Exception("ROIs have not been defined cannot save session")
        if isinstance(roi_ids, str):
            roi_ids = [roi_ids]
        # Save extracted shoreline info to session directory
        session_name = self.get_session_name()
        for roi_id in roi_ids:
            ROI_directory = self.rois.roi_settings[roi_id]["sitename"]
            # create session directory
            session_path = file_utilities.create_session_path(
                session_name, ROI_directory
            )
            # save source data
            self.save_config(session_path, roi_ids=roi_ids)
            # save extracted shorelines
            extracted_shoreline = self.rois.get_extracted_shoreline(roi_id)
            logger.info(f"Extracted shorelines for ROI {roi_id}: {extracted_shoreline}")
            if extracted_shoreline is None:
                logger.info(f"No extracted shorelines for ROI: {roi_id}")
                continue
            # save the geojson and json files for extracted shorelines
            common.save_extracted_shorelines(extracted_shoreline, session_path)

            # save transects to session folder
            if save_transects:
                self.save_transect_timeseries(session_path, extracted_shoreline, roi_id)

    def save_transect_timeseries(
        self,
        session_path: str,
        extracted_shoreline: "extracted_shoreline.Extracted_Shoreline",
        roi_id: str = "",
    ) -> None:
        """
        Saves the transect time series for a given session and ROI.

        Args:
            session_path (str): Path to the session directory.
            extracted_shoreline (Extracted_Shoreline): The extracted shoreline object.
            roi_id (str, optional): The ROI ID.

        Returns:
            None
        """
        # save transects to session folder
        if extracted_shoreline is None:
            logger.info(f"No extracted shorelines for roi {roi_id}")
            return
        # get extracted_shorelines from extracted shoreline object in rois
        extracted_shorelines_dict = extracted_shoreline.dictionary
        # if no shorelines were extracted then skip
        if extracted_shorelines_dict == {}:
            logger.info(f"No extracted shorelines for roi {roi_id}")
            return
        if self.rois is not None:
            cross_shore_distance = self.rois.get_cross_shore_distances(roi_id)
        # if no cross distance was 0 then skip
        if cross_shore_distance == 0:
            print(
                f"ROI: {roi_id} had no time-series of shoreline change along transects"
            )
            logger.info(f"ROI: {roi_id} cross distance is 0")
            return
        # get the setting that control whether shoreline intersection points that are not on the transects are kept
        drop_intersection_pts = self.get_settings().get("drop_intersection_pts", False)
        common.save_transects(
            session_path,
            cross_shore_distance,
            extracted_shorelines_dict,
            self.get_settings(),
            self.transects.gdf,
            drop_intersection_pts,
        )

    def remove_all(self) -> None:
        """Remove all features from the map"""
        self.remove_bbox()
        self.remove_shoreline_extraction_area()
        self.remove_shoreline()
        self.remove_transects()
        self.remove_all_rois()
        self.remove_layer_by_name("geodataframe")
        self.remove_extracted_shorelines()

    def remove_extracted_shorelines(self) -> None:
        """Removes all extracted shorelines from the map and removes extracted shorelines from ROIs"""
        # empty extracted shorelines dictionary
        if self.rois is not None:
            self.rois.remove_extracted_shorelines(remove_all=True)
        # remove extracted shoreline vectors from the map
        self.remove_extracted_shoreline_layers()
        self.id_container.ids = []
        self.extract_shorelines_container.clear()

    def remove_extracted_shoreline_layers(self) -> None:
        """Removes all extracted shoreline layers from the map."""
        self.remove_layer_by_name("delete")
        self.remove_layer_by_name("extracted shoreline")

    def clear_draw_control(self) -> None:
        """Clears the draw control on the map."""
        if self.draw_control is not None:
            self.draw_control.clear()

    def remove_bbox(self) -> None:
        """Removes the bounding box from the map and clears the draw control."""
        self.clear_draw_control()

        if self.map is not None:
            existing_layer = self.map.find_layer(Bounding_Box.LAYER_NAME)
            if existing_layer is not None:
                self.map.remove_layer(existing_layer)
        self.bbox = None

    def remove_shoreline_extraction_area(self) -> None:
        """
        Removes the shoreline_extraction_area from the map and clears the draw control.

        If a shoreline_extraction_area exists, it is deleted. The draw control is also cleared.
        Additionally, if a layer with the name Shoreline_Extraction_Area.LAYER_NAME exists in the map,
        it is removed.
        """
        if self.shoreline_extraction_area is not None:
            del self.shoreline_extraction_area
        self.clear_draw_control()

        if self.map is not None:
            existing_layer = self.map.find_layer(Shoreline_Extraction_Area.LAYER_NAME)
            if existing_layer is not None:
                self.map.remove_layer(existing_layer)
        self.shoreline_extraction_area = None

    def remove_layer_by_name(self, layer_name: str) -> None:
        """
        Removes a layer from the map by its name.

        Args:
            layer_name (str): The name of the layer to remove.

        Returns:
            None
        """
        if self.map is None:
            return
        existing_layer = self.map.find_layer(layer_name)
        if existing_layer is not None:
            self.map.remove(existing_layer)

    def remove_shoreline(self) -> None:
        """Removes the shoreline from the map."""
        del self.shoreline
        self.remove_layer_by_name(Shoreline.LAYER_NAME)
        self.shoreline = None

    def remove_transects(self) -> None:
        """Removes transects from the map."""
        self.transects = None
        self.remove_layer_by_name(Transects.LAYER_NAME)

    def replace_layer_by_name(
        self,
        layer_name: str,
        new_layer: Optional[GeoJSON],
        on_hover: Optional[Callable] = None,
        on_click: Optional[Callable] = None,
    ) -> None:
        """
        Replaces a layer on the map by its name with a new layer.

        Adds on_hover and on_click callable functions
        as handlers for hover and click events on new_layer

        Args:
            layer_name (str): The name of the layer to replace.
            new_layer (Optional[GeoJSON]): The new GeoJSON layer to add to the map. If None, the function does nothing.
            on_hover: Optional hover callback.
            on_click: Optional click callback. This function should take the event and the feature as inputs.

        Returns:
            None
        """
        if new_layer is None or self.map is None:
            return
        self.remove_layer_by_name(layer_name)
        # when feature is hovered over on_hover function is called
        if on_hover is not None:
            new_layer.on_hover(on_hover)
        if on_click is not None:
            # when feature is clicked on on_click function is called
            new_layer.on_click(on_click)
        self.map.add_layer(new_layer)

    def remove_all_rois(self) -> None:
        """Removes all ROIs from the map

        Side Effects:
            - Clears the selected_set of ROI IDs.
            - Sets rois to None.
        """
        # Remove the selected and unselected rois
        if self.map is not None:
            self.remove_layer_by_name(ROI.SELECTED_LAYER_NAME)
            self.remove_layer_by_name(ROI.LAYER_NAME)
        # clear all the ids from the selected set
        self.selected_set = set()
        del self.rois
        self.rois = None

    def remove_selected_shorelines(self) -> None:
        """
        Removes selected shorelines from the map.

        Side Effects:
            - Clears the selected_shorelines_set of shoreline IDs.
            - Removes the selected shorelines from the shoreline dataframe.
            - Reloads the remaining shorelines on the map.
        """
        logger.info("Removing selected shorelines from map")
        # Remove the selected and unselected rois
        if self.map is not None:
            self.remove_layer_by_name(SELECTED_LAYER_NAME)
            self.remove_layer_by_name(Shoreline.LAYER_NAME)
        # delete selected ROIs from the shoreline dataframe
        if self.shoreline:
            self.shoreline.remove_by_id(self.selected_shorelines_set)
        # clear all the ids from the shoreline selected set
        self.selected_shorelines_set = set()
        # reload rest of shorelines on map now that selected shorelines have been removed
        if hasattr(self.shoreline, "gdf"):
            self.load_feature_on_map(
                "shoreline",
                gdf=self.shoreline.gdf,
                zoom_to_bounds=True,  # type: ignore
            )

    def remove_selected_rois(self) -> None:
        """
        Removes selected ROIs from the map.

        Side Effects:
            - Clears the selected_set of ROI IDs.
            - Removes the selected ROIs from the ROIs dataframe.
            - Reloads the remaining ROIs on the map.
        """
        # Remove the selected and unselected rois
        if self.map is not None:
            self.remove_layer_by_name(ROI.SELECTED_LAYER_NAME)
            self.remove_layer_by_name(ROI.LAYER_NAME)
        # delete selected ROIs from dataframe
        if self.rois:
            self.rois.remove_by_id(self.selected_set)
        # clear all the ids from the selected set
        self.selected_set = set()
        # reload rest of ROIs on map
        if hasattr(self.rois, "gdf"):
            self.load_feature_on_map("roi", gdf=self.rois.gdf, zoom_to_bounds=True)  # type: ignore

    def create_DrawControl(self, draw_control: DrawControl) -> DrawControl:
        """
        Creates and configures a DrawControl for the map.
        Only polygon and rectangle drawing tools are enabled.

        Args:
            draw_control (DrawControl): The DrawControl object to configure.

        Returns:
            DrawControl: The configured DrawControl object.
        """
        draw_control.polyline = {}
        draw_control.circlemarker = {}
        draw_control.polygon = {
            "shapeOptions": {
                "fillColor": "black",
                "color": "black",
                "fillOpacity": 0.1,
                "Opacity": 0.1,
            },
            "drawError": {"color": "#dd253b", "message": "Ops!"},
            "allowIntersection": False,
            "transform": True,
        }
        draw_control.rectangle = {
            "shapeOptions": {
                "fillColor": "black",
                "color": "black",
                "fillOpacity": 0.1,
                "Opacity": 0.1,
            },
            "drawError": {"color": "#dd253b", "message": "Ops!"},
            "allowIntersection": False,
            "transform": True,
        }
        return draw_control

    def _handle_shoreline_extraction_area(self, gdf: gpd.GeoDataFrame) -> None:
        """
        Handler that creates or updates the shoreline extraction area.
        Adds the shoreline extraction area to the map and clears the draw control.

        Args:
            gdf (gpd.GeoDataFrame): The GeoDataFrame containing the shoreline extraction area geometry.
        """
        from coastseg.shoreline_extraction_area import Shoreline_Extraction_Area

        # Create or update shoreline extraction area
        if self.shoreline_extraction_area is None:
            self.shoreline_extraction_area = Shoreline_Extraction_Area(gdf)
        else:  # append the new geometry to the existing gdf
            self.shoreline_extraction_area.gdf = pd.concat(
                [self.shoreline_extraction_area.gdf, gdf], ignore_index=True
            )

        # Add layer to map and cleanup
        self.add_feature_on_map(
            self.shoreline_extraction_area,
            self.shoreline_extraction_area.LAYER_NAME,
            self.shoreline_extraction_area.LAYER_NAME,
        )
        self.clear_draw_control()

    def _handle_bbox_creation(self, geometry: dict, gdf: gpd.GeoDataFrame) -> None:
        """
        Handle bounding box creation or update.

        Args:
            geometry (dict): The geometry of the bounding box. Received from the draw control.
            gdf (gpd.GeoDataFrame): The GeoDataFrame containing the bounding box geometry.
        """
        bbox_area = common.get_area(geometry)
        try:
            Bounding_Box.check_bbox_size(bbox_area)
            # Success case: create and load new bbox
            self.load_feature_on_map("bbox", gdf=gdf)
        except (
            exceptions.BboxTooLargeError,
            exceptions.BboxTooSmallError,
        ) as bbox_error:
            # Handle both bbox size errors uniformly
            self.remove_bbox()
            exception_handler.handle_bbox_error(bbox_error, self.warning_box)

    def handle_draw(self, target, **kwargs) -> None:
        """
        Handles draw events on the map.

        This method is designed to be used as an event handler for ipyleaflet's DrawControl.
        The target parameter is the DrawControl object that triggered the event.

        Args:
            target: The DrawControl object that triggered the draw event.
            **kwargs: Additional keyword arguments from the event handler.
                Expected kwargs include:
                - action (str): The draw action performed (e.g., 'created', 'deleted').
                - geo_json (dict): The GeoJSON data of the drawn feature.

        Returns:
            None

        Side Effects:
            - Adds or removes the drawn feature from the map.
            - Updates the internal state of the DrawControl.
            - Clears the DrawControl after processing the draw event.
            - Validates the size of the drawn bounding box.
            - Adds the drawn shoreline extraction area or bounding box to the map.
        """
        print(f"Draw event received with kwargs: {kwargs}")
        if self.draw_control is None:
            return

        if self.draw_control.last_action == "created":
            # get the last drawn geometry
            geometry = self.draw_control.last_draw["geometry"]
            if not isinstance(geometry, dict):
                logger.warning(
                    f"Drawn geometry is not a valid GeoJSON dictionary. Geometry: {geometry}"
                )
                return
            gdf = gpd.GeoDataFrame({"geometry": [shape(geometry)]}, crs="EPSG:4326")

            # Create a bbox or shoreline extraction area based on the current mode
            if self.drawing_shoreline_extraction_area:
                self._handle_shoreline_extraction_area(gdf)
            else:  # assume drawing bounding box control is on if not drawing shoreline extraction area
                self._handle_bbox_creation(geometry, gdf)

        if self.draw_control.last_action == "deleted":
            self.remove_bbox()

    def load_extracted_shoreline_by_id(
        self, selected_id: str, row_number: int = 0
    ) -> None:
        """
        Loads an extracted shoreline by ROI ID and row number.

        Args:
            selected_id (str): The ROI ID to load extracted shorelines for.
            row_number (int, optional): The row number of the region of interest to plot. Defaults to 0.

        Returns:
            None
        """
        # remove any existing extracted shorelines
        self.remove_extracted_shoreline_layers()
        # get the extracted shorelines for the selected roi
        if self.rois is not None:
            extracted_shorelines = self.rois.get_extracted_shoreline(selected_id)
            # if extracted shorelines exist then load them onto map, if none exist nothing loads
            self.load_extracted_shorelines_on_map(extracted_shorelines, row_number)

    def update_roi_ids_with_shorelines(self) -> list[str]:
        """
        Updates and returns the list of ROI IDs that have extracted shorelines.

        Side Effects:
            - Updates the id_container.ids with the ROI IDs that have extracted shorelines.
            - Updates the extract_shorelines_container.roi_ids_list with the ROI IDs that have extracted shorelines.

        Returns:
            list[str]: List of ROI IDs with extracted shorelines.
        """
        # Get the list of the ROI IDs that have extracted shorelines
        ids_with_extracted_shorelines = self.get_roi_ids(has_shorelines=True)
        logger.info(f"ids_with_extracted_shorelines: {ids_with_extracted_shorelines}")
        # if no ROIs have extracted shorelines, return otherwise load extracted shorelines for the first ROI ID with extracted shorelines
        if not ids_with_extracted_shorelines:
            self.id_container.ids = []
            self.extract_shorelines_container.roi_ids_list = []
            self.extract_shorelines_container.load_list = []
            self.extract_shorelines_container.trash_list = []
            logger.warning("No ROIs found with extracted shorelines.")
            return []
        # save the ROI IDs that had extracted shoreline to observable variables self.id_container.ids and self.extract_shorelines_container.roi_ids_list
        self.id_container.ids = list(ids_with_extracted_shorelines)
        self.extract_shorelines_container.roi_ids_list = list(
            ids_with_extracted_shorelines
        )
        return ids_with_extracted_shorelines

    def update_loadable_shorelines(
        self, selected_id: str
    ) -> Optional[extracted_shoreline.Extracted_Shoreline]:
        """
        Updates and returns the loadable extracted shorelines for a selected ROI ID.

        Side Effects:
            - Updates the extract_shorelines_container.load_list with the dates of the extracted shorelines
            - Updates the extract_shorelines_container.trash_list with the dates of the extracted shorelines

        Args:
            selected_id (str): The ROI ID.

        Returns:
            Extracted_Shoreline: The extracted shoreline object for the selected ROI.
        """
        # get the extracted shoreline for the selected roi's id
        if self.rois is None:
            self.extract_shorelines_container.load_list = []
            self.extract_shorelines_container.trash_list = []
            return None

        extracted_shorelines = self.rois.get_extracted_shoreline(selected_id)
        # if extracted shorelines exist, load them onto map, if none exist nothing loads
        if hasattr(extracted_shorelines, "gdf"):
            # sort the extracted shoreline gdf by date
            if not extracted_shorelines.gdf.empty:
                extracted_shorelines.gdf = extracted_shorelines.gdf.sort_values(
                    by=["date"]
                )
                if extracted_shorelines.gdf["date"].dtype == "object":
                    # If the "date" column is already of string type, concatenate directly
                    # formatted_dates = extracted_shorelines.gdf["date"]
                    formatted_dates = extracted_shorelines.gdf.apply(
                        lambda row: f"{row['satname']}_{row['date']}", axis=1
                    )
                else:
                    # If the "date" column is not of string type, convert to string with the required format
                    formatted_dates = extracted_shorelines.gdf.apply(
                        lambda row: f"{row['satname']}_{row['date'].strftime('%Y-%m-%d %H:%M:%S')}",
                        axis=1,
                    )
                self.extract_shorelines_container.load_list = []
                # only get the unique dates
                self.extract_shorelines_container.load_list = np.unique(
                    formatted_dates
                ).tolist()
                self.extract_shorelines_container.trash_list = []
        else:
            logger.warning(f"No shorelines extracted for ROI {selected_id}")
            # if the selected ROI has no extracted shorelines, clear the load list & trash list
            self.extract_shorelines_container.load_list = []
            self.extract_shorelines_container.trash_list = []
            return None
        return extracted_shorelines

    def load_extracted_shorelines_on_map(
        self,
        extracted_shorelines: Optional[extracted_shoreline.Extracted_Shoreline],
        row_number: int = 0,
    ) -> None:
        """
        Loads a stylized extracted shorelines onto the map for a given row number.

        Args:
            extracted_shorelines (Extracted_Shoreline): The extracted shoreline object.
            row_number (int, optional): The row number in the extracted shoreline GeoDataFrame to plot. Defaults to 0.
                For example, if there are 10 extracted shorelines for the ROI, and row_number is 2,
                the third extracted shoreline will be plotted.

        Returns:
            None
        """
        if extracted_shorelines is None:
            return
        if extracted_shorelines.gdf.empty:
            logger.info(
                f"No extracted shorelines for ROI {extracted_shorelines.roi_id}"
            )
            return
        # check if row number exists in gdf
        if row_number >= len(extracted_shorelines.gdf):
            logger.warning(
                f"Row number {row_number} does not exist in extracted shoreline gdf using row number 0 instead"
            )
            row_number = 0
        # load the selected extracted shoreline layer onto the map
        self.load_extracted_shoreline_layer(
            extracted_shorelines.gdf.iloc[[row_number]],
            layer_name="extracted shoreline",
            colormap="viridis",
        )

    def load_feature_on_map(
        self,
        feature_name: str,
        file: str = "",
        gdf: Optional[gpd.GeoDataFrame] = None,
        **kwargs,
    ) -> None:
        """
        Loads a feature onto the map from a file or GeoDataFrame.

        Args:
            feature_name (str): The name of the feature. eg. "shoreline", "transects", "bbox", "rois"
            file (str, optional): Path to the feature file, typically a geojson file.
            gdf (gpd.GeoDataFrame, optional): The feature GeoDataFrame to load the feature from.
            **kwargs: Additional keyword arguments.
        """
        # create the feature based on the feature name either from file, gdf or load a default feature(if available in the region)
        new_feature = self._get_feature(feature_name, file, gdf, **kwargs)

        if new_feature is not None:
            # load the features onto the map
            self.add_feature_on_map(
                new_feature,
                feature_name,
                **kwargs,
            )

    def _get_feature(
        self,
        feature_name: str,
        file: str = "",
        gdf: Optional["gpd.GeoDataFrame"] = None,
        **kwargs,
    ) -> Optional[Feature]:
        """
        Retrieves a feature based on the given feature name, file path, or GeoDataFrame.

        Args:
            feature_name (str): The name of the feature.
            file (str, optional): Path to the feature file to load the feature from.
            gdf (gpd.GeoDataFrame, optional): The feature GeoDataFrame to load the feature from.
            **kwargs: Additional keyword arguments to be passed to the feature loading methods.

        Returns:
            (Optional[Feature]): The loaded feature. Either an ROI, bbox, shoreline,shoreline extraction area or transects object.
            If not feature can be created returns None.

        """
        if file:
            return self.load_feature_from_file(feature_name, file, **kwargs)
        elif isinstance(gdf, gpd.GeoDataFrame):
            if gdf.empty:
                logger.info(f"No {feature_name} was empty")
                return
            else:
                return self.load_feature_from_gdf(feature_name, gdf, **kwargs)
        else:
            # if gdf is None then create the feature from scratch by loading a default
            return self.factory.make_feature(self, feature_name, gdf, **kwargs)

    def make_feature(
        self, feature_name: str, gdf: Optional[gpd.GeoDataFrame] = None, **kwargs
    ) -> Optional[Feature]:
        """
        Creates a Feature object from a GeoDataFrame of the specified feature type.
        If a geodataframe is not provided then a default feature is created using the feature_name.

        Args:
            feature_name (str): The name of the feature. eg. "shoreline", "transects", "bbox", "rois"
            gdf (Optional[gpd.GeoDataFrame]): The feature GeoDataFrame to create the feature from.
            **kwargs: Additional keyword arguments.

        Returns:
            Feature (Optional[Feature]): The created Feature object.
        """
        # Ensure the gdf is not empty
        if gdf is not None and gdf.empty:
            logger.info(
                f"The provided gdf for {feature_name} was empty and a new feature was not created"
            )
            return
        # create the feature and add it to the class
        new_feature = self.factory.make_feature(self, feature_name, gdf, **kwargs)
        return new_feature

    def load_feature_from_file(
        self, feature_name: str, file: str, **kwargs
    ) -> Optional[Feature]:
        """
        Loads a feature from a file of type feature_name and returns a Feature object.

        Available feature types are "shoreline","transects","bbox", or "rois".
        Must be lowercase.

        Args:
            feature_name (str): The name of the feature. eg. "shoreline", "transects", "bbox", "rois"
            file (str): Path to the feature file, typically a geojson file.
            **kwargs: Additional keyword arguments.

        Returns:
            Feature (Optional[Feature]): The loaded Feature object.
        """
        # Load GeoDataFrame if file is provided
        gdf = geodata_processing.load_geodataframe_from_file(
            file, feature_type=feature_name
        )
        # Ensure the gdf is not empty
        if gdf is not None and gdf.empty:
            logger.info(f"No {feature_name} was empty or None")
            return

        # create the feature and add it to the class
        new_feature = self.factory.make_feature(self, feature_name, gdf, **kwargs)
        return new_feature

    def load_feature_from_gdf(
        self, feature_name: str, gdf: Optional[gpd.GeoDataFrame], **kwargs
    ) -> Optional[Feature]:
        """
        Loads a feature from a GeoDataFrame of type feature_name and returns a Feature object.

        Available feature types are "shoreline","transects","bbox", or "rois".
        Must be lowercase.

        Args:
            feature_name (str): The name of the feature. eg. "shoreline", "transects", "bbox", "rois"
            gdf (Optional[gpd.GeoDataFrame]): The feature GeoDataFrame.
            **kwargs: Additional keyword arguments.

        Returns:
            Feature (Optional[Feature]): The loaded Feature object.
        """
        # Ensure the gdf is not empty
        if gdf is not None and gdf.empty:
            logger.info(f"No {feature_name} was empty")
            return

        # create the feature and add it to the class
        new_feature = self.factory.make_feature(self, feature_name, gdf, **kwargs)
        return new_feature

    def add_feature_on_map(
        self,
        new_feature,
        feature_name: str,
        layer_name: str = "",
        **kwargs,
    ) -> None:
        """
        Adds a feature to the map and sets up its on_click and on_hover handlers.

        Args:
            new_feature: The feature object to add to the map.
            feature_name (str): The name of the feature.
            layer_name (str, optional): The name of the layer.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        if new_feature is None:
            logger.warning(f"No {feature_name} was None")
            return
        # get on hover and on click handlers for feature
        on_hover = self.get_on_hover_handler(feature_name)
        on_click = self.get_on_click_handler(feature_name)
        # if layer name is not given use the layer name of the feature
        if not layer_name and hasattr(new_feature, "LAYER_NAME"):
            layer_name = new_feature.LAYER_NAME
        if self.map is not None:
            if hasattr(new_feature, "gdf") and hasattr(new_feature.gdf, "total_bounds"):
                bounds = new_feature.gdf.total_bounds
                self.map.zoom_to_bounds(bounds)
        self.load_on_map(new_feature, layer_name, on_hover, on_click)

    def get_on_click_handler(self, feature_name: str) -> Optional[Callable]:
        """
        Returns the on-click handler function for a given feature name.

        Args:
            feature_name (str): The name of the feature for which the click handler will be applied.

        Returns:
            Callable: The on-click handler function that handles mouse click events for the given feature.
        """
        on_click = None
        if "roi" in feature_name.lower():
            on_click = self.geojson_onclick_handler
        elif "shoreline" in feature_name.lower():
            on_click = self.shoreline_onclick_handler
        return on_click

    def get_on_hover_handler(self, feature_name: str) -> Optional[Callable]:
        """
        Returns the on-hover handler function for a given feature name.

        Args:
            feature_name (str): The name of the feature for which the hover handler will be applied.

        Returns:
            Callable: The on-hover handler function that updates the HTML display.
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
        self,
        feature: Feature,
        layer_name: str,
        on_hover: Optional[Callable] = None,
        on_click: Optional[Callable] = None,
    ) -> None:
        """
        Loads a feature onto the map by creating a new layer and replacing the existing one.

        Args:
            feature (Feature): The feature object to load.
            layer_name (str): The name of the layer.
            on_hover: Optional hover callback.
            on_click: Optional click callback.

        Raises:
            Exception: raised if feature layer is empty
        """
        # style and add the feature to the map
        new_layer = self.create_layer(feature, layer_name)
        # Replace old feature layer with new feature layer
        self.replace_layer_by_name(
            layer_name, new_layer, on_hover=on_hover, on_click=on_click
        )

    def create_layer(self, feature: Feature, layer_name: str) -> Optional[GeoJSON]:
        """
        Creates a GeoJSON layer from the given feature with the specified layer name.

        Args:
            feature (Feature): The feature object.
            layer_name (str): The name of the layer.

        Returns:
            Optional[GeoJSON]: The styled GeoJSON layer or None if the feature is empty.
        """
        if not hasattr(feature, "gdf"):
            logger.warning("Cannot add an empty GeoDataFrame layer to the map.")
            print("Cannot add an empty layer to the map.")
            return None
        if feature.gdf is None:  # type: ignore
            logger.warning("Cannot add an empty GeoDataFrame layer to the map.")
            print("Cannot add an empty layer to the map.")
            return None
        if feature.gdf.empty:  # type: ignore
            logger.warning("Cannot add an empty GeoDataFrame layer to the map.")
            print("Cannot add an empty layer to the map.")
            return None
        else:
            styled_layer = feature.style_layer(feature.gdf, layer_name)  # type: ignore
        return styled_layer

    def geojson_onclick_handler(
        self,
        event: Optional[str] = None,
        id: Optional[str] = None,
        properties: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Handles click events on GeoJSON features specifically for ROIs.

        Side Effects:
            - Adds the clicked feature's ID to the selected_set.
            - Replaces the current selected layer with a new one that includes the recently clicked GeoJSON.

        Args:
            event (Optional[str]): The event type. Tpically 'click'.
            id (Optional[str]): The feature ID.
            properties (Optional[dict]): The GeoJSON properties of the clicked feature.
            **kwargs: Additional arguments.

        Returns:
            None
        """
        if properties is None:
            return
        # Add id of clicked ROI to selected_set
        self.selected_set.add(str(properties["id"]))
        # remove old selected layer
        self.remove_layer_by_name(ROI.SELECTED_LAYER_NAME)
        selected_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(
                self.selected_set, ROI.LAYER_NAME
            ),
            name=ROI.SELECTED_LAYER_NAME,
            hover_style={"fillColor": "blue", "fillOpacity": 0.1, "color": "aqua"},
        )
        self.replace_layer_by_name(
            ROI.SELECTED_LAYER_NAME,
            selected_layer,
            on_click=self.selected_onclick_handler,
            on_hover=self.update_roi_html,
        )

    def shoreline_onclick_handler(
        self,
        event: Optional[str] = None,
        id: Optional[int] = None,
        properties: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Handles click events for when unselected geojson is clicked.

        Side Effects:
            - Adds object's id to selected_objects_set.
            - Replaces current selected layer with a new one that includes recently clicked geojson.

        Args:
            event (Optional[str]): The event type. In this case, it is expected to be 'click'.
            id (Optional[int]): The feature ID.
            properties (Optional[dict]): The GeoJSON properties of the clicked feature.
            **kwargs: Additional arguments.

        Returns:
            None
        """
        if properties is None:
            return
        # Add id of clicked shape to selected_set
        self.selected_shorelines_set.add(str(properties["id"]))
        # remove old selected layer
        self.remove_layer_by_name(SELECTED_LAYER_NAME)
        selected_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(
                self.selected_shorelines_set, Shoreline.LAYER_NAME
            ),
            name=SELECTED_LAYER_NAME,
            hover_style={"fillColor": "orange", "fillOpacity": 0.1, "color": "orange"},
        )
        self.replace_layer_by_name(
            SELECTED_LAYER_NAME,
            selected_layer,
            on_click=self.selected_shoreline_onclick_handler,
            on_hover=None,
        )

    def selected_shoreline_onclick_handler(
        self,
        event: Optional[str] = None,
        id: Optional[str] = None,
        properties: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Handles click events on selected shoreline features. Creates a new selected layer without the clicked feature.

        Removes clicked layer's cid from the selected_set and replaces the select layer with a new one with
        the clicked layer removed from select_layer.

        Args:
            event (Optional[str]): The event type. In this case, it is expected to be 'click'.
            id (Optional[str]): The feature ID.
            properties (Optional[dict]): Feature properties.
            **kwargs: Additional arguments.

        Returns:
            None
        """
        if properties is None:
            return

        # Remove the current layers cid from selected set
        self.selected_shorelines_set.remove(str(properties["id"]))
        self.remove_layer_by_name(SELECTED_LAYER_NAME)
        # Recreate selected layers without layer that was removed
        selected_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(
                self.selected_shorelines_set, Shoreline.LAYER_NAME
            ),
            name=SELECTED_LAYER_NAME,
            hover_style={"fillColor": "orange", "fillOpacity": 0.1, "color": "orange"},
        )
        self.replace_layer_by_name(
            SELECTED_LAYER_NAME,
            selected_layer,
            on_click=self.selected_shoreline_onclick_handler,
            on_hover=None,
        )

    def selected_onclick_handler(
        self,
        event: Optional[str] = None,
        id: Optional[str] = None,
        properties: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Handles click events on selected ROI features. Creates a new selected layer without the clicked feature.

        Args:
            event (Optional[str]): The event type.
            id (Optional[str]): The feature ID.
            properties (Optional[dict]): The GeoJSON properties of the clicked feature.
            **kwargs: Additional arguments.

        Side Effects:
            - Removes clicked layer's cid from the selected_set.
            - Replaces the select layer with a new one with the clicked layer removed from select_layer.

        Returns:
            None
        """
        if properties is None:
            return
        # Remove the current layers cid from selected set
        self.selected_set.remove(properties["id"])
        self.remove_layer_by_name(ROI.SELECTED_LAYER_NAME)
        # Recreate selected layers without layer that was removed
        selected_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(
                self.selected_set, ROI.LAYER_NAME
            ),
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
    ) -> None:
        """
        Saves a feature to a file.

        Side Effects:
            - If the feature is an ROI, only the selected ROIs are saved.

        Args:
            feature (Union[Bounding_Box, Shoreline, Transects, ROI]): The feature object to save.
            feature_type (str, optional): The type of the feature. e.g. Bounding_Box, Shoreline, Transects, or ROI.

        Returns:
            None
        """
        exception_handler.can_feature_save_to_file(feature, feature_type)
        logger.info(f"Saving feature type({feature}) to file")
        if isinstance(feature, ROI):
            # raise exception if no rois were selected
            exception_handler.check_selected_set(self.selected_set)
            # save only the selected ROIs to file
            feature.gdf[feature.gdf["id"].isin(self.selected_set)].to_file(
                feature.filename, driver="GeoJSON"
            )
            print(f"Saved selected ROIs to {feature.filename}")
            logger.info(f"Save {feature.LAYER_NAME} to {feature.filename}")
        else:
            if hasattr(feature, "gdf"):
                feature.gdf.to_file(feature.filename, driver="GeoJSON")  # type: ignore
                print(f"Save {feature.LAYER_NAME} to {feature.filename}")
                logger.info(f"Save {feature.LAYER_NAME} to {feature.filename}")
            else:
                logger.warning(f"Empty {feature.LAYER_NAME} cannot be saved to file")
                print(f"Empty {feature.LAYER_NAME} cannot be saved to file")

    def convert_selected_set_to_geojson(
        self, selected_set: set, layer_name: str, style: Optional[Dict] = None
    ) -> Dict:
        """
        Returns a GeoJSON FeatureCollection (dict) containing only the features from the
        specified layer whose 'properties.id' is present in `selected_set`.

        Args:
            selected_set (set): Set of ids of the selected features.
            layer_name (str): The name of the layer to read the geometries from.
            style (Optional[Dict]): The style dictionary to be applied to each selected feature. If no style is provided then a default style is used:
                style = {
                    "color": "blue",
                    "weight": 2,
                    "fillColor": "blue",
                    "fillOpacity": 0.1,
                }
        Returns:
            Dict: GeoJSON dictionary containing a FeatureCollection for all the selected features whose ids are in selected_set
        """
        empty_collection = {"type": "FeatureCollection", "features": []}
        if self.map is None:
            return empty_collection
        style = style or {
            "color": "blue",
            "weight": 2,
            "fillColor": "blue",
            "fillOpacity": 0.1,
        }
        layer = self.map.find_layer(layer_name)

        # if layer does not exist throw an error
        if layer is not None:
            exception_handler.check_empty_layer(layer, layer_name)

        # Get features from the map layer
        layer_data = getattr(layer, "data", {})
        features = (
            layer_data.get("features", []) if isinstance(layer_data, dict) else []
        )

        # Normalize IDs to strings for comparison
        selected_ids = {str(id_) for id_ in selected_set}

        # Copy only selected features with id in selected_set and add style to each selected feature
        result_features = []
        for feature in features:
            feature_id = feature.get("properties", {}).get("id")
            if feature_id and str(feature_id) in selected_ids:
                # Shallow copy with new style
                new_feature = {**feature}
                new_feature["properties"] = {**feature["properties"], "style": style}
                result_features.append(new_feature)

        return {"type": "FeatureCollection", "features": result_features}
