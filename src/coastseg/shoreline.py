# Standard library imports
import logging
import os
from typing import Dict, Optional

# Internal dependencies imports
from coastseg.exceptions import DownloadError
from coastseg.common import (
    download_url,
    preprocess_geodataframe,
    create_unique_ids,
)

# External dependencies imports
import geopandas as gpd
from pandas import concat

# from fiona.errors import DriverError
from ipyleaflet import GeoJSON

logger = logging.getLogger(__name__)

# only export Shoreline class
__all__ = ["Shoreline"]


class Shoreline:
    """Shoreline: contains the shorelines within a region specified by bbox (bounding box)"""

    LAYER_NAME = "shoreline"

    def __init__(
        self,
        bbox: gpd.GeoDataFrame = None,
        shoreline: gpd.GeoDataFrame = None,
        filename: str = None,
    ):
        self.gdf = gpd.GeoDataFrame()
        self.filename = filename if filename else "shoreline.geojson"
        self.initialize_shorelines(bbox,shoreline)


    def __str__(self):
        return f"Shoreline: geodataframe {self.gdf}"

    def __repr__(self):
        return f"Shoreline: geodataframe {self.gdf}"

    def initialize_shorelines(self, bbox: Optional[gpd.GeoDataFrame] = None, shorelines: Optional[gpd.GeoDataFrame] = None):
        if shorelines is not None:
            self.initialize_shorelines_with_shorelines(shorelines)

        elif bbox is not None:
            self.initialize_shoreliness_with_bbox(bbox)


    def initialize_shorelines_with_shorelines(self, shorelines: gpd.GeoDataFrame):
        """
        Initalize shorelines with the provided shorelines in a geodataframe
        """
        if not shorelines.empty:
            columns_to_keep = [
            "id", "geometry", "river_label", "ERODIBILITY", "CSU_ID",
            "turbid_label", "slope_label", "sinuosity_label",
            "TIDAL_RANGE", "MEAN_SIG_WAVEHEIGHT"
            ]
            if not shorelines.crs:
                logger.warning(f"shorelines did not have a crs converting to crs 4326 \n {shorelines}")
                shorelines.set_crs('EPSG:4326', inplace=True)
            shorelines= preprocess_geodataframe(shorelines,columns_to_keep,create_ids=True)
            # make sure all the ids are unique
            shorelines = create_unique_ids(shorelines,prefix_length=3)
            # @todo add the shorelines to the current dataframe
            # @todo make sure none of the ids already exist in the dataframe. this can be a flag to turn an exception on/off
            self.gdf = shorelines

    def initialize_shoreliness_with_bbox(self, bbox: gpd.GeoDataFrame):
        """ 
        Load shorelines within the bounding box. The shorelines will be clipped to the bounding box.
        Args:
            bbox (gpd.GeoDataFrame): bounding box
        """
        if not bbox.empty:
            self.gdf = self.create_geodataframe(bbox)

    def create_geodataframe(self, bbox: gpd.GeoDataFrame, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """
        Creates a GeoDataFrame with the specified CRS, containing shorelines that intersect with the given bounding box.
        Downloads the shorelines from online.
        Args:
            bbox (gpd.GeoDataFrame): Bounding box being searched for shorelines.
            crs (str, optional): Coordinate reference system string. Defaults to 'EPSG:4326'.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with geometry column = rectangle and given CRS.
        """
        shorelines_gdf = gpd.GeoDataFrame()
        intersecting_files = get_intersecting_files(bbox)

        if not intersecting_files:
            logger.error(f"No intersecting shorelines found. BBox: {bbox}")
            raise ValueError("No intersecting shorelines found. Try loading your own or draw a new bounding box.")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Download any missing shoreline files
        shoreline_files = self.get_shoreline_files(intersecting_files, script_dir)

        if not shoreline_files:
            logger.error(f"No shoreline files found at {script_dir}")
            raise FileNotFoundError(f"No shoreline files found at {script_dir}")

        # Read in each shoreline file and clip it to the bounding box
        columns_to_keep = [
            "id", "geometry", "river_label", "ERODIBILITY", "CSU_ID",
            "turbid_label", "slope_label", "sinuosity_label",
            "TIDAL_RANGE", "MEAN_SIG_WAVEHEIGHT"
        ]

        for shoreline_file in shoreline_files:
            shoreline = gpd.read_file(shoreline_file, mask=bbox).to_crs(crs)
            # try:
            # shoreline = gpd.read_file(shoreline_file, mask=bbox).to_crs(crs)
            # except DriverError as driver_error:
            #     print(driver_error)
            #     continue
            shoreline = preprocess_geodataframe(shoreline, columns_to_keep)
            clipped_shoreline = gpd.clip(shoreline, bbox).to_crs(crs)

            shorelines_gdf = concat([shorelines_gdf, clipped_shoreline], ignore_index=True)

        shorelines_gdf = preprocess_geodataframe(shorelines_gdf, columns_to_keep)
        # make sure all the ids are unique
        shorelines_gdf = create_unique_ids(shorelines_gdf,prefix_length=3)

        if not shorelines_gdf.empty:
            shorelines_gdf.to_crs(crs, inplace=True)

        return shorelines_gdf


    def get_shoreline_files(
        self, intersecting_shoreline_files: Dict[str, str], script_dir: str
    ) -> None:
        """Downloads missing shoreline files.

        Args:
            intersecting_shoreline_files (Dict[str, str]): Dictionary mapping shoreline filenames to dataset IDs.
            script_dir (str): The path to the script's directory.
        """
        available_files = []
        for filename, dataset_id in intersecting_shoreline_files.items():
            shoreline_path = os.path.abspath(
                os.path.join(script_dir, "shorelines", filename)
            )
            if not os.path.exists(shoreline_path):
                try:
                    self.download_shoreline(filename, dataset_id)
                except DownloadError as download_exception:
                    logger.warning(download_exception)
                    print(download_exception)
            if os.path.exists(shoreline_path):
                available_files.append(shoreline_path)
        return available_files

    def download_missing_shorelines(
        self, intersecting_shoreline_files: Dict[str, str], script_dir: str
    ) -> None:
        """Downloads missing shoreline files.

        Args:
            intersecting_shoreline_files (Dict[str, str]): Dictionary mapping shoreline filenames to dataset IDs.
            script_dir (str): The path to the script's directory.
        """
        for filename, dataset_id in intersecting_shoreline_files.items():
            shoreline_path = os.path.abspath(
                os.path.join(script_dir, "shorelines", filename)
            )
            if not os.path.exists(shoreline_path):
                try:
                    self.download_shoreline(filename, dataset_id)
                except DownloadError as download_exception:
                    print(download_exception)

    def style_layer(self, geojson: dict, layer_name: str) -> "ipyleaflet.GeoJSON":
        """Return styled GeoJson object with layer name

        Args:
            geojson (dict): geojson dictionary to be styled
            layer_name(str): name of the GeoJSON layer

        Returns:
            "ipyleaflet.GeoJSON": shoreline as GeoJSON layer styled with yellow dashes
        """
        assert geojson != {}, "ERROR.\n Empty geojson cannot be drawn onto  map"
        return GeoJSON(
            data=geojson,
            name=layer_name,
            style={
                "color": "black",
                "fill_color": "black",
                "opacity": 1,
                "dashArray": "5",
                "fillOpacity": 0.5,
                "weight": 4,
            },
            hover_style={"color": "white", "dashArray": "4", "fillOpacity": 0.7},
        )

    def download_shoreline(self, filename: str, dataset_id: str = "7761607"):
        """Downloads the shoreline file from zenodo
        Args:
            filename (str): name of file to download
            dataset_id (str, optional): zenodo id of file. Defaults to '7761607'."""
        root_url = "https://zenodo.org/record/" + dataset_id + "/files/"
        # Create the directory to hold the downloaded shorelines from Zenodo
        script_dir = os.path.dirname(os.path.abspath(__file__))
        shoreline_dir = os.path.abspath(os.path.join(script_dir, "shorelines"))
        if not os.path.exists(shoreline_dir):
            os.mkdir(shoreline_dir)
            logger.info(f"Created shoreline directory: {shoreline_dir}")
        # location to save the downloaded shoreline file
        shoreline_file = os.path.abspath(os.path.join(shoreline_dir, filename))
        # Download the model from Zenodo
        if not os.path.exists(shoreline_file):
            url = root_url + filename + "?download=1"
            logger.info(f"Retrieving: {url}")
            logger.info(f"Retrieving file: {shoreline_file}")
            print(f"Retrieving: {url}")
            print(f"Retrieving file: {shoreline_file}")
            download_url(url, shoreline_file, filename=filename)


def get_intersecting_files(bbox: gpd.GeoDataFrame) -> Dict[str, str]:
    """Given a bounding box (bbox), returns a dictionary of shoreline filenames whose
    contents intersect with bbox, mapped to their dataset IDs.

    Args:
        bbox (geopandas.GeoDataFrame): bounding box being searched for shorelines

    Returns:
        dict: intersecting_files containing filenames whose contents intersect with bbox
    """
    WORLD_DATASET_ID = "7814755"

    # DataFrames containing total bounding box for each shoreline file
    world_total_bounds_df = load_total_bounds_df("world", bbox)
    # Create a list of tuples containing the DataFrames and their dataset IDs
    total_bounds_dfs = [
        (world_total_bounds_df, WORLD_DATASET_ID),
    ]

    intersecting_files = {}
    # Add filenames of interesting shoreline in both the usa and world shorelines to intersecting_files
    for bounds_df, dataset_id in total_bounds_dfs:
        if not bounds_df.empty:
            filenames = bounds_df.index
            # Create a dictionary mapping filenames to their dataset IDs
            filenames_and_ids = zip(filenames, [dataset_id] * len(filenames))
            # Add the filenames and their dataset IDs to intersecting_files
            intersecting_files.update(dict(filenames_and_ids))
    logger.info(
        f"Found {len(intersecting_files)} intersecting files\n {intersecting_files}"
    )
    return intersecting_files


def load_total_bounds_df(
    location: str = "usa", mask: gpd.GeoDataFrame = None
) -> gpd.GeoDataFrame:
    """
    Returns dataframe containing total bounds for each set of shorelines in the geojson file specified by location.
    Args:
        location (str, optional): Determines whether USA or world shoreline bounding boxes are loaded. Defaults to 'usa'.
            Can be either 'world' or 'usa'.
        mask (gpd.GeoDataFrame, optional): A GeoDataFrame to use as a mask when reading the file. Defaults to None.
    Returns:
        gpd.GeoDataFrame: Returns geodataframe containing total bounds for each set of shorelines.
    """
    # Load in the total bounding box from csv
    # Create the directory to hold the downloaded shorelines from Zenodo
    script_dir = os.path.dirname(os.path.abspath(__file__))

    bounding_box_dir = os.path.abspath(os.path.join(script_dir, "bounding_boxes"))
    if not os.path.exists(bounding_box_dir):
        os.mkdir(bounding_box_dir)
    # load different csv files depending on location given
    if location == "usa":
        gdf_file = "usa_shorelines_bounding_boxes.geojson"
    elif location == "world":
        gdf_file = "world_reference_shorelines_bboxes.geojson"

    gdf_location = os.path.join(bounding_box_dir, gdf_file)
    total_bounds_df = gpd.read_file(gdf_location, mask=mask)
    total_bounds_df.index = total_bounds_df["filename"]
    if "filename" in total_bounds_df.columns:
        total_bounds_df.drop("filename", axis=1, inplace=True)
    return total_bounds_df
