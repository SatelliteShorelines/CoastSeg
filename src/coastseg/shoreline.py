# Standard library imports
import logging
import os
from typing import Any, Dict, List, Optional, Callable


# Internal dependencies imports
from coastseg import exception_handler
from coastseg.exceptions import DownloadError
from coastseg.common import (
    download_url,
    preprocess_geodataframe,
    create_unique_ids,
)
from coastseg.common import validate_geometry_types
from coastseg.feature import Feature

# External dependencies imports
import geopandas as gpd
import pandas as pd
from pandas import concat

# from fiona.errors import DriverError
from ipyleaflet import GeoJSON

logger = logging.getLogger(__name__)

# only export Shoreline class
__all__ = ["Shoreline", "ShorelineServices"]


class ShorelineServices:
    def __init__(
        self,
        download_service: Callable[[str, str, Optional[str]], Any] = None,
        preprocess_service=None,
        create_unique_ids_service=None,
    ):
        self.download_service = download_service or download_url
        self.preprocess_service = preprocess_service or preprocess_geodataframe
        self.create_ids_service = create_unique_ids_service or create_unique_ids


class Shoreline(Feature):
    """Shoreline: contains the shorelines within a region specified by bbox (bounding box)"""

    LAYER_NAME = "shoreline"
    SELECTED_LAYER_NAME = "Selected Shorelines"

    def __init__(
        self,
        bbox: gpd.GeoDataFrame = None,
        shoreline: gpd.GeoDataFrame = None,
        filename: str = None,
        services: ShorelineServices = None,
        download_location: str = None,
    ):
        # function to download shoreline files by default use download_url
        services = services or ShorelineServices()
        self.download_service = services.download_service
        self.preprocess_service = services.preprocess_service
        self.create_ids_service = services.create_ids_service

        # location to create shorelines directory and download shorelines to
        self._download_location = download_location or os.path.dirname(
            os.path.abspath(__file__)
        )

        self.gdf = gpd.GeoDataFrame()
        self.filename = filename if filename else "shoreline.geojson"
        self.initialize_shorelines(bbox, shoreline)

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        if not isinstance(value, str):
            raise ValueError("Filename must be a string.")
        if not value.endswith(".geojson"):
            raise ValueError("Filename must end with '.geojson'.")
        self._filename = value

    def __str__(self):
        # Get column names and their data types
        col_info = self.gdf.dtypes.apply(lambda x: x.name).to_string()
        # Get first 3 rows as a string
        first_rows = self.gdf
        geom_str = ""
        if isinstance(self.gdf, gpd.GeoDataFrame):
            if "geometry" in self.gdf.columns:
                first_rows = self.gdf.head(3).drop(columns="geometry").to_string()
            if not self.gdf.empty:
                geom_str = str(self.gdf.iloc[0]["geometry"])[:100] + "...)"
        # Get CRS information
        if self.gdf.empty:
            crs_info = "CRS: None"
        else:
            if self.gdf is not None and hasattr(self.gdf, 'crs'):
                crs_info = f"CRS: {self.gdf.crs}" if self.gdf.crs else "CRS: None"
            else:
                crs_info = "CRS: None"
        ids = []
        if "id" in self.gdf.columns:
            ids = self.gdf["id"].astype(str)
        return f"Shoreline:\nself.gdf:\n\n{crs_info}\n- Columns and Data Types:\n{col_info}\n\n- First 3 Rows:\n{first_rows}\n geometry: {geom_str}\nIDs:\n{ids}"

    def __repr__(self):
        # Get column names and their data types
        col_info = self.gdf.dtypes.apply(lambda x: x.name).to_string()
        # Get first 3 rows as a string
        first_rows = self.gdf
        geom_str = ""
        if isinstance(self.gdf, gpd.GeoDataFrame):
            if "geometry" in self.gdf.columns:
                first_rows = self.gdf.head(3).drop(columns="geometry").to_string()
            if not self.gdf.empty:
                geom_str = str(self.gdf.iloc[0]["geometry"])[:100] + "...)"
        # Get CRS information
        if self.gdf.empty:
            crs_info = "CRS: None"
        else:
            if self.gdf is not None and hasattr(self.gdf, 'crs'):
                crs_info = f"CRS: {self.gdf.crs}" if self.gdf.crs else "CRS: None"
            else:
                crs_info = "CRS: None"
                
        ids = []
        if "id" in self.gdf.columns:
            ids = self.gdf["id"].astype(str)
        return f"Shoreline:\nself.gdf:\n\n{crs_info}\n- Columns and Data Types:\n{col_info}\n\n- First 3 Rows:\n{first_rows}\n geometry: {geom_str}\nIDs:\n{ids}"
    def initialize_shorelines(
        self,
        bbox: Optional[gpd.GeoDataFrame] = None,
        shorelines: Optional[gpd.GeoDataFrame] = None,
    ):
        if shorelines is not None:
            self.initialize_shorelines_with_shorelines(shorelines)

        elif bbox is not None:
            self.initialize_shorelines_with_bbox(bbox)

    def remove_by_id(
        self, ids_to_drop: list | set | tuple | str | int
    ) -> gpd.GeoDataFrame:
        if self.gdf.empty or "id" not in self.gdf.columns or ids_to_drop is None:
            return self.gdf
        if isinstance(ids_to_drop, (str, int)):
            ids_to_drop = [
                str(ids_to_drop)
            ]  # Convert to list and ensure ids are strings
        # Ensure all elements in ids_to_drop are strings for consistent comparison
        ids_to_drop = set(map(str, ids_to_drop))
        logger.info(f"ids_to_drop from shoreline: {ids_to_drop}")
        # drop the ids from the geodataframe
        self.gdf = self.gdf[~self.gdf["id"].astype(str).isin(ids_to_drop)]
        return self.gdf

    def initialize_shorelines_with_shorelines(self, shorelines: gpd.GeoDataFrame):
        """
        Initalize shorelines with the provided shorelines in a geodataframe
        """
        if not isinstance(shorelines, gpd.GeoDataFrame):
            raise ValueError("Shorelines must be a geodataframe")
        elif shorelines.empty:
            raise logger.warning("Shorelines cannot be an empty geodataframe")
        else:
            columns_to_keep = [
                "id",
                "geometry",
                "river_label",
                "ERODIBILITY",
                "CSU_ID",
                "turbid_label",
                "slope_label",
                "sinuosity_label",
                "TIDAL_RANGE",
                "MEAN_SIG_WAVEHEIGHT",
            ]

            if not shorelines.crs:
                logger.warning(
                    f"shorelines did not have a crs converting to crs 4326 \n {shorelines}"
                )
                shorelines.set_crs("EPSG:4326", inplace=True)
            shorelines = self.preprocess_service(
                shorelines, columns_to_keep, create_ids=True, output_crs="EPSG:4326"
            )
            validate_geometry_types(
                shorelines,
                set(["LineString", "MultiLineString"]),
                feature_type="shoreline",
                help_message=f"The uploaded shorelines need to be LineStrings.",
            )
            # make sure all the ids are unique with 3 random chars in front of id number
            shorelines = self.create_ids_service(shorelines, 3)
            self.gdf = shorelines

    def initialize_shorelines_with_bbox(self, bbox: gpd.GeoDataFrame):
        """
        Creates a geodataframe with shorelines within the bounding box. The shorelines will be clipped to the bounding box.
        Args:
            bbox (gpd.GeoDataFrame): bounding box
        """
        if not bbox.empty:
            shoreline_files = self.get_intersecting_shoreline_files(bbox)
            # if no shorelines were found to intersect with the bounding box raise an exception
            if not shoreline_files:
                exception_handler.check_if_default_feature_available(None, "shoreline")
                
            self.gdf = self.create_geodataframe(bbox, shoreline_files)

    def get_clipped_shoreline(
        self, shoreline_file: str, bbox: gpd.GeoDataFrame, columns_to_keep: List[str]
    ):
        """Read a shoreline file, preprocess it, and clip it to the bounding box."""
        shoreline = gpd.read_file(shoreline_file, mask=bbox)
        shoreline = self.preprocess_service(shoreline, columns_to_keep)
        validate_geometry_types(
            shoreline, set(["LineString", "MultiLineString"]), feature_type="shoreline"
        )
        return gpd.clip(shoreline, bbox)

    def get_intersecting_shoreline_files(
        self, bbox: gpd.GeoDataFrame, bounding_boxes_location: str = ""
    ) -> List[str]:
        """
        Retrieves a list of intersecting shoreline files based on the given bounding box.

        Args:
            bbox (gpd.GeoDataFrame): The bounding box to use for finding intersecting shoreline files.
            bounding_boxes_location (str, optional): The location to store the bounding box files. If not provided,
                it defaults to the download location specified during object initialization.

        Returns:
            List[str]: A list of intersecting shoreline file paths.

        Raises:
            ValueError: If no intersecting shorelines were available within the bounding box.
            FileNotFoundError: If no shoreline files were found at the download location.
        """
        # load the intersecting shoreline files
        bounding_boxes_location = (
            bounding_boxes_location
            if bounding_boxes_location
            else os.path.join(self._download_location, "bounding_boxes")
        )
        os.makedirs(bounding_boxes_location, exist_ok=True)
        intersecting_files = get_intersecting_files(bbox, bounding_boxes_location)

        if not intersecting_files:
            logger.warning("No intersecting shoreline files were found.")
            return []

        # Download any missing shoreline files
        shoreline_files = self.get_shoreline_files(
            intersecting_files, self._download_location
        )
        if not shoreline_files:
            raise FileNotFoundError(
                f"No shoreline files were found at {self._download_location}."
            )
        return shoreline_files

    def create_geodataframe(
        self, bbox: gpd.GeoDataFrame, shoreline_files: List[str], crs: str = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        """
        Creates a GeoDataFrame with the specified CRS, containing shorelines that intersect with the given bounding box.
        Downloads the shorelines from online.
        Args:
            bbox (gpd.GeoDataFrame): Bounding box being searched for shorelines.
            shoreline_files (List[str]): List of filepaths for available shoreline files.
            crs (str, optional): Coordinate reference system string. Defaults to 'EPSG:4326'.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with geometry column = rectangle and given CRS.
        """
        # Read in each shoreline file and clip it to the bounding box
        columns_to_keep = [
            "id",
            "geometry",
            "river_label",
            "ERODIBILITY",
            "CSU_ID",
            "turbid_label",
            "slope_label",
            "sinuosity_label",
            "TIDAL_RANGE",
            "MEAN_SIG_WAVEHEIGHT",
        ]

        if not shoreline_files:
            logger.error(f"No shoreline files were provided to read shorelines from")
            raise FileNotFoundError(
                f"No shoreline files were found at {self._download_location}."
            )

        shorelines_gdf = gpd.GeoDataFrame()
        shorelines = [
            self.get_clipped_shoreline(file, bbox, columns_to_keep)
            for file in shoreline_files
        ]
        # shorelines_gdf = concat(shorelines, ignore_index=True)
        # Drop columns where all values are NA
        shorelines = [df.dropna(axis=1, how='all') for df in shorelines]

        # Concatenate the DataFrames
        shorelines_gdf = pd.concat(shorelines, ignore_index=True)
        # clean the shoreline geodataframe
        shorelines_gdf = self.preprocess_service(shorelines_gdf, columns_to_keep)
        validate_geometry_types(
            shorelines_gdf,
            set(["LineString", "MultiLineString"]),
            feature_type="shoreline",
        )
        # make sure all the ids are unique
        shorelines_gdf = self.create_ids_service(shorelines_gdf, 3)

        if not shorelines_gdf.empty:
            shorelines_gdf.to_crs(crs, inplace=True)

        return shorelines_gdf

    def get_shoreline_files(
        self, intersecting_shoreline_files: Dict[str, str], download_location: str
    ) -> List[str]:
        """Downloads missing shoreline files.

        Args:
            intersecting_shoreline_files (Dict[str, str]): Dictionary mapping shoreline filenames to dataset IDs.
            download_location (str): full path to location where the shorelines directory should be created
        Returns:
            List[str]: List of filepaths for available shoreline files.
        """
        available_files = []

        # Ensure the directory to hold the downloaded shorelines from Zenodo exists
        shoreline_dir = os.path.abspath(os.path.join(download_location, "shorelines"))
        os.makedirs(shoreline_dir, exist_ok=True)

        for filename, dataset_id in intersecting_shoreline_files.items():
            shoreline_path = os.path.join(shoreline_dir, filename)
            if not os.path.exists(shoreline_path):
                try:
                    self.download_shoreline(filename, shoreline_path, dataset_id)
                    available_files.append(shoreline_path)
                except DownloadError as download_exception:
                    logger.error(
                        f"{download_exception} Shoreline {filename} failed to download."
                    )
                    print(
                        f"{download_exception} Shoreline {filename} failed to download."
                    )
                    # raise download_exception(f"Shoreline {filename} failed to download.")
            else:
                # if the shoreline file already exists then add it to the list of available files
                available_files.append(shoreline_path)
        return available_files

    def style_layer(self, geojson: dict, layer_name: str) -> "ipyleaflet.GeoJSON":
        """Return styled GeoJson object with layer name

        Args:
            geojson (dict): geojson dictionary to be styled
            layer_name(str): name of the GeoJSON layer

        Returns:
            "ipyleaflet.GeoJSON": shoreline as GeoJSON layer styled with yellow dashes
        """
        style={
            "color": "black",
            "fill_color": "black",
            "opacity": 1,
            "dashArray": "5",
            "fillOpacity": 0.5,
            "weight": 4,
        }
        hover_style={"color": "white", "dashArray": "4", "fillOpacity": 0.7}
        return super().style_layer(geojson, layer_name, style=style, hover_style=hover_style)
    
        # assert geojson != {}, "ERROR.\n Empty geojson cannot be drawn onto  map"
        # return GeoJSON(
        #     data=geojson,
        #     name=layer_name,
        #     style={
        #         "color": "black",
        #         "fill_color": "black",
        #         "opacity": 1,
        #         "dashArray": "5",
        #         "fillOpacity": 0.5,
        #         "weight": 4,
        #     },
        #     hover_style={"color": "white", "dashArray": "4", "fillOpacity": 0.7},
        # )

    def download_shoreline(
        self, filename: str, save_location: str, dataset_id: str = "7814755"
    ):
        """Downloads the shoreline file from zenodo
        Args:
            filename (str): name of file to download
            save_location (str): full path to location to save the downloaded shoreline file
            dataset_id (str, optional): zenodo id of file. Defaults to '7814755'.
        """

        # Construct the download URL
        root_url = "https://zenodo.org/record/"
        url = construct_download_url(root_url, dataset_id, filename)

        # Download shorelines from Zenodo
        logger.info(f"Retrieving file: {save_location} from {url}")
        self.download_service(url, save_location, filename=filename)


# helper functions
def construct_download_url(root_url: str, dataset_id: str, filename: str) -> str:
    """Constructs the download URL."""
    return f"{root_url}{dataset_id}/files/{filename}?download=1"


def get_intersecting_files(
    bbox: gpd.GeoDataFrame, bounding_boxes_location: str
) -> Dict[str, str]:
    """Given a bounding box (bbox), returns a dictionary of shoreline filenames whose
    contents intersect with bbox, mapped to their dataset IDs.

    Args:
        bbox (geopandas.GeoDataFrame): bounding box being searched for shorelines
        bounding_boxes_location(str): full path to the location of the bounding_boxes directory
    Returns:
        dict: intersecting_files containing filenames whose contents intersect with bbox
    """
    WORLD_DATASET_ID = "7814755"

    # DataFrames containing total bounding box for each shoreline file
    world_total_bounds_df = load_total_bounds_df(bounding_boxes_location, "world", bbox)
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
    logger.debug(
        f"Found {len(intersecting_files)} intersecting files\n {intersecting_files}"
    )
    return intersecting_files


def load_total_bounds_df(
    bounding_boxes_location: str,
    location: str = "usa",
    mask: gpd.GeoDataFrame = None,
) -> gpd.GeoDataFrame:
    """
    Returns dataframe containing total bounds for each set of shorelines in the geojson file specified by location.
    Args:
        bounding_boxes_location(str): full path to the location of the bounding_boxes directory
        location (str, optional): Determines whether USA or world shoreline bounding boxes are loaded. Defaults to 'usa'.
            Can be either 'world' or 'usa'.
        mask (gpd.GeoDataFrame, optional): A GeoDataFrame to use as a mask when reading the file. Defaults to None.

    Returns:
        gpd.GeoDataFrame: Returns geodataframe containing total bounds for each set of shorelines.
    """
    # load in different  total bounding box different depending on location given
    if location == "usa":
        gdf_file = "usa_shorelines_bounding_boxes.geojson"
    elif location == "world":
        gdf_file = "world_reference_shorelines_bboxes.geojson"

    gdf_location = os.path.join(bounding_boxes_location, gdf_file)
    total_bounds_df = gpd.read_file(gdf_location, mask=mask)
    total_bounds_df.index = total_bounds_df["filename"]
    if "filename" in total_bounds_df.columns:
        total_bounds_df.drop("filename", axis=1, inplace=True)
    return total_bounds_df
