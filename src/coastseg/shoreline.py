"""
CoastSeg Shoreline Management.

Streamlined tools to download, validate, clip, and visualize shoreline geometries
(LineString/MultiLineString) with CRS handling (EPSG:4326), unique ID generation,
and simple ipyleaflet styling. Automatically handles Zenodo data fetching.

Classes:
  - Shoreline: manages shoreline data and operations.

Functions:
  - construct_download_url: helper for Zenodo URL construction.

Key Features:
  - Streamlined loading: automatically finds intersecting files and downloads as needed
  - Robust error handling: continues processing even if some files fail
  - Efficient caching: reuses already downloaded files
"""

# Standard library imports
import logging
import os
from typing import Any, Callable, Dict, List, Optional

# External dependencies imports
import geopandas as gpd
from ipyleaflet import GeoJSON

# Internal dependencies imports
from coastseg import exception_handler
from coastseg.common import (
    download_url,
)
from coastseg.exceptions import DownloadError
from coastseg.feature import Feature

logger = logging.getLogger(__name__)

# only export Shoreline class
__all__ = ["Shoreline"]


class Shoreline(Feature):
    """
    Represents shoreline data within a specified region.

    This class manages shoreline geometries (LineStrings and MultiLineStrings) within
    a bounding box, providing functionality for downloading, preprocessing, clipping,
    and styling shoreline data from various sources including Zenodo datasets.

    Attributes:
        LAYER_NAME (str): Default layer name for shoreline display.
        SELECTED_LAYER_NAME (str): Layer name for selected shorelines.
        gdf (gpd.GeoDataFrame): GeoDataFrame containing shoreline geometries.
        filename (str): Name of the shoreline file.
    """

    DATASET_ID = "7814755"
    LAYER_NAME = "shoreline"
    SELECTED_LAYER_NAME = "Selected Shorelines"
    # Read in each shoreline file and clip it to the bounding box
    COLUMNS_TO_KEEP = [
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

    def __init__(
        self,
        bbox: Optional[gpd.GeoDataFrame] = None,
        shoreline: Optional[gpd.GeoDataFrame] = None,
        filename: Optional[str] = None,
        bounds_file: Optional[str] = None,
        shorelines_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize a Shoreline object with optional data sources.

        Args:
            bbox: Bounding box to find and clip shorelines. Triggers automatic
                file discovery and download from Zenodo datasets.
            shoreline: Existing shoreline data to use directly. Takes precedence
                over bbox if both are provided.
            filename: Name for the shoreline file. Defaults to 'shoreline.geojson'.
            bounds_file: Path to GeoJSON file containing bounding boxes for shoreline files.
                Defaults to 'coastseg/bounding_boxes/world_reference_shorelines_bboxes.geojson'.
            shorelines_directory: Directory for shoreline files. Defaults to 'coastseg/shorelines'.

        Raises:
            ValueError: If shoreline data is invalid or bbox processing fails.
            FileNotFoundError: If no intersecting shoreline data can be found.
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        # coastseg/bounding_boxes by default
        # this is the location of the bounds file. Its each shoreline file name and the bounds each covers
        self.bounds_file = bounds_file or os.path.join(
            self.base_dir, "bounding_boxes", "world_reference_shorelines_bboxes.geojson"
        )
        # coastseg/shorelines by default
        # this is the location of the shoreline files to load the reference shoreline from
        self.shoreline_dir = shorelines_dir or os.path.join(self.base_dir, "shorelines")

        # initialize the shorelines
        super().__init__(filename or "shoreline.geojson")
        self.initialize_shorelines(bbox, shoreline)

    def __repr__(self) -> str:
        """
        Return a string representation of the Shoreline object.

        Returns:
            String representation including Shoreline-specific and Feature information.
        """
        # Get base feature representation
        base_repr = super().__repr__()

        # Add shoreline-specific information
        geometry_preview = ""
        if not self.gdf.empty and "geometry" in self.gdf.columns:
            geom = self.gdf.iloc[0]["geometry"]
            geometry_preview = f", geometry_preview={str(geom)[:50]}..."

        return f"Shoreline(ids={self.ids()}{geometry_preview}, base={base_repr})"

    __str__ = __repr__

    def initialize_shorelines(
        self,
        bbox: Optional[gpd.GeoDataFrame] = None,
        shorelines: Optional[gpd.GeoDataFrame] = None,
    ) -> None:
        """
        Initialize shoreline data from either existing shorelines or a bounding box.

        Args:
            bbox: Bounding box to download and clip shorelines.
            shorelines: Existing shoreline data to use directly.

        Note:
            If both are provided, shorelines takes precedence.
        """
        if shorelines is not None:
            self.gdf = self.initialize_shorelines_with_shorelines(shorelines)
        elif bbox is not None:
            self.gdf = self.initialize_shorelines_with_bbox(bbox)
        else:
            # Initialize with empty GeoDataFrame
            self.gdf = gpd.GeoDataFrame()

    def initialize_shorelines_with_shorelines(
        self, shorelines: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Initialize shorelines using provided shoreline GeoDataFrame.

        Args:
            shorelines: GeoDataFrame containing shoreline geometries.

        Returns:
            Cleaned and validated GeoDataFrame of shorelines.

        Raises:
            ValueError: If shorelines is not a GeoDataFrame.

        Note:
            - Sets CRS to EPSG:4326 if not already set
            - Validates geometry types are LineString or MultiLineString
            - Creates unique IDs for all features
            - Keeps only specified columns related to shoreline properties
        """
        if not isinstance(shorelines, gpd.GeoDataFrame):
            raise ValueError("Shorelines must be a GeoDataFrame")
        if shorelines.empty:
            logger.warning("Empty shorelines provided")
            return shorelines

        return self.clean_gdf(
            self.ensure_crs(shorelines),
            columns_to_keep=self.COLUMNS_TO_KEEP,
            output_crs=self.DEFAULT_CRS,
            create_ids_flag=True,
            geometry_types=("LineString", "MultiLineString"),
            feature_type="shoreline",
            unique_ids=True,
            ids_as_str=True,
            help_message="Shorelines must be LineString or MultiLineString geometries",
        )

    def initialize_shorelines_with_bbox(
        self, bbox: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Initialize shorelines by downloading and clipping to a bounding box.

        Args:
            bbox: Bounding box geometry for clipping shorelines.

        Returns:
            GeoDataFrame of shorelines clipped to the bounding box.

        Raises:
            ValueError: If bbox is None or empty.
            FileNotFoundError: If no intersecting shoreline files are found.
        """
        if bbox is None or bbox.empty:
            raise ValueError("Bounding box cannot be None or empty")

        try:
            shoreline_files = self.get_intersecting_shoreline_files(bbox)
            if not shoreline_files:
                exception_handler.check_if_default_feature_available(None, "shoreline")
            return self.create_geodataframe(bbox, shoreline_files)
        except Exception as e:
            logger.error(f"Failed to initialize shorelines from bbox: {e}")
            raise

    def get_clipped_shoreline(
        self, shoreline_file: str, bbox: gpd.GeoDataFrame, columns_to_keep: List[str]
    ) -> gpd.GeoDataFrame:
        """
        Read, preprocess, and clip a shoreline file to the bounding box.

        Args:
            shoreline_file: Path to the shoreline file to read.
            bbox: Bounding box for clipping.
            columns_to_keep: List of column names to retain.

        Returns:
            Clipped shoreline GeoDataFrame with validated geometries.
        """
        return self.read_masked_clean(
            shoreline_file,
            mask=bbox,
            columns_to_keep=columns_to_keep,
            geometry_types=("LineString", "MultiLineString"),
            feature_type="shoreline",
            output_crs=self.DEFAULT_CRS,
        )

    def get_intersecting_shoreline_files(
        self,
        bbox: gpd.GeoDataFrame,
    ) -> List[str]:
        """
        Get list of shoreline files that intersect with the bounding box.

        Args:
            bbox: Bounding box to use for finding intersecting files.

        Returns:
            List of local file paths for intersecting shorelines.

        Raises:
            FileNotFoundError: If no intersecting shoreline files are found.
        """
        try:
            bounds_df = gpd.read_file(self.bounds_file, mask=bbox)
            if "filename" in bounds_df.columns:
                bounds_df = bounds_df.set_index("filename")
        except Exception as e:
            logger.warning(f"Could not load bounding box file {self.bounds_file}: {e}")
            return []

        if bounds_df.empty:
            logger.warning("No intersecting shoreline files found")
            return []

        # Process each intersecting file
        available_files = []

        for filename in bounds_df.index:
            shoreline_path = os.path.join(self.shoreline_dir, filename)

            # Check if file exists locally, download if needed
            if os.path.exists(shoreline_path):
                available_files.append(shoreline_path)
            else:
                try:
                    self.download_shoreline(filename, shoreline_path, self.DATASET_ID)
                    available_files.append(shoreline_path)
                except DownloadError as e:
                    logger.error(f"Failed to download {filename}: {e}")
                    # Continue with other files rather than failing completely

        if not available_files:
            raise FileNotFoundError(
                "No shoreline files could be obtained for the given area"
            )

        logger.info(f"Found {len(available_files)} available shoreline files")
        return available_files

    def create_geodataframe(
        self, bbox: gpd.GeoDataFrame, shoreline_files: List[str], crs: str = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        """
        Create a GeoDataFrame containing shorelines that intersect with the bounding box.

        Reads multiple shoreline files, clips them to the bounding box, and combines
        them into a single GeoDataFrame with unique IDs and proper CRS.

        Args:
            bbox: Bounding box for clipping shorelines.
            shoreline_files: List of file paths to shoreline data.
            crs: Target coordinate reference system. Defaults to 'EPSG:4326'.

        Returns:
            Combined shoreline data clipped to bbox with unique IDs.

        Raises:
            FileNotFoundError: If no shoreline files are provided.
        """
        if not shoreline_files:
            logger.error(
                "No shoreline files were provided for creating the Shoreline GeoDataFrame"
            )
            raise FileNotFoundError(
                "No shoreline files were provided for creating the Shoreline GeoDataFrame"
            )

        # Read and clip all shoreline files in parallel-friendly way
        shorelines = []
        for file_path in shoreline_files:
            try:
                clipped_shoreline = self.get_clipped_shoreline(
                    file_path, bbox, self.COLUMNS_TO_KEEP
                )
                if not clipped_shoreline.empty:
                    shorelines.append(clipped_shoreline)
            except Exception as e:
                logger.warning(f"Could not process shoreline file {file_path}: {e}")
                # Continue with other files

        if not shorelines:
            raise FileNotFoundError(
                "No valid shoreline data found after processing. Check the logs for details."
            )

        # Use parent class method to concatenate and clean
        combined_gdf = self.concat_clean(
            shorelines, ignore_index=True, drop_all_na=True
        )

        return self.clean_gdf(
            combined_gdf,
            columns_to_keep=self.COLUMNS_TO_KEEP,
            output_crs=crs,
            create_ids_flag=True,
            geometry_types=("LineString", "MultiLineString"),
            feature_type="shoreline",
            unique_ids=True,
            ids_as_str=True,
            help_message="Shorelines must be LineString or MultiLineString geometries",
        )

    def style_layer(self, geojson: Dict[str, Any], layer_name: str) -> GeoJSON:
        """
        Return a styled GeoJSON layer for shoreline visualization.

        Creates a black dashed line style appropriate for shoreline display
        on interactive maps with hover effects.

        Args:
            geojson (Dict[str, Any]): GeoJSON dictionary containing shoreline data.
            layer_name (str): Name for the GeoJSON layer.

        Returns:
            Any: Styled GeoJSON layer (ipyleaflet.GeoJSON) ready for map display.
        """
        style = {
            "color": "black",
            "fill_color": "black",
            "opacity": 1,
            "dashArray": "5",
            "fillOpacity": 0.5,
            "weight": 4,
        }
        hover_style = {"color": "white", "dashArray": "4", "fillOpacity": 0.7}
        return super().style_layer(
            geojson, layer_name, style=style, hover_style=hover_style
        )

    def download_shoreline(
        self,
        filename: str,
        save_location: str,
        dataset_id: str = "7814755",
        download_function: Callable = download_url,
    ) -> None:
        """
        Download shoreline file from Zenodo dataset.

        Args:
            filename: Name of the file to download from Zenodo.
            save_location: Full path where the downloaded file should be saved.
            dataset_id: Zenodo dataset ID. Defaults to '7814755' (world shorelines).
            download_function: Function to use for downloading. Defaults to download_url.

        Raises:
            DownloadError: If the download fails.
        """
        # Construct URL and download
        url = construct_download_url("https://zenodo.org/record/", dataset_id, filename)

        logger.info(f"Downloading {filename} from {url} to {save_location}")
        download_function(url, save_location, filename)


# helper functions
def construct_download_url(root_url: str, dataset_id: str, filename: str) -> str:
    """
    Construct a download URL for Zenodo dataset files.

    Args:
        root_url: Base URL for the Zenodo repository.
        dataset_id: Zenodo dataset identifier.
        filename: Name of the file to download.

    Returns:
        Complete download URL with download parameter.

    Example:
        >>> construct_download_url("https://zenodo.org/record/", "7814755", "shoreline.geojson")
        "https://zenodo.org/record/7814755/files/shoreline.geojson?download=1"
    """
    return f"{root_url}{dataset_id}/files/{filename}?download=1"
