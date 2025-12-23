"""
CoastSeg Shoreline Class and Utilities

Streamlined tools to download, validate, clip, and visualize shoreline geometries
(LineString/MultiLineString) with CRS handling (EPSG:4326), unique ID generation,
and simple ipyleaflet styling. Automatically handles Zenodo data fetching.

Classes:
  - Shoreline: manages shoreline data and operations.

Functions:
  - construct_download_url: helper for Zenodo URL construction.
  - raise_download_shoreline_error: helper to raise download errors with instructions.

Key Features:
  - Streamlined loading: automatically finds intersecting files and downloads as needed
  - Robust error handling: continues processing even if some files fail
  - Efficient caching: reuses already downloaded files
"""

# Standard library imports
import logging
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

# External dependencies imports
import geopandas as gpd
from ipyleaflet import GeoJSON

# Internal dependencies imports
from coastseg import exception_handler
from coastseg.common import (
    download_url,
    check_url_status,
)
from coastseg.exceptions import DownloadError, DownloadShorelineError
from coastseg.feature import Feature
from coastseg import core_utilities

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
    ZENODO_URL = "https://zenodo.org/record/7814755"
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
            shorelines_dir: Directory to read/write shoreline files.
                If not provided, CoastSeg will try to create a project-level 'shorelines'
                directory and fall back to the packaged shoreline directory on failure.

        Raises:
            ValueError: If shoreline data is invalid or bbox processing fails.
            FileNotFoundError: If no intersecting shoreline data can be found.
        """
        # base directory is coastseg package location
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        # coastseg/bounding_boxes by default
        # this is the location of the bounds file. Its each shoreline file name and the bounds each covers
        self.bounds_file = bounds_file or os.path.join(
            self.base_dir, "bounding_boxes", "world_reference_shorelines_bboxes.geojson"
        )
        self.shoreline_pkg_dir = self.get_shoreline_package_dir()
        self.shoreline_dir = self._resolve_shoreline_dir(shorelines_dir)
        logger.info("Using shoreline directory: %s", self.shoreline_dir)
        # initialize the shorelines
        super().__init__(filename or "shoreline.geojson")
        self.initialize_shorelines(bbox, shoreline)

    def _resolve_shoreline_dir(self, shorelines_dir: Optional[str]) -> str:
        """Resolve the directory used to store shoreline files.

        Precedence:
        1) Explicit `shorelines_dir` (created if missing when possible)
        2) Project-level `CoastSeg/shorelines` (via `make_shoreline_dir()`)
        3) Packaged shoreline directory (read-only fallback)

        Args:
            shorelines_dir (Optional[str]): User-provided shoreline directory.

        Returns:
            str: Directory path used for shoreline storage.
        """
        if shorelines_dir:
            try:
                os.makedirs(shorelines_dir, exist_ok=True)
            except OSError as e:
                logger.warning(
                    "Could not create shorelines_dir '%s' (%s). Falling back.",
                    shorelines_dir,
                    e,
                )
            else:
                return shorelines_dir

        try:
            return make_shoreline_dir()
        except Exception as e:
            logger.error(
                "Defaulting to package shoreline directory. Could not create shoreline directory: %s",
                e,
            )
            return self.shoreline_pkg_dir

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
        Initialize shorelines by downloading them from zenodo and clipping to a bounding box.

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
            # get the shoreline files that intersect with the bbox
            shoreline_files = self.get_intersecting_shoreline_files(bbox)
            if (
                not shoreline_files
            ):  # if no default shorelines existed within bbox then tell the user to use their own
                exception_handler.check_if_default_feature_available(None, "shoreline")
            return self.create_geodataframe(bbox, shoreline_files)
        except Exception as e:
            logger.error(f"Failed to initialize shorelines from bbox: {e}")
            raise e

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

    def read_bounding_boxes(
        self, geojson_file: str, bbox: gpd.GeoDataFrame
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Read bounding boxes from a GeoJSON file that intersect with a given bounding box.
        The bounding boxes file contains the extents of all shoreline files available and are indexed by filename.
        Eg. shoreline_1.geojson, geometry: POLYGON((...))
        Args:
            geojson_file (str): Path to the GeoJSON file containing bounding boxes.
            bbox (gpd.GeoDataFrame): GeoDataFrame representing the bounding box to use as a mask
                for filtering the geometries from the GeoJSON file.
        Returns:
            Optional[gpd.GeoDataFrame]: A GeoDataFrame containing the bounding boxes that intersect
                with the provided bbox, with 'filename' set as the index if the column exists.
                Returns None if the file cannot be loaded or if no intersecting geometries are found.
        Raises:
            Logs a warning if the GeoJSON file cannot be loaded or if no intersecting shoreline
            files are found.
        """
        try:
            bounds_df = gpd.read_file(geojson_file, mask=bbox)
            if "filename" in bounds_df.columns:
                bounds_df = bounds_df.set_index("filename")
        except Exception as e:
            logger.warning(f"Could not load bounding box file {geojson_file}: {e}")
            return None

        if bounds_df.empty:
            logger.warning("No intersecting shoreline files found")
            return None

        return bounds_df

    def get_shoreline_package_dir(self) -> str:
        """
        Get the default shoreline directory within the CoastSeg package.

        Returns:
            str: Path to the CoastSeg shoreline directory.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        shoreline_dir = os.path.join(base_dir, "shorelines")
        return shoreline_dir

    def get_local_shoreline_files(
        self,
        bounds_file: str,
        bbox: gpd.GeoDataFrame,
        dirs_to_check: Optional[List[str]] = None,
    ) -> tuple:
        """
        Get list of local shoreline files that intersect with the specified bounding box and a list of filenames to download.
        This method reads a bounding boxes file to determine which shoreline files intersect
        with the given bbox, then checks which of those files exist locally in the shoreline
        directory. If they do not exist locally, their filenames are added to a list for downloading.
        Args:
            bounds_file (str): Path to the file containing bounding box information for
                shoreline files.
            bbox (gpd.GeoDataFrame): Bounding box GeoDataFrame to check for intersection. This is the area the user wants to load shorelines within
            dirs_to_check (Optional[List[str]]): List of directories to check for shoreline files.
                If None, defaults to checking only the main shoreline directory.
        Returns:
            tuple: A tuple containing two lists:
                - A list of file paths (str) to locally available shoreline files that
                  intersect with the specified bounding box.
                - A list of filenames (str) that need to be downloaded because they are not available locally.
        """
        bounds_df = self.read_bounding_boxes(bounds_file, bbox)
        if bounds_df is None:
            return [], []

        search_dirs = dirs_to_check or [self.shoreline_dir]

        def first_existing_path(filename: str) -> Optional[str]:
            for shoreline_dir in search_dirs:
                shoreline_path = os.path.join(shoreline_dir, filename)
                if os.path.exists(shoreline_path):
                    return shoreline_path
            return None

        # Check for each shoreline file if its available locally
        available_files: List[str] = []
        files_to_download: List[str] = []

        for filename in bounds_df.index:
            shoreline_path = first_existing_path(filename)
            if shoreline_path:
                available_files.append(shoreline_path)
            else:
                files_to_download.append(filename)

        return available_files, files_to_download

    def download_all_shorelines(self, filenames: List[str], output_dir) -> list:
        """
        Download the requested files from Zenodo dataset and save to output directory.
        Validates Zenodo API availability before each download and if rate limited
        raises a DownloadShorelineError with remaining files to download so the user can downlaod them manually.

        Args:
            filenames: List of shoreline file names to download.
        Raises:
            DownloadError: If any download fails.
            DownloadShorelineError: If the zenodo api returns the status 429
        """
        downloaded_files = []
        filenames_stack = (
            filenames.copy()
        )  # make a stack so its easier to track the files the user need to download for the error message

        # Gets the filename from the stack as long as one is available and if zenodo is not avilable prints error message with remaining files to download
        while filenames_stack:
            # check if the api is available before downloading each file
            status_code = check_url_status(self.ZENODO_URL)
            if status_code == 429:
                logger.info(
                    f"Zenodo returned 429 (rate limited) for url {self.ZENODO_URL}."
                )
                raise_download_shoreline_error(
                    self.shoreline_dir, filenames_stack, self.DATASET_ID
                )
            else:
                filename = (
                    filenames_stack.pop()
                )  # now that we are sure zenodo is available lets remove the file to download from the stack
                # attempt the download
                shoreline_path = os.path.join(output_dir, filename)
                try:
                    self.download_shoreline(filename, shoreline_path, self.DATASET_ID)
                    downloaded_files.append(shoreline_path)
                except DownloadError as e:
                    logger.error(f"Failed to download {filename}: {e}")
                    # Continue with other files rather than failing completely

        return downloaded_files

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
            DownloadShorelineError: If shoreline files cannot be downloaded due to rate limiting.
        """
        ## Check at <coastseg package location>/shorelines first then user shoreline dir
        search_dirs = [
            self.shoreline_pkg_dir,
            self.shoreline_dir,
        ]  # search the user shoreline dir first then the coastseg shoreline dir

        # Get local shoreline files that intersect with the bbox and the list of filenames to download
        available_files, files_to_download = self.get_local_shoreline_files(
            self.bounds_file, bbox, search_dirs
        )

        # if there are files to download then download them now
        if files_to_download:
            # once that is done see if the zenodo api is available
            status_code = check_url_status(self.ZENODO_URL)
            logger.info(f"Zenodo status code: {status_code}")
            # now download all the files that need to be downloaded if zenodo is available

            if status_code == 200:
                downloaded_files = self.download_all_shorelines(
                    files_to_download, self.shoreline_dir
                )
                available_files.extend(downloaded_files)
            else:  # if zenodo is not available then have the user manually download the files
                raise_download_shoreline_error(
                    self.shoreline_dir, files_to_download, self.DATASET_ID
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
        logger.info(
            f"Preparing to download shoreline file: {filename} for dataset {dataset_id} from Zenodo"
        )
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


def raise_download_shoreline_error(
    shoreline_dir: Union[os.PathLike, str], files_to_download: Iterable[str], dataset_id
) -> None:
    """
    Raises a DownloadShorelineError with instructions for manual file downloads.

    This function is called when shoreline files cannot be automatically downloaded from Zenodo
    (typically due to rate limiting). It constructs download URLs for the required files and
    raises an error with HTML-formatted instructions for manual download.

    Args:
        shoreline_dir (Union[os.PathLike, str]): The directory where the shoreline files should be saved.
        files_to_download (Iterable[str]): An iterable of filenames that need to be downloaded.
        dataset_id: The Zenodo dataset identifier used to construct the download URLs.

    Raises:
        DownloadShorelineError: Always raised with a message containing clickable download links
            and instructions for manual file download and placement.

    Note:
        The error message includes HTML-formatted links that will be clickable in contexts
        that support HTML rendering.
    """
    save_location = os.path.abspath(shoreline_dir)
    # Make a list of the urls you need
    urls = [
        construct_download_url(
            "https://zenodo.org/record/",
            dataset_id,
            filename,  # type: ignore
        )
        for filename in files_to_download
    ]
    # Convert URLs to clickable HTML links
    clickable_urls = [
        f'<a href="{u}" target="_blank" rel="noopener noreferrer">{u}</a>' for u in urls
    ]
    # Trigger a custom popup to tell the user to click each link to download them then save
    raise DownloadShorelineError(
        "Error these shoreline files cannot be downloaded from Zenodo due to rate limiting: "
        f"Please download the file(s) manually from the links below and place them in {save_location}.<br>"
        "Once you have downloaded the files close this popup and click 'Load Shoreline' again.<br>"
        + "<br>".join(clickable_urls)
    )


def make_shoreline_dir() -> str:
    """Create and return the CoastSeg project-level shorelines directory.

    Returns:
        str: Absolute path to the `CoastSeg/shorelines` directory.

    Raises:
        OSError: If the directory cannot be created.
    """
    base_dir = core_utilities.get_base_dir()  # Get CoastSeg project repo
    shoreline_dir = os.path.join(base_dir, "shorelines")
    # might be a good idea to set shoreline_dir to self.shoreline_dir
    os.makedirs(shoreline_dir, exist_ok=True)
    return shoreline_dir
