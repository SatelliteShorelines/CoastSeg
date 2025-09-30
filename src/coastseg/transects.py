# Standard library
import logging
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Union

# Third-party
import geopandas as gpd
import pandas as pd
from ipyleaflet import GeoJSON
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

# Project / local
from coastseg.feature import Feature

logger = logging.getLogger(__name__)


def drop_columns(
    gdf: gpd.GeoDataFrame, columns_to_drop: Optional[List[str]] = None
) -> gpd.GeoDataFrame:
    """
    Drop specified columns from a GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame.
        columns_to_drop (Optional[List[str]], optional): Column names to drop.
            If None, drops default oceanographic columns.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with specified columns removed.
    """
    if columns_to_drop is None:
        columns_to_drop = [
            "MEAN_SIG_WAVEHEIGHT",
            "TIDAL_RANGE",
            "ERODIBILITY",
            "river_label",
            "sinuosity_label",
            "slope_label",
            "turbid_label",
        ]
    for col in columns_to_drop:
        if col in gdf.columns:
            gdf.drop(columns=[col], inplace=True)
    return gdf


def create_transects_with_arrowheads(
    gdf: gpd.GeoDataFrame,
    arrow_length: float = 0.0004,
    arrow_angle: int = 30,
    columns_to_drop: Optional[list[str]] = None,
) -> gpd.GeoDataFrame:
    """
    Creates transects with arrowheads to indicate direction.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing transects with LineString geometries.
        arrow_length (float, optional): Length of arrowhead in CRS units. Defaults to 0.0004.
        arrow_angle (int, optional): Angle of arrowhead in degrees. Defaults to 30.
        columns_to_drop (Optional[list[str]], optional): List of columns to drop from the GeoDataFrame.
            If None, drops default oceanographic columns.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with merged geometries of transects and arrowheads.
    """
    gdf_copy = gdf.to_crs("EPSG:4326")
    # remove unneeded columns
    gdf_copy = drop_columns(gdf_copy, columns_to_drop)

    # Create arrowheads for each transect
    gdf_copy["arrowheads"] = gdf_copy["geometry"].apply(
        lambda x: create_arrowhead(
            x, arrow_length=arrow_length, arrow_angle=arrow_angle
        )  # type: ignore
    )
    # Merge each transect with its arrowhead
    gdf_copy["merged"] = gdf_copy.apply(
        lambda row: unary_union([row["geometry"], row["arrowheads"]]), axis=1
    )
    gdf_copy.rename(
        columns={"geometry": "transect_geometry", "merged": "geometry"}, inplace=True
    )
    gdf_copy.drop(
        columns=["arrowheads", "transect_geometry"], errors="ignore", inplace=True
    )

    return gdf_copy


# Function to create an arrowhead as a triangle polygon this works in crs 4326
def create_arrowhead(
    line: LineString, arrow_length: float = 0.0004, arrow_angle: float = 30
) -> Polygon:
    """
    Create an arrowhead polygon at the end of a line.

    Args:
        line (LineString): Line geometry to create arrowhead for.
        arrow_length (float, optional): Arrowhead length in CRS units. Defaults to 0.0004 for CRS EPSG:4326.
        arrow_angle (float, optional): Arrowhead angle in degrees. Defaults to 30.

    Returns:
        Polygon: Triangular arrowhead polygon positioned at line end.
    """
    # Get the last segment of the line
    p1, p2 = line.coords[-2], line.coords[-1]
    # Calculate the angle of the line
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

    # Calculate the points of the arrowhead
    arrow_angle_rad = math.radians(arrow_angle)
    left_angle = angle - math.pi + arrow_angle_rad
    right_angle = angle - math.pi - arrow_angle_rad

    left_point = (
        p2[0] + arrow_length * math.cos(left_angle),
        p2[1] + arrow_length * math.sin(left_angle),
    )
    right_point = (
        p2[0] + arrow_length * math.cos(right_angle),
        p2[1] + arrow_length * math.sin(right_angle),
    )

    return Polygon([p2, left_point, right_point])


def load_intersecting_transects(
    rectangle: gpd.GeoDataFrame,
    transect_files: List[str],
    transect_dir: str,
    columns_to_keep: Iterable[str] = {"id", "geometry", "slope"},
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Load transects that intersect with a rectangular bounding box.

    Args:
        rectangle: GeoDataFrame containing rectangular geometry for filtering.
        transect_files: List of GeoJSON filenames to read.
        transect_dir: Directory path containing GeoJSON transect files.
        columns_to_keep: Column names to retain.
        **kwargs: Additional keyword arguments.

    Keyword Args:
        crs (str, optional): Target CRS. Defaults to "EPSG:4326".

    Returns:
        Transects that intersect with rectangle, cleaned and validated.
    """
    crs = kwargs.get("crs", "EPSG:4326")

    # Ensure rectangle has proper CRS
    rectangle = Feature.ensure_crs(rectangle, crs)

    # Load intersecting files
    gdfs = []
    for transect_file in transect_files:
        transect_path = os.path.join(transect_dir, transect_file)
        if not os.path.exists(transect_path):
            logger.warning("Transect file %s does not exist", transect_path)
            continue

        try:
            # Use Feature.read_masked_clean for consistent processing
            gdf = Feature.read_masked_clean(
                transect_path,
                mask=rectangle,
                columns_to_keep=columns_to_keep,
                geometry_types=("LineString", "MultiLineString"),
                feature_type="transects",
                output_crs=crs,
            )

            if not gdf.empty:
                logger.info(
                    f"Added transects from {os.path.splitext(transect_file)[0]}"
                )
                gdfs.append(gdf)

        except Exception as e:
            logger.warning(f"Failed to load {transect_file}: {e}")

    # Combine all loaded transects using Feature utility
    if not gdfs:
        return gpd.GeoDataFrame(
            columns=list(columns_to_keep),
            geometry=gpd.GeoSeries(dtype="geometry"),
            crs=crs,
        )

    combined_gdf = Feature.concat_clean(gdfs)

    # Use Feature.clean_gdf with unique_ids=True instead of direct create_unique_ids call
    return Feature.clean_gdf(
        combined_gdf,
        columns_to_keep=columns_to_keep,
        output_crs=crs,
        create_ids_flag=False,  # Already have IDs from read_masked_clean
        geometry_types=("LineString", "MultiLineString"),
        feature_type="transects",
        unique_ids=True,  # This handles unique ID generation internally
        ids_as_str=True,
    )


class Transects(Feature):
    """Manages coastal transect data with streamlined processing and visualization.

    Provides automated workflow for loading, filtering, and processing transect
    geometries with robust error handling and efficient spatial operations.
    Leverages Feature parent class utilities for consistent data handling.

    Attributes:
        LAYER_NAME: Default layer name for visualization.
        COLUMNS_TO_KEEP: Essential columns preserved during processing.

    Examples:
        Create from bounding box:
        >>> bbox_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs="EPSG:4326")
        >>> transects = Transects.from_bbox(bbox_gdf)

        Create from existing data:
        >>> transects = Transects.from_gdf(existing_transects_gdf)

        Load from files:
        >>> files = ["transects1.geojson", "transects2.geojson"]
        >>> transects = Transects.from_files(files)
    """

    LAYER_NAME = "transects"
    COLUMNS_TO_KEEP = set(
        [
            "id",
            "geometry",
            "slope",
            "distance",
            "feature_x",
            "feature_y",
            "nearest_x",
            "nearest_y",
        ]
    )

    # COLUMNS_TO_KEEP
    # ---------------
    # id: unique identifier for each transect
    # geometry: the geometric shape, position, and configuration of the transect
    # slope: represents the beach face slope, used for tidal correction of transect-based data
    # distance: represents the distance in degrees between the slope datum location and the transect location
    # feature_x: x-coordinate of the transect location
    # feature_y: y-coordinate of the transect location
    # nearest_x: x-coordinate of the nearest slope location to the transect
    # nearest_y: y-coordinate of the nearest slope location to the transect

    def __init__(
        self,
        bbox: Optional[gpd.GeoDataFrame] = None,
        transects: Optional[gpd.GeoDataFrame] = None,
        filename: Optional[str] = None,
    ) -> None:
        """
        Initialize a Transects object for coastal transect analysis.

        Args:
            bbox: Bounding box geometry to filter transects. Triggers automatic
                file discovery and spatial filtering.
            transects: Existing transect data to use directly. Takes precedence
                over bbox if both provided.
            filename: Filename for saving/loading. Defaults to "transects.geojson".

        Note:
            Transects takes precedence over bbox if both are provided.
            Uses Feature parent class utilities for consistent data processing.
        """
        super().__init__(filename or "transects.geojson")

        # Initialize based on provided data
        if transects is not None:
            self.gdf = self._process_transects(transects)
        elif bbox is not None:
            self.gdf = self._create_from_bbox(bbox)
        else:
            # Create empty GeoDataFrame with proper geometry column
            self.gdf = gpd.GeoDataFrame(
                columns=list(self.COLUMNS_TO_KEEP),
                geometry=gpd.GeoSeries(dtype="geometry"),
                crs=self.DEFAULT_CRS,
            )

    def _process_transects(self, transects: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Process provided transect data using Feature parent class utilities.

        Args:
            transects: Existing transect data with LineString/MultiLineString geometries.

        Returns:
            Cleaned and validated GeoDataFrame of transects.

        Raises:
            ValueError: If transects contain invalid geometry types.
        """
        return self.clean_gdf(
            self.ensure_crs(transects),
            columns_to_keep=list(self.COLUMNS_TO_KEEP),
            output_crs=self.DEFAULT_CRS,
            create_ids_flag=True,
            geometry_types=("LineString", "MultiLineString"),
            feature_type="transects",
            unique_ids=True,
            ids_as_str=True,
            help_message="The uploaded transects need to be LineStrings.",
        )

    @staticmethod
    def get_transects_directory() -> str:
        """
        Get the directory path where transect files are stored.

        Returns:
            str: Absolute path to the transects directory.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        transect_dir = os.path.abspath(os.path.join(script_dir, "transects"))
        return transect_dir

    def _create_from_bbox(self, bbox: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Create a GeoDataFrame of transects that intersect with a bounding box.

        Searches for transect files in the data directory, identifies those
        that spatially intersect with the provided bounding box, and loads them into
        a GeoDataFrame. The resulting transects are validated, and converted to the
        specified coordinate reference system.

        Args:
            bbox: Bounding box geometry to filter transects.

        Returns:
            Cleaned GeoDataFrame of intersecting transects.
        """
        # Get intersecting files and load them
        intersecting_files = self.get_intersecting_files(bbox)
        gdf = load_intersecting_transects(
            bbox[["geometry"]],  # geometry only
            intersecting_files,
            self.get_transects_directory(),
            columns_to_keep=self.COLUMNS_TO_KEEP,
        )

        if gdf.empty:
            logger.warning(f"No transects found within bounding box: {bbox.bounds}")

        return gdf  # Already cleaned by load_intersecting_transects

    @classmethod
    def from_bbox(
        cls, bbox: gpd.GeoDataFrame, filename: Optional[str] = None
    ) -> "Transects":
        """
        Create a Transects instance from a bounding box.

        Args:
            bbox: Bounding box for spatial filtering.
            filename: Optional output filename.

        Returns:
            Transects instance with intersecting data.
        """
        return cls(bbox=bbox, filename=filename)

    @classmethod
    def from_gdf(
        cls, gdf: gpd.GeoDataFrame, filename: Optional[str] = None
    ) -> "Transects":
        """
        Create a Transects instance from an existing GeoDataFrame.

        Args:
            gdf: Existing transect data.
            filename: Optional output filename.

        Returns:
            Transects instance with provided data.
        """
        return cls(transects=gdf, filename=filename)

    @classmethod
    def from_files(
        cls, transect_files: List[str], filename: Optional[str] = None
    ) -> "Transects":
        """
        Create a Transects instance from transect files.

        Args:
            transect_files: List of transect file paths.
            filename: Optional output filename.

        Returns:
            Transects instance with combined data.
        """
        gdfs = []

        for file in transect_files:
            try:
                gdf = gpd.read_file(file)
                if not gdf.empty:
                    gdfs.append(gdf)
            except Exception as e:
                logger.warning(f"Failed to load {file}: {e}")

        combined_gdf = (
            cls.concat_clean(gdfs)
            if gdfs
            else gpd.GeoDataFrame(
                columns=list(cls.COLUMNS_TO_KEEP),
                geometry=gpd.GeoSeries(dtype="geometry"),
                crs=cls.DEFAULT_CRS,
            )
        )
        return cls(transects=combined_gdf, filename=filename)

    def style_layer(
        self,
        data: Union[Dict[str, Any], gpd.GeoDataFrame],
        layer_name: str = "transects",
    ) -> GeoJSON:
        """
        Create a styled GeoJSON layer for transects visualization.
        Creates arrowheads to indicate transect direction with origin being the starting point on land.

        Args:
            data: Transect data to style (GeoDataFrame or GeoJSON dict).
            layer_name: Layer name for visualization.

        Returns:
            Styled GeoJSON layer with arrows indicating transect direction.
        """
        geojson = data
        # if the data is a gdf then convert to geojson with arrowheads, otherwise just use the geojson as is
        if isinstance(data, gpd.GeoDataFrame):
            gdf = create_transects_with_arrowheads(data, arrow_angle=30)
            geojson = self.to_geojson(gdf)

        style = {
            "color": "grey",
            "fill_color": "grey",
            "opacity": 1,
            "fillOpacity": 0.2,
            "weight": 2,
        }
        hover_style = {"color": "blue", "fillOpacity": 0.7}

        return super().style_layer(
            geojson, layer_name, style=style, hover_style=hover_style
        )

    def get_intersecting_files(self, bbox_gdf: gpd.GeoDataFrame) -> List[str]:
        """
        Get transect filenames that spatially intersect with the bounding box.

        Args:
            bbox_gdf (gpd.GeoDataFrame): Bounding box geometry to test for intersection.

        Returns:
            List[str]: Transect filenames that intersect with the bounding box.
        """
        # dataframe containing total bounding box for each transects file
        total_bounds_df = self.load_total_bounds_df()
        # filenames where transects/shoreline's bbox intersect bounding box drawn by user
        intersecting_files = []
        for filename in total_bounds_df.index:
            minx, miny, maxx, maxy = total_bounds_df.loc[filename]
            intersection_df = bbox_gdf.cx[minx:maxx, miny:maxy]
            # save filenames where gpd_bbox & bounding box for set of transects intersect
            if not intersection_df.empty:
                intersecting_files.append(filename)
        return intersecting_files

    def load_total_bounds_df(self) -> pd.DataFrame:
        """
        Load DataFrame containing bounding box information for all transect files.

        Returns:
            pd.DataFrame: DataFrame with filenames as index and bounding box coordinates
                (minx, miny, maxx, maxy) for each transect file.

        Raises:
            FileNotFoundError: If bounding boxes CSV file is not found.
        """
        # Load in the total bounding box from csv
        # Create the directory to hold the downloaded shorelines from Zenodo
        script_dir = os.path.dirname(os.path.abspath(__file__))
        bounding_box_dir = os.path.abspath(os.path.join(script_dir, "bounding_boxes"))
        if not os.path.exists(bounding_box_dir):
            os.mkdir(bounding_box_dir)

        transects_csv = os.path.join(bounding_box_dir, "transects_bounding_boxes.csv")
        if not os.path.exists(transects_csv):
            print("Did not find transects csv at ", transects_csv)
            return pd.DataFrame()  # Return empty DataFrame if file not found
        else:
            total_bounds_df = pd.read_csv(transects_csv)

        # Set filename as index for efficient lookup
        total_bounds_df = total_bounds_df.set_index("filename")
        return total_bounds_df
