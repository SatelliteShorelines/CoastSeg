# Standard library imports
from __future__ import annotations
import datetime
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

if TYPE_CHECKING:
    from coastseg.extracted_shoreline import Extracted_Shoreline

# External dependencies imports
import geopandas as gpd
from ipyleaflet import GeoJSON
from shapely import geometry

# Internal dependencies imports
from coastseg import common, exceptions
from coastseg.feature import Feature

logger = logging.getLogger(__name__)

__all__ = ["ROI"]


def get_ids_with_invalid_area(
    gdf: gpd.GeoDataFrame, max_area: float = 98000000, min_area: float = 0
) -> List[str]:
    """
    Get indices of geometries with areas outside the specified range.

    Projects to appropriate UTM CRS for accurate area calculation.

    Args:
        gdf: GeoDataFrame with geometries to check.
        max_area: Maximum allowable area in square meters.
        min_area: Minimum allowable area in square meters.

    Returns:
        List of indices for geometries with invalid areas.

    Raises:
        TypeError: If gdf is not a GeoDataFrame.
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("Input must be a GeoDataFrame")

    if gdf.empty:
        return []

    # Project to UTM for accurate area calculation
    projected = gdf.to_crs(gdf.estimate_utm_crs())
    areas = projected.area

    # Return indices where area is outside valid range
    invalid_mask = (areas > max_area) | (areas < min_area)
    return gdf.index[invalid_mask].tolist()


class ROI(Feature):
    """A class that controls all the ROIs on the map"""

    LAYER_NAME = "ROIs"
    SELECTED_LAYER_NAME = "Selected ROIs"
    MAX_SIZE = 98000000  # 98km^2 area
    MIN_SIZE = 0

    def __init__(
        self,
        bbox: Optional[gpd.GeoDataFrame] = None,
        shoreline: Optional[gpd.GeoDataFrame] = None,
        rois_gdf: Optional[gpd.GeoDataFrame] = None,
        square_len_lg: float = 0,
        square_len_sm: float = 0,
        filename: Optional[str] = None,
    ) -> None:
        """
        Initialize the ROI object.

        Args:
            bbox: Bounding box GeoDataFrame.
            shoreline: Shoreline GeoDataFrame.
            rois_gdf: Existing ROIs GeoDataFrame.
            square_len_lg: Large square length for ROI generation.
            square_len_sm: Small square length for ROI generation.
            filename: Filename for saving ROIs.
        """
        # Initialize parent class
        super().__init__(filename or "rois.geojson")

        # Initialize ROI-specific attributes
        self.roi_settings: Dict[str, Any] = {}
        self.extracted_shorelines: Dict[str, Extracted_Shoreline] = {}
        self.cross_shore_distances: Dict[str, Dict[str, Any]] = {}

        # Initialize GeoDataFrame based on provided data
        if rois_gdf is not None:
            self.gdf = self._initialize_from_roi_gdf(rois_gdf)
        else:
            self.gdf = self._initialize_from_bbox_and_shoreline(
                bbox, shoreline, square_len_lg, square_len_sm
            )

    def __repr__(self) -> str:
        """
        Returns string representation of the ROI object.
        """
        # Get column names and their data types
        col_info = self.gdf.dtypes.apply(lambda x: x.name).to_string()
        # Get first 5 rows as a string
        first_rows = self.gdf.head().to_string()
        # Get CRS information
        crs_info = f"CRS: {self.gdf.crs}" if self.gdf.crs else "CRS: None"
        extracted_shoreline_info = ""
        for key in self.extracted_shorelines.keys():
            if hasattr(self.extracted_shorelines[key], "gdf") and (
                isinstance(self.extracted_shorelines[key].gdf, gpd.GeoDataFrame)
            ):
                if not self.extracted_shorelines[key].gdf.empty:
                    extracted_shoreline_info.join(
                        f"ROI ID {key}:\n{len(self.extracted_shorelines[key].gdf)}\n"
                    )
        return f"ROI:\nROI IDs: {self.get_ids()}\nROI IDs with extracted shorelines: {extracted_shoreline_info}\nROI IDs with shoreline transect intersections: {list(self.cross_shore_distances.keys())}\n gdf:\n{crs_info}\nColumns and Data Types:\n{col_info}\n\nFirst 5 Rows:\n{first_rows}"

    __str__ = __repr__

    def remove_by_id(
        self, ids_to_drop: Union[List[str], Set[str], Tuple[str, ...], str, int]
    ) -> gpd.GeoDataFrame:
        """
        Remove ROIs by their IDs and clean up associated data.

        Args:
            ids_to_drop: IDs to remove, can be single or collection.

        Returns:
            Updated GeoDataFrame after removal.
        """
        # Use parent class method for the basic removal
        result_gdf = super().remove_by_id(ids_to_drop)

        # Clean up ROI-specific data
        if ids_to_drop is not None:
            if isinstance(ids_to_drop, (str, int)):
                ids_to_drop = [str(ids_to_drop)]
            ids_to_drop_set = set(map(str, ids_to_drop))

            logger.info(f"Removing ROI-specific data for IDs: {ids_to_drop_set}")
            # Remove corresponding extracted shorelines and cross shore distances
            for roi_id in ids_to_drop_set:
                self.remove_extracted_shorelines(roi_id)
                self.remove_cross_shore_distance(roi_id)

        self.gdf = result_gdf
        return result_gdf.copy(deep=True)

    def _initialize_from_roi_gdf(self, rois_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Initialize GeoDataFrame from existing ROIs.

        Args:
            rois_gdf: GeoDataFrame of ROIs.

        Returns:
            Cleaned and validated GeoDataFrame of ROIs.

        Raises:
            exceptions.InvalidSize: If all ROIs have invalid sizes.
        """
        gdf = self.clean_gdf(
            self.ensure_crs(rois_gdf),
            columns_to_keep=("id", "geometry"),
            output_crs=self.DEFAULT_CRS,
            create_ids_flag=True,
            geometry_types=("Polygon", "MultiPolygon"),
            feature_type="ROI",
            unique_ids=True,
            ids_as_str=True,
        )
        return self.validate_ROI_sizes(gdf)

    def validate_ROI_sizes(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Validate ROI sizes and remove invalid ones.

        Args:
            gdf: GeoDataFrame to validate.

        Returns:
            GeoDataFrame with valid ROIs only.

        Raises:
            exceptions.InvalidSize: If all ROIs are invalid.
        """
        drop_index = get_ids_with_invalid_area(
            gdf, max_area=self.MAX_SIZE, min_area=self.MIN_SIZE
        )
        if drop_index:
            logger.info(f"Dropping ROIs with invalid size: {drop_index}")
            gdf = gdf.drop(index=drop_index)

        if gdf.empty:
            raise exceptions.InvalidSize(
                "All ROI(s) have invalid sizes.",
                "ROI",
                max_size=self.MAX_SIZE,
                min_size=self.MIN_SIZE,
            )

        return gdf

    def _initialize_from_bbox_and_shoreline(
        self,
        bbox: Optional[gpd.GeoDataFrame],
        shoreline: Optional[gpd.GeoDataFrame],
        square_len_lg: float,
        square_len_sm: float,
    ) -> gpd.GeoDataFrame:
        """
        Initialize GeoDataFrame from bounding box and shoreline intersection.

        Args:
            bbox: Bounding box GeoDataFrame.
            shoreline: Shoreline GeoDataFrame.
            square_len_lg: Large square side length in meters.
            square_len_sm: Small square side length in meters.

        Returns:
            GeoDataFrame of ROIs intersecting with the shoreline.

        Raises:
            exceptions.Object_Not_Found: If bbox or shoreline is missing/empty.
            ValueError: If both square lengths are zero.
        """
        # Validate inputs
        if bbox is None or bbox.empty:
            raise exceptions.Object_Not_Found("bounding box")
        if shoreline is None or shoreline.empty:
            raise exceptions.Object_Not_Found("shorelines")
        if square_len_sm == square_len_lg == 0:
            raise ValueError("At least one square size must be greater than 0")

        return self.create_geodataframe(bbox, shoreline, square_len_lg, square_len_sm)

    def get_roi_settings(
        self, roi_id: Union[str, Iterable[str]] = ""
    ) -> Dict[str, Any]:
        """
        Retrieve settings for specific ROI(s) or all settings.

        Args:
            roi_id: ROI ID, collection of IDs, or empty string for all settings.

        Returns:
            Dictionary of ROI settings.

        Raises:
            TypeError: If roi_id is not a string or iterable of strings.
        """
        if not hasattr(self, "roi_settings") or self.roi_settings is None:
            self.roi_settings = {}

        if roi_id is None or roi_id == "":
            logger.info(f"Returning all ROI settings: {len(self.roi_settings)} items")
            return self.roi_settings

        if isinstance(roi_id, str):
            result = self.roi_settings.get(roi_id, {})
            logger.info(f"ROI settings for {roi_id}: {bool(result)}")
            return result

        if isinstance(roi_id, Iterable) and not isinstance(roi_id, (str, bytes)):
            result = {}
            for id_str in roi_id:
                if not isinstance(id_str, str):
                    raise TypeError(f"Each ROI ID must be a string, got {type(id_str)}")
                if id_str in self.roi_settings:
                    result[id_str] = self.roi_settings[id_str]
            return result

        raise TypeError("roi_id must be a string or iterable of strings")

    def set_roi_settings(self, roi_settings: Dict[str, Any]) -> None:
        """
        Sets ROI settings dictionary.

        Args:
            roi_settings: Dict of settings.

        Raises:
            ValueError: If roi_settings is None.
            TypeError: If roi_settings is not dict.
        """
        if roi_settings is None:
            raise ValueError("roi_settings cannot be None")
        if not isinstance(roi_settings, dict):
            raise TypeError("roi_settings must be a dictionary")

        logger.info(f"Saving roi_settings {roi_settings}")
        self.roi_settings = roi_settings

    def update_roi_settings(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update ROI settings with new values.

        Args:
            new_settings: Dictionary of new settings to merge.

        Returns:
            Updated settings dictionary.

        Raises:
            ValueError: If new_settings is None.
            TypeError: If new_settings is not a dictionary.
        """
        if new_settings is None:
            raise ValueError("new_settings cannot be None")
        if not isinstance(new_settings, dict):
            raise TypeError("new_settings must be a dictionary")

        logger.info(f"Updating ROI settings with {len(new_settings)} items")

        if not hasattr(self, "roi_settings") or self.roi_settings is None:
            self.roi_settings = {}

        self.roi_settings.update(new_settings)
        return self.roi_settings

    def get_extracted_shoreline(self, roi_id: str) -> Optional[Extracted_Shoreline]:
        """
        Returns extracted shoreline for ROI ID.

        Args:
            roi_id: ROI ID.

        Returns:
            Extracted shoreline or None.
        """
        return self.extracted_shorelines.get(roi_id, None)

    def get_ids(self) -> List[str]:
        """
        Get list of all ROI IDs.

        Returns:
            List of ROI ID strings.
        """
        return self.ids()

    def get_ids_with_extracted_shorelines(self) -> List[str]:
        """
        Returns list of ROI IDs with extracted shorelines.
        """
        return list(self.get_all_extracted_shorelines().keys())

    @classmethod
    def extract_roi_by_id(
        cls, gdf: gpd.GeoDataFrame, roi_id: Union[str, int]
    ) -> gpd.GeoDataFrame:
        """
        Extract a single ROI from GeoDataFrame by ID.

        Args:
            gdf: GeoDataFrame containing ROIs to extract from.
            roi_id: ID of the ROI to extract. If None, returns original GeoDataFrame.

        Returns:
            GeoDataFrame with a single ROI matching the given ID.

        Raises:
            exceptions.Id_Not_Found: If the ID doesn't exist in the GeoDataFrame or GeoDataFrame is empty.

        Example:
            >>> import geopandas as gpd
            >>> from shapely.geometry import Polygon
            >>> from coastseg.roi import ROI
            >>> 
            >>> # Create test GeoDataFrame with ROIs
            >>> roi_data = {
            ...     'id': ['roi1', 'roi2'],
            ...     'geometry': [
            ...         Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            ...         Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
            ...     ]
            ... }
            >>> rois_gdf = gpd.GeoDataFrame(roi_data, crs='EPSG:4326')
            >>> 
            >>> # Extract single ROI by ID
            >>> single_roi = ROI.extract_roi_by_id(rois_gdf, 'roi1')
            >>> print(single_roi['id'].iloc[0])
            roi1
        """
        if roi_id is None:
            return gdf

        # Select a single roi by id
        single_roi = gdf[gdf["id"].astype(str) == str(roi_id)]
        
        # If the id was not found in the GeoDataFrame raise an exception
        if single_roi.empty:
            logger.error(f"Id: {roi_id} was not found in {gdf}")
            # Convert to int if possible for the exception, otherwise use None
            try:
                id_as_int = int(roi_id)
                raise exceptions.Id_Not_Found(id_as_int)
            except (ValueError, TypeError):
                raise exceptions.Id_Not_Found(None, f"The ROI id '{roi_id}' does not exist.")
        
        logger.info(f"single_roi: {single_roi}")
        return single_roi

    def add_geodataframe(self, gdf: gpd.GeoDataFrame) -> "ROI":
        """
        Add GeoDataFrame to existing ROI object.

        Args:
            gdf: GeoDataFrame to add.

        Returns:
            Updated ROI object.
        """
        if gdf.empty:
            return self

        # Clean and validate the new GeoDataFrame
        cleaned_gdf = self.clean_gdf(
            gdf,
            columns_to_keep=("id", "geometry"),
            output_crs=self.DEFAULT_CRS,
            create_ids_flag=True,
            geometry_types=("Polygon", "MultiPolygon"),
            feature_type="ROI",
            unique_ids=True,
            ids_as_str=True,
        )

        # Validate sizes
        cleaned_gdf = self.validate_ROI_sizes(cleaned_gdf)

        # Combine and remove duplicates
        self.gdf = self.concat_clean(
            [self.gdf, cleaned_gdf], ignore_index=True, drop_all_na=True
        ).drop_duplicates(subset=["id", "geometry"], keep="first")

        return self

    def get_all_extracted_shorelines(
        self,
    ) -> Dict[str, Extracted_Shoreline]:
        """
        Returns dict of all extracted shorelines.
        """
        if not hasattr(self, "extracted_shorelines"):
            self.extracted_shorelines = {}
        return self.extracted_shorelines

    def remove_extracted_shorelines(
        self, roi_id: Optional[Union[str, int]] = None, remove_all: bool = False
    ) -> None:
        """
        Remove extracted shoreline(s) for specified ROI ID or all.

        Args:
            roi_id: ROI ID to remove shorelines for.
            remove_all: If True, remove all extracted shorelines.
        """
        if remove_all:
            self.extracted_shorelines.clear()
        elif roi_id is not None:
            roi_id_str = str(roi_id)
            if roi_id_str in self.extracted_shorelines:
                del self.extracted_shorelines[roi_id_str]

    def remove_selected_shorelines(
        self, roi_id: str, dates: List[datetime.datetime], satellites: List[str]
    ) -> None:
        """
        Removes selected shorelines for ROI.

        Args:
            roi_id (str): The ID of the ROI.
            dates (list[datetime.datetime]): A list of dates for which the shorelines should be removed.
            satellites (list[str]): A list of satellite names for which the shorelines should be removed.

        Returns:
            None
        """
        if roi_id in self.get_ids_with_extracted_shorelines():
            extracted_shoreline = self.get_extracted_shoreline(roi_id)
            if extracted_shoreline is not None:
                extracted_shoreline.remove_selected_shorelines(dates, satellites)

    def add_extracted_shoreline(
        self,
        extracted_shoreline: Extracted_Shoreline,
        roi_id: str,
    ) -> None:
        """
        Adds extracted shoreline for ROI ID.

        Args:
            extracted_shoreline: Extracted shoreline object.
            roi_id: ROI ID.
        """
        self.extracted_shorelines[roi_id] = extracted_shoreline
        logger.info(f"New extracted shoreline added for ROI {roi_id}")
        # logger.info(f"New extracted shoreline added for ROI {roi_id}: {self.extracted_shorelines}")

    def get_cross_shore_distances(self, roi_id: str) -> Dict[str, Any]:
        """
        Get cross shore distances for specified ROI ID.

        Args:
            roi_id: ROI ID to get distances for.

        Returns:
            Dictionary of cross shore distances for the ROI.
        """
        result = self.cross_shore_distances.get(roi_id, {})
        logger.debug(
            f"ROI {roi_id}: {'found' if result else 'no'} cross shore distances"
        )
        return result

    def add_cross_shore_distances(
        self, cross_shore_distance: Dict[str, Any], roi_id: str
    ) -> None:
        """
        Adds cross shore distances for ROI ID.

        Args:
            cross_shore_distance: Dict of distances.
            roi_id: ROI ID.
        """
        self.cross_shore_distances[roi_id] = cross_shore_distance

    def get_all_cross_shore_distances(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns dict of all cross shore distances.
        """
        return self.cross_shore_distances

    def remove_cross_shore_distance(
        self, roi_id: Optional[str] = None, remove_all: bool = False
    ) -> None:
        """
        Remove cross shore distance(s) for specified ROI ID or all.

        Args:
            roi_id: ROI ID to remove distances for.
            remove_all: If True, remove all cross shore distances.
        """
        if remove_all:
            self.cross_shore_distances.clear()
        elif roi_id is not None and roi_id in self.cross_shore_distances:
            del self.cross_shore_distances[roi_id]

    def create_geodataframe(
        self,
        bbox: gpd.GeoDataFrame,
        shoreline: gpd.GeoDataFrame,
        large_length: float = 7500,
        small_length: float = 5000,
    ) -> gpd.GeoDataFrame:
        """
        Generate ROIs along shoreline using fishnet method.

        Args:
            bbox: Bounding box GeoDataFrame.
            shoreline: Shoreline GeoDataFrame.
            large_length: Large square side length in meters.
            small_length: Small square side length in meters.

        Returns:
            GeoDataFrame of ROIs intersecting the shoreline.

        Raises:
            ValueError: If bbox or shoreline is None.
        """
        if bbox is None or shoreline is None:
            raise ValueError("bbox and shoreline must not be None")

        # Determine fishnet configuration
        fishnets = []
        if large_length > 0:
            fishnets.append(self.get_fishnet_gdf(bbox, shoreline, large_length))
        if small_length > 0:
            fishnets.append(self.get_fishnet_gdf(bbox, shoreline, small_length))

        if not fishnets:
            raise ValueError("At least one of large_length or small_length must be > 0")

        # Combine fishnets if multiple sizes specified
        fishnet_gdf = self.concat_clean(fishnets, ignore_index=True)

        # Clean and validate the final GeoDataFrame
        return self.clean_gdf(
            fishnet_gdf,
            columns_to_keep=("id", "geometry"),
            output_crs=self.DEFAULT_CRS,
            create_ids_flag=True,
            geometry_types=("Polygon", "MultiPolygon"),
            feature_type="rois",
            unique_ids=True,
            ids_as_str=True,
        )

    def style_layer(self, geojson: Dict[str, Any], layer_name: str) -> GeoJSON:
        """
        Returns styled GeoJSON layer.

        Args:
            geojson: GeoJSON dict.
            layer_name: Layer name.

        Returns:
            Styled GeoJSON layer.
        """
        return super().style_layer(
            geojson,
            layer_name,
            hover_style={"color": "red", "fillOpacity": 0.1, "fillColor": "crimson"},
        )

    def fishnet_intersection(
        self, fishnet: gpd.GeoDataFrame, data: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Find intersection between fishnet grid and data geometries.

        Args:
            fishnet: Fishnet grid GeoDataFrame.
            data: Data GeoDataFrame to intersect with.

        Returns:
            GeoDataFrame containing intersecting geometries.
        """
        # Perform a spatial join between the fishnet and data to find the intersecting geometries
        intersection_gdf = gpd.sjoin(
            left_df=fishnet, right_df=data, how="inner", predicate="intersects"
        )

        # Keep only geometry and remove duplicates
        result = intersection_gdf[["geometry"]].drop_duplicates(
            subset=["geometry"], keep="first"
        )

        return result.reset_index(drop=True)

    def create_rois(
        self,
        bbox: gpd.GeoDataFrame,
        square_size: float,
        input_epsg: Union[str, int] = "epsg:4326",
        output_epsg: str = "epsg:4326",
    ) -> gpd.GeoDataFrame:
        """
        Create fishnet of square ROIs from bounding box.

        Args:
            bbox: Bounding box GeoDataFrame.
            square_size: Square side length in meters.
            input_epsg: Input EPSG code (unused, determined automatically).
            output_epsg: Output EPSG code for final projection.

        Returns:
            GeoDataFrame containing square ROI geometries.
        """
        # Determine appropriate UTM projection for accurate area calculations
        projected_epsg = common.get_epsg_from_geometry(bbox.iloc[0]["geometry"])
        logger.debug(f"Using projected EPSG: {projected_epsg}")

        # Project to UTM and create fishnet
        projected_bbox = bbox.to_crs(projected_epsg)
        return self.create_fishnet(
            projected_bbox, projected_epsg, output_epsg, square_size
        )

    def create_fishnet(
        self,
        bbox_gdf: gpd.GeoDataFrame,
        input_epsg: Union[str, int],
        output_epsg: str,
        square_size: float = 1000,
    ) -> gpd.GeoDataFrame:
        """
        Create a fishnet grid of square geometries within bounding box.

        Args:
            bbox_gdf: Bounding box GeoDataFrame.
            input_epsg: Input EPSG code for the bounding box.
            output_epsg: Output EPSG code for final projection.
            square_size: Square side length in meters.

        Returns:
            GeoDataFrame containing fishnet grid geometries.
        """
        minX, minY, maxX, maxY = bbox_gdf.total_bounds

        # Generate grid of squares
        geometries = []
        y = minY
        while y <= maxY:
            x = minX
            while x <= maxX:
                square = geometry.Polygon(
                    [
                        (x, y),
                        (x, y + square_size),
                        (x + square_size, y + square_size),
                        (x + square_size, y),
                        (x, y),
                    ]
                )
                geometries.append(square)
                x += square_size
            y += square_size

        # Create GeoDataFrame and project to output CRS
        fishnet = gpd.GeoDataFrame(geometries, columns=["geometry"], crs=input_epsg)
        logger.debug(f"Created {len(geometries)} fishnet squares")

        return fishnet.to_crs(output_epsg)

    def get_fishnet_gdf(
        self,
        bbox_gdf: gpd.GeoDataFrame,
        shoreline_gdf: gpd.GeoDataFrame,
        square_length: float = 1000,
    ) -> gpd.GeoDataFrame:
        """
        Generate fishnet grid that intersects with shoreline.

        Args:
            bbox_gdf: Bounding box GeoDataFrame.
            shoreline_gdf: Shoreline GeoDataFrame to intersect with.
            square_length: Square side length in meters.

        Returns:
            GeoDataFrame containing fishnet squares intersecting shoreline.

        Raises:
            ValueError: If bbox_gdf or shoreline_gdf is None.
        """
        if bbox_gdf is None or shoreline_gdf is None:
            raise ValueError("Both bbox_gdf and shoreline_gdf are required")

        # Create fishnet and find intersection with shoreline
        fishnet = self.create_rois(bbox_gdf, square_length)
        return self.fishnet_intersection(fishnet, shoreline_gdf)
