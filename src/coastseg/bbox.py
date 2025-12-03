"""Bounding box feature utilities for CoastSeg.

Defines the Bounding_Box class for managing rectangular areas of interest (AOI).
Supports initialization from GeoJSON or GeoDataFrame, validates geometry types,
normalizes to EPSG:4326, and provides styled map layers with area constraints.

Example:
    >>> from coastseg.bbox import Bounding_Box
    >>> import geopandas as gpd
    >>> # Create from GeoJSON polygon
    >>> polygon = {
    ...     "type": "Polygon",
    ...     "coordinates": [[[-118.5, 33.5], [-118.5, 33.6], [-118.4, 33.6], [-118.4, 33.5], [-118.5, 33.5]]]
    ... }
    >>> bbox = Bounding_Box(rectangle=polygon)
    >>> print(f"Area: {bbox.gdf.to_crs('EPSG:3857').area.iloc[0]:.2f} m²")
    >>> map_layer = bbox.style_layer(bbox.gdf.__geo_interface__, "My AOI")
"""

# Standard library imports
from typing import Any, Dict, Union

# External dependencies imports
import geopandas as gpd
from ipyleaflet import GeoJSON

from coastseg.feature import Feature

__all__ = ["Bounding_Box"]


class Bounding_Box(Feature):
    """A user-drawn rectangular Area of Interest (AOI).

    Wraps a GeoDataFrame that stores a single Polygon/MultiPolygon representing a
    bounding box, normalized to EPSG:4326 and validated for geometry type.

    Attributes:
        gdf (gpd.GeoDataFrame): The underlying GeoDataFrame (single geometry) with
            CRS set to EPSG:4326.
        filename (str): Default filename to use when persisting the feature.
        MAX_AREA (int): Maximum allowed area of the box in square meters.
        MIN_AREA (int): Minimum allowed area of the box in square meters.
        LAYER_NAME (str): Default display name for map layers.

    Raises:
        Exception: If rectangle is neither a GeoDataFrame nor a GeoJSON-like
            mapping.
    """

    MAX_AREA: int = 100000000000  # UNITS = Sq. Meters
    MIN_AREA: int = 1000  # UNITS = Sq. Meters
    LAYER_NAME: str = "Bbox"

    def __init__(
        self,
        rectangle: Union[Dict[str, Any], gpd.GeoDataFrame],
        filename: str = "bbox.geojson",
    ) -> None:
        """
        Initialize Bounding_Box from geometry.

        Args:
            rectangle (Dict[str, Any] | gpd.GeoDataFrame): The bounding geometry. When a dict
                is provided, it must be a GeoJSON-like mapping representing a single
                Polygon or MultiPolygon in EPSG:4326 (or accompanied by the
                provided crs in :meth:`create_geodataframe`). When a GeoDataFrame
                is provided, it will be cleaned and reprojected to EPSG:4326.
            filename (str): Default filename.

        Raises:
            Exception: If rectangle type invalid.
        """
        super().__init__(filename)
        if isinstance(rectangle, gpd.GeoDataFrame):
            gdf = rectangle
        elif isinstance(rectangle, dict):
            gdf = self.gdf_from_mapping(rectangle, crs=Feature.DEFAULT_CRS)
        else:
            raise TypeError("rectangle must be GeoDataFrame or GeoJSON-like dict")

        self.gdf = self.clean_gdf(
            self.ensure_crs(gdf),
            columns_to_keep=("geometry",),
            output_crs=self.DEFAULT_CRS,
            geometry_types=("Polygon", "MultiPolygon"),
            feature_type="Bounding Box",
        )

    def style_layer(self, geojson: Dict[str, Any], layer_name: str) -> GeoJSON:
        """
        Return styled GeoJSON layer for map.

        Args:
            geojson: GeoJSON to render.
            layer_name: Display name.

        Returns:
            Styled GeoJSON layer.
        """
        style = {
            "color": "#75b671",
            "fill_color": "#75b671",
            "opacity": 1,
            "fillOpacity": 0.1,
            "weight": 3,
        }
        return super().style_layer(geojson, layer_name, style=style, hover_style={})

    @staticmethod
    def check_bbox_size(bbox_area: Union[int, float]) -> None:
        """
        Validate bounding box area.

        Args:
            bbox_area: Area in square meters.

        Raises:
            BboxTooLargeError: If exceeds max area.
            BboxTooSmallError: If below min area.
        """
        from .exceptions import BboxTooLargeError, BboxTooSmallError

        Feature.check_size(
            bbox_area,
            min_area=Bounding_Box.MIN_AREA,
            max_area=Bounding_Box.MAX_AREA,
            too_small_exc=BboxTooSmallError,
            too_large_exc=BboxTooLargeError,
        )
