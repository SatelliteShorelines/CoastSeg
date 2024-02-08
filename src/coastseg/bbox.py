# Standard library imports
from typing import Optional, Union

# Internal dependencies imports
from .exceptions import BboxTooLargeError, BboxTooSmallError
from coastseg.common import preprocess_geodataframe, validate_geometry_types
from coastseg.feature import Feature

# External dependencies imports
import geopandas as gpd
from shapely.geometry import shape
from ipyleaflet import GeoJSON

__all__ = ["Bounding_Box"]


class Bounding_Box(Feature):
    """Bounding_Box

    A Bounding Box drawn by user.
    """

    MAX_AREA = 100000000000  # UNITS = Sq. Meters
    MIN_AREA = 1000  # UNITS = Sq. Meters
    LAYER_NAME = "Bbox"

    def __init__(self, rectangle: Union[dict, gpd.GeoDataFrame], filename: str = None):
        self.gdf = None
        self.filename = filename if filename else "bbox.geojson"
        if isinstance(rectangle, gpd.GeoDataFrame):
            self.gdf = self._initialize_from_gdf(rectangle)
        elif isinstance(rectangle, dict):
            self.gdf = self.create_geodataframe(rectangle)
        else:
            raise Exception(
                "Invalid rectangle provided to BBox must be either a geodataframe or dict"
            )

    def __str__(self):
        return f"BBox: geodataframe {self.gdf}"

    def __repr__(self):
        return f"BBox: geodataframe {self.gdf}"

    def _initialize_from_gdf(self, bbox_gdf: gpd.GeoDataFrame) -> None:
        """
        Initialize the `gdf` attribute from a GeoDataFrame containing a bounding box.

        Args:
            rois_gdf: A GeoDataFrame containing a bounding box.

        Returns:
            None.

        Raises:
            None.
        """
        # clean the geodataframe
        bbox_gdf = preprocess_geodataframe(
            bbox_gdf,
            columns_to_keep=["geometry"],
            create_ids=False,
            output_crs="EPSG:4326",
        )
        validate_geometry_types(
            bbox_gdf, set(["Polygon", "MultiPolygon"]), feature_type="Bounding Box"
        )
        return bbox_gdf

    def create_geodataframe(
        self, rectangle: dict, crs: Optional[str] = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        """Creates a geodataframe in crs "EPSG:4326" with the provided geometry in rectangle. The
        The geometry must in CRS epsg 4326 or the code will not work properly
        Args:
            rectangle (dict): geojson dictionary
            crs (str, optional): coordinate reference system string. Defaults to 'EPSG:4326'.

        Returns:
            gpd.GeoDataFrame: geodataframe with geometry column = rectangle with crs = "EPSG:4326"
        """
        geom = [shape(rectangle)]
        geojson_bbox = gpd.GeoDataFrame({"geometry": geom})
        geojson_bbox.set_crs(crs, inplace=True)
        # clean the geodataframe
        geojson_bbox = preprocess_geodataframe(
            geojson_bbox,
            columns_to_keep=["geometry"],
            create_ids=False,
            output_crs="EPSG:4326",
        )
        validate_geometry_types(
            geojson_bbox, set(["Polygon", "MultiPolygon"]), feature_type="Bounding Box"
        )
        return geojson_bbox

    def style_layer(self, geojson: dict, layer_name: str) -> "ipyleaflet.GeoJSON":
        """Return styled GeoJson object with layer name

        Args:
            geojson (dict): geojson dictionary to be styled
            layer_name(str): name of the GeoJSON layer
        Returns:
            "ipyleaflet.GeoJSON": shoreline as GeoJSON layer styled with yellow dashes
        """
        style={
            "color": "#75b671",
            "fill_color": "#75b671",
            "opacity": 1,
            "fillOpacity": 0.1,
            "weight": 3,
        }
        return super().style_layer(geojson, layer_name, style=style, hover_style=None)
        # if geojson == {}:
        #     raise Exception("ERROR.\n Empty geojson cannot be drawn onto map")
        # return GeoJSON(
        #     data=geojson,
        #     name=layer_name,
        #     style={
        #         "color": "#75b671",
        #         "fill_color": "#75b671",
        #         "opacity": 1,
        #         "fillOpacity": 0.1,
        #         "weight": 3,
        #     },
        # )

    def check_bbox_size(bbox_area: float):
        """ "Raises an exception if the size of the bounding box is too large or small."""
        # Check if the size is greater than MAX_BBOX_SIZE
        if bbox_area > Bounding_Box.MAX_AREA:
            raise BboxTooLargeError()
        # Check if size smaller than MIN_BBOX_SIZE
        elif bbox_area < Bounding_Box.MIN_AREA:
            raise BboxTooSmallError()
