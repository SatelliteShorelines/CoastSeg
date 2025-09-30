from typing import Optional

import geopandas as gpd
from ipyleaflet import GeoJSON

from coastseg.feature import Feature


class Shoreline_Extraction_Area(Feature):
    LAYER_NAME = "shoreline_extraction_area"

    def __init__(
        self,
        gdf: Optional[gpd.GeoDataFrame] = None,
        filename: Optional[str] = None,
    ):
        """Initialize the Shoreline_Extraction_Area object"""
        super().__init__(filename or "shoreline_extraction_area.geojson")

        if gdf is None:
            gdf = gpd.GeoDataFrame(
                columns=["geometry"], geometry="geometry", crs=self.DEFAULT_CRS
            )
        self.gdf = self.clean_gdf(
            self.ensure_crs(gdf, self.DEFAULT_CRS),
            columns_to_keep=("geometry",),
            output_crs=self.DEFAULT_CRS,
            geometry_types=("Polygon", "MultiPolygon"),
        )

    def style_layer(self, geojson: dict, layer_name: str) -> GeoJSON:
        """Return styled GeoJson object with layer name

        Args:
            geojson (dict): geojson dictionary to be styled
            layer_name(str): name of the GeoJSON layer
        Returns:
            GeoJSON: shoreline as GeoJSON layer styled with yellow dashes
        """
        style = {
            "color": "#cb42f5",  # purple
            "fill_color": "#cb42f5",
            "opacity": 1,
            "fillOpacity": 0.1,
            "weight": 3,
        }
        return super().style_layer(geojson, layer_name, style=style, hover_style=None)
