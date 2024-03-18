from coastseg.feature import Feature
import geopandas as gpd

class Shoreline_Extraction_Area(Feature):
    LAYER_NAME = "shoreline_extraction_area"
    
    def __init__(
        self,
        gdf: gpd.GeoDataFrame = None,
        filename: str = None,
    ):
        """Initialize the Shoreline_Extraction_Area object"""
        if gdf is not None:
            self.gdf = gdf
        else:
            self.gdf = gpd.GeoDataFrame()
        self.filename = filename if filename else "shoreline_extraction_area.geojson"
    
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
    
    def style_layer(self, geojson: dict, layer_name: str) -> "ipyleaflet.GeoJSON":
        """Return styled GeoJson object with layer name

        Args:
            geojson (dict): geojson dictionary to be styled
            layer_name(str): name of the GeoJSON layer
        Returns:
            "ipyleaflet.GeoJSON": shoreline as GeoJSON layer styled with yellow dashes
        """
        style={
            "color": "#cb42f5", #purple
            "fill_color": "#cb42f5",
            "opacity": 1,
            "fillOpacity": 0.1,
            "weight": 3,
        }
        return super().style_layer(geojson, layer_name, style=style, hover_style=None)
    
    
    def __str__(self):
        # Get CRS information
        if self.gdf.empty:
            crs_info = "CRS: None"
        else:
            if self.gdf is not None and hasattr(self.gdf, 'crs'):
                crs_info = f"CRS: {self.gdf.crs}" if self.gdf.crs else "CRS: None"
            else:
                crs_info = "CRS: None"
        return f"Reference Shoreline Buffer:\nself.gdf:\n\n{crs_info}\n- {self.gdf}"

    def __repr__(self):
        # Get CRS information
        if self.gdf.empty:
            crs_info = "CRS: None"
        else:
            if self.gdf is not None and hasattr(self.gdf, 'crs'):
                crs_info = f"CRS: {self.gdf.crs}" if self.gdf.crs else "CRS: None"
            else:
                crs_info = "CRS: None"
        return f"Reference Shoreline Buffer:\nself.gdf:\n\n{crs_info}\n- {self.gdf}"