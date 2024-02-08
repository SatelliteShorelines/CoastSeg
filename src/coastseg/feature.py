from abc import ABC, abstractmethod
from ipyleaflet import GeoJSON
import geopandas as gpd
import json

class Feature(ABC):
    def style_layer(self, data: dict, layer_name: str,style:dict=None,hover_style:dict={}) -> GeoJSON:
        """Return styled GeoJson object with layer name.
        Default style is grey solid lines and red on hover.

        Args:
            data (dict or geodataframe): The geojson dictionary or geodataframe to be styled.
            layer_name(str): name of the GeoJSON layer
        Returns:
            "ipyleaflet.GeoJSON": ROIs as GeoJson layer styled with yellow dashes
        """
        if isinstance(data, dict):
            geojson = data
        elif isinstance(data,gpd.GeoDataFrame):
            geojson  = json.loads(data.to_json())
 
        assert (
            geojson != {}
        ), f"ERROR.\n Empty {layer_name} geojson cannot be drawn onto map"
    
        if hover_style is None:
            hover_style= {}
        if style is None:
            style={
                "color": "#555555",
                "fill_color": "#555555",
                "fillOpacity": 0.1,
                "weight": 1,
            }
        
        return GeoJSON(
            data=geojson,
            name=layer_name,
            style=style,
            hover_style=hover_style,
        )