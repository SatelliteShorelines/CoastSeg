# Standard library imports
import logging
from typing import Union
# Internal dependencies imports
from .exceptions import BboxTooLargeError, BboxTooSmallError
# External dependencies imports
import geopandas as gpd
from shapely.geometry import shape


logger = logging.getLogger(__name__)
logger.info("I am a log from %s",__name__)

class Bounding_Box():
    """Bounding_Box 

    _extended_summary_
    """
    MAX_AREA = 3000000000   # UNITS = Sq. Meters
    MIN_AREA = 9000         # UNITS = Sq. Meters
   
    def __init__(self, rectangle: Union[dict, gpd.GeoDataFrame], filename:str=None):
        self.gdf=None
        self.filename="bbox.geojson"
        
        if type(rectangle) == type(gpd.GeoDataFrame()):
            self.gdf = rectangle
        elif type(rectangle) == dict:
            self.gdf = self.create_geodataframe(rectangle)
        if filename:
            self.filename=filename
            
    def create_geodataframe(self, rectangle:dict, crs:str='EPSG:4326') -> gpd.GeoDataFrame:
        """Creates a geodataframe with the crs specified by crs
        Args:
            rectangle (dict): geojson dictionary
            crs (str, optional): coordinate reference system string. Defaults to 'EPSG:4326'.

        Returns:
            gpd.GeoDataFrame: geodataframe with geometry column = rectangle and given crs
        """        
        geom = [shape(rectangle) ]
        geojson_bbox = gpd.GeoDataFrame({'geometry': geom})
        geojson_bbox.crs = crs
        return geojson_bbox
            
    def check_bbox_size(bbox_area: float):
        """"Raises an exception if the size of the bounding box is too large or small."""
        # Check if the size is greater than MAX_BBOX_SIZE
        if bbox_area > Bounding_Box.MAX_AREA:
            raise BboxTooLargeError
        # Check if size smaller than MIN_BBOX_SIZE
        elif bbox_area < Bounding_Box.MIN_AREA:
            raise BboxTooSmallError



