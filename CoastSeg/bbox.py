# Standard library imports
import os
from datetime import datetime
import logging
# External dependencies imports
from area import area
import geopandas as gpd
from shapely.geometry import shape
# from CoastSeg.coastseg_logs import log_file


logger = logging.getLogger(__name__)
logger.info("I am a log from %s",__name__)

# Internal dependencies imports
from .exceptions import BboxTooLargeError, BboxTooSmallError


def calculate_area_bbox(bbox: dict):
    "Calculates the area of the geojson polygon using the same method as geojson.io"
    bbox_area = round(area(bbox), 2)
    return bbox_area


def read_gpd_file(filename: str) -> "geopandas.geodataframe.GeoDataFrame":
    """
    Returns geodataframe from geopandas geodataframe file
    """
    logger.info("read_gpd_file")
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            gpd_data = gpd.read_file(f)
    else:
        print('File does not exist. Please download the coastline_vector necessary here: https://geodata.lib.berkeley.edu/catalog/stanford-xv279yj9196 ')
        raise FileNotFoundError
    return gpd_data


def check_bbox_size(bbox_area: float, shapes_list: list):
    """"Raises an exception and clears the map if the size of the bounding box is too large or small."""
    # UNITS = Sq. Meters
    MAX_BBOX_SIZE = 3000000000
    MIN_BBOX_SIZE = 9000
    # Check if the size is greater than MAX_BBOX_SIZE
    if bbox_area > MAX_BBOX_SIZE:
        shapes_list = []
        raise BboxTooLargeError
    # Check if size smaller than MIN_BBOX_SIZE
    elif bbox_area < MIN_BBOX_SIZE:
        shapes_list = []
        raise BboxTooSmallError


def validate_bbox_size(shapes_list: list):
    assert shapes_list != [], "ERROR.\nYou must draw a bounding box somewhere on the coast first."
    last_index = len(shapes_list) - 1
    # The last index in the shapes_list was the last shape drawn
    bbox_area = calculate_area_bbox(shapes_list[last_index])
    check_bbox_size(bbox_area, shapes_list)
    return shapes_list[last_index]


def clip_to_bbox( gdf_to_clip:'geopandas.geodataframe.GeoDataFrame', bbox_gdf:'geopandas.geodataframe.GeoDataFrame')->'geopandas.geodataframe.GeoDataFrame':        
    """Clip gdf_to_clip to bbox_gdf. Only data within bbox will be kept.
    Args:
        gdf_to_clip (geopandas.geodataframe.GeoDataFrame): geodataframe to be clipped to bbox
        bbox_gdf (geopandas.geodataframe.GeoDataFrame): drawn bbox
    Returns:
        geopandas.geodataframe.GeoDataFrame: clipped geodata within bbox
    """        
    transects_in_bbox = gpd.clip(gdf_to_clip, bbox_gdf)
    transects_in_bbox = transects_in_bbox.to_crs('EPSG:4326')
    return transects_in_bbox

def create_geodataframe_from_bbox(
        shapes_list: list) -> "geopandas.geodataframe.GeoDataFrame":
    """ Create a geodataframe from the bounding box's geometry"""
    assert shapes_list != [], "ERROR.\nYou must draw a bounding box somewhere on the coast first."
    geom = [shape(i) for i in shapes_list]
    geojson_bbox = gpd.GeoDataFrame({'geometry': geom})
    geojson_bbox.crs = 'EPSG:4326'
    return geojson_bbox
