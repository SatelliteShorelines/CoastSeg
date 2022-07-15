# Standard library imports
import json
import os

# External dependencies imports
from area import area
import geopandas as gpd
from shapely.geometry import shape
from shapely import geometry
from ipyleaflet import GeoJSON

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


def create_geodataframe_from_bbox(
        shapes_list: list) -> "geopandas.geodataframe.GeoDataFrame":
    """ Create a geodataframe from the bounding box's geometry"""
    assert shapes_list != [], "ERROR.\nYou must draw a bounding box somewhere on the coast first."
    geom = [shape(i) for i in shapes_list]
    geojson_bbox = gpd.GeoDataFrame({'geometry': geom})
    geojson_bbox.crs = 'EPSG:4326'
    return geojson_bbox


# def clip_coastline_to_bbox(
#         coastline_vector: "geopandas.geodataframe.GeoDataFrame",
#         geojson_bbox: "geopandas.geodataframe.GeoDataFrame") -> "geopandas.geodataframe.GeoDataFrame":
#     """
#     Clips the geojson coastline_vector within geojson_bbox,the geojson bounding box.
#      Arguments:
#     -----------
#     coastline_vector: geopandas.geodataframe.GeoDataFrame
#         GeoDataFrame contains the coastline's geometry
#     geojson_bbox: "geopandas.geodataframe.GeoDataFrame"
#         GeoDataFrame containing the bounding box geometry
#     Returns
#     --------
#     roi_coast : "geopandas.geodataframe.GeoDataFrame"
#         roi_coast a GeoDataFrame that holds the clipped portion of the coastline_vector within geojson_bbox
#     """
#     assert coastline_vector.empty != True, "ERROR: Empty shoreline dataframe"
#     # clip coastal polyline
#     roi_coast = gpd.clip(coastline_vector, geojson_bbox)
#     roi_coast = roi_coast.to_crs('EPSG:4326')
#     return roi_coast

# @ todo reowork this 
def get_coastline_gpd(shoreline_file: str, bbox: list):
    """
         geojson containing the clipped portion of the coastline in bounding box

     Arguments:
    -----------
    shoreline_file: str
        location of geojson file containing shoreline data
    bbox: dict
        dict containing geojson of the bounding box drawn by the user

    Returns:
    --------
    roi_coast: dict
        contains the clipped portion of the coastline in bounding box

    """
    # Read the coastline vector from a geopandas geodataframe file
    shoreline = read_gpd_file(shoreline_file)
    # Convert bbox to GDP
    geojson_bbox = create_geodataframe_from_bbox(bbox)
    # Clip coastline to bbox
    roi_coast_gdp = clip_coastline_to_bbox(shoreline, geojson_bbox)
    return roi_coast_gdp

#@todo remove this
# def get_coastline(shoreline_file: str, bbox: list):
#     """
#          geojson containing the clipped portion of the coastline in bounding box

#      Arguments:
#     -----------
#     shoreline_file: str
#         location of geojson file containing shoreline data
#     bbox: dict
#         dict containing geojson of the bounding box drawn by the user

#     Returns:
#     --------
#     roi_coast: dict
#         contains the clipped portion of the coastline in bounding box

#     """
#     # Read the coastline vector from a geopandas geodataframe file
#     shoreline = read_gpd_file(shoreline_file)
#     # Convert bbox to GDP
#     geojson_bbox = create_geodataframe_from_bbox(bbox)
#     # Clip coastline to bbox
#     roi_coast_gdp = clip_coastline_to_bbox(shoreline, geojson_bbox)
#     # Convert coastline geodataframe(gdp) to json string then to dictionary
#     roi_coast = json.loads(roi_coast_gdp.to_json())
#     return roi_coast


def get_coastline_for_map(coast_geojson: dict):
    """Returns a GeoJSON object that can be added to the map """
    assert coast_geojson != {}, "ERROR.\n Empty geojson cannot be drawn onto  map"
    return GeoJSON(
        data=coast_geojson,
        style={
            'color': 'yellow',
            'fill_color': 'yellow',
            'opacity': 1,
            'dashArray': '5',
            'fillOpacity': 0.5,
            'weight': 4},
        hover_style={
            'color': 'white',
            'dashArray': '4',
            'fillOpacity': 0.7},
    )
