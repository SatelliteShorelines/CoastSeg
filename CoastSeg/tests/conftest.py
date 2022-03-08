# This file is meant to hold fixtures that can be used for testing
# These fixtures set up data that can be used as inputs for the tests, so that no code is repeated
import pytest
import os
import geopandas as gpd
from shapely.geometry import shape
import json
# MAKE SURE TO DOCUMENT WHAT EACH FIXTURE  IS INTENDED FOR
# FIXTURES NEEDED
# ----------------------------
# valid shapes_list
# valid large shapes_list
# valid small shapes_list
# invalid large shapes_list
# invalid small shapes_list

# -----------------------------------
# valid rois
# Valid roi selections for each shapes_list
# invalid rois

@pytest.fixture
def valid_shapeslist() -> list:
    return [{'type': 'Polygon', 'coordinates': [[[-121.609594, 36.007493],[-121.609594, 36.053998],[-121.533957, 36.053998],[-121.533957, 36.007493],[-121.609594, 36.007493]]]}]

@pytest.fixture
def valid_geojson_bbox_geodataframe(valid_shapeslist):
    shapes_list = valid_shapeslist
    geom = [shape(shapes_list[0])]
    geojson_bbox = gpd.GeoDataFrame({'geometry': geom})
    geojson_bbox.crs = 'EPSG:4326'
    return geojson_bbox

@pytest.fixture
def valid_roi_coastline_json(valid_clipped_geodataframe):
    roi_coast = json.loads(valid_clipped_geodataframe.to_json())
    return roi_coast


@pytest.fixture
def valid_shoreline_geodataframe(get_shoreline_file):
    shoreline_file = get_shoreline_file
    if os.path.exists(shoreline_file):
        with open(shoreline_file, 'r') as f:
            shoreline = gpd.read_file(f)
    else:
        raise FileNotFoundError
    return shoreline


@pytest.fixture
def valid_clipped_geodataframe(valid_geojson_bbox_geodataframe, valid_shoreline_geodataframe):
    coastline_vector=valid_shoreline_geodataframe
    geojson_bbox=valid_geojson_bbox_geodataframe
    roi_coast = gpd.clip(coastline_vector, geojson_bbox)
    roi_coast = roi_coast.to_crs('EPSG:4326')
    return roi_coast

# @pytest.fixture
# def invalid_shapeslist() -> list:
#     return [{'type': 'Polygon', 'coordinates': [[[-121.533957, 36.053998],[-121.533957, 36.007493],[-121.609594, 36.007493]]]}]


@pytest.fixture
def get_shoreline_file() -> str:
    shoreline_file=os.getcwd()+os.sep+"third_party_data"+os.sep+"stanford-xv279yj9196-geojson.json"
    assert os.path.exists(shoreline_file)
    return shoreline_file