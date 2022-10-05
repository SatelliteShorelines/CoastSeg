# This file is meant to hold fixtures that can be used for testing
# These fixtures set up data that can be used as inputs for the tests, so that no code is repeated
import pytest
import os
import geopandas as gpd
from shapely.geometry import shape
import json
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def valid_bbox_gdf()->gpd.GeoDataFrame:
    file_path=os.path.abspath(os.path.join(script_dir,'test_data','valid_bbox.geojson'))
    with open(file_path, 'r') as f:
        gpd_data = gpd.read_file(f)
    return gpd_data

@pytest.fixture
def valid_bbox_geojson()->dict:
    file_path=os.path.abspath(os.path.join(script_dir,'test_data','valid_bbox.geojson'))
    with open(file_path, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)
    return data
