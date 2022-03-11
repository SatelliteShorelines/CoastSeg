# This file is meant to hold fixtures that can be used for testing
# These fixtures set up data that can be used as inputs for the tests, so that no code is repeated
import pytest
import os
import geopandas as gpd
from shapely.geometry import shape
import json
import pickle
import download_roi


@pytest.fixture
def expected_shapes_list() -> list:
    """ Returns a list of all the true expected shapes list (aka the bbox)"""
    shapes_list_pkl = './CoastSeg/tests/test_data/ca_simple_shapes_list.pkl'
    with open(shapes_list_pkl, "rb") as file:
        simple_shapes = pickle.load(file)

    shapes_list_pkl = './CoastSeg/tests/test_data/island_shapes_list.pkl'   
    with open(shapes_list_pkl, "rb") as file:
        island_shapes = pickle.load(file)

    shapes_list_pkl = './CoastSeg/tests/test_data/duck_shapes_list.pkl'
    with open(shapes_list_pkl, "rb") as file:
        duck_shapes = pickle.load(file)

    shapes_list_pkl = './CoastSeg/tests/test_data/sf_complex_shapes_list.pkl'   
    with open(shapes_list_pkl, "rb") as file:
        sf_complex_shapes = pickle.load(file)        

    expected_shapes_list=[island_shapes, simple_shapes, duck_shapes, sf_complex_shapes]
    return expected_shapes_list


@pytest.fixture
def expected_lines_list() -> list:
    """ Returns a list of all the true expected lines list (aka all the lines that makeup the coastline)"""
    lines_list_pkl = './CoastSeg/tests/test_data/ca_simple_lines_list.pkl'
    with open(lines_list_pkl, "rb") as file:
        simple_lines = pickle.load(file)

    lines_list_pkl = './CoastSeg/tests/test_data/island_lines_list.pkl'   
    with open(lines_list_pkl, "rb") as file:
        island_lines = pickle.load(file)

    lines_list_pkl = './CoastSeg/tests/test_data/duck_lines_list.pkl'
    with open(lines_list_pkl, "rb") as file:
        duck_lines = pickle.load(file)

    lines_list_pkl = './CoastSeg/tests/test_data/sf_complex_lines_list.pkl'   
    with open(lines_list_pkl, "rb") as file:
        sf_complex_lines = pickle.load(file)        

    expected_lines_list=[island_lines, simple_lines, duck_lines, sf_complex_lines]
    return expected_lines_list


@pytest.fixture
def expected_coastline_list() -> list:
    """ Returns a list of all the true expected coastlines (aka coastline in the bbox)"""
    roi_coastline_pkl = './CoastSeg/tests/test_data/ca_simple_coastline.pkl'
    with open(roi_coastline_pkl, "rb") as poly_file:
        roi_simple_coastline = pickle.load(poly_file)
    roi_coastline_pkl = './CoastSeg/tests/test_data/island_coastline.pkl'   
    with open(roi_coastline_pkl, "rb") as poly_file:
        roi_island_coastline = pickle.load(poly_file)
    roi_coastline_pkl = './CoastSeg/tests/test_data/duck_coastline.pkl'
    with open(roi_coastline_pkl, "rb") as poly_file:
        duck_coastline = pickle.load(poly_file)
    roi_coastline_pkl = './CoastSeg/tests/test_data/sf_complex_coastline.pkl'
    with open(roi_coastline_pkl, "rb") as poly_file:
        sf_complex_coastline = pickle.load(poly_file)

    coastline_list=[roi_island_coastline,roi_simple_coastline,duck_coastline,sf_complex_coastline]
    return coastline_list


@pytest.fixture
def expected_multipoint_list() -> list:
    """ Returns a list of all the true expected multipoints list (aka the the points that make up  each linestring)
    As of 3/11/2022 this  is  only for the duck coastline
    Returned type is a list of multipoint lists"""
    multipoint_list_pkl = './CoastSeg/tests/test_data/duck_multipoint_list.pkl'
    with open(multipoint_list_pkl, "rb") as file:
        expected_multipoint_list = pickle.load(file)
    return expected_multipoint_list


@pytest.fixture
def expected_tuples_list() -> list:
    """ Returns a list of all the true expected tuples list (aka the the points that make up  each linestring)
    As of 3/11/2022 this  is  only for the duck coastline
    Returned type is a list of tuple lists"""
    tuples_list_pkl ='./CoastSeg/tests/test_data/duck_tuples_list.pkl'
    with open(tuples_list_pkl, "rb") as file:
        expected_tuples_list = pickle.load(file)
    return expected_tuples_list


@pytest.fixture
def expected_geojson_polygons_list() -> list:
    """ Returns a list of all the true expected tuples list (aka the the points that make up  each linestring)
    As of 3/11/2022 this  is  only for the duck coastline
    Returned type is a list of geojson polygons"""
    geojson_polygons_pkl ='./CoastSeg/tests/test_data/duck_geojson_polygons.pkl'
    with open(geojson_polygons_pkl, "rb") as file:
        expected_geojson_polygons_list = pickle.load(file)
    return expected_geojson_polygons_list    

@pytest.fixture
def expected_geojson_file() -> list:
    """ As of 3/11/2022 this  is  only for the duck coastline
    Returned type geojson file contents"""
    duck_geojson_file = './CoastSeg/tests/test_data/official_roi_duck.geojson'
    assert os.path.exists(duck_geojson_file),f"File {duck_geojson_file} not found"
    expected_duck_geojson = download_roi.read_geojson_file(duck_geojson_file)
    return expected_duck_geojson

@pytest.fixture
def expected_geojson_bbox_geodataframe(expected_shapes_list):
    shapes_list = expected_shapes_list[0]
    geom = [shape(shapes_list[0])]
    geojson_bbox = gpd.GeoDataFrame({'geometry': geom})
    geojson_bbox.crs = 'EPSG:4326'
    return geojson_bbox

@pytest.fixture
def expected_roi_coastline_json(expected_clipped_geodataframe):
    roi_coast = json.loads(expected_clipped_geodataframe.to_json())
    return roi_coast


@pytest.fixture
def expected_shoreline_geodataframe(expected_shoreline_file):
    shoreline_file = expected_shoreline_file
    if os.path.exists(shoreline_file):
        with open(shoreline_file, 'r') as f:
            shoreline = gpd.read_file(f)
    else:
        raise FileNotFoundError
    return shoreline


@pytest.fixture
def expected_clipped_geodataframe(expected_geojson_bbox_geodataframe, expected_shoreline_geodataframe):
    coastline_vector = expected_shoreline_geodataframe
    geojson_bbox = expected_geojson_bbox_geodataframe
    roi_coast = gpd.clip(coastline_vector, geojson_bbox)
    roi_coast = roi_coast.to_crs('EPSG:4326')
    return roi_coast


@pytest.fixture
def expected_shoreline_file() -> str:
    shoreline_file=os.getcwd()+os.sep+"third_party_data"+os.sep+"stanford-xv279yj9196-geojson.json"
    assert os.path.exists(shoreline_file)
    return shoreline_file