# from CoastSeg import make_overlapping_roi
from CoastSeg import bbox
from CoastSeg import exceptions
import geopandas as gpd
import pytest


# Tests for bbox.check_bbox_size()
# ----------------------------------------------------------
# Test MAX box size
def test_max_bbox_size():
    sample_shapes_list = [10]
    with pytest.raises(exceptions.BboxTooLargeError):
        bbox.check_bbox_size(3000000001, sample_shapes_list)


# test MIN box size
def test_min_bbox_size():
    sample_shapes_list = [10]
    with pytest.raises(exceptions.BboxTooSmallError):
        bbox.check_bbox_size(3, sample_shapes_list)


# test valid box size
def test_valid_bbox_size():
    sample_shapes_list = [10]
    bbox.check_bbox_size(300000, sample_shapes_list)


# Tests for bbox.calculate_area_bbox()
# -------------------------------------------------------------
# 1. test valid shapeslist
def test_valid_calculate_area_bbox(valid_shapeslist):
    sample_shapes_list = valid_shapeslist
    area = bbox.calculate_area_bbox(sample_shapes_list[0])
    assert area == 35250443.43,\
        f"Area was : {area}"


# Test bbox.validate_bbox_size()
# ------------------------------------------------------
# 1. test empty shapes list throws
def test_empty_shapeslist_validate_bbox_size():
    with pytest.raises(AssertionError):
        bbox.validate_bbox_size([])

# 2. test valid shapes_list


def test_valid_validate_bbox_size(valid_shapeslist):
    sample_shapes_list = valid_shapeslist
    returned_shapes_list = bbox.validate_bbox_size(sample_shapes_list)
    print(returned_shapes_list)
    print(sample_shapes_list)
    assert(returned_shapes_list == sample_shapes_list[0])

# 3. test invalid shapes_list
# def test_invalid_validate_bbox_size(invalid_shapeslist):
#     sample_shapes_list = invalid_shapeslist
#     returned_shapes_list = bbox.validate_bbox_size(sample_shapes_list)
#     print(returned_shapes_list)
#     print(sample_shapes_list)
#     assert(returned_shapes_list == sample_shapes_list[0])

# Test bbox.read_gpd_file()
# ------------------------------------------------------
# 1. test for filename that doesn't exist


def test_bad_filename_read_gpd_file():
    with pytest.raises(FileNotFoundError):
        bbox.read_gpd_file("badfilename.json")


# 2. Test with valid file
def test_valid_filename_read_gpd_file(get_shoreline_file):
    shoreline_file = get_shoreline_file
    bbox.read_gpd_file(shoreline_file)


# Test bbox.create_geodataframe_from_bbox
# ------------------------------------------------------
# 1. test for empty shapes_list
def test_empty_create_geodataframe_from_bbox():
    with pytest.raises(AssertionError):
        bbox.create_geodataframe_from_bbox([])

# 2. test with valid shapes_list


def test_valid_create_geodataframe_from_bbox(
        valid_shapeslist, valid_geojson_bbox_geodataframe):
    shapes_list = valid_shapeslist
    geojson_bbox = bbox.create_geodataframe_from_bbox(shapes_list)
    assert isinstance(geojson_bbox, gpd.geodataframe.GeoDataFrame)
    assert geojson_bbox.crs == 'EPSG:4326'
    valid_geojson_bbox_geodataframe = valid_geojson_bbox_geodataframe
    assert valid_geojson_bbox_geodataframe.equals(geojson_bbox)


# Test bbox.clip_coastline_to_bbox
# ------------------------------------------------------
# 1. test for empty shapes_list
def test_empty_clip_coastline_to_bbox(valid_geojson_bbox_geodataframe):
    empty_geodataframe = gpd.GeoDataFrame({})
    valid_geojson_bbox_geodataframe = valid_geojson_bbox_geodataframe
    with pytest.raises(AssertionError):
        bbox.clip_coastline_to_bbox(
            empty_geodataframe,
            valid_geojson_bbox_geodataframe)

# 2. test with valid shapes_list


def test_valid_clip_coastline_to_bbox(
        valid_shoreline_geodataframe,
        valid_geojson_bbox_geodataframe,
        valid_clipped_geodataframe):
    geojson_bbox_geodataframe = valid_geojson_bbox_geodataframe
    shoreline_geodataframe = valid_shoreline_geodataframe
    clipped_geojson_bbox = bbox.clip_coastline_to_bbox(
        shoreline_geodataframe, geojson_bbox_geodataframe)
    assert isinstance(clipped_geojson_bbox, gpd.geodataframe.GeoDataFrame)
    assert clipped_geojson_bbox.crs == 'EPSG:4326'
    assert clipped_geojson_bbox.equals(valid_clipped_geodataframe)


# Test bbox.get_coastline
# ------------------------------------------------------
# 1. test for empty shapes_list
def test_empty_get_coastline(get_shoreline_file):
    with pytest.raises(AssertionError):
        bbox.get_coastline(get_shoreline_file, [])

# 2. test for invalid shoreline file


def test_invalid_get_coastline():
    with pytest.raises(FileNotFoundError):
        bbox.get_coastline("badfilename.json", [])

# 3. test with valid shapes_list and valid shoreline file


def test_valid_get_coastline(get_shoreline_file, valid_shapeslist,valid_roi_coastline_json):
    json_coastline=bbox.get_coastline(get_shoreline_file, valid_shapeslist)
    assert  json_coastline == valid_roi_coastline_json

# Test bbox.get_coastline_for_map
# ------------------------------------------------------
# 1. test for empty geojson dict


def test_empty_get_coastline_for_map():
    with pytest.raises(AssertionError):
        bbox.get_coastline_for_map({})
