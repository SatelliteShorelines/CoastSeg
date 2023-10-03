import pytest
import os
import json
import datetime
import tempfile
import shutil

from coastseg import exceptions
from coastseg import common
from coastseg import file_utilities

from shapely.geometry import LineString, MultiLineString, Point, Polygon
import geopandas as gpd
import numpy as np
from shapely import geometry
import pandas as pd
import pytest
from unittest.mock import patch


def test_filter_partial_images():
    # Mock rectangle with known size 1 degree by 1 degree at the equator.
    # This is approximately 111 km by 111 km.
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    polygon = Polygon(coords)
    roi_gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:4326")

    directory = "/path/to/directory"

    with patch("coastseg.common.filter_images") as mock_filter:
        common.filter_partial_images(roi_gdf, directory)

        # area calculations don't exactly match because the geometry is converted to the most accurate UTM zone to get the least distorted area
        expected_min_area = 7393.52
        expected_max_area = 18483.80
        actual_min_area, actual_max_area, actual_directory = mock_filter.call_args[0]
        # Assert directory is the same
        assert actual_directory == directory

        # Assert areas are almost equal within a tolerance
        np.testing.assert_almost_equal(actual_min_area, expected_min_area, decimal=2)
        np.testing.assert_almost_equal(actual_max_area, expected_max_area, decimal=2)


def test_non_existent_directory():
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    polygon = Polygon(coords)
    roi_gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:4326")

    directory = "/non/existent/directory"

    with pytest.raises(FileNotFoundError):
        common.filter_partial_images(roi_gdf, directory)


def test_get_roi_area():
    # Mock rectangle with known size 1 degree by 1 degree at the equator.
    # This is approximately 111 km by 111 km (for simplicity, we're treating it as a square).
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    polygon = Polygon(coords)
    gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:4326")

    # The expected area is approximately 111 * 111 = 12321 km^2.
    # We'll use `pytest.approx` to handle floating-point precision issues.
    assert common.get_roi_area(gdf) == pytest.approx(12321, 1e-3)


def test_empty_geodataframe():
    gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    with pytest.raises(IndexError):
        common.get_roi_area(gdf)


def test_invalid_projection():
    # Mock rectangle, but with a crs that won't be transformed by our get_epsg_from_geometry function
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    polygon = Polygon(coords)
    gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:3857")

    assert common.get_roi_area(gdf) == pytest.approx(9.95e-7, 1e-3)


def test_filter_images_existing_directory(setup_image_directory):
    bad_images = common.filter_images(
        0.8, 1.5, setup_image_directory, setup_image_directory.join("bad")
    )
    assert len(bad_images) == 2

    assert (
        os.path.join(setup_image_directory, "dummy_prefix_S2_image.jpg") in bad_images
    )
    assert (
        os.path.join(setup_image_directory, "dummy_prefix_L5_image.jpg") in bad_images
    )


def test_filter_images_non_existing_directory():
    with pytest.raises(FileNotFoundError):
        common.filter_images(0.8, 1.5, "non_existing_path", "some_output_path")


def test_filter_images_no_jpg_files_found(tmpdir):
    bad_files = common.filter_images(0.8, 1.5, tmpdir, tmpdir.join("bad"))
    assert len(bad_files) == 0


def test_filter_images_no_output_directory_provided(setup_image_directory):
    bad_images = common.filter_images(0.8, 1.5, setup_image_directory)
    assert len(bad_images) == 2
    assert (
        os.path.join(setup_image_directory, "dummy_prefix_S2_image.jpg") in bad_images
    )
    assert (
        os.path.join(setup_image_directory, "dummy_prefix_L5_image.jpg") in bad_images
    )
    assert os.path.exists(
        os.path.join(setup_image_directory, "bad", "dummy_prefix_S2_image.jpg")
    )
    assert os.path.exists(
        os.path.join(setup_image_directory, "bad", "dummy_prefix_L5_image.jpg")
    )


def test_valid_input():
    rois = gpd.GeoDataFrame({"geometry": [Point(1, 1)]}, crs="EPSG:4326")

    result = common.create_config_gdf(rois)
    assert not result.empty
    assert result.crs == "EPSG:4326"
    assert "type" in result.columns
    assert result["type"][0] == "roi"


def test_with_multiple_gdfs():
    rois = gpd.GeoDataFrame({"geometry": [Point(1, 1), Point(2, 2)]}, crs="EPSG:4326")

    shorelines = gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 0), (1, 1)])]}, crs="EPSG:4326"
    )

    result = common.create_config_gdf(rois, shorelines_gdf=shorelines, epsg_code=3857)

    assert not result.empty
    assert result.crs == "EPSG:3857"
    assert result["type"].nunique() == 2
    assert set(result["type"].unique()) == {"roi", "shoreline"}


def test_with_empty_rois_and_no_epsg():
    with pytest.raises(ValueError):
        result = common.create_config_gdf(None)


def test_with_empty_rois_and_valid_epsg():
    result = common.create_config_gdf(None, epsg_code=3857)
    assert result.empty
    assert result.crs == "EPSG:3857"


def test_with_all_gdfs():
    rois = gpd.GeoDataFrame({"geometry": [Point(1, 1), Point(2, 2)]}, crs="EPSG:4326")

    shorelines = gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 0), (1, 1)])]}, crs="EPSG:4326"
    )

    transects = gpd.GeoDataFrame(
        {"geometry": [LineString([(2, 2), (3, 3)])]}, crs="EPSG:4326"
    )

    bbox = gpd.GeoDataFrame({"geometry": [Point(3, 3), Point(4, 4)]}, crs="EPSG:4326")

    result = common.create_config_gdf(
        rois,
        shorelines_gdf=shorelines,
        transects_gdf=transects,
        bbox_gdf=bbox,
        epsg_code=3857,
    )

    assert not result.empty
    assert result.crs == "EPSG:3857"
    assert result["type"].nunique() == 4
    assert set(result["type"].unique()) == {"roi", "shoreline", "transect", "bbox"}


def test_set_crs_for_non_empty_gdf_with_same_epsg():
    # Create a GeoDataFrame with some data
    gdf = gpd.GeoDataFrame({"geometry": [Point(1, 1)]}, crs="EPSG:4326")

    result = common.set_crs_or_initialize_empty(gdf, "EPSG:4326")

    # Check that the resulting GeoDataFrame has the correct CRS
    assert result.crs == "EPSG:4326"
    # Check that the data remains unchanged
    assert not result.empty


def test_set_crs_for_non_empty_gdf():
    # Create a GeoDataFrame with some data
    gdf = gpd.GeoDataFrame({"geometry": [Point(1, 1)]}, crs="EPSG:4326")

    result = common.set_crs_or_initialize_empty(gdf, "EPSG:3857")

    # Check that the resulting GeoDataFrame has the correct CRS
    assert result.crs == "EPSG:3857"
    # Check that the data remains unchanged
    assert not result.empty


def test_initialize_empty_for_none_gdf():
    result = common.set_crs_or_initialize_empty(None, "EPSG:3857")

    # Check that the resulting GeoDataFrame is empty but has the correct CRS
    assert result.crs == "EPSG:3857"
    assert result.empty


def test_initialize_empty_for_empty_gdf():
    gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    result = common.set_crs_or_initialize_empty(gdf, "EPSG:3857")

    # Check that the resulting GeoDataFrame is empty but has the correct CRS
    assert result.crs == "EPSG:3857"
    assert result.empty


def test_raise_exception_on_invalid_crs():
    gdf = gpd.GeoDataFrame({"geometry": [Point(1, 1)]}, crs="EPSG:4326")

    with pytest.raises(Exception):
        # Attempt to convert to a non-existent EPSG code
        common.set_crs_or_initialize_empty(gdf, "EPSG:999999")


# Test data
@pytest.fixture(scope="function")
def empty_ROI_directory(request):
    temp_dir = tempfile.mkdtemp()
    directory_name = request.param if hasattr(request, "param") else ""

    # Create subdirectories
    os.makedirs(
        os.path.join(temp_dir, directory_name, "jpg_files", "preprocessed", "RGB")
    )

    filenames = []

    for fn in filenames:
        with open(
            os.path.join(
                temp_dir, directory_name, "jpg_files", "preprocessed", "RGB", fn
            ),
            "w",
        ) as f:
            f.write("test content")

    yield temp_dir

    shutil.rmtree(temp_dir)


# Test data
@pytest.fixture(scope="function")
def sample_ROI_directory(request):
    temp_dir = tempfile.mkdtemp()
    directory_name = request.param if hasattr(request, "param") else ""

    # Create subdirectories
    os.makedirs(
        os.path.join(temp_dir, directory_name, "jpg_files", "preprocessed", "RGB")
    )

    filenames = [
        "20210101_sat1_L5.jpg",
        "20210101_sat1_L7.jpg",
        "20210101_sat1_L8.jpg",
        "20210101_sat1_L9.jpg",
        "20210101_sat1_S2.jpg",
        "20210101_wrong_name.jpg",
        "wrong_name_2.jpg",
    ]

    for fn in filenames:
        with open(
            os.path.join(
                temp_dir, directory_name, "jpg_files", "preprocessed", "RGB", fn
            ),
            "w",
        ) as f:
            f.write("test content")

    yield temp_dir

    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def sample_directory():
    temp_dir = tempfile.mkdtemp()

    filenames = [
        "20210101_sat1_L5.jpg",
        "20210101_sat1_L7.jpg",
        "20210101_sat1_L8.jpg",
        "20210101_sat1_L9.jpg",
        "20210101_sat1_S2.jpg",
        "20210101_wrong_name.jpg",
        "wrong_name_2.jpg",
    ]

    for fn in filenames:
        with open(os.path.join(temp_dir, fn), "w") as f:
            f.write("test content")

    yield temp_dir

    shutil.rmtree(temp_dir)


@pytest.fixture(scope="module")
def expected_satellites():
    return {
        "L5": {"20210101_L5_site1_ms.tif"},
        "L7": {"20210101_L7_site1_ms.tif"},
        "L8": {"20210101_L8_site1_ms.tif"},
        "L9": {"20210101_L9_site1_ms.tif"},
        "S2": {"20210101_S2_site1_ms.tif"},
    }


# Test data
@pytest.fixture(scope="module")
def sample_metadata():
    metadata = {
        "L5": {
            "filenames": ["20210101_L5_site1_ms.tif", "20220101_L5_site1_ms.tif"],
            "acc_georef": [9.185, 10.185],
            "epsg": [32618, 32618],
            "dates": [
                "datetime.datetime(2020, 1, 5, 15, 33, 53, tzinfo=<UTC>)",
                "datetime.datetime(2020, 1, 21, 15, 33, 50, tzinfo=<UTC>)",
            ],
        },
        "L7": {
            "filenames": ["20210101_L7_site1_ms.tif", "20220101_L7_site1_ms.tif"],
            "acc_georef": [7.441, 5.693],
            "epsg": [32618, 32618],
            "dates": [
                "datetime.datetime(2020, 1, 5, 15, 33, 53, tzinfo=<UTC>)",
                "datetime.datetime(2020, 2, 22, 15, 33, 41, tzinfo=<UTC>)",
            ],
        },
        "sitename": "site1",
    }
    return metadata


@pytest.fixture(scope="module")
def expected_filtered_metadata():
    metadata = {
        "L5": {
            "filenames": ["20210101_L5_site1_ms.tif"],
            "acc_georef": [9.185],
            "epsg": [32618],
            "dates": ["datetime.datetime(2020, 1, 5, 15, 33, 53, tzinfo=<UTC>)"],
        },
        "L7": {
            "filenames": ["20210101_L7_site1_ms.tif"],
            "acc_georef": [7.441],
            "epsg": [32618],
            "dates": [
                "datetime.datetime(2020, 1, 5, 15, 33, 53, tzinfo=<UTC>)",
            ],
        },
        "sitename": "site1",
    }
    return metadata


@pytest.fixture(scope="module")
def expected_empty_filtered_metadata():
    metadata = {
        "L5": {
            "filenames": [],
            "acc_georef": [],
            "epsg": [],
            "dates": [],
        },
        "L7": {
            "filenames": [],
            "acc_georef": [],
            "epsg": [],
            "dates": [],
        },
        "sitename": "site1",
    }
    return metadata


@pytest.mark.parametrize("sample_ROI_directory", ["site1"], indirect=True)
def test_filter_metadata(
    sample_ROI_directory, sample_metadata, expected_filtered_metadata
):
    result = common.filter_metadata(sample_metadata, "site1", sample_ROI_directory)

    assert (
        result == expected_filtered_metadata
    ), "The output filtered metadata is not as expected."


@pytest.mark.parametrize("empty_ROI_directory", ["site1"], indirect=True)
def test_empty_roi_directory_filter_metadata(
    empty_ROI_directory,
    sample_metadata,
    expected_empty_filtered_metadata,
    sample_directory,
):
    # if the RGB directory but exists and is empty then no filenames in result should be empty lists
    result = common.filter_metadata(sample_metadata, "site1", empty_ROI_directory)
    assert (
        result == expected_empty_filtered_metadata
    ), "The output filtered metadata is not as expected."

    # should raise an error if the RGB directory within the ROI directory does not exist
    with pytest.raises(Exception):
        common.filter_metadata(sample_metadata, "site1", sample_directory)


# Test cases
def test_get_filtered_files_dict(sample_directory, expected_satellites):
    result = common.get_filtered_files_dict(sample_directory, "jpg", "site1")

    assert (
        result == expected_satellites
    ), "The output file name dictionary is not as expected."


# Test for preprocessing function
def test_preprocess_geodataframe():
    # Test when data is empty
    empty_data = gpd.GeoDataFrame()
    result = common.preprocess_geodataframe(empty_data)
    assert result.empty, "Result should be empty when data is empty."

    # Test when data contains "ID" column
    data_with_ID = gpd.GeoDataFrame({"ID": [1, 2, 3], "geometry": [None, None, None]})
    result = common.preprocess_geodataframe(data_with_ID)
    assert "id" in result.columns, "Column 'ID' should be renamed to 'id'."
    assert "ID" not in result.columns, "Column 'ID' should not exist in the result."

    # Test when data does not contain "id" column
    data_without_id = gpd.GeoDataFrame(
        {"foo": [1, 2, 3], "geometry": [None, None, None]}
    )
    result = common.preprocess_geodataframe(data_without_id, create_ids=True)
    assert "id" in result.columns, "Column 'id' should be added when it does not exist."

    # Test when data does not contain "id" column
    data_without_id = gpd.GeoDataFrame(
        {"foo": [1, 2, 3], "geometry": [None, None, None]}
    )
    result = common.preprocess_geodataframe(data_without_id, create_ids=False)
    assert (
        "id" not in result.columns
    ), "Column 'id' should be added when it does not exist."

    # Test when columns_to_keep is specified
    data_with_extra_columns = gpd.GeoDataFrame(
        {"id": [1, 2, 3], "foo": ["a", "b", "c"], "geometry": [None, None, None]}
    )
    columns_to_keep = ["id", "geometry"]
    result = common.preprocess_geodataframe(data_with_extra_columns, columns_to_keep)
    assert set(result.columns) == set(
        columns_to_keep
    ), "Only specified columns should be kept."


# @todo add a test like this back when you are ready
# def test_duplicate_ids_exception():
#     data_with_duplicate_ids = gpd.GeoDataFrame({'id': ['abc', 'abc', 'def'], 'geometry': [None, None, None]})
#     with pytest.raises(exceptions.Duplicate_ID_Exception):
#         common.preprocess_geodataframe(data_with_duplicate_ids, create_ids=False)


# Test for remove_z_coordinates function
def test_remove_z_coordinates():
    # Test when geodf is empty
    empty_data = gpd.GeoDataFrame()
    result = common.remove_z_coordinates(empty_data)
    assert result.empty, "Result should be empty when geodf is empty."

    # Test when geodf contains geometries with z coordinates
    data_with_z = gpd.GeoDataFrame({"geometry": [Point(1, 2, 3), Point(4, 5, 6)]})
    result = common.remove_z_coordinates(data_with_z)
    assert not any(
        geom.has_z for geom in result.geometry
    ), "All z coordinates should be removed."

    # Test when geodf contains MultiLineStrings
    data_with_multilinestrings = gpd.GeoDataFrame(
        {
            "geometry": [MultiLineString([((1, 2), (3, 4)), ((5, 6), (7, 8))])],
        }
    )
    result = common.remove_z_coordinates(data_with_multilinestrings)
    assert all(
        isinstance(geom, LineString) for geom in result.geometry
    ), "All MultiLineStrings should be exploded into LineStrings."


def test_get_transect_points_dict(valid_transects_gdf):
    """Tests get_transect_points_dict to see if it returns a valid dictionary when given
    transects geodataframe and an id
    Args:
        valid_transects_gdf (geodataframe): transects geodataframe with ids:[17,30,35]
    """
    roi_id = "17"
    transects_dict = common.get_transect_points_dict(valid_transects_gdf)
    # simulate how roi transect ids would be created
    transect_ids = valid_transects_gdf["id"].to_list()
    roi_transect_ids = [tid for tid in transect_ids]

    assert isinstance(transects_dict, dict)
    assert set(transects_dict.keys()).issubset(set(roi_transect_ids))
    assert isinstance(transects_dict[roi_transect_ids[0]], np.ndarray)


def test_do_rois_filepaths_exist(tmp_path):
    # should return false when a filepath exist
    good_filepath = tmp_path
    roi_settings = {"1": {"filepath": str(good_filepath)}}
    return_value = common.do_rois_filepaths_exist(
        roi_settings, list(roi_settings.keys())
    )
    assert return_value == True
    # should return false when all filepaths exist
    good_filepath = tmp_path
    roi_settings = {
        "1": {"filepath": str(good_filepath)},
        "2": {"filepath": str(good_filepath)},
    }
    return_value = common.do_rois_filepaths_exist(
        roi_settings, list(roi_settings.keys())
    )
    assert return_value == True
    # should return false when a filepath doesn't exist
    bad_filepath = tmp_path / "fake"
    roi_settings = {"1": {"filepath": str(bad_filepath)}}
    return_value = common.do_rois_filepaths_exist(
        roi_settings, list(roi_settings.keys())
    )
    assert return_value == False
    # should return false when one filepath exist and one filepath doesn't exist
    roi_settings = {
        "1": {"filepath": str(good_filepath)},
        "2": {"filepath": str(bad_filepath)},
    }
    return_value = common.do_rois_filepaths_exist(
        roi_settings, list(roi_settings.keys())
    )
    assert return_value == False


def test_were_rois_downloaded_empty_roi_settings():
    actual_value = common.were_rois_downloaded(None, None)
    assert actual_value == False
    actual_value = common.were_rois_downloaded({}, None)
    assert actual_value == False


def test_do_rois_have_sitenames(valid_roi_settings, roi_settings_empty_sitenames):
    """Test if do_rois_have_sitenames returns true when
    each roi's 'sitename' != "" and false when each roi's 'sitename' == ""

    Args:
        valid_roi_settings (dict): roi_settings with ids["2","3","5"] and valid sitenames
        roi_settings_empty_sitenames(dict): roi_settings with ids["2"] and sitenames = ""
    """
    # ids of rois in valid_roi_settings
    roi_ids = ["2", "3", "5"]
    # when sitenames are not empty strings should return true
    actual_value = common.do_rois_have_sitenames(valid_roi_settings, roi_ids)
    assert actual_value == True

    roi_ids = ["2"]
    # when sitenames are not empty strings should return False
    actual_value = common.do_rois_have_sitenames(roi_settings_empty_sitenames, roi_ids)
    assert actual_value == False


def test_were_rois_downloaded(valid_roi_settings, roi_settings_empty_sitenames):
    """Test if were_rois_downloaded returns true when sitenames != "" and false sitenames == ""

    Args:
        valid_roi_settings (dict): roi_settings with ids["2","3","5"] and valid sitenames
        roi_settings_empty_sitenames(dict): roi_settings with ids["2"] and sitenames = ""
    """
    # ids of rois in valid_roi_settings
    roi_ids = ["2", "3", "5"]
    # when sitenames are not empty strings should return true
    actual_value = common.were_rois_downloaded(valid_roi_settings, roi_ids)
    assert actual_value == True

    roi_ids = ["2"]
    # when sitenames are not empty strings should return False
    actual_value = common.were_rois_downloaded(roi_settings_empty_sitenames, roi_ids)
    assert actual_value == False


def test_create_roi_settings(valid_selected_ROI_layer_data, valid_settings):
    """test if valid roi_settings dictionary is created when provided settings and a single selected ROI

    Args:
        valid_selected_ROI_layer_data (dict): geojson for ROI with id ="0"
        valid_settings (dict):
            download settings for ROI
            "sat_list": ["L5", "L7", "L8"],
            "landsat_collection": "C01",
            "dates": ["2018-12-01", "2019-03-01"],
    """
    filepath = r"C\:Sharon"
    date_str = datetime.date(2019, 4, 13).strftime("%m-%d-%y__%I_%M_%S")
    actual_roi_settings = common.create_roi_settings(
        valid_settings, valid_selected_ROI_layer_data, filepath, date_str
    )
    expected_roi_id = valid_selected_ROI_layer_data["features"][0]["properties"]["id"]
    assert isinstance(actual_roi_settings, dict)
    assert expected_roi_id in actual_roi_settings
    assert set(actual_roi_settings[expected_roi_id]["dates"]) == set(
        valid_settings["dates"]
    )
    assert set(actual_roi_settings[expected_roi_id]["sat_list"]) == set(
        valid_settings["sat_list"]
    )
    assert (
        actual_roi_settings[expected_roi_id]["landsat_collection"]
        == valid_settings["landsat_collection"]
    )
    assert actual_roi_settings[expected_roi_id]["roi_id"] == expected_roi_id
    assert actual_roi_settings[expected_roi_id]["filepath"] == filepath


def test_config_dict_to_file(tmp_path):
    # test if file named config.json is saved when dictionary is passed
    config = {
        "0": {
            "dates": ["2018-12-01", "2019-03-01"],
            "sat_list": ["L8"],
            "roi_id": "0",
            "polygon": [
                [
                    [-124.19437983778509, 40.82355301978889],
                    [-124.19502680580241, 40.859579119105774],
                    [-124.14757559660633, 40.86006100475558],
                    [-124.14695430388457, 40.82403429740862],
                    [-124.19437983778509, 40.82355301978889],
                ]
            ],
            "landsat_collection": "C01",
            "sitename": "ID_0_datetime11-01-22__03_54_47",
            "filepath": "C:\\1_USGS\\CoastSeg\\repos\\2_CoastSeg\\CoastSeg_fork\\Seg2Map\\data",
        }
    }
    filepath = tmp_path
    file_utilities.config_to_file(config, filepath)
    assert tmp_path.exists()
    expected_path = tmp_path / "config.json"
    assert expected_path.exists()
    with open(expected_path, "r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    # test if roi id was saved as key and key fields exist
    assert "0" in data
    assert "dates" in data["0"]
    assert "sat_list" in data["0"]
    assert "roi_id" in data["0"]
    assert "polygon" in data["0"]
    assert "landsat_collection" in data["0"]
    assert "sitename" in data["0"]
    assert "filepath" in data["0"]


def test_config_geodataframe_to_file(tmp_path):
    # test if file named config_gdf.geojson is saved when geodataframe is passed
    d = {"col1": ["name1"], "geometry": [geometry.Point(1, 2)]}
    config = gpd.GeoDataFrame(d, crs="EPSG:4326")
    filepath = tmp_path
    file_utilities.config_to_file(config, filepath)
    assert tmp_path.exists()
    expected_path = tmp_path / "config_gdf.geojson"
    assert expected_path.exists()


def test_create_config_gdf(valid_rois_gdf, valid_shoreline_gdf, valid_transects_gdf):
    # test if a gdf is created with all the rois, shorelines and transects
    epsg_code = "epsg:4326"
    if valid_rois_gdf is not None:
        if not valid_rois_gdf.empty:
            epsg_code = valid_rois_gdf.crs
    actual_gdf = common.create_config_gdf(
        valid_rois_gdf, valid_shoreline_gdf, valid_transects_gdf, epsg_code=epsg_code
    )
    assert "type" in actual_gdf.columns
    assert actual_gdf[actual_gdf["type"] == "transect"].empty == False
    assert actual_gdf[actual_gdf["type"] == "shoreline"].empty == False
    assert actual_gdf[actual_gdf["type"] == "roi"].empty == False

    # test if a gdf is created with all the rois, transects if shorelines is None
    shorelines_gdf = None
    epsg_code = "epsg:4326"
    if valid_rois_gdf is not None:
        if not valid_rois_gdf.empty:
            epsg_code = valid_rois_gdf.crs
    actual_gdf = common.create_config_gdf(
        valid_rois_gdf, shorelines_gdf, valid_transects_gdf, epsg_code=epsg_code
    )
    assert "type" in actual_gdf.columns
    assert actual_gdf[actual_gdf["type"] == "transect"].empty == False
    assert actual_gdf[actual_gdf["type"] == "shoreline"].empty == True
    assert actual_gdf[actual_gdf["type"] == "roi"].empty == False
    # test if a gdf is created with all the rois if  transects and shorelines is None
    transects_gdf = None
    epsg_code = "epsg:4326"
    if valid_rois_gdf is not None:
        if not valid_rois_gdf.empty:
            epsg_code = valid_rois_gdf.crs
    actual_gdf = common.create_config_gdf(
        valid_rois_gdf, shorelines_gdf, transects_gdf, epsg_code=epsg_code
    )
    assert "type" in actual_gdf.columns
    assert actual_gdf[actual_gdf["type"] == "transect"].empty == True
    assert actual_gdf[actual_gdf["type"] == "shoreline"].empty == True
    assert actual_gdf[actual_gdf["type"] == "roi"].empty == False


def test_get_epsg_from_geometry_northern_hemisphere():
    # Rectangle around New York City (North)
    coords = [
        (-74.2591, 40.4774),
        (-74.2591, 40.9176),
        (-73.7004, 40.9176),
        (-73.7004, 40.4774),
    ]
    polygon = Polygon(coords)
    assert common.get_epsg_from_geometry(polygon) == 32618


def test_get_epsg_from_geometry_southern_hemisphere():
    # Rectangle around Buenos Aires (South)
    coords = [
        (-58.5034, -34.7054),
        (-58.5034, -34.5265),
        (-58.3399, -34.5265),
        (-58.3399, -34.7054),
    ]
    polygon = Polygon(coords)
    assert common.get_epsg_from_geometry(polygon) == 32721


def test_convert_wgs_to_utm():
    # tests if valid espg code is returned by lon lat coordinates
    lon = 150
    lat = 100
    actual_espg = common.convert_wgs_to_utm(lon, lat)
    assert isinstance(actual_espg, str)
    assert actual_espg.startswith("326")
    lat = -20
    actual_espg = common.convert_wgs_to_utm(lon, lat)
    assert isinstance(actual_espg, str)
    assert actual_espg.startswith("327")


def test_convert_wgs_to_utm_northern_hemisphere():
    # New York City (North)
    lon, lat = -74.006, 40.7128
    assert common.convert_wgs_to_utm(lon, lat) == "32618"


def test_convert_wgs_to_utm_southern_hemisphere():
    # Buenos Aires (South)
    lon, lat = -58.3816, -34.6037
    assert common.convert_wgs_to_utm(lon, lat) == "32721"


def test_convert_wgs_to_utm_boundary():
    # On Prime Meridian, in Ghana (North)
    lon, lat = 0, 7.9465
    assert common.convert_wgs_to_utm(lon, lat) == "32631"


def test_get_most_accurate_epsg_from_int():
    # Mock bbox around New York City (North, EPSG:4326 -> EPSG:32618)
    coords = [
        (-74.2591, 40.4774),
        (-74.2591, 40.9176),
        (-73.7004, 40.9176),
        (-73.7004, 40.4774),
    ]
    polygon = Polygon(coords)
    bbox = gpd.GeoDataFrame({"geometry": [polygon]})
    assert common.get_most_accurate_epsg(4326, bbox) == 32618


def test_get_most_accurate_epsg_from_str():
    # Mock bbox around Buenos Aires (South, EPSG:4326 -> EPSG:32721)
    coords = [
        (-58.5034, -34.7054),
        (-58.5034, -34.5265),
        (-58.3399, -34.5265),
        (-58.3399, -34.7054),
    ]
    polygon = Polygon(coords)
    bbox = gpd.GeoDataFrame({"geometry": [polygon]})
    assert common.get_most_accurate_epsg("epsg:4326", bbox) == 32721


def test_get_most_accurate_epsg_unchanged():
    # Mock bbox around New York City
    coords = [
        (-74.2591, 40.4774),
        (-74.2591, 40.9176),
        (-73.7004, 40.9176),
        (-73.7004, 40.4774),
    ]
    polygon = Polygon(coords)
    bbox = gpd.GeoDataFrame({"geometry": [polygon]})
    assert common.get_most_accurate_epsg(3857, bbox) == 3857


def test_get_center_point():
    """test correct center of rectangle is returned"""
    expected_coords = [(4.0, 5.0), (4.0, 6.0), (8.0, 6.0), (8.0, 5.0), (4.0, 5.0)]
    center_x, center_y = common.get_center_point(expected_coords)
    assert center_x == 6
    assert center_y == 5.5


def test_create_json_config(
    valid_settings,
    valid_roi_settings,
):
    # test if valid json style config is created when inputs dictionary contains multiple entries
    actual_config = common.create_json_config(valid_roi_settings, valid_settings)
    expected_roi_ids = list(valid_roi_settings.keys())

    assert isinstance(actual_config, dict)
    assert "settings" in actual_config.keys()
    assert "roi_ids" in actual_config.keys()
    assert isinstance(actual_config["roi_ids"], list)
    assert isinstance(actual_config["settings"], dict)
    assert actual_config["roi_ids"] == expected_roi_ids
    for key in expected_roi_ids:
        assert isinstance(actual_config[str(key)], dict)


def test_create_json_config_single_input(valid_settings, valid_single_roi_settings):
    # test if valid json style config is created when inputs dictionary contains only one entry
    actual_config = common.create_json_config(valid_single_roi_settings, valid_settings)
    expected_roi_ids = list(valid_single_roi_settings.keys())

    assert isinstance(actual_config, dict)
    assert "settings" in actual_config.keys()
    assert "roi_ids" in actual_config.keys()
    assert isinstance(actual_config["roi_ids"], list)
    assert isinstance(actual_config["settings"], dict)
    assert actual_config["roi_ids"] == expected_roi_ids
    for key in expected_roi_ids:
        assert isinstance(actual_config[str(key)], dict)


def test_extract_roi_by_id(valid_rois_gdf):
    # test if valid gdf is returned when id within gdf is given
    roi_id = 17
    actual_roi = common.extract_roi_by_id(valid_rois_gdf, roi_id)
    assert isinstance(actual_roi, gpd.GeoDataFrame)
    assert actual_roi[actual_roi["id"].astype(int) == roi_id].empty == False
    expected_roi = valid_rois_gdf[valid_rois_gdf["id"].astype(int) == roi_id]
    assert actual_roi["geometry"][0] == expected_roi["geometry"][0]
    assert actual_roi["id"][0] == expected_roi["id"][0]
