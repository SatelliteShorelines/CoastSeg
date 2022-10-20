# This file is meant to hold fixtures that can be used for testing
# These fixtures set up data that can be used as inputs for the tests, so that no code is repeated
import os
import json
import pytest
import geopandas as gpd
from shapely.geometry import shape
from src.coastseg import roi

script_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def config_dict() -> dict:
    """returns a valid dictionary of settings"""
    settings = {16: {}}
    return settings


@pytest.fixture
def valid_settings() -> dict:
    """returns a valid dictionary of settings"""
    settings = {
        "sat_list": ["L5", "L7", "L8"],
        "collection": "C01",
        "dates": ["2018-12-01", "2019-03-01"],
        "cloud_thresh": 0.5,
        "dist_clouds": 300,
        "output_epsg": 3857,
        "check_detection": False,
        "adjust_detection": False,
        "save_figure": True,
        "min_beach_area": 4500,
        "buffer_size": 150,
        "min_length_sl": 200,
        "cloud_mask_issue": False,
        "sand_color": "default",
        "pan_off": "False",
        "create_plot": False,
        "max_dist_ref": 25,
        "along_dist": 25,
    }
    return settings


@pytest.fixture
def valid_inputs() -> dict:
    """returns a valid dictionary of inputs"""
    inputs = {
        "dates": ["2018-12-01", "2019-03-01"],
        "sat_list": ["S2"],
        "sitename": "ID02022-10-07__15_hr_42_min59sec",
        "filepath": "C:\\1_USGS\\CoastSeg\\repos\\2_CoastSeg\\CoastSeg_fork\\Seg2Map\\data",
        "roi_id": 16,
        "polygon": [
            [
                [-124.19135332563553, 40.866019455883986],
                [-124.19199962055943, 40.902045328874756],
                [-124.14451802259211, 40.90252637117114],
                [-124.14389745792991, 40.866499891350706],
                [-124.19135332563553, 40.866019455883986],
            ]
        ],
        "landsat_collection": "C01",
    }
    return inputs


@pytest.fixture
def valid_bbox_gdf() -> gpd.GeoDataFrame:
    """returns the contents of valid_bbox.geojson as a gpd.GeoDataFrame"""
    file_path = os.path.abspath(
        os.path.join(script_dir, "test_data", "valid_bbox.geojson")
    )
    with open(file_path, "r") as f:
        gpd_data = gpd.read_file(f)
    return gpd_data


@pytest.fixture
def valid_bbox_geojson() -> dict:
    """returns the contents of valid_bbox.geojson as a geojson dictionary
    ROIs with ids:[17,30,35]"""
    file_path = os.path.abspath(
        os.path.join(script_dir, "test_data", "valid_bbox.geojson")
    )
    with open(file_path, "r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    return data


@pytest.fixture
def valid_rois_gdf() -> gpd.GeoDataFrame:
    """returns the contents of valid_rois.geojson as a gpd.GeoDataFrame
    ROIs with ids:[17,30,35]"""
    file_path = os.path.abspath(
        os.path.join(script_dir, "test_data", "valid_rois.geojson")
    )
    with open(file_path, "r") as f:
        gpd_data = gpd.read_file(f)
    return gpd_data


@pytest.fixture
def valid_rois_geojson() -> dict:
    """returns the contents of valid_rois.geojson as a geojson dictionary
    ROIs with ids:[17,30,35]"""
    file_path = os.path.abspath(
        os.path.join(script_dir, "test_data", "valid_rois.geojson")
    )
    with open(file_path, "r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    return data


@pytest.fixture
def transect_compatible_roi() -> gpd.GeoDataFrame:
    """returns the contents of valid_rois.geojson as a gpd.GeoDataFrame
    ROI ids:[23,29]"""
    file_path = os.path.abspath(
        os.path.join(script_dir, "test_data", "transect_compatible_rois.geojson")
    )
    with open(file_path, "r") as f:
        gpd_data = gpd.read_file(f)
    return gpd_data


@pytest.fixture
def valid_shoreline_gdf() -> gpd.GeoDataFrame:
    """returns the contents of valid_rois.geojson as a gpd.GeoDataFrame
    corresponds to ROI ids:[23,29]"""
    file_path = os.path.abspath(
        os.path.join(script_dir, "test_data", "transect_compatible_shoreline.geojson")
    )
    with open(file_path, "r") as f:
        gpd_data = gpd.read_file(f)
    return gpd_data


@pytest.fixture
def valid_transects_gdf() -> gpd.GeoDataFrame:
    """returns the contents of transects.geojson as a gpd.GeoDataFrame
    These transects are compatible with bbox, shorelines and transects
    ROIs with ids:[17,30,35]"""
    file_path = os.path.abspath(
        os.path.join(script_dir, "test_data", "transect_compatible_transects.geojson")
    )
    with open(file_path, "r") as f:
        gpd_data = gpd.read_file(f)
    return gpd_data


@pytest.fixture
def valid_bbox_gdf() -> gpd.GeoDataFrame:
    """returns the contents of bbox.geojson as a gpd.GeoDataFrame
    current espg code : 4326
    most accurate espg code: 32610
    ROIs with ids:[17,30,35]"""
    file_path = os.path.abspath(
        os.path.join(script_dir, "test_data", "transect_compatible_bbox.geojson")
    )
    with open(file_path, "r") as f:
        gpd_data = gpd.read_file(f)
    return gpd_data


@pytest.fixture
def valid_ROI(transect_compatible_roi) -> gpd.GeoDataFrame:
    """returns a valid instance of ROI current espg code : 4326 ROIs with ids:[17,30,35]"""
    return roi.ROI(rois_gdf=transect_compatible_roi)


@pytest.fixture
def valid_settings() -> dict:
    return {
        "sat_list": ["L8"],
        "collection": "C01",
        "dates": ["2018-12-01", "2019-03-01"],
        "cloud_thresh": 0.5,
        "dist_clouds": 300,
        "output_epsg": 3857,
        "check_detection": False,
        "adjust_detection": False,
        "save_figure": True,
        "min_beach_area": 4500,
        "buffer_size": 550,
        "min_length_sl": 100,
        "cloud_mask_issue": False,
        "sand_color": "default",
        "pan_off": "False",
        "create_plot": False,
        "max_dist_ref": 25,
        "along_dist": 25,
    }


@pytest.fixture
def valid_single_inputs_dict() -> dict:
    """Returns valid inputs dict with two roi id '2' and '5'

    Returns:
        dict: valid inputs dict with two roi id '2' and '5'
    """
    return {
        "2": {
            "dates": ["2018-12-01", "2019-03-01"],
            "sat_list": ["L8"],
            "sitename": "ID_2_datetime10-19-22__04_00_34",
            "filepath": "C:\\1_USGS\\CoastSeg\\repos\\2_CoastSeg\\CoastSeg_fork\\Seg2Map\\data",
            "roi_id": "2",
            "polygon": [
                [
                    [-124.16930255115336, 40.8665390046026],
                    [-124.16950858759564, 40.878247531017706],
                    [-124.15408259844114, 40.878402930533994],
                    [-124.1538792781699, 40.8666943403763],
                    [-124.16930255115336, 40.8665390046026],
                ]
            ],
            "landsat_collection": "C01",
        },
    }


@pytest.fixture
def valid_inputs_dict() -> dict:
    """Returns valid inputs dict with two roi id '2' and '5'

    Returns:
        dict: valid inputs dict with two roi id '2' and '5'
    """
    return {
        "2": {
            "dates": ["2018-12-01", "2019-03-01"],
            "sat_list": ["L8"],
            "sitename": "ID_2_datetime10-19-22__04_00_34",
            "filepath": "C:\\1_USGS\\CoastSeg\\repos\\2_CoastSeg\\CoastSeg_fork\\Seg2Map\\data",
            "roi_id": "2",
            "polygon": [
                [
                    [-124.16930255115336, 40.8665390046026],
                    [-124.16950858759564, 40.878247531017706],
                    [-124.15408259844114, 40.878402930533994],
                    [-124.1538792781699, 40.8666943403763],
                    [-124.16930255115336, 40.8665390046026],
                ]
            ],
            "landsat_collection": "C01",
        },
        "3": {
            "dates": ["2018-12-01", "2019-03-01"],
            "sat_list": ["L8"],
            "sitename": "ID_3_datetime10-19-22__04_00_34",
            "filepath": "C:\\1_USGS\\CoastSeg\\repos\\2_CoastSeg\\CoastSeg_fork\\Seg2Map\\data",
            "roi_id": "3",
            "polygon": [
                [
                    [-124.16950858759564, 40.878247531017706],
                    [-124.16971474532464, 40.88995603272874],
                    [-124.15428603840094, 40.890111496009816],
                    [-124.15408259844114, 40.878402930533994],
                    [-124.16950858759564, 40.878247531017706],
                ]
            ],
            "landsat_collection": "C01",
        },
        "5": {
            "dates": ["2018-12-01", "2019-03-01"],
            "sat_list": ["L8"],
            "sitename": "ID_5_datetime10-19-22__04_00_34",
            "filepath": "C:\\1_USGS\\CoastSeg\\repos\\2_CoastSeg\\CoastSeg_fork\\Seg2Map\\data",
            "roi_id": "5",
            "polygon": [
                [
                    [-124.15428603840094, 40.890111496009816],
                    [-124.15448959812942, 40.90182003680178],
                    [-124.13905805198854, 40.901973499567674],
                    [-124.13885721170861, 40.89026489583505],
                    [-124.15428603840094, 40.890111496009816],
                ]
            ],
            "landsat_collection": "C01",
        },
    }


@pytest.fixture
def valid_master_config() -> dict:
    """Returns a complete master config with roi_ids=['2', '3', '5'], settings, and
    a key for each roi and their associated inputs
    Returns:
        dict: master_config with roi_ids=['2', '3', '5']
    """
    return {
        "roi_ids": ["2", "3", "5"],
        "settings": {
            "sat_list": ["L8"],
            "collection": "C01",
            "dates": ["2018-12-01", "2019-03-01"],
            "cloud_thresh": 0.5,
            "dist_clouds": 300,
            "output_epsg": 3857,
            "check_detection": False,
            "adjust_detection": False,
            "save_figure": True,
            "min_beach_area": 4500,
            "buffer_size": 550,
            "min_length_sl": 100,
            "cloud_mask_issue": False,
            "sand_color": "default",
            "pan_off": "False",
            "create_plot": False,
            "max_dist_ref": 25,
            "along_dist": 25,
        },
        "2": {
            "dates": ["2018-12-01", "2019-03-01"],
            "sat_list": ["L8"],
            "sitename": "ID_2_datetime10-19-22__04_00_34",
            "filepath": "C:\\1_USGS\\CoastSeg\\repos\\2_CoastSeg\\CoastSeg_fork\\Seg2Map\\data",
            "roi_id": "2",
            "polygon": [
                [
                    [-124.16930255115336, 40.8665390046026],
                    [-124.16950858759564, 40.878247531017706],
                    [-124.15408259844114, 40.878402930533994],
                    [-124.1538792781699, 40.8666943403763],
                    [-124.16930255115336, 40.8665390046026],
                ]
            ],
            "landsat_collection": "C01",
        },
        "3": {
            "dates": ["2018-12-01", "2019-03-01"],
            "sat_list": ["L8"],
            "sitename": "ID_3_datetime10-19-22__04_00_34",
            "filepath": "C:\\1_USGS\\CoastSeg\\repos\\2_CoastSeg\\CoastSeg_fork\\Seg2Map\\data",
            "roi_id": "3",
            "polygon": [
                [
                    [-124.16950858759564, 40.878247531017706],
                    [-124.16971474532464, 40.88995603272874],
                    [-124.15428603840094, 40.890111496009816],
                    [-124.15408259844114, 40.878402930533994],
                    [-124.16950858759564, 40.878247531017706],
                ]
            ],
            "landsat_collection": "C01",
        },
        "5": {
            "dates": ["2018-12-01", "2019-03-01"],
            "sat_list": ["L8"],
            "sitename": "ID_5_datetime10-19-22__04_00_34",
            "filepath": "C:\\1_USGS\\CoastSeg\\repos\\2_CoastSeg\\CoastSeg_fork\\Seg2Map\\data",
            "roi_id": "5",
            "polygon": [
                [
                    [-124.15428603840094, 40.890111496009816],
                    [-124.15448959812942, 40.90182003680178],
                    [-124.13905805198854, 40.901973499567674],
                    [-124.13885721170861, 40.89026489583505],
                    [-124.15428603840094, 40.890111496009816],
                ]
            ],
            "landsat_collection": "C01",
        },
    }
