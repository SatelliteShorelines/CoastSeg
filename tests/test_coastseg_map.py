import json
import pytest
from coastseg import shoreline
from coastseg import transects
from coastseg import roi
from coastseg import exceptions
from coastseg import coastseg_map
from coastseg import common
from leafmap import Map
import geopandas as gpd
from ipyleaflet import GeoJSON

def test_load_json_config_without_rois(valid_coastseg_map_with_settings):
    # test if exception is raised when coastseg_map has no ROIs
    actual_coastsegmap=valid_coastseg_map_with_settings
    with pytest.raises(Exception):
        actual_coastsegmap.load_json_config("")

def test_load_json_config_downloaded(valid_coastseg_map_with_settings,
                          valid_rois_filepath,
                          downloaded_config_json_filepath):
    # create instance of Coastseg_Map with settings and ROIs initially loaded
    actual_coastsegmap=valid_coastseg_map_with_settings
    actual_coastsegmap.load_rois_on_map(file=valid_rois_filepath)
    # test if settings are correctly loaded when valid json config loaded with 'filepath' & 'sitename' keys is loaded
    actual_coastsegmap.load_json_config(downloaded_config_json_filepath)
    assert isinstance(actual_coastsegmap.rois.roi_settings,dict)
    actual_config=common.read_json_file(downloaded_config_json_filepath)
    for key in actual_config["roi_ids"]:
        assert key in actual_coastsegmap.rois.roi_settings
  
def test_load_json_config(valid_coastseg_map_with_settings,
                          valid_rois_filepath,
                          config_json_filepath):
    # create instance of Coastseg_Map with settings and ROIs initially loaded
    actual_coastsegmap=valid_coastseg_map_with_settings
    actual_coastsegmap.load_rois_on_map(file=valid_rois_filepath)
    # test if settings are correctly loaded when valid json config without 'filepath' & 'sitename' keys is loaded
    actual_coastsegmap.load_json_config(config_json_filepath)
    assert isinstance(actual_coastsegmap.rois.roi_settings,dict)
    actual_config=common.read_json_file(config_json_filepath)
    for key in actual_config["roi_ids"]:
        assert key in actual_coastsegmap.rois.roi_settings
    
        
    

def test_valid_shoreline_gdf(valid_shoreline_gdf: gpd.GeoDataFrame):
    """tests if a Shoreline will be created from a valid shoreline thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): a valid shoreline as a gpd.GeoDataFrame
    """
    expected_shoreline = shoreline.Shoreline(shoreline=valid_shoreline_gdf)
    assert isinstance(expected_shoreline, shoreline.Shoreline)
    assert expected_shoreline.gdf is not None
    assert expected_shoreline.filename == "shoreline.geojson"


def test_valid_transects_gdf(valid_transects_gdf: gpd.GeoDataFrame):
    """tests if a Transects will be created from a valid transects thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): valid transects as a gpd.GeoDataFrame
    """
    expected_transects = transects.Transects(transects=valid_transects_gdf)
    assert isinstance(expected_transects, transects.Transects)
    assert expected_transects.gdf is not None
    assert expected_transects.filename == "transects.geojson"


def test_transect_compatible_roi(transect_compatible_roi: gpd.GeoDataFrame):
    """tests if a ROI will be created from valid rois thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): alid rois as a gpd.GeoDataFrame
    """
    expected_roi = roi.ROI(rois_gdf=transect_compatible_roi)
    assert isinstance(expected_roi, roi.ROI)
    assert expected_roi.gdf is not None
    assert expected_roi.filename == "rois.geojson"


def test_transect_compatible_roi(transect_compatible_roi: gpd.GeoDataFrame):
    """tests if a ROI will be created from valid rois thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): alid rois as a gpd.GeoDataFrame
    """
    expected_roi = roi.ROI(rois_gdf=transect_compatible_roi)
    assert isinstance(expected_roi, roi.ROI)
    assert expected_roi.gdf is not None
    assert expected_roi.filename == "rois.geojson"


def test_coastseg_map():
    """tests a CoastSeg_Map object is created"""
    coastsegmap = coastseg_map.CoastSeg_Map()
    assert isinstance(coastsegmap, coastseg_map.CoastSeg_Map)
    assert isinstance(coastsegmap.map, Map)
    assert hasattr(coastsegmap, "draw_control")


def test_coastseg_map_settings():
    """tests if a ROI will be created from valid rois thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): alid rois as a gpd.GeoDataFrame
    """
    coastsegmap = coastseg_map.CoastSeg_Map()
    dates = ["2018-12-01", "2019-03-01"]
    collection = "C01"
    sat_list = ["L8"]
    pre_process_settings = {
        # general parameters:
        "cloud_thresh": 0.5,  # threshold on maximum cloud cover
        "dist_clouds": 300,  # ditance around clouds where shoreline can't be mapped
        "output_epsg": 3857,  # epsg code of spatial reference system desired for the output
        # quality control:
        "check_detection": True,  # if True, shows each shoreline detection to the user for validation
        "adjust_detection": False,  # if True, allows user to adjust the postion of each shoreline by changing the threhold
        "save_figure": True,  # if True, saves a figure showing the mapped shoreline for each image
        # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
        "min_beach_area": 4500,  # minimum area (in metres^2) for an object to be labelled as a beach
        "buffer_size": 150,  # radius (in metres) of the buffer around sandy pixels considered in the shoreline detection
        "min_length_sl": 200,  # minimum length (in metres) of shoreline perimeter to be valid
        "cloud_mask_issue": False,  # switch this parameter to True if sand pixels are masked (in black) on many images
        "sand_color": "default",  # 'default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
        "pan_off": "False",  # if True, no pan-sharpening is performed on Landsat 7,8 and 9 imagery
        "create_plot": False,  # True create a matplotlib plot of the image with the datetime as the title
        "max_dist_ref": 25,
    }
    coastsegmap.save_settings(
        sat_list=sat_list, collection=collection, dates=dates, **pre_process_settings
    )
    actual_settings = set(list(coastsegmap.settings.keys()))
    expected_settings = set(list(pre_process_settings.keys()))
    assert expected_settings.issubset(actual_settings)
    assert set(["dates", "collection", "sat_list"]).issubset(actual_settings)
    
def test_load_rois_on_map_with_file(valid_coastseg_map_with_settings,
                                    valid_rois_filepath,
                                    valid_rois_gdf):
    """tests if a ROI will be created from geojson file and added to the map 
    Args:
        valid_coastseg_map_with_settings (Coastseg_Map): valid instance of coastseg map with settings already loaded
        valid_rois_filepath (str): filepath to geojson file containing valid rois
    """
    actual_coastsegmap = valid_coastseg_map_with_settings
    # test if rois will be correctly loaded onto map
    actual_coastsegmap.load_rois_on_map(file=valid_rois_filepath)
    assert actual_coastsegmap.rois is not None
    assert isinstance(actual_coastsegmap.rois, roi.ROI)
    # test if rois geodataframe was created correctly
    assert isinstance(actual_coastsegmap.rois.gdf, gpd.GeoDataFrame)
    assert actual_coastsegmap.rois.gdf.equals(valid_rois_gdf)
    # test if roi layer was added to map
    assert actual_coastsegmap.ROI_layer is not None
    existing_layer = actual_coastsegmap.map.find_layer(roi.ROI.LAYER_NAME)
    assert existing_layer is not None
    
    
