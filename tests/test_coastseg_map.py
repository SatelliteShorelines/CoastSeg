import json
import pytest
from src.coastseg import shoreline
from src.coastseg import transects
from src.coastseg import roi
from src.coastseg import exceptions
from src.coastseg import coastseg_map
from leafmap import Map
import geopandas as gpd
from ipyleaflet import GeoJSON

def test_valid_shoreline_gdf(valid_shoreline_gdf:gpd.GeoDataFrame):
    """tests if a Shoreline will be created from a valid shoreline thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): a valid shoreline as a gpd.GeoDataFrame
    """    
    expected_shoreline = shoreline.Shoreline(shoreline = valid_shoreline_gdf)
    assert isinstance(expected_shoreline,shoreline.Shoreline)
    assert expected_shoreline.gdf is not None
    assert expected_shoreline.filename == "shoreline.geojson"

def test_valid_transects_gdf(transect_compatible_transects_gdf:gpd.GeoDataFrame):
    """tests if a Transects will be created from a valid transects thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): valid transects as a gpd.GeoDataFrame
    """    
    expected_transects = transects.Transects(transects = transect_compatible_transects_gdf)
    assert isinstance(expected_transects,transects.Transects)
    assert expected_transects.gdf is not None
    assert expected_transects.filename == "transects.geojson"

def test_valid_roi_gdf(transect_compatible_rois_gdf:gpd.GeoDataFrame):
    """tests if a ROI will be created from valid rois thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): alid rois as a gpd.GeoDataFrame
    """    
    expected_roi = roi.ROI(rois_gdf = transect_compatible_rois_gdf)
    assert isinstance(expected_roi,roi.ROI)
    assert expected_roi.gdf is not None
    assert expected_roi.filename == "rois.geojson"
    
def test_valid_roi_gdf(transect_compatible_rois_gdf:gpd.GeoDataFrame):
    """tests if a ROI will be created from valid rois thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): alid rois as a gpd.GeoDataFrame
    """    
    expected_roi = roi.ROI(rois_gdf = transect_compatible_rois_gdf)
    assert isinstance(expected_roi,roi.ROI)
    assert expected_roi.gdf is not None
    assert expected_roi.filename == "rois.geojson"
    
    
def test_coastseg_map():
    """tests a CoastSeg_Map object is created
    """    
    coastsegmap = coastseg_map.CoastSeg_Map()
    assert isinstance(coastsegmap,coastseg_map.CoastSeg_Map)
    assert isinstance(coastsegmap.map,Map)
    assert hasattr(coastsegmap,"draw_control")
    
def test__coastseg_map_settings():
    """tests if a ROI will be created from valid rois thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): alid rois as a gpd.GeoDataFrame
    """    
    coastsegmap = coastseg_map.CoastSeg_Map()
    dates = ['2018-12-01', '2019-03-01']
    collection = 'C01'
    sat_list=  ['L8']
    pre_process_settings = { 
        # general parameters:
        'cloud_thresh': 0.5,        # threshold on maximum cloud cover
        'dist_clouds': 300,        # ditance around clouds where shoreline can't be mapped
        'output_epsg': 3857,        # epsg code of spatial reference system desired for the output   
        # quality control:
        'check_detection': True,    # if True, shows each shoreline detection to the user for validation
        'adjust_detection': False,  # if True, allows user to adjust the postion of each shoreline by changing the threhold
        'save_figure': True,        # if True, saves a figure showing the mapped shoreline for each image
        # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
        'min_beach_area': 4500,     # minimum area (in metres^2) for an object to be labelled as a beach
        'buffer_size': 150,         # radius (in metres) of the buffer around sandy pixels considered in the shoreline detection
        'min_length_sl': 200,       # minimum length (in metres) of shoreline perimeter to be valid
        'cloud_mask_issue': False,  # switch this parameter to True if sand pixels are masked (in black) on many images  
        'sand_color': 'default',    # 'default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
        'pan_off':'False',          # if True, no pan-sharpening is performed on Landsat 7,8 and 9 imagery
        'create_plot':False,        # True create a matplotlib plot of the image with the datetime as the title
        'max_dist_ref':25
    }
    coastsegmap.save_settings(sat_list=sat_list, collection = collection,dates=dates,**pre_process_settings)
    actual_settings = set(list(coastsegmap.settings.keys()))
    expected_settings = set(list(pre_process_settings.keys()))
    assert expected_settings.issubset(actual_settings)
    assert set(['dates','collection','sat_list']).issubset(actual_settings)

