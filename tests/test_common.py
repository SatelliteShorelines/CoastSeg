import json

from src.coastseg import exceptions
from src.coastseg import common

import geopandas as gpd
from shapely import geometry
import pytest

def test_create_config_gdf(valid_rois_gdf, transect_compatible_shoreline_gdf, transect_compatible_transects_gdf):
    # test if a gdf is created with all the rois, shorelines and transects
    actual_gdf=common.create_config_gdf(valid_rois_gdf,transect_compatible_shoreline_gdf,transect_compatible_transects_gdf)
    assert 'type' in actual_gdf.columns
    assert actual_gdf[actual_gdf['type']=='transect'].empty == False
    assert actual_gdf[actual_gdf['type']=='shoreline'].empty == False
    assert actual_gdf[actual_gdf['type']=='roi'].empty == False
    
    # test if a gdf is created with all the rois, transects if shorelines is None
    shorelines_gdf=None
    actual_gdf=common.create_config_gdf(valid_rois_gdf,shorelines_gdf,transect_compatible_transects_gdf)
    assert 'type' in actual_gdf.columns
    assert actual_gdf[actual_gdf['type']=='transect'].empty == False
    assert actual_gdf[actual_gdf['type']=='shoreline'].empty == True
    assert actual_gdf[actual_gdf['type']=='roi'].empty == False
    # test if a gdf is created with all the rois if  transects and shorelines is None
    transects_gdf = None
    actual_gdf=common.create_config_gdf(valid_rois_gdf,shorelines_gdf,transects_gdf)
    assert 'type' in actual_gdf.columns
    assert actual_gdf[actual_gdf['type']=='transect'].empty == True
    assert actual_gdf[actual_gdf['type']=='shoreline'].empty == True
    assert actual_gdf[actual_gdf['type']=='roi'].empty == False


def test_create_config_dict(valid_inputs, valid_settings)->dict:
    # test adding an ROI's data to an empty config dictionary
    master_config = {}
    actual_config = common.create_json_config(master_config, valid_inputs, valid_settings)
    assert isinstance(actual_config,dict)
    expected_roi = valid_inputs['roi_id']
    # test the roi id was added as a key to config
    assert expected_roi in actual_config
    # test roi id is in the config list of all roi_ids
    assert actual_config['roi_ids'] == [expected_roi]
    assert actual_config[expected_roi]['inputs']  == valid_inputs
    assert actual_config[expected_roi]['settings'] == valid_settings
    
    # test adding another ROI's data to a non-empty config dictionary
    master_config = {'roi_ids':[23,24],
                        23:{
                            'settings':[],
                            'inputs':[]
                        },
                            24:{
                            'settings':[],
                            'inputs':[]
                        }}
    actual_config = common.create_json_config(master_config, valid_inputs, valid_settings)
    assert isinstance(actual_config,dict)
    expected_roi = valid_inputs['roi_id']
    # test the id was added as a key to config
    assert expected_roi in actual_config
    # test all roi ids are in the config list of all roi_ids
    assert actual_config['roi_ids'] == [23,24,expected_roi]
    assert actual_config[expected_roi]['inputs']  == valid_inputs
    assert actual_config[expected_roi]['settings'] == valid_settings
    # ensure the other record in config dict were not changed
    assert actual_config[23]['inputs']  == []
    assert actual_config[23]['settings'] == []
    # test that all the roi ids are in config
    assert 23 in actual_config
    assert 24 in actual_config
    
    



def test_gdf_to_polygon(valid_bbox_gdf,valid_rois_gdf):
    # test if it returns a shapely.geometry.Polygon() given geodataframe and no id given
    polygon = common.convert_gdf_to_polygon(valid_bbox_gdf,None)
    bbox_dict = json.loads((valid_bbox_gdf["geometry"].to_json()))
    assert isinstance(polygon, geometry.Polygon)
    assert geometry.Polygon(bbox_dict["features"][0]["geometry"]['coordinates'][0])
    # test if it returns the correct shapely.geometry.Polygon() given a specific id in the geodataframe 
    id = 17
    polygon = common.convert_gdf_to_polygon(valid_rois_gdf,id)
    roi_dict =json.loads(valid_rois_gdf[valid_rois_gdf['id']==id]["geometry"].to_json())
    assert isinstance(polygon, geometry.Polygon)
    assert geometry.Polygon(roi_dict["features"][0]["geometry"]['coordinates'][0])

def test_gdf_to_polygon_invalid(valid_rois_gdf):
    # should raise exception if id is not in the geodataframe
    with pytest.raises(Exception):
        id = 18
        polygon = common.convert_gdf_to_polygon(valid_rois_gdf,id)