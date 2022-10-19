import json
import math
import pytest
from src.coastseg import shoreline
from src.coastseg import transects
from src.coastseg import roi
from src.coastseg import exceptions
from src.coastseg import coastseg_map
from leafmap import Map
import geopandas as gpd
import pyproj
from ipyleaflet import GeoJSON

def test_create_geodataframe(valid_bbox_gdf,valid_shoreline_gdf):
    large_len = 1000
    small_len = 750
    input_espg = "epsg:4326"
    actual_roi = roi.ROI(bbox=valid_bbox_gdf,
                         shoreline = valid_shoreline_gdf,
                         square_len_lg =large_len,
                         square_len_sm = small_len)
    
    actual_gdf = actual_roi.create_geodataframe(bbox = valid_bbox_gdf,
                                   shoreline = valid_shoreline_gdf,
                                   large_length = large_len,
                                   small_length= small_len,
                                   crs= input_espg)
    
    assert isinstance(actual_gdf,gpd.GeoDataFrame)
    assert set(actual_gdf.columns) == set(['geometry', 'id'])
    assert actual_gdf.dtypes['geometry'] == 'geometry'
    assert actual_gdf.dtypes['id'] == 'int32'
    # drop unneeded columns before checking
    valid_shoreline_gdf = valid_shoreline_gdf.drop(valid_shoreline_gdf.columns.difference(['geometry']), 'columns')
    # Validate any shorelines intersect any squares in actual_gdf
    intersection_gdf = valid_shoreline_gdf.sjoin(actual_gdf, how="inner", predicate='intersects')
    assert intersection_gdf.empty == False
    
def test_fishnet_intersection(valid_bbox_gdf,valid_shoreline_gdf):
    # tests if a valid fishnet geodataframe intersects with given shoreline geodataframe
    square_size=1000
    output_espg="epsg:4326"
    
    actual_roi = roi.ROI(bbox=valid_bbox_gdf,
                         shoreline = valid_shoreline_gdf,
                         square_len_lg =square_size,
                         square_len_sm = square_size)
    
    fishnet_gdf = actual_roi.create_rois(valid_bbox_gdf,square_size)
    # check if fishnet intersects the shoreline
    fishnet_gdf = actual_roi.fishnet_intersection(fishnet_gdf,valid_shoreline_gdf)
    assert isinstance(fishnet_gdf,gpd.GeoDataFrame)
    assert fishnet_gdf.empty == False
    assert isinstance(fishnet_gdf.crs,pyproj.CRS)
    assert fishnet_gdf.crs == output_espg
    assert set(fishnet_gdf.columns) == set(['geometry'])
    # drop unneeded columns before checking
    valid_shoreline_gdf = valid_shoreline_gdf.drop(valid_shoreline_gdf.columns.difference(['geometry']), 'columns')
    # check if any shorelines intersect any squares in fishnet_gdf
    intersection_gdf = valid_shoreline_gdf.sjoin(fishnet_gdf, how="inner", predicate='intersects')
    assert intersection_gdf.empty == False

def test_get_fishnet(valid_bbox_gdf,valid_shoreline_gdf):
    # tests if a valid fishnet geodataframe intersects with given shoreline geodataframe
    square_size=1000
    output_espg="epsg:4326"
    
    actual_roi = roi.ROI(bbox=valid_bbox_gdf,
                         shoreline = valid_shoreline_gdf,
                         square_len_lg =square_size,
                         square_len_sm = square_size)
    
    # check if fishnet intersects the shoreline
    fishnet_gdf = actual_roi.get_fishnet_gdf(bbox_gdf = valid_bbox_gdf,
                               shoreline_gdf = valid_shoreline_gdf,
                               square_length = square_size ) 
    assert isinstance(fishnet_gdf,gpd.GeoDataFrame)
    assert isinstance(fishnet_gdf.crs,pyproj.CRS)
    assert fishnet_gdf.crs == output_espg
    assert set(fishnet_gdf.columns) == set(['geometry'])
    # drop unneeded columns before checking
    valid_shoreline_gdf = valid_shoreline_gdf.drop(valid_shoreline_gdf.columns.difference(['geometry']), 'columns')
    # check if any shorelines intersect any squares in fishnet_gdf
    intersection_gdf = valid_shoreline_gdf.sjoin(fishnet_gdf, how="inner", predicate='intersects')
    assert intersection_gdf.empty == False
    
def test_roi_missing_lengths(valid_bbox_gdf,valid_shoreline_gdf):
    # test with missing square lengths
    with pytest.raises(Exception):
        roi.ROI(bbox=valid_bbox_gdf,shoreline = valid_shoreline_gdf)

def test_bad_roi_initialization(valid_bbox_gdf):
    empty_gdf = gpd.GeoDataFrame()
    # test with missing shoreline
    with pytest.raises(exceptions.Object_Not_Found):
            roi.ROI(bbox=valid_bbox_gdf)
    # test with missing bbox and shoreline
    with pytest.raises(exceptions.Object_Not_Found):
            roi.ROI()
    # test with empty bbox
    with pytest.raises(exceptions.Object_Not_Found):
            roi.ROI(bbox= empty_gdf)    
    # test with empty shoreline
    with pytest.raises(exceptions.Object_Not_Found):
            roi.ROI(bbox=valid_bbox_gdf, shoreline=empty_gdf)             
            
def test_roi_from_bbox_and_shorelines(valid_bbox_gdf,valid_shoreline_gdf):
    large_len = 1000
    small_len = 750
    actual_roi = roi.ROI(bbox=valid_bbox_gdf,
                         shoreline = valid_shoreline_gdf,
                         square_len_lg =large_len,
                         square_len_sm = small_len)
    
    assert isinstance(actual_roi,roi.ROI)
    assert isinstance(actual_roi.gdf,gpd.GeoDataFrame)
    assert set(actual_roi.gdf.columns) == set(['geometry', 'id'])
    assert actual_roi.filename == "rois.geojson"
    assert hasattr(actual_roi, "extracted_shorelines")
    assert hasattr(actual_roi, "cross_distance_transects")
    assert hasattr(actual_roi, "inputs_dict")
    assert hasattr(actual_roi, "master_config")
 
def test_create_fishnet(valid_bbox_gdf,valid_shoreline_gdf):
    # tests if a valid geodataframe is created with square sizes approx. equal to given square_size
    square_size=1000
    input_espg = 'epsg:32610'
    output_espg="epsg:4326"
    
    actual_roi = roi.ROI(bbox=valid_bbox_gdf,
                         shoreline = valid_shoreline_gdf,
                         square_len_lg =square_size,
                         square_len_sm = square_size)
    
    # convert bbox to input_espg to most accurate espg to create fishnet with
    valid_bbox_gdf = valid_bbox_gdf.to_crs(input_espg)
    assert (valid_bbox_gdf.crs == 'epsg:32610')
    
    actual_fishnet = actual_roi.create_fishnet(valid_bbox_gdf,
                                               input_espg=input_espg,
                                               output_espg=output_espg,
                                               square_size=square_size)
    assert isinstance(actual_fishnet,gpd.GeoDataFrame)
    assert set(actual_fishnet.columns) == set(['geometry'])
    assert isinstance(actual_fishnet.crs,pyproj.CRS)
    assert actual_fishnet.crs == output_espg
    # reproject back to input_espg to check if square sizes are correct
    actual_fishnet = actual_fishnet.to_crs(input_espg)
    # pick a square out of the fishnet ensure is approx. equal to square size
    actual_lengths = tuple(map(lambda x: x.length/4 ,actual_fishnet['geometry']))
    # check if actual lengths are close to square_size length
    is_actual_length_correct = all(tuple(map(lambda x: math.isclose(x,square_size,rel_tol=1e-04) ,actual_lengths)))
    # assert all actual lengths are close to square_size length
    assert is_actual_length_correct == True

def test_create_rois(valid_bbox_gdf,valid_shoreline_gdf):
    square_size=1000
    # espg code of the valid_bbox_gdf
    input_espg = "epsg:4326"
    output_espg ="epsg:4326"
    actual_roi = roi.ROI(bbox=valid_bbox_gdf,
                         shoreline = valid_shoreline_gdf,
                         square_len_lg =square_size,
                         square_len_sm = square_size)
    
    actual_roi_gdf=actual_roi.create_rois(bbox=valid_bbox_gdf,
                                          input_espg =input_espg,
                                          output_espg=output_espg,
                                          square_size=square_size)
    assert isinstance(actual_roi_gdf,gpd.GeoDataFrame)
    assert isinstance(actual_roi_gdf.crs,pyproj.CRS)
    assert actual_roi_gdf.crs == output_espg
    assert set(actual_roi_gdf.columns) == set(['geometry'])    

def test_valid_roi_gdf(transect_compatible_rois_gdf:gpd.GeoDataFrame):
    """tests if a ROI will be created from valid rois of type gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): alid rois as a gpd.GeoDataFrame
    """    
    actual_roi = roi.ROI(rois_gdf = transect_compatible_rois_gdf)
    assert isinstance(actual_roi,roi.ROI)
    assert isinstance(actual_roi.gdf,gpd.GeoDataFrame)
    assert actual_roi.filename == "rois.geojson"
    assert hasattr(actual_roi, "extracted_shorelines")
    assert hasattr(actual_roi, "cross_distance_transects")
    assert hasattr(actual_roi, "inputs_dict")
    assert hasattr(actual_roi, "master_config")

def test_update_extracted_shorelines(transect_compatible_rois_gdf:gpd.GeoDataFrame):
    """tests if a ROI will be created from valid rois of type gpd.GeoDataFrame
    Args:
       transect_compatible_rois_gdf (gpd.GeoDataFrame): valid rois as a gpd.GeoDataFrame
    """    
    actual_roi = roi.ROI(rois_gdf = transect_compatible_rois_gdf)
    expected_dict={23:{'filename': 
         ['ms.tif',
          '2019.tif'], 
         'cloud_cover': [0.14, 0.0],
         'geoaccuracy': [7.9, 9.72],
         'idx': [4, 6], 'MNDWI_threshold': [-0.231, -0.3],
         'satname': ['L8', 'L8']}}
    actual_roi.update_extracted_shorelines(expected_dict)
    assert actual_roi.extracted_shorelines != {}
    assert actual_roi.extracted_shorelines == expected_dict
    
def test_set_inputs_dict(transect_compatible_rois_gdf:gpd.GeoDataFrame):
    """tests if a ROI will be created from valid rois thats a gpd.GeoDataFrame
    Args:
        transect_compatible_rois_gdf (gpd.GeoDataFrame): valid rois as a gpd.GeoDataFrame
    """    
    actual_roi = roi.ROI(rois_gdf = transect_compatible_rois_gdf)
    expected_dict = {23: {'dates': ['2018-12-01', '2019-03-01'],
                    'sat_list': ['L8'],
                    'sitename': 'ID02022-10-07__09_hr_38_min37sec',
                    'filepath': 'C:\\1_USGS\\CoastSeg\\repos\\2_CoastSeg\\CoastSeg_fork\\Seg2Map\\data',
                    'roi_id': 23,
                    'polygon': [[[-124.1662679688807, 40.863030239542944],
                        [-124.16690059058178, 40.89905645671534],
                        [-124.11942071317034, 40.89952713781644],
                        [-124.11881381876809, 40.863500326870245],
                        [-124.1662679688807, 40.863030239542944]]],
                    'landsat_collection': 'C01'},
                    39: {'dates': ['2018-12-01', '2019-03-01'],
                    'sat_list': ['L8'],
                    'sitename': 'ID12022-10-07__09_hr_38_min37sec',
                    'filepath': 'C:\\1_USGS\\CoastSeg\\repos\\2_CoastSeg\\CoastSeg_fork\\Seg2Map\\data',
                    'roi_id': 39,
                    'polygon': [[[-124.16690059058178, 40.89905645671534],
                        [-124.1675343590045, 40.93508244001033],
                        [-124.12002870768146, 40.9355537155221],
                        [-124.11942071317034, 40.89952713781644],
                        [-124.16690059058178, 40.89905645671534]]],
                    'landsat_collection': 'C01'}}
    actual_roi.set_inputs_dict(expected_dict)
    assert actual_roi.inputs_dict != {}
    assert actual_roi.inputs_dict == expected_dict 
    
    


