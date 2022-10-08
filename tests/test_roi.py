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

def test_valid_roi_gdf(transect_compatible_rois_gdf:gpd.GeoDataFrame):
    """tests if a ROI will be created from valid rois thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): alid rois as a gpd.GeoDataFrame
    """    
    actual_roi = roi.ROI(rois_gdf = transect_compatible_rois_gdf)
    assert isinstance(actual_roi,roi.ROI)
    assert actual_roi.gdf is not None
    assert actual_roi.filename == "rois.geojson"
    assert hasattr(actual_roi, "extracted_shorelines")
    assert hasattr(actual_roi, "cross_distance_transects")

def test_valid_roi_gdf(transect_compatible_rois_gdf:gpd.GeoDataFrame):
    """tests if a ROI will be created from valid rois thats a gpd.GeoDataFrame
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
    
    


