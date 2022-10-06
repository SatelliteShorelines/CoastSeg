import json

from src.coastseg import exceptions
from src.coastseg import common

import geopandas as gpd
from shapely import geometry
import pytest



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