import pytest
import pkg_resources
import json
import geopandas as gpd
import os
from shapely.geometry import Polygon
from coastseg.transects import Transects,load_intersecting_transects

def test_load_intersecting_transects():
    rectangle = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 1), (1, 0)])])
    file_path = pkg_resources.resource_filename('coastseg', 'transects')
    transect_dir = file_path
    transect_files = os.listdir(transect_dir)
    result = load_intersecting_transects(rectangle, transect_files, transect_dir)
    assert isinstance(result, gpd.GeoDataFrame), 'Output should be a GeoDataFrame'
    assert all(col in result.columns for col in ['id', 'geometry', 'slope']), 'Output columns should contain "id", "geometry", and "slope"'

def test_transects_init(valid_bbox_gdf : gpd.GeoDataFrame):
    transects = Transects(bbox= valid_bbox_gdf)
    assert isinstance(transects, Transects), 'Output should be an instance of Transects class'
    assert isinstance(transects.gdf, gpd.GeoDataFrame), 'Transects attribute gdf should be a GeoDataFrame'

def test_transects_process_provided_transects():
    transects = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 1), (1, 0)])])
    transects['id'] = 'test_id'
    transects['slope'] = 1.0
    transects_obj = Transects(transects=transects)
    assert not transects_obj.gdf.empty, 'gdf should not be empty after processing provided transects'

def test_transects_process_bbox(valid_bbox_gdf : gpd.GeoDataFrame):
    transects_obj = Transects(bbox= valid_bbox_gdf)
    assert not transects_obj.gdf.empty, 'gdf should not be empty after processing bbox'

def test_transects_with_valid_transects(valid_transects_gdf):
    transects_obj = Transects(transects=valid_transects_gdf)
    columns_to_keep = ['id', 'geometry','slope']
    assert not transects_obj.gdf.empty, 'gdf should not be empty after processing provided transects'
    assert set(transects_obj.gdf.columns) == set(columns_to_keep), 'gdf should contain columns id, slope and geometry'
    assert 'usa_CA_0288-0122' in list(transects_obj.gdf['id'])
    assert not any(transects_obj.gdf['id'].duplicated()) == True

