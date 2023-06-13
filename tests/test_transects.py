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

# 1. load transects from a bbox geodataframe in crs 4326
def test_transects_init(valid_bbox_gdf : gpd.GeoDataFrame):
    actual_transects = Transects(bbox= valid_bbox_gdf)
    assert isinstance(actual_transects, Transects), 'Output should be an instance of Transects class'
    assert isinstance(actual_transects.gdf, gpd.GeoDataFrame), 'Transects attribute gdf should be a GeoDataFrame'
    assert not actual_transects.gdf.empty, 'gdf should not be empty after processing bbox'
    assert actual_transects.gdf.crs.to_string() =='EPSG:4326'
    assert 'id' in actual_transects.gdf.columns
    assert not any(actual_transects.gdf['id'].duplicated())


# 2. load transects from a bbox geodataframe with no crs
def test_transects_process_provided_transects():
    transects = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 1), (1, 0)])])
    transects['id'] = 'test_id'
    transects['slope'] = 1.0
    actual_transects = Transects(transects=transects)
    assert isinstance(actual_transects, Transects), 'Output should be an instance of Transects class'
    assert not actual_transects.gdf.empty, 'gdf should not be empty after processing provided transects'
    assert actual_transects.gdf.crs.to_string() =='EPSG:4326'
    assert 'id' in  actual_transects.gdf.columns
    assert not any( actual_transects.gdf['id'].duplicated())

# 3. load transects from a bbox  geodataframe with a CRS 4327
def test_transects_process_provided_transects_in_different_crs():
    transects = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 1), (1, 0)])])
    transects.set_crs('EPSG:4327', inplace=True)
    transects['id'] = 'test_id'
    transects['slope'] = 1.0
    actual_transects = Transects(transects=transects)
    assert isinstance(actual_transects, Transects), 'Output should be an instance of Transects class'
    assert not actual_transects.gdf.empty, 'gdf should not be empty after processing provided transects'
    assert actual_transects.gdf.crs.to_string() =='EPSG:4326'
    assert 'id' in  actual_transects.gdf.columns
    assert not any( actual_transects.gdf['id'].duplicated())


# 4. load transects from a transects geodataframe with a CRS 4326
def test_transects_with_valid_transects(valid_transects_gdf):
    actual_transects = Transects(transects=valid_transects_gdf)
    columns_to_keep = ['id', 'geometry','slope']
    assert not actual_transects.gdf.empty, 'gdf should not be empty after processing provided transects'
    assert set(actual_transects.gdf.columns) == set(columns_to_keep), 'gdf should contain columns id, slope and geometry'
    assert 'usa_CA_0288-0122' in list(actual_transects.gdf['id'])
    assert not any(actual_transects.gdf['id'].duplicated())
    assert actual_transects.gdf.crs.to_string() =='EPSG:4326'
    assert 'id' in  actual_transects.gdf.columns
    assert not any( actual_transects.gdf['id'].duplicated())

# 5. load transects from a transects  geodataframe with a CRS 4327
def test_transects_with_transects_different_crs(valid_transects_gdf):
    # change the crs of the geodataframe
    transects_diff_crs=valid_transects_gdf.to_crs('EPSG:4326',inplace=False)
    actual_transects = Transects(transects=transects_diff_crs)
    columns_to_keep = ['id', 'geometry','slope']
    assert not actual_transects.gdf.empty, 'gdf should not be empty after processing provided transects'
    assert set(actual_transects.gdf.columns) == set(columns_to_keep), 'gdf should contain columns id, slope and geometry'
    assert 'usa_CA_0288-0122' in list(actual_transects.gdf['id'])
    assert not any(actual_transects.gdf['id'].duplicated())
    assert actual_transects.gdf.crs.to_string() =='EPSG:4326'
    assert 'id' in  actual_transects.gdf.columns
    assert not any( actual_transects.gdf['id'].duplicated())


