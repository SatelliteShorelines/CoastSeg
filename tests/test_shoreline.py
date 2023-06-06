import pytest
import geopandas as gpd
from coastseg.shoreline import Shoreline

from coastseg import exceptions

def test_shoreline_initialization():
    shoreline = Shoreline()
    assert isinstance(shoreline, Shoreline)
    assert isinstance(shoreline.gdf, gpd.GeoDataFrame)

# 1. load shorelines from a shorelines geodataframe with a CRS 4326 with no id
def test_initialize_shorelines_with_provided_shorelines(valid_shoreline_gdf):
    actual_shoreline = Shoreline(shoreline=valid_shoreline_gdf)
    assert not actual_shoreline.gdf.empty
    assert "id" in actual_shoreline.gdf.columns
    columns_to_keep = [
            "id", "geometry", "river_label", "ERODIBILITY", "CSU_ID",
            "turbid_label", "slope_label", "sinuosity_label",
            "TIDAL_RANGE", "MEAN_SIG_WAVEHEIGHT"
        ]
    assert all(col in actual_shoreline.gdf.columns for col in columns_to_keep), "Not all columns are present in shoreline.gdf.columns"
    assert not any(actual_shoreline.gdf['id'].duplicated()) == True
    assert actual_shoreline.gdf.crs.to_string() =='EPSG:4326'

# 2. load shorelines from a shorelines geodataframe with a CRS 4327 with no id
def test_initialize_shorelines_with_wrong_CRS(valid_shoreline_gdf):
    # change the crs of the geodataframe
    shorelines_diff_crs=valid_shoreline_gdf.to_crs('EPSG:4326',inplace=False)
    actual_shoreline = Shoreline(shoreline=shorelines_diff_crs)
    assert not actual_shoreline.gdf.empty
    assert "id" in actual_shoreline.gdf.columns
    columns_to_keep = [
            "id", "geometry", "river_label", "ERODIBILITY", "CSU_ID",
            "turbid_label", "slope_label", "sinuosity_label",
            "TIDAL_RANGE", "MEAN_SIG_WAVEHEIGHT"
        ]
    assert all(col in actual_shoreline.gdf.columns for col in columns_to_keep), "Not all columns are present in shoreline.gdf.columns"
    assert not any(actual_shoreline.gdf['id'].duplicated()) == True
    assert actual_shoreline.gdf.crs.to_string() =='EPSG:4326'

# 3. load shorelines from a shorelines geodataframe with empty ids
def test_initialize_shorelines_with_empty_id_column(valid_shoreline_gdf):
    # change the crs of the geodataframe
    shorelines_diff_crs=valid_shoreline_gdf.to_crs('EPSG:4326',inplace=False)
    # make id column empty
    shorelines_diff_crs = shorelines_diff_crs.assign(id=None)
    actual_shoreline = Shoreline(shoreline=shorelines_diff_crs)
    assert not actual_shoreline.gdf.empty
    assert "id" in actual_shoreline.gdf.columns
    columns_to_keep = [
            "id", "geometry", "river_label", "ERODIBILITY", "CSU_ID",
            "turbid_label", "slope_label", "sinuosity_label",
            "TIDAL_RANGE", "MEAN_SIG_WAVEHEIGHT"
        ]
    assert all(col in actual_shoreline.gdf.columns for col in columns_to_keep), "Not all columns are present in shoreline.gdf.columns"
    assert not any(actual_shoreline.gdf['id'].duplicated()) == True
    assert actual_shoreline.gdf.crs.to_string() =='EPSG:4326'

# 4. load shorelines from a shorelines geodataframe with identical ids
def test_initialize_shorelines_with_identical_ids(valid_shoreline_gdf):
    # change the crs of the geodataframe
    shorelines_diff_crs=valid_shoreline_gdf.to_crs('EPSG:4326',inplace=False)
    # make id column empty
    shorelines_diff_crs = shorelines_diff_crs.assign(id='bad_id')
    actual_shoreline = Shoreline(shoreline=shorelines_diff_crs)
    assert not actual_shoreline.gdf.empty
    assert "id" in actual_shoreline.gdf.columns
    columns_to_keep = [
            "id", "geometry", "river_label", "ERODIBILITY", "CSU_ID",
            "turbid_label", "slope_label", "sinuosity_label",
            "TIDAL_RANGE", "MEAN_SIG_WAVEHEIGHT"
        ]
    assert all(col in actual_shoreline.gdf.columns for col in columns_to_keep), "Not all columns are present in shoreline.gdf.columns"
    assert not any(actual_shoreline.gdf['id'].duplicated()) == True
    assert actual_shoreline.gdf.crs.to_string() =='EPSG:4326'


def test_initialize_shorelines_with_bbox(valid_bbox_gdf):
    shoreline = Shoreline(bbox=valid_bbox_gdf)

    assert not shoreline.gdf.empty
    assert "id" in shoreline.gdf.columns
    columns_to_keep = [
            "id", "geometry", "river_label", "ERODIBILITY", "CSU_ID",
            "turbid_label", "slope_label", "sinuosity_label",
            "TIDAL_RANGE", "MEAN_SIG_WAVEHEIGHT"
        ]
    assert all(col in shoreline.gdf.columns for col in columns_to_keep), "Not all columns are present in shoreline.gdf.columns"
    assert not any(shoreline.gdf['id'].duplicated()) == True

def test_style_layer():
    layer_name ="test_layer"
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [125.6, 10.1]
                },
                "properties": {
                    "name": "test"
                }
            }
        ]
    }
    shoreline = Shoreline()
    layer = shoreline.style_layer(geojson_data, layer_name )
    
    assert layer.name == layer_name 
    assert layer.data['features'][0]["geometry"] == geojson_data['features'][0]["geometry"]
    assert layer.style


# # you can also mock some methods that depend on external data, like downloading from the internet
# def test_download_shoreline(mocker):
#     mock_download = mocker.patch("coastseg.common.download_url", return_value=None)
#     shoreline = Shoreline()
#     with exceptions.DownloadError:
#         shoreline.download_shoreline("test_file.geojson")

#     mock_download.assert_called_once_with("https://zenodo.org/record/7761607/files/test_file.geojson?download=1", mocker.ANY, filename="test_file.geojson")

