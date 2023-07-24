import os
import pytest
from unittest.mock import MagicMock
import geopandas as gpd
from shapely.geometry import Polygon
from coastseg.shoreline import Shoreline, ShorelineServices

from coastseg import exceptions

from unittest.mock import MagicMock

# def test_create_geodataframe(valid_bbox_gdf:gpd.GeoDataFrame,):

#     # Mock services
#     services = ShorelineServices()
#     services.download_service = MagicMock()
#     services.preprocess_service = MagicMock()
#     services.create_ids_service = MagicMock()

#     # Mock data
#     bbox_data = valid_bbox_gdf

#     shoreline_files_data = ['file1.geojson', 'file2.geojson']

#     shoreline = Shoreline(bbox=bbox_data, services=services)

#     # Test with a known bounding box and a known set of shoreline files
#     gdf = shoreline.create_geodataframe(bbox_data, shoreline_files_data)
#     services.preprocess_service.assert_called()
#     services.create_ids_service.assert_called()
#     assert isinstance(gdf, gpd.GeoDataFrame)

#     # Test with an empty list of shoreline files
#     with pytest.raises(FileNotFoundError):
#         gdf = shoreline.create_geodataframe(bbox_data, [])

#     # Test with an empty GeoDataFrame
#     empty_bbox_data = gpd.GeoDataFrame()
#     gdf = shoreline.create_geodataframe(empty_bbox_data, shoreline_files_data)
#     assert gdf.empty


# def test_ShorelineServices():
#     # Create a mock object for ShorelineServices
#     mock_services = MagicMock(ShorelineServices)

#     # Now you can set return values for the mock object's methods
#     mock_services.download_service.return_value = "download_service"
#     mock_services.preprocess_service.return_value = "preprocess_service"
#     mock_services.create_ids_service.return_value = "create_unique_ids_service"

#     # Now you can use the mock object in your tests
#     assert mock_services.download_service('a', 'b', 'c') == "download_service"
#     assert mock_services.preprocess_service('test') == "preprocess_service"
#     assert mock_services.create_ids_service('test') == "create_unique_ids_service"


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
        "id",
        "geometry",
        "river_label",
        "ERODIBILITY",
        "CSU_ID",
        "turbid_label",
        "slope_label",
        "sinuosity_label",
        "TIDAL_RANGE",
        "MEAN_SIG_WAVEHEIGHT",
    ]
    assert all(
        col in actual_shoreline.gdf.columns for col in columns_to_keep
    ), "Not all columns are present in shoreline.gdf.columns"
    assert not any(actual_shoreline.gdf["id"].duplicated()) == True
    assert actual_shoreline.gdf.crs.to_string() == "EPSG:4326"


# 2. load shorelines from a shorelines geodataframe with a CRS 4327 with no id
def test_initialize_shorelines_with_wrong_CRS(valid_shoreline_gdf):
    # change the crs of the geodataframe
    shorelines_diff_crs = valid_shoreline_gdf.to_crs("EPSG:4326", inplace=False)
    actual_shoreline = Shoreline(shoreline=shorelines_diff_crs)
    assert not actual_shoreline.gdf.empty
    assert "id" in actual_shoreline.gdf.columns
    columns_to_keep = [
        "id",
        "geometry",
        "river_label",
        "ERODIBILITY",
        "CSU_ID",
        "turbid_label",
        "slope_label",
        "sinuosity_label",
        "TIDAL_RANGE",
        "MEAN_SIG_WAVEHEIGHT",
    ]
    assert all(
        col in actual_shoreline.gdf.columns for col in columns_to_keep
    ), "Not all columns are present in shoreline.gdf.columns"
    assert not any(actual_shoreline.gdf["id"].duplicated()) == True
    assert actual_shoreline.gdf.crs.to_string() == "EPSG:4326"


# 3. load shorelines from a shorelines geodataframe with empty ids
def test_initialize_shorelines_with_empty_id_column(valid_shoreline_gdf):
    # change the crs of the geodataframe
    shorelines_diff_crs = valid_shoreline_gdf.to_crs("EPSG:4326", inplace=False)
    # make id column empty
    shorelines_diff_crs = shorelines_diff_crs.assign(id=None)
    actual_shoreline = Shoreline(shoreline=shorelines_diff_crs)
    assert not actual_shoreline.gdf.empty
    assert "id" in actual_shoreline.gdf.columns
    columns_to_keep = [
        "id",
        "geometry",
        "river_label",
        "ERODIBILITY",
        "CSU_ID",
        "turbid_label",
        "slope_label",
        "sinuosity_label",
        "TIDAL_RANGE",
        "MEAN_SIG_WAVEHEIGHT",
    ]
    assert all(
        col in actual_shoreline.gdf.columns for col in columns_to_keep
    ), "Not all columns are present in shoreline.gdf.columns"
    assert not any(actual_shoreline.gdf["id"].duplicated()) == True
    assert actual_shoreline.gdf.crs.to_string() == "EPSG:4326"


# 4. load shorelines from a shorelines geodataframe with identical ids
def test_initialize_shorelines_with_identical_ids(valid_shoreline_gdf):
    # change the crs of the geodataframe
    shorelines_diff_crs = valid_shoreline_gdf.to_crs("EPSG:4326", inplace=False)
    # make id column empty
    shorelines_diff_crs = shorelines_diff_crs.assign(id="bad_id")
    actual_shoreline = Shoreline(shoreline=shorelines_diff_crs)
    assert not actual_shoreline.gdf.empty
    assert "id" in actual_shoreline.gdf.columns
    columns_to_keep = [
        "id",
        "geometry",
        "river_label",
        "ERODIBILITY",
        "CSU_ID",
        "turbid_label",
        "slope_label",
        "sinuosity_label",
        "TIDAL_RANGE",
        "MEAN_SIG_WAVEHEIGHT",
    ]
    assert all(
        col in actual_shoreline.gdf.columns for col in columns_to_keep
    ), "Not all columns are present in shoreline.gdf.columns"
    assert not any(actual_shoreline.gdf["id"].duplicated()) == True
    assert actual_shoreline.gdf.crs.to_string() == "EPSG:4326"


def test_initialize_shorelines_with_bbox(valid_bbox_gdf):
    shoreline = Shoreline(bbox=valid_bbox_gdf)

    assert not shoreline.gdf.empty
    assert "id" in shoreline.gdf.columns
    columns_to_keep = [
        "id",
        "geometry",
        "river_label",
        "ERODIBILITY",
        "CSU_ID",
        "turbid_label",
        "slope_label",
        "sinuosity_label",
        "TIDAL_RANGE",
        "MEAN_SIG_WAVEHEIGHT",
    ]
    assert all(
        col in shoreline.gdf.columns for col in columns_to_keep
    ), "Not all columns are present in shoreline.gdf.columns"
    assert not any(shoreline.gdf["id"].duplicated()) == True


def test_style_layer():
    layer_name = "test_layer"
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [125.6, 10.1]},
                "properties": {"name": "test"},
            }
        ],
    }
    shoreline = Shoreline()
    layer = shoreline.style_layer(geojson_data, layer_name)

    assert layer.name == layer_name
    assert (
        layer.data["features"][0]["geometry"] == geojson_data["features"][0]["geometry"]
    )
    assert layer.style


# # you can also mock some methods that depend on external data, like downloading from the internet
# def test_download_shoreline(mocker):
#     mock_download = mocker.patch("coastseg.common.download_url", return_value=None)
#     shoreline = Shoreline()
#     with exceptions.DownloadError:
#         shoreline.download_shoreline("test_file.geojson")

#     mock_download.assert_called_once_with("https://zenodo.org/record/7761607/files/test_file.geojson?download=1", mocker.ANY, filename="test_file.geojson")
