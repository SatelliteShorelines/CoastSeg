import pytest
import geopandas as gpd
from shapely.geometry import Point
from coastseg import geodata_processing
from unittest.mock import patch, Mock

# Mock Data Setup

data = {
    "type": [
        "roi",
        "rois",
        "shoreline",
        "shorelines",
        "transect",
        "transects",
        "bbox",
        "other",
    ],
    "geometry": [
        Point(1, 1),
        Point(2, 2),
        Point(3, 3),
        Point(4, 4),
        Point(5, 5),
        Point(6, 6),
        Point(7, 7),
        Point(8, 8),
    ],
}

mock_gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

# Tests


def test_feature_type_roi():
    result = geodata_processing.extract_feature_from_geodataframe(mock_gdf, "roi")
    assert set(result["type"].to_list()) == {"roi", "rois"}


def test_feature_type_rois():
    result = geodata_processing.extract_feature_from_geodataframe(mock_gdf, "rois")
    assert set(result["type"].to_list()) == {"roi", "rois"}


def test_feature_type_shoreline():
    result = geodata_processing.extract_feature_from_geodataframe(mock_gdf, "shoreline")
    assert set(result["type"].to_list()) == {"shoreline", "shorelines"}


def test_feature_type_shorelines():
    result = geodata_processing.extract_feature_from_geodataframe(
        mock_gdf, "shorelines"
    )
    assert set(result["type"].to_list()) == {"shoreline", "shorelines"}


def test_feature_type_transect():
    result = geodata_processing.extract_feature_from_geodataframe(mock_gdf, "transect")
    assert set(result["type"].to_list()) == {"transect", "transects"}


def test_feature_type_transects():
    result = geodata_processing.extract_feature_from_geodataframe(mock_gdf, "transects")
    assert set(result["type"].to_list()) == {"transect", "transects"}


def test_feature_type_bbox():
    result = geodata_processing.extract_feature_from_geodataframe(mock_gdf, "bbox")
    assert set(result["type"].to_list()) == {"bbox"}


def test_invalid_feature_type():
    # Let's use a feature type not in the mock data, like 'invalidtype'.
    result = geodata_processing.extract_feature_from_geodataframe(
        mock_gdf, "invalidtype"
    )
    assert result.empty  # This checks that the resulting DataFrame is empty


def test_load_valid_file(valid_geojson_path):
    result = geodata_processing.load_geodataframe_from_file(valid_geojson_path, "roi")
    assert set(result["type"].to_list()) == {"roi", "rois"}


def test_load_invalid_file(empty_geojson_path):
    with pytest.raises(ValueError, match=r"Empty .* file provided"):
        geodata_processing.load_geodataframe_from_file(empty_geojson_path, "roi")


def test_load_file_missing_rois(config_gdf_missing_rois_path):
    with pytest.raises(ValueError, match=r"Empty .* file provided"):
        geodata_processing.load_geodataframe_from_file(
            config_gdf_missing_rois_path, "roi"
        )


@patch("geopandas.read_file")
def test_load_nonexistent_file(mock_read_file):
    mock_read_file.side_effect = FileNotFoundError
    with pytest.raises(FileNotFoundError):
        geodata_processing.load_geodataframe_from_file(
            "nonexistent_path.geojson", "roi"
        )


def test_load_valid_file_non_config(non_config_geojson_path):
    result = geodata_processing.load_geodataframe_from_file(
        non_config_geojson_path, "roi"
    )
    assert not result.empty
