import os
import json
import tempfile
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry import Polygon
from coastseg.tide_correction import get_tide_predictions
from coastseg.tide_correction import save_transect_settings
from coastseg.tide_correction import load_regions_from_geojson
from unittest.mock import patch

def test_save_transect_settings():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a settings file in the temporary directory
        settings_file = os.path.join(tmpdir, "transects_settings.json")
        with open(settings_file, "w") as f:
            json.dump({"reference_elevation": 0, "beach_slope": 0}, f)

        # Call the function to update the settings
        save_transect_settings(tmpdir, 1.23, 4.56)

        # Check that the settings were updated correctly
        with open(settings_file, "r") as f:
            settings = json.load(f)
        assert settings["reference_elevation"] == 1.23
        assert settings["beach_slope"] == 4.56

def test_save_transect_settings_no_file():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # The settings file does not exist initially
        settings_file = os.path.join(tmpdir, "transects_settings.json")

        # Call the function to create and update the settings
        save_transect_settings(tmpdir, 1.23, 4.56)

        # Check that the settings file was created with the correct values
        with open(settings_file, "r") as f:
            settings = json.load(f)
        assert settings["reference_elevation"] == 1.23
        assert settings["beach_slope"] == 4.56


def test_load_regions_from_geojson():
    # Create a temporary GeoJSON file
    with tempfile.NamedTemporaryFile(suffix=".geojson") as tmp:
        geojson_path = tmp.name

        # Create a GeoDataFrame with some test data
        regions_gdf = gpd.GeoDataFrame(
            {
                'region_id': [1, 2, 3],
                'geometry': [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
                    Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                ]
            }
        )

        # Save the GeoDataFrame to the temporary GeoJSON file
        regions_gdf.to_file(geojson_path, driver='GeoJSON')

        # Call the function to load the regions
        loaded_regions_gdf = load_regions_from_geojson(geojson_path)

        # Check that the loaded regions is a GeoDataFrame
        assert isinstance(loaded_regions_gdf, gpd.GeoDataFrame)

        # Check that the 'region_id' column is added
        assert 'region_id' in loaded_regions_gdf.columns

        # Check that the number of regions is correct
        assert len(loaded_regions_gdf) == 3
        
        
def test_get_tide_predictions():
    # Test data setup
    x, y = 1.0, 2.0
    timeseries_df = pd.DataFrame({
        "dates": pd.date_range("2021-01-01", periods=3),
        "transect1": [0.5, 0.6, 0.7]
    })
    model_region_directory = "path/to/model/region"
    # invalid transect ID
    transect_id = "transect3"

    # Call the function
    # Mock the model_tides function for invalid transect ID
    with patch('coastseg.tide_correction.model_tides') as mock_model_tides:
        mock_model_tides.return_value = pd.DataFrame({"tide": [1.0, 2.0, 3.0],
                                                      "transect_id": np.repeat(transect_id,3),
                                                      "dates":timeseries_df.dates,
                                                      "x": np.repeat(x, len(timeseries_df)),
                                                      "y": np.repeat(y, len(timeseries_df))})
        result = get_tide_predictions(x, y, timeseries_df, model_region_directory, transect_id)
        assert result is None
        
    # Mock the model_tides function with no transect ID
    with patch('coastseg.tide_correction.model_tides') as mock_model_tides:
        mock_model_tides.return_value = pd.DataFrame({"tide": [1.0, 2.0, 3.0],
                                                      "dates":timeseries_df.dates,
                                                      "x": np.repeat(x, len(timeseries_df)),
                                                      "y": np.repeat(y, len(timeseries_df))})
        result = get_tide_predictions(x, y, timeseries_df, model_region_directory, "")
        # Assertions
        assert 'tide' in result.columns, "Result should contain 'tide' column"
        assert 'transect_id' not in result.columns, "Result should contain 'transect id' column"
        assert 'dates' in result.columns, "Result should contain 'dates' column"
        assert 'x' in result.columns, "Result should contain 'x' column"
        assert 'y' in result.columns, "Result should contain 'y' column"
        
        assert len(result) == 3, "Result should have 3 rows of predictions"
        assert mock_model_tides.called, "model_tides function should be called"
        assert (result['x'] == x).all(), "All 'x' values should be equal to input x"
        assert (result['y'] == y).all(), "All 'y' values should be equal to input y"

def test_get_tide_predictions_valid_transect():
    # Test data setup
    x, y = 1.0, 2.0
    timeseries_df = pd.DataFrame({
        "dates": pd.date_range("2021-01-01", periods=3),
        "transect1": [0.5, 0.6, 0.7]
    })
    model_region_directory = "path/to/model/region"
    transect_id = "transect1"

    # Call the function
    # Mock the model_tides function
    with patch('coastseg.tide_correction.model_tides') as mock_model_tides:
        mock_model_tides.return_value = pd.DataFrame({"tide": [1.0, 2.0, 3.0],
                                                      "transect_id": np.repeat(transect_id,3),
                                                      "dates":timeseries_df.dates,
                                                      "x": np.repeat(x, len(timeseries_df)),
                                                      "y": np.repeat(y, len(timeseries_df))})
        result = get_tide_predictions(x, y, timeseries_df, model_region_directory, transect_id)
        # Assertions
        assert 'tide' in result.columns, "Result should contain 'tide' column"
        assert 'transect_id' in result.columns, "Result should contain 'transect id' column"
        assert 'dates' in result.columns, "Result should contain 'dates' column"
        assert 'x' in result.columns, "Result should contain 'x' column"
        assert 'y' in result.columns, "Result should contain 'y' column"
        
        assert len(result) == 3, "Result should have 3 rows of predictions"
        assert mock_model_tides.called, "model_tides function should be called"
        assert (result['x'] == x).all(), "All 'x' values should be equal to input x"
        assert (result['y'] == y).all(), "All 'y' values should be equal to input y"
        assert (result['transect_id'] == transect_id).all(), "All 'transect_id' values should be equal to input transect_id"
