import pytest
from coastseg import extracted_shoreline
import geopandas as gpd
import numpy as np


def test_is_list_empty():
    # empty list to test if it detects it as empty
    empty_list = [np.ndarray(shape=(0)), np.ndarray(shape=(0))]
    assert extracted_shoreline.is_list_empty(empty_list) == True
    # half empty list to test if it detects it as not empty
    non_empty_list = [np.ndarray(shape=(1, 2)), np.ndarray(shape=(0))]
    assert extracted_shoreline.is_list_empty(non_empty_list) == False
    # full list to test if it detects it as not empty
    non_empty_list = [np.ndarray(shape=(2)), np.ndarray(shape=(2))]
    assert extracted_shoreline.is_list_empty(non_empty_list) == False


def test_get_colors():
    length = 4
    actual_list = extracted_shoreline.get_colors(length)
    assert len(actual_list) == length
    assert isinstance(actual_list, list)
    assert isinstance(actual_list[0], str)


def test_init_invalid_inputs(valid_roi_settings, valid_shoreline_gdf, valid_settings):
    # Test initialize Extracted_Shoreline with invalid ROI id
    invalid_roi_id = 4
    roi_id = "2"
    invalid_shoreline = None
    empty_shoreline = gpd.GeoDataFrame()
    shoreline = valid_shoreline_gdf
    invalid_roi_settings = None
    roi_settings = valid_roi_settings
    invalid_settings = None
    settings = valid_settings

    with pytest.raises(ValueError):
        extracted_shorelines = extracted_shoreline.Extracted_Shoreline()               
        extracted_shorelines.create_extracted_shorlines(
            invalid_roi_id, shoreline, roi_settings, settings
        )


    # Test initialize Extracted_Shoreline with invalid shoreline
    with pytest.raises(ValueError):
        extracted_shorelines = extracted_shoreline.Extracted_Shoreline()               
        extracted_shorelines.create_extracted_shorlines(
            roi_id, invalid_shoreline, roi_settings, settings
        )
    # Test initialize Extracted_Shoreline with empty shoreline
    with pytest.raises(ValueError):
        extracted_shorelines = extracted_shoreline.Extracted_Shoreline()               
        extracted_shorelines.create_extracted_shorlines(
            roi_id, empty_shoreline, roi_settings, settings
        )

    # Test initialize Extracted_Shoreline with invalid roi_settings
    with pytest.raises(ValueError):
        extracted_shorelines = extracted_shoreline.Extracted_Shoreline()               
        extracted_shorelines.create_extracted_shorlines(
            roi_id, shoreline, invalid_roi_settings, settings
        )       
    # Test initialize Extracted_Shoreline with empty roi_settings
    with pytest.raises(ValueError):
        extracted_shorelines = extracted_shoreline.Extracted_Shoreline()               
        extracted_shorelines.create_extracted_shorlines(
            roi_id, shoreline, {}, settings
        )     
    # Test initialize Extracted_Shoreline with invalid settings
    with pytest.raises(ValueError):
        extracted_shorelines = extracted_shoreline.Extracted_Shoreline()               
        extracted_shorelines.create_extracted_shorlines(
             roi_id, shoreline, roi_settings, invalid_settings
        ) 
    # Test initialize Extracted_Shoreline with empty settings
    with pytest.raises(ValueError):
        extracted_shorelines = extracted_shoreline.Extracted_Shoreline()               
        extracted_shorelines.create_extracted_shorlines(
             roi_id, shoreline, roi_settings, {}
        ) 
