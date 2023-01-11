import pytest
from coastseg import extracted_shoreline
import geopandas as gpd


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
        extracted_shoreline.Extracted_Shoreline(
            invalid_roi_id, shoreline, roi_settings, settings
        )

    # Test initialize Extracted_Shoreline with invalid shoreline
    with pytest.raises(ValueError):
        extracted_shoreline.Extracted_Shoreline(
            roi_id, invalid_shoreline, roi_settings, settings
        )
    # Test initialize Extracted_Shoreline with empty shoreline
    with pytest.raises(ValueError):
        extracted_shoreline.Extracted_Shoreline(
            roi_id, empty_shoreline, roi_settings, settings
        )

    # Test initialize Extracted_Shoreline with invalid roi_settings
    with pytest.raises(ValueError):
        extracted_shoreline.Extracted_Shoreline(
            roi_id, shoreline, invalid_roi_settings, settings
        )
    # Test initialize Extracted_Shoreline with empty roi_settings
    with pytest.raises(ValueError):
        extracted_shoreline.Extracted_Shoreline(roi_id, shoreline, {}, settings)

    # Test initialize Extracted_Shoreline with invalid settings
    with pytest.raises(ValueError):
        extracted_shoreline.Extracted_Shoreline(
            roi_id, shoreline, roi_settings, invalid_settings
        )
    # Test initialize Extracted_Shoreline with empty settings
    with pytest.raises(ValueError):
        extracted_shoreline.Extracted_Shoreline(roi_id, shoreline, roi_settings, {})

