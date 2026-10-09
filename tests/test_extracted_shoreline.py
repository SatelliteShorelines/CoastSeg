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
        extracted_shorelines.create_extracted_shorelines(
            invalid_roi_id, shoreline, roi_settings, settings
        )

    # Test initialize Extracted_Shoreline with invalid shoreline
    with pytest.raises(ValueError):
        extracted_shorelines = extracted_shoreline.Extracted_Shoreline()
        extracted_shorelines.create_extracted_shorelines(
            roi_id, invalid_shoreline, roi_settings, settings
        )
    # Test initialize Extracted_Shoreline with empty shoreline
    with pytest.raises(ValueError):
        extracted_shorelines = extracted_shoreline.Extracted_Shoreline()
        extracted_shorelines.create_extracted_shorelines(
            roi_id, empty_shoreline, roi_settings, settings
        )

    # Test initialize Extracted_Shoreline with invalid roi_settings
    with pytest.raises(ValueError):
        extracted_shorelines = extracted_shoreline.Extracted_Shoreline()
        extracted_shorelines.create_extracted_shorelines(
            roi_id, shoreline, invalid_roi_settings, settings
        )
    # Test initialize Extracted_Shoreline with empty roi_settings
    with pytest.raises(ValueError):
        extracted_shorelines = extracted_shoreline.Extracted_Shoreline()
        extracted_shorelines.create_extracted_shorelines(
            roi_id, shoreline, {}, settings
        )
    # Test initialize Extracted_Shoreline with invalid settings
    with pytest.raises(ValueError):
        extracted_shorelines = extracted_shoreline.Extracted_Shoreline()
        extracted_shorelines.create_extracted_shorelines(
            roi_id, shoreline, roi_settings, invalid_settings
        )
    # Test initialize Extracted_Shoreline with empty settings
    with pytest.raises(ValueError):
        extracted_shorelines = extracted_shoreline.Extracted_Shoreline()
        extracted_shorelines.create_extracted_shorelines(
            roi_id, shoreline, roi_settings, {}
        )


def shoreline_crossing_roi_twice():
    """A reference shoreline clipped to an ROI it leaves and re-enters, so the
    clip returns two separate lines."""
    from shapely.geometry import LineString, box

    crs = "EPSG:32618"
    roi = gpd.GeoDataFrame(geometry=[box(0, 0, 1000, 1000)], crs=crs)
    # runs along the bottom, leaves through the right edge, comes back in at the top
    line = LineString(
        [(100, 100), (900, 100), (1500, 100), (1500, 900), (900, 900), (100, 900)]
    )
    shoreline = gpd.GeoDataFrame(geometry=[line], crs=crs)
    return gpd.clip(shoreline, roi), crs


def test_get_reference_shoreline_parts_keeps_separate_lines_apart():
    clipped, crs = shoreline_crossing_roi_twice()

    parts = extracted_shoreline.get_reference_shoreline_parts(clipped, crs)

    assert len(parts) == 2
    for part in parts:
        assert part.ndim == 2 and part.shape[1] == 3
        assert (part[:, 2] == 0).all()
        # neither line strays outside the ROI to reach the other one
        assert part[:, 0].max() <= 1000
    # one line along the bottom of the ROI and one along the top
    assert sorted(float(part[:, 1].mean()) for part in parts) == [100.0, 900.0]


def test_get_reference_shoreline_parts_holds_the_same_points_as_the_flat_array():
    clipped, crs = shoreline_crossing_roi_twice()

    parts = extracted_shoreline.get_reference_shoreline_parts(clipped, crs)
    flat = extracted_shoreline.get_reference_shoreline(clipped, crs)

    assert np.array_equal(np.vstack(parts), flat)


def test_get_reference_shoreline_parts_single_line_is_one_part():
    from shapely.geometry import LineString

    crs = "EPSG:32618"
    shoreline = gpd.GeoDataFrame(
        geometry=[LineString([(100, 100), (500, 300), (900, 100)])], crs=crs
    )

    parts = extracted_shoreline.get_reference_shoreline_parts(shoreline, crs)

    assert len(parts) == 1
    assert np.array_equal(
        parts[0], extracted_shoreline.get_reference_shoreline(shoreline, crs)
    )


def test_get_reference_shoreline_parts_drops_z_coordinates():
    from shapely.geometry import LineString

    crs = "EPSG:32618"
    shoreline = gpd.GeoDataFrame(
        geometry=[LineString([(100, 100, 5), (500, 300, 5), (900, 100, 5)])], crs=crs
    )

    parts = extracted_shoreline.get_reference_shoreline_parts(shoreline, crs)

    assert parts[0].shape == (3, 3)
    assert (parts[0][:, 2] == 0).all()


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({"pan_off": "False"}, False),
        ({"pan_off": "True"}, True),
        ({"pan_off": False}, False),
        ({"pan_off": True}, True),
        ({}, False),
    ],
)
def test_create_shoreline_settings_passes_pan_off_as_a_boolean(settings, expected):
    """coastsat skips pansharpening when pan_off is truthy, and the string "False" is."""
    shoreline_settings = extracted_shoreline.Extracted_Shoreline().create_shoreline_settings(
        settings,
        {"roi_id": "abc1", "sitename": "site", "filepath": "data"},
        reference_shoreline={},
    )
    assert shoreline_settings["pan_off"] is expected


@pytest.mark.parametrize("pan_off, expected", [("False", False), ("True", True)])
def test_zoo_model_set_settings_stores_pan_off_as_a_boolean(pan_off, expected):
    from coastseg import zoo_model

    model = zoo_model.Zoo_Model()
    assert model.get_settings()["pan_off"] is False
    assert model.set_settings(pan_off=pan_off)["pan_off"] is expected
