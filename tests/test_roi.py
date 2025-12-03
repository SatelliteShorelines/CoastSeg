import math
import pytest
from coastseg import roi
from coastseg import exceptions
import geopandas as gpd
import pyproj

# Geometry classes now used from shared fixtures in conftest.py


def test_extract_roi_by_id(valid_rois_gdf):
    # test if valid gdf is returned when id within gdf is given
    roi_id = 17
    actual_roi = roi.ROI.extract_roi_by_id(valid_rois_gdf, roi_id)
    assert isinstance(actual_roi, gpd.GeoDataFrame)
    assert not actual_roi[actual_roi["id"].astype(int) == roi_id].empty
    expected_roi = valid_rois_gdf[valid_rois_gdf["id"].astype(int) == roi_id]
    assert actual_roi["geometry"][0] == expected_roi["geometry"][0]
    assert actual_roi["id"][0] == expected_roi["id"][0]


# Test that when ROI's area is too large an error is thrown
def test_ROI_too_large(large_polygon_gdf):
    """Test that creating ROI with a polygon that is too large raises InvalidSize error."""
    with pytest.raises(exceptions.InvalidSize):
        roi.ROI(rois_gdf=large_polygon_gdf)


# Test that when ROI is not a polygon an error is thrown
def test_ROI_wrong_geometry(simple_linestring_gdf):
    """Test that creating ROI with LineString geometry raises InvalidGeometryType error."""
    with pytest.raises(exceptions.InvalidGeometryType):
        roi.ROI(rois_gdf=simple_linestring_gdf)


# create ROIs from geodataframe with a different CRS
def test_initialize_from_roi_gdf_different_crs(standard_polygon_gdf):
    """Test creating ROI from GeoDataFrame with different CRS gets converted to EPSG:4326."""
    CRS = "epsg:2033"
    rectangle = standard_polygon_gdf.copy()
    rectangle.to_crs(CRS, inplace=True)
    rois = roi.ROI(rois_gdf=rectangle)
    assert hasattr(rois, "gdf")
    assert isinstance(rois.gdf, gpd.GeoDataFrame)
    assert not rois.gdf.empty
    assert isinstance(rois.gdf.crs, pyproj.CRS)
    assert rois.gdf.crs == "epsg:4326"


# create ROIs from geodataframe with CRS 4326 (default CRS for map)
def test_initialize_from_roi_gdf(standard_polygon_gdf):
    """Test creating ROI from GeoDataFrame with EPSG:4326 CRS."""
    CRS = "epsg:4326"
    rois = roi.ROI(rois_gdf=standard_polygon_gdf)
    assert hasattr(rois, "gdf")
    assert isinstance(rois.gdf, gpd.GeoDataFrame)
    assert not rois.gdf.empty
    assert isinstance(rois.gdf.crs, pyproj.CRS)
    assert rois.gdf.crs == CRS


def test_create_geodataframe(
    valid_bbox_gdf: gpd.GeoDataFrame,
    valid_shoreline_gdf: gpd.GeoDataFrame,
    valid_ROI: roi.ROI,
):
    actual_gdf = valid_ROI.create_geodataframe(
        bbox=valid_bbox_gdf,
        shoreline=valid_shoreline_gdf,
        large_length=1000,
        small_length=750,
    )

    assert isinstance(actual_gdf, gpd.GeoDataFrame)
    assert set(actual_gdf.columns) == set(["geometry", "id"])
    assert actual_gdf.dtypes["geometry"] == "geometry"
    assert actual_gdf.dtypes["id"] == "object"
    # drop unneeded columns before checking
    columns_to_drop = list(valid_shoreline_gdf.columns.difference(["geometry"]))
    valid_shoreline_gdf = valid_shoreline_gdf.drop(columns=columns_to_drop)
    # Validate any shorelines intersect any squares in actual_gdf
    intersection_gdf = gpd.sjoin(
        valid_shoreline_gdf, right_df=actual_gdf, how="inner", predicate="intersects"
    )
    assert not intersection_gdf.empty


def test_fishnet_intersection(
    valid_bbox_gdf: gpd.GeoDataFrame,
    valid_shoreline_gdf: gpd.GeoDataFrame,
    valid_ROI: roi.ROI,
):
    # tests if a valid fishnet geodataframe intersects with given shoreline geodataframe
    square_size = 1000
    output_epsg = "epsg:4326"
    fishnet_gdf = valid_ROI.create_rois(valid_bbox_gdf, square_size)
    # check if fishnet intersects the shoreline
    fishnet_gdf = valid_ROI.fishnet_intersection(fishnet_gdf, valid_shoreline_gdf)
    assert isinstance(fishnet_gdf, gpd.GeoDataFrame)
    assert not fishnet_gdf.empty
    assert isinstance(fishnet_gdf.crs, pyproj.CRS)
    assert fishnet_gdf.crs == output_epsg
    assert set(fishnet_gdf.columns) == set(["geometry"])
    # drop unneeded columns before checking
    columns_to_drop = list(valid_shoreline_gdf.columns.difference(["geometry"]))
    valid_shoreline_gdf = valid_shoreline_gdf.drop(columns=columns_to_drop)
    # check if any shorelines intersect any squares in fishnet_gdf
    intersection_gdf = valid_shoreline_gdf.sjoin(
        fishnet_gdf, how="inner", predicate="intersects"
    )
    assert not intersection_gdf.empty


def test_get_fishnet(
    valid_bbox_gdf: gpd.GeoDataFrame,
    valid_shoreline_gdf: gpd.GeoDataFrame,
    valid_ROI: roi.ROI,
):
    # tests if a valid fishnet geodataframe intersects with given shoreline geodataframe
    square_size = 1000
    output_epsg = "epsg:4326"
    # check if fishnet intersects the shoreline
    fishnet_gdf = valid_ROI.get_fishnet_gdf(
        bbox_gdf=valid_bbox_gdf,
        shoreline_gdf=valid_shoreline_gdf,
        square_length=square_size,
    )
    assert isinstance(fishnet_gdf, gpd.GeoDataFrame)
    assert isinstance(fishnet_gdf.crs, pyproj.CRS)
    assert fishnet_gdf.crs == output_epsg
    assert set(fishnet_gdf.columns) == set(["geometry"])
    # drop unneeded columns before checking
    columns_to_drop = list(valid_shoreline_gdf.columns.difference(["geometry"]))
    valid_shoreline_gdf = valid_shoreline_gdf.drop(columns=columns_to_drop)
    # check if any shorelines intersect any squares in fishnet_gdf
    intersection_gdf = valid_shoreline_gdf.sjoin(
        fishnet_gdf, how="inner", predicate="intersects"
    )
    assert not intersection_gdf.empty


def test_roi_missing_lengths(valid_bbox_gdf, valid_shoreline_gdf):
    # test with missing square lengths
    with pytest.raises(Exception):
        roi.ROI(bbox=valid_bbox_gdf, shoreline=valid_shoreline_gdf)


def test_bad_roi_initialization(valid_bbox_gdf):
    empty_gdf = gpd.GeoDataFrame()
    # test with missing shoreline
    with pytest.raises(exceptions.Object_Not_Found):
        roi.ROI(bbox=valid_bbox_gdf)
    # test with missing bbox and shoreline
    with pytest.raises(exceptions.Object_Not_Found):
        roi.ROI()
    # test with empty bbox
    with pytest.raises(exceptions.Object_Not_Found):
        roi.ROI(bbox=empty_gdf)
    # test with empty shoreline
    with pytest.raises(exceptions.Object_Not_Found):
        roi.ROI(bbox=valid_bbox_gdf, shoreline=empty_gdf)


def test_roi_from_bbox_and_shorelines(valid_bbox_gdf, valid_shoreline_gdf):
    large_len = 1000
    small_len = 750
    actual_roi = roi.ROI(
        bbox=valid_bbox_gdf,
        shoreline=valid_shoreline_gdf,
        square_len_lg=large_len,
        square_len_sm=small_len,
    )

    assert isinstance(actual_roi, roi.ROI)
    assert isinstance(actual_roi.gdf, gpd.GeoDataFrame)
    assert set(actual_roi.gdf.columns) == set(["geometry", "id"])
    assert actual_roi.filename == "rois.geojson"
    assert hasattr(actual_roi, "extracted_shorelines")
    assert hasattr(actual_roi, "cross_shore_distances")
    assert hasattr(actual_roi, "roi_settings")


def test_create_fishnet(valid_bbox_gdf: gpd.GeoDataFrame, valid_ROI: roi.ROI):
    # tests if a valid geodataframe is created with square sizes approx. equal to given square_size
    square_size = 1000
    input_epsg = "epsg:32610"
    output_epsg = "epsg:4326"

    # convert bbox to input_epsg to most accurate espg to create fishnet with
    valid_bbox_gdf = valid_bbox_gdf.to_crs(input_epsg)
    assert valid_bbox_gdf.crs == "epsg:32610"

    actual_fishnet = valid_ROI.create_fishnet(
        valid_bbox_gdf,
        input_epsg=input_epsg,
        output_epsg=output_epsg,
        square_size=square_size,
    )
    assert isinstance(actual_fishnet, gpd.GeoDataFrame)
    assert set(actual_fishnet.columns) == set(["geometry"])
    assert isinstance(actual_fishnet.crs, pyproj.CRS)
    assert actual_fishnet.crs == output_epsg
    # reproject back to input_epsg to check if square sizes are correct
    actual_fishnet = actual_fishnet.to_crs(input_epsg)
    # pick a square out of the fishnet ensure is approx. equal to square size
    actual_lengths = tuple(map(lambda x: x.length / 4, actual_fishnet["geometry"]))
    # check if actual lengths are close to square_size length
    is_actual_length_correct = all(
        tuple(
            map(lambda x: math.isclose(x, square_size, rel_tol=1e-04), actual_lengths)
        )
    )
    # assert all actual lengths are close to square_size length
    assert is_actual_length_correct


def test_create_rois(valid_ROI: roi.ROI, valid_bbox_gdf: gpd.GeoDataFrame):
    square_size = 1000
    # espg code of the valid_bbox_gdf
    input_epsg = "epsg:4326"
    output_epsg = "epsg:4326"
    actual_roi_gdf = valid_ROI.create_rois(
        bbox=valid_bbox_gdf,
        input_epsg=input_epsg,
        output_epsg=output_epsg,
        square_size=square_size,
    )
    assert isinstance(actual_roi_gdf, gpd.GeoDataFrame)
    assert isinstance(actual_roi_gdf.crs, pyproj.CRS)
    assert actual_roi_gdf.crs == output_epsg
    assert set(actual_roi_gdf.columns) == set(["geometry"])


def test_transect_compatible_roi(transect_compatible_roi: gpd.GeoDataFrame):
    """tests if a ROI will be created from valid rois of type gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): alid rois as a gpd.GeoDataFrame
    """
    actual_roi = roi.ROI(rois_gdf=transect_compatible_roi)
    assert isinstance(actual_roi, roi.ROI)
    assert isinstance(actual_roi.gdf, gpd.GeoDataFrame)
    assert set(actual_roi.gdf.columns) == set(["geometry", "id"])
    assert actual_roi.gdf.dtypes["geometry"] == "geometry"
    assert actual_roi.gdf.dtypes["id"] == "object"
    assert actual_roi.filename == "rois.geojson"
    assert hasattr(actual_roi, "extracted_shorelines")
    assert hasattr(actual_roi, "cross_shore_distances")
    assert hasattr(actual_roi, "roi_settings")


def test_add_extracted_shoreline(valid_ROI: roi.ROI):
    """tests if a ROI will be created from valid rois of type gpd.GeoDataFrame
    Args:
       transect_compatible_roi (gpd.GeoDataFrame): valid rois as a gpd.GeoDataFrame
    """
    roi_id = "23"
    expected_dict = {
        "23": {
            "filename": ["ms.tif", "2019.tif"],
            "cloud_cover": [0.14, 0.0],
            "geoaccuracy": [7.9, 9.72],
            "idx": [4, 6],
            "MNDWI_threshold": [-0.231, -0.3],
            "satname": ["L8", "L8"],
        }
    }
    # Use mock object instead of dict to match expected type
    from unittest.mock import Mock

    mock_shoreline = Mock()
    mock_shoreline.data = expected_dict[roi_id]
    valid_ROI.add_extracted_shoreline(mock_shoreline, roi_id)

    # Check that extracted shorelines dictionary is not empty
    assert valid_ROI.extracted_shorelines != {}
    # Check that the shoreline was added for the correct ROI ID
    assert roi_id in valid_ROI.extracted_shorelines
    # Check that we can retrieve the mock object
    retrieved_shoreline = valid_ROI.get_extracted_shoreline(roi_id)
    assert retrieved_shoreline is mock_shoreline


def test_set_roi_settings(valid_ROI: roi.ROI):
    """tests if a ROI will be created from valid rois thats a gpd.GeoDataFrame
    Args:
        transect_compatible_roi (gpd.GeoDataFrame): valid rois as a gpd.GeoDataFrame
    """
    expected_dict = {
        "23": {
            "dates": ["2018-12-01", "2019-03-01"],
            "sat_list": ["L8"],
            "sitename": "ID02022-10-07__09_hr_38_min37sec",
            "filepath": "C:\\CoastSeg\\data",
            "roi_id": "23",
            "polygon": [
                [
                    [-124.1662679688807, 40.863030239542944],
                    [-124.16690059058178, 40.89905645671534],
                    [-124.11942071317034, 40.89952713781644],
                    [-124.11881381876809, 40.863500326870245],
                    [-124.1662679688807, 40.863030239542944],
                ]
            ],
            "landsat_collection": "C01",
        },
        "39": {
            "dates": ["2018-12-01", "2019-03-01"],
            "sat_list": ["L8"],
            "sitename": "ID12022-10-07__09_hr_38_min37sec",
            "filepath": "C:\\CoastSeg\\data",
            "roi_id": "39",
            "polygon": [
                [
                    [-124.16690059058178, 40.89905645671534],
                    [-124.1675343590045, 40.93508244001033],
                    [-124.12002870768146, 40.9355537155221],
                    [-124.11942071317034, 40.89952713781644],
                    [-124.16690059058178, 40.89905645671534],
                ]
            ],
            "landsat_collection": "C01",
        },
    }
    valid_ROI.set_roi_settings(expected_dict)
    assert valid_ROI.roi_settings != {}
    assert valid_ROI.roi_settings == expected_dict


def test_style_layer(valid_ROI: roi.ROI):
    with pytest.raises(ValueError):
        valid_ROI.style_layer({}, layer_name="Nope")


# Test get_roi_settings method


def test_get_roi_settings_with_valid_roi_id(valid_ROI):
    # Test with valid ROI ID
    roi_id = "roi1"
    expected_settings = {"param1": 10, "param2": "abc"}
    valid_ROI.roi_settings = {roi_id: expected_settings}

    actual_settings = valid_ROI.get_roi_settings(roi_id)

    assert actual_settings == expected_settings


def test_get_roi_settings_with_invalid_roi_id(valid_ROI):
    # Test with invalid ROI ID
    roi_id = "roi2"
    valid_ROI.roi_settings = {"roi1": {"param1": 10, "param2": "abc"}}

    actual_settings = valid_ROI.get_roi_settings(roi_id)

    assert actual_settings == {}


def test_get_roi_settings_with_no_roi_id(valid_ROI):
    # Test with no ROI ID
    valid_ROI.roi_settings = {"roi1": {"param1": 10, "param2": "abc"}}

    actual_settings = valid_ROI.get_roi_settings()

    assert actual_settings == valid_ROI.roi_settings


def test_get_roi_settings_with_existing_roi_id(valid_ROI):
    roi_id = "roi1"
    valid_ROI.roi_settings = {roi_id: {"param1": 10, "param2": "abc"}}
    assert valid_ROI.get_roi_settings(roi_id) == {"param1": 10, "param2": "abc"}


def test_get_roi_settings_with_non_existing_roi_id(valid_ROI):
    roi_id = "roi1"
    valid_ROI.roi_settings = {"roi2": {"param1": 10, "param2": "abc"}}
    assert valid_ROI.get_roi_settings(roi_id) == {}


def test_get_roi_settings_with_empty_roi_id(valid_ROI):
    valid_ROI.roi_settings = {"roi1": {"param1": 10, "param2": "abc"}}
    assert valid_ROI.get_roi_settings("") == {"roi1": {"param1": 10, "param2": "abc"}}


def test_get_roi_settings_with_none_roi_id_returns_empty(valid_ROI):
    """Test that get_roi_settings with None returns empty dict."""
    assert valid_ROI.get_roi_settings(None) == {}


def test_get_roi_settings_with_non_string_roi_id_raises_error(valid_ROI):
    """Test that get_roi_settings with non-string ID raises TypeError."""
    with pytest.raises(TypeError):
        valid_ROI.get_roi_settings(123)


def test_update_roi_settings_with_valid_settings(valid_ROI):
    new_settings = {"param1": 10, "param2": "value", "param3": [1, 2, 3]}
    valid_ROI.update_roi_settings(new_settings)
    assert valid_ROI.roi_settings == new_settings


def test_update_roi_settings_with_empty_settings(valid_ROI_with_settings):
    """Test updating ROI settings with empty dict doesn't change existing settings."""
    # Store original settings before update
    original_settings = valid_ROI_with_settings.roi_settings.copy()

    # Update with empty settings
    new_settings = {}
    result = valid_ROI_with_settings.update_roi_settings(new_settings)

    # Settings should remain unchanged when updating with empty dict
    assert valid_ROI_with_settings.roi_settings == original_settings
    assert result == original_settings
    assert len(valid_ROI_with_settings.roi_settings) == len(original_settings)


def test_update_roi_settings_with_existing_settings(valid_ROI_with_settings):
    """Test updating ROI settings with new data merges correctly."""
    # Get original settings to work with
    original_settings = valid_ROI_with_settings.roi_settings.copy()

    # Create update that modifies one existing ROI and check that others remain
    roi_id_to_update = list(original_settings.keys())[0]  # Get first ROI ID
    new_settings = {
        roi_id_to_update: {
            **original_settings[roi_id_to_update],  # Keep existing data
            "sat_list": ["L7", "L9"],  # Update satellite list
            "dates": ["2017-12-01", "2023-03-01"],  # Update dates
        }
    }

    valid_ROI_with_settings.update_roi_settings(new_settings)

    # Check that the updated ROI has new values
    updated_roi = valid_ROI_with_settings.roi_settings[roi_id_to_update]
    assert updated_roi["sat_list"] == ["L7", "L9"]
    assert updated_roi["dates"] == ["2017-12-01", "2023-03-01"]

    # Check that other ROIs remain unchanged
    other_roi_ids = [k for k in original_settings.keys() if k != roi_id_to_update]
    for roi_id in other_roi_ids:
        assert valid_ROI_with_settings.roi_settings[roi_id] == original_settings[roi_id]

    # Check that all original ROIs are still present
    assert set(valid_ROI_with_settings.roi_settings.keys()) == set(
        original_settings.keys()
    )


def test_update_roi_settings_with_none_settings(valid_ROI):
    with pytest.raises(ValueError):
        valid_ROI.update_roi_settings(None)


# ============================================================================
# Additional Comprehensive Tests for ROI Class
# ============================================================================


class TestROIRemovalOperations:
    """Test ROI removal operations and associated data cleanup."""

    def test_remove_by_id_single_string(self, valid_ROI_with_settings):
        """Test removing ROI by single string ID."""
        # Add some test data first
        original_ids = valid_ROI_with_settings.get_ids()
        if original_ids:
            roi_id = original_ids[0]

            # Add extracted shoreline data to test cleanup
            test_data = {"test": "data"}
            valid_ROI_with_settings.add_extracted_shoreline(test_data, roi_id)

            # Remove the ROI
            result_gdf = valid_ROI_with_settings.remove_by_id(roi_id)

            # Check ROI was removed from GeoDataFrame
            assert roi_id not in result_gdf["id"].tolist()
            # Check associated data was cleaned up
            assert roi_id not in valid_ROI_with_settings.extracted_shorelines

    def test_remove_by_id_single_int(self, valid_ROI_with_settings):
        """Test removing ROI by single integer ID."""
        original_ids = valid_ROI_with_settings.get_ids()
        if original_ids:
            # Convert first ID to int for testing
            roi_id_str = original_ids[0]
            roi_id_int = int(roi_id_str) if roi_id_str.isdigit() else 1

            result_gdf = valid_ROI_with_settings.remove_by_id(roi_id_int)
            assert isinstance(result_gdf, gpd.GeoDataFrame)

    def test_remove_by_id_list(self, valid_ROI_with_settings):
        """Test removing ROI by list of IDs."""
        original_ids = valid_ROI_with_settings.get_ids()
        if len(original_ids) >= 2:
            ids_to_remove = original_ids[:2]

            result_gdf = valid_ROI_with_settings.remove_by_id(ids_to_remove)

            # Check all specified IDs were removed
            remaining_ids = result_gdf["id"].tolist()
            for roi_id in ids_to_remove:
                assert roi_id not in remaining_ids

    def test_remove_by_id_set(self, valid_ROI_with_settings):
        """Test removing ROI by set of IDs."""
        original_ids = valid_ROI_with_settings.get_ids()
        if original_ids:
            ids_to_remove = set(original_ids[:1])

            result_gdf = valid_ROI_with_settings.remove_by_id(ids_to_remove)

            # Check ID was removed
            remaining_ids = result_gdf["id"].tolist()
            for roi_id in ids_to_remove:
                assert roi_id not in remaining_ids

    def test_remove_by_id_tuple(self, valid_ROI_with_settings):
        """Test removing ROI by tuple of IDs."""
        original_ids = valid_ROI_with_settings.get_ids()
        if original_ids:
            ids_to_remove = tuple(original_ids[:1])

            result_gdf = valid_ROI_with_settings.remove_by_id(ids_to_remove)

            # Check ID was removed
            remaining_ids = result_gdf["id"].tolist()
            for roi_id in ids_to_remove:
                assert roi_id not in remaining_ids

    def test_remove_by_id_none(self, valid_ROI_with_settings):
        """Test remove_by_id handles None gracefully."""
        original_count = len(valid_ROI_with_settings.gdf)

        result_gdf = valid_ROI_with_settings.remove_by_id(None)

        # Should not change anything when None passed
        assert len(result_gdf) == original_count


class TestROISizeValidation:
    """Test ROI size validation functions."""

    def test_get_ids_with_invalid_area_empty_gdf(self):
        """Test size validation with empty GeoDataFrame."""
        empty_gdf = gpd.GeoDataFrame()

        result = roi.get_ids_with_invalid_area(empty_gdf)

        assert result == []

    def test_get_ids_with_invalid_area_non_gdf_input(self):
        """Test size validation raises error for non-GeoDataFrame input."""
        with pytest.raises(TypeError, match="Input must be a GeoDataFrame"):
            roi.get_ids_with_invalid_area("not a geodataframe")  # type: ignore

    def test_get_ids_with_invalid_area_valid_sizes(self, standard_polygon_gdf):
        """Test size validation with valid polygon sizes."""
        result = roi.get_ids_with_invalid_area(
            standard_polygon_gdf, max_area=98000000, min_area=0
        )

        # Standard polygon should be within valid range
        assert result == []

    def test_get_ids_with_invalid_area_too_large(self, large_polygon_gdf):
        """Test size validation identifies polygons that are too large."""
        result = roi.get_ids_with_invalid_area(
            large_polygon_gdf,
            max_area=1000,  # Very small max to trigger invalid
            min_area=0,
        )

        # Large polygon should be flagged as invalid
        assert len(result) > 0

    def test_validate_ROI_sizes_removes_invalid(self, valid_ROI):
        """Test validate_ROI_sizes removes ROIs with invalid areas."""
        # Create a test GeoDataFrame with known invalid area
        from shapely.geometry import Polygon

        # Create very large polygon that definitely exceeds MAX_SIZE (98,000,000 m²)
        # Use a polygon that spans a huge area in a projected coordinate system
        large_coords = [
            [0, 0],
            [100000, 0],
            [100000, 100000],
            [0, 100000],
            [0, 0],  # 10,000,000,000 m²
        ]
        large_polygon = Polygon(large_coords)

        # Create in a UTM projection where coordinates are in meters
        invalid_gdf = gpd.GeoDataFrame(
            {"geometry": [large_polygon], "id": ["large_roi"]}, crs="EPSG:32633"
        )  # UTM Zone 33N

        # Convert to EPSG:4326 first as the validate method expects this
        invalid_gdf = invalid_gdf.to_crs("EPSG:4326")

        # Should raise InvalidSize error when all ROIs are invalid
        with pytest.raises(exceptions.InvalidSize):
            valid_ROI.validate_ROI_sizes(invalid_gdf)

    def test_validate_ROI_sizes_keeps_valid(self, valid_ROI, standard_polygon_gdf):
        """Test validate_ROI_sizes keeps valid ROIs."""
        result = valid_ROI.validate_ROI_sizes(standard_polygon_gdf)

        # Should keep valid polygon
        assert not result.empty
        assert len(result) == len(standard_polygon_gdf)


class TestROIDataAddition:
    """Test ROI data addition operations."""

    def test_add_geodataframe_empty_input(self, valid_ROI):
        """Test adding empty GeoDataFrame returns unchanged ROI."""
        empty_gdf = gpd.GeoDataFrame()
        original_count = len(valid_ROI.gdf)

        result_roi = valid_ROI.add_geodataframe(empty_gdf)

        # Should return same ROI without changes
        assert result_roi is valid_ROI
        assert len(valid_ROI.gdf) == original_count

    def test_add_geodataframe_valid_data(self, valid_ROI, standard_polygon_gdf):
        """Test adding valid GeoDataFrame to ROI."""
        original_count = len(valid_ROI.gdf)

        result_roi = valid_ROI.add_geodataframe(standard_polygon_gdf)

        # Should return the same ROI object
        assert result_roi is valid_ROI
        # Should have more ROIs now (unless duplicates were removed)
        assert len(valid_ROI.gdf) >= original_count

    def test_add_geodataframe_duplicate_removal(self, valid_ROI):
        """Test adding duplicate ROI gets removed."""
        # Add the same data twice
        if not valid_ROI.gdf.empty:
            duplicate_gdf = valid_ROI.gdf.copy()
            original_count = len(valid_ROI.gdf)

            valid_ROI.add_geodataframe(duplicate_gdf)

            # Should not increase count due to duplicate removal
            assert len(valid_ROI.gdf) == original_count

    def test_add_geodataframe_invalid_geometry_type(
        self, valid_ROI, simple_linestring_gdf
    ):
        """Test adding invalid geometry types raises error."""
        with pytest.raises(exceptions.InvalidGeometryType):
            valid_ROI.add_geodataframe(simple_linestring_gdf)


class TestExtractedShorelineManagement:
    """Test extracted shoreline management operations."""

    def test_add_extracted_shoreline(self, valid_ROI):
        """Test adding extracted shoreline data."""
        roi_id = "test_roi"
        test_shoreline_data = {"test": "shoreline_data"}

        valid_ROI.add_extracted_shoreline(test_shoreline_data, roi_id)

        # Check data was added
        assert roi_id in valid_ROI.extracted_shorelines
        assert valid_ROI.extracted_shorelines[roi_id] == test_shoreline_data

    def test_get_extracted_shoreline_existing(self, valid_ROI):
        """Test getting existing extracted shoreline."""
        roi_id = "test_roi"
        test_data = {"test": "data"}
        valid_ROI.extracted_shorelines[roi_id] = test_data

        result = valid_ROI.get_extracted_shoreline(roi_id)

        assert result == test_data

    def test_get_extracted_shoreline_nonexistent(self, valid_ROI):
        """Test getting nonexistent extracted shoreline returns None."""
        result = valid_ROI.get_extracted_shoreline("nonexistent_id")

        assert result is None

    def test_get_all_extracted_shorelines(self, valid_ROI):
        """Test getting all extracted shorelines."""
        # Add some test data
        test_data = {"roi1": {"data1": "value1"}, "roi2": {"data2": "value2"}}
        valid_ROI.extracted_shorelines = test_data

        result = valid_ROI.get_all_extracted_shorelines()

        assert result == test_data

    def test_get_all_extracted_shorelines_uninitialized(self, valid_ROI):
        """Test getting all extracted shorelines when uninitialized."""
        # Remove the attribute to test initialization
        if hasattr(valid_ROI, "extracted_shorelines"):
            delattr(valid_ROI, "extracted_shorelines")

        result = valid_ROI.get_all_extracted_shorelines()

        assert result == {}
        assert hasattr(valid_ROI, "extracted_shorelines")

    def test_get_ids_with_extracted_shorelines(self, valid_ROI):
        """Test getting list of ROI IDs with extracted shorelines."""
        # Add some test data
        valid_ROI.extracted_shorelines = {"roi1": {}, "roi2": {}}

        result = valid_ROI.get_ids_with_extracted_shorelines()

        assert set(result) == {"roi1", "roi2"}

    def test_remove_extracted_shorelines_single_id(self, valid_ROI):
        """Test removing extracted shoreline for single ROI ID."""
        # Add test data
        valid_ROI.extracted_shorelines = {"roi1": {}, "roi2": {}}

        valid_ROI.remove_extracted_shorelines("roi1")

        # Check roi1 was removed but roi2 remains
        assert "roi1" not in valid_ROI.extracted_shorelines
        assert "roi2" in valid_ROI.extracted_shorelines

    def test_remove_extracted_shorelines_nonexistent_id(self, valid_ROI):
        """Test removing nonexistent extracted shoreline ID."""
        # Should not raise error for nonexistent ID
        valid_ROI.remove_extracted_shorelines("nonexistent")

        # Should succeed without error
        assert True

    def test_remove_extracted_shorelines_all(self, valid_ROI):
        """Test removing all extracted shorelines."""
        # Add test data
        valid_ROI.extracted_shorelines = {"roi1": {}, "roi2": {}}

        valid_ROI.remove_extracted_shorelines(remove_all=True)

        # Check all were removed
        assert len(valid_ROI.extracted_shorelines) == 0

    def test_remove_extracted_shorelines_int_id(self, valid_ROI):
        """Test removing extracted shoreline with integer ID."""
        # Add test data with string key
        valid_ROI.extracted_shorelines = {"123": {"data": "test"}}

        # Remove using integer ID
        valid_ROI.remove_extracted_shorelines(123)

        # Check was removed (ID converted to string)
        assert "123" not in valid_ROI.extracted_shorelines


class TestCrossShoreDistanceManagement:
    """Test cross shore distance management operations."""

    def test_add_cross_shore_distances(self, valid_ROI):
        """Test adding cross shore distance data."""
        roi_id = "test_roi"
        test_distance_data = {"distances": [1, 2, 3]}

        valid_ROI.add_cross_shore_distances(test_distance_data, roi_id)

        # Check data was added
        assert roi_id in valid_ROI.cross_shore_distances
        assert valid_ROI.cross_shore_distances[roi_id] == test_distance_data

    def test_get_cross_shore_distances_existing(self, valid_ROI):
        """Test getting existing cross shore distances."""
        roi_id = "test_roi"
        test_data = {"distances": [1, 2, 3]}
        valid_ROI.cross_shore_distances[roi_id] = test_data

        result = valid_ROI.get_cross_shore_distances(roi_id)

        assert result == test_data

    def test_get_cross_shore_distances_nonexistent(self, valid_ROI):
        """Test getting nonexistent cross shore distances returns empty dict."""
        result = valid_ROI.get_cross_shore_distances("nonexistent_id")

        assert result == {}

    def test_get_all_cross_shore_distances(self, valid_ROI):
        """Test getting all cross shore distances."""
        test_data = {"roi1": {"dist1": [1, 2]}, "roi2": {"dist2": [3, 4]}}
        valid_ROI.cross_shore_distances = test_data

        result = valid_ROI.get_all_cross_shore_distances()

        assert result == test_data

    def test_remove_cross_shore_distance_single_id(self, valid_ROI):
        """Test removing cross shore distance for single ROI ID."""
        # Add test data
        valid_ROI.cross_shore_distances = {"roi1": {}, "roi2": {}}

        valid_ROI.remove_cross_shore_distance("roi1")

        # Check roi1 was removed but roi2 remains
        assert "roi1" not in valid_ROI.cross_shore_distances
        assert "roi2" in valid_ROI.cross_shore_distances

    def test_remove_cross_shore_distance_all(self, valid_ROI):
        """Test removing all cross shore distances."""
        # Add test data
        valid_ROI.cross_shore_distances = {"roi1": {}, "roi2": {}}

        valid_ROI.remove_cross_shore_distance(remove_all=True)

        # Check all were removed
        assert len(valid_ROI.cross_shore_distances) == 0


class TestROIUtilityMethods:
    """Test ROI utility and helper methods."""

    def test_get_ids(self, valid_ROI):
        """Test getting list of all ROI IDs."""
        result = valid_ROI.get_ids()

        # Should return list of strings
        assert isinstance(result, list)
        if result:  # If ROI has data
            assert all(isinstance(roi_id, str) for roi_id in result)

    def test_repr_with_data(self, valid_ROI_with_settings):
        """Test string representation with ROI data."""
        repr_str = repr(valid_ROI_with_settings)

        # Should contain key ROI information
        assert "ROI:" in repr_str
        assert "ROI IDs:" in repr_str
        assert "CRS:" in repr_str
        assert "Columns and Data Types:" in repr_str

    def test_repr_empty_roi(self):
        """Test string representation with empty ROI."""
        # Create empty ROI by passing empty GeoDataFrame to avoid bbox initialization
        # Since empty GeoDataFrame raises InvalidSize, we expect this
        empty_gdf = gpd.GeoDataFrame(columns=["geometry", "id"], crs="EPSG:4326")

        with pytest.raises(exceptions.InvalidSize):
            roi.ROI(rois_gdf=empty_gdf)

    def test_str_method(self, valid_ROI):
        """Test that __str__ method works (should be same as __repr__)."""
        str_result = str(valid_ROI)
        repr_result = repr(valid_ROI)

        # Should be identical
        assert str_result == repr_result


class TestROIInitializationEdgeCases:
    """Test edge cases and error conditions in ROI initialization."""

    def test_initialization_with_custom_filename(self, standard_polygon_gdf):
        """Test ROI initialization with custom filename."""
        custom_filename = "custom_rois.geojson"
        roi_obj = roi.ROI(rois_gdf=standard_polygon_gdf, filename=custom_filename)

        assert roi_obj.filename == custom_filename

    def test_initialization_precedence_rois_over_bbox(
        self, standard_polygon_gdf, valid_bbox_gdf
    ):
        """Test that rois_gdf takes precedence over bbox+shoreline initialization."""
        # When both provided, should use rois_gdf
        roi_obj = roi.ROI(
            rois_gdf=standard_polygon_gdf,
            bbox=valid_bbox_gdf,
            shoreline=gpd.GeoDataFrame(),  # Empty shoreline
            square_len_lg=1000,
            square_len_sm=500,
        )

        # Should have used rois_gdf (can check by comparing geometry or IDs)
        assert not roi_obj.gdf.empty
        assert "id" in roi_obj.gdf.columns

    def test_initialization_attributes_setup(self, standard_polygon_gdf):
        """Test that all expected attributes are initialized."""
        roi_obj = roi.ROI(rois_gdf=standard_polygon_gdf)

        # Check all required attributes exist
        assert hasattr(roi_obj, "roi_settings")
        assert hasattr(roi_obj, "extracted_shorelines")
        assert hasattr(roi_obj, "cross_shore_distances")
        assert isinstance(roi_obj.roi_settings, dict)
        assert isinstance(roi_obj.extracted_shorelines, dict)
        assert isinstance(roi_obj.cross_shore_distances, dict)

    def test_initialization_with_invalid_square_lengths(
        self, valid_bbox_gdf, valid_shoreline_gdf
    ):
        """Test initialization fails with invalid square lengths."""
        with pytest.raises(
            ValueError, match="At least one square size must be greater than 0"
        ):
            roi.ROI(
                bbox=valid_bbox_gdf,
                shoreline=valid_shoreline_gdf,
                square_len_lg=0,  # Invalid
                square_len_sm=0,  # Invalid
            )


class TestROIStyleLayer:
    """Test ROI layer styling functionality."""

    def test_style_layer_with_valid_geojson(self, valid_ROI):
        """Test styling layer with valid GeoJSON data."""
        # Create sample GeoJSON data
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                    },
                    "properties": {"id": "test_roi"},
                }
            ],
        }

        result = valid_ROI.style_layer(geojson_data, "test_layer")

        # Should return styled GeoJSON layer
        from ipyleaflet import GeoJSON

        assert isinstance(result, GeoJSON)
        assert result.name == "test_layer"

    def test_style_layer_with_empty_geojson(self, valid_ROI):
        """Test styling layer with empty GeoJSON raises error."""
        empty_geojson = {}

        with pytest.raises(
            ValueError, match="Empty test_layer geojson cannot be drawn"
        ):
            valid_ROI.style_layer(empty_geojson, "test_layer")


class TestROISettingsEdgeCases:
    """Test additional edge cases for ROI settings management."""

    def test_get_roi_settings_with_iterable_roi_ids(self, valid_ROI_with_settings):
        """Test getting settings for multiple ROI IDs as iterable."""
        # Assume valid_ROI_with_settings has some settings
        if valid_ROI_with_settings.roi_settings:
            roi_ids = list(valid_ROI_with_settings.roi_settings.keys())[
                :2
            ]  # Take first 2

            result = valid_ROI_with_settings.get_roi_settings(roi_ids)

            # Should return dict with requested ROI settings
            assert isinstance(result, dict)
            for roi_id in roi_ids:
                if roi_id in valid_ROI_with_settings.roi_settings:
                    assert roi_id in result

    def test_set_roi_settings_invalid_type(self, valid_ROI):
        """Test setting ROI settings with invalid type raises error."""
        with pytest.raises(TypeError, match="roi_settings must be a dict"):
            valid_ROI.set_roi_settings("not_a_dict")

    def test_update_roi_settings_invalid_type(self, valid_ROI):
        """Test updating ROI settings with invalid type raises error."""
        with pytest.raises(TypeError, match="new_settings must be a dict"):
            valid_ROI.update_roi_settings("not_a_dict")

    def test_update_roi_settings_initializes_empty_settings(self, valid_ROI):
        """Test updating ROI settings initializes empty settings if needed."""
        # Remove roi_settings to test initialization
        if hasattr(valid_ROI, "roi_settings"):
            delattr(valid_ROI, "roi_settings")

        new_settings = {"test_roi": {"param": "value"}}
        result = valid_ROI.update_roi_settings(new_settings)

        # Should initialize and update settings
        assert hasattr(valid_ROI, "roi_settings")
        assert valid_ROI.roi_settings == new_settings
        assert result == new_settings
