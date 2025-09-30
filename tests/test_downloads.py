from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from coastseg import downloads


@pytest.fixture
def polygon():
    return [
        [
            [151.2957545, -33.7390216],
            [151.312234, -33.7390216],
            [151.312234, -33.7012561],
            [151.2957545, -33.7012561],
            [151.2957545, -33.7390216],
        ]
    ]


@pytest.fixture
def start_date():
    return "2017-12-01"


@pytest.fixture
def end_date():
    return "2018-01-01"


@pytest.fixture
def mock_ee():
    with patch("coastseg.downloads.ee") as mock_ee:
        yield mock_ee


def test_count_images_in_ee_collection(mock_ee, polygon, start_date, end_date):
    """Test counting images in EE collection with proper mocking."""
    # Mock the EE initialization check to pass
    mock_ee.ImageCollection.return_value = MagicMock()
    mock_ee.Geometry.Polygon.return_value = MagicMock()

    # Mock get_collection_by_tier to return a mock collection
    with (
        patch("coastseg.downloads.get_collection_by_tier") as mock_get_collection,
        patch("coastseg.downloads.filter_collection_by_coverage") as mock_filter,
    ):
        mock_collection = MagicMock()
        mock_collection.size.return_value.getInfo.return_value = 10
        mock_get_collection.return_value = mock_collection

        # Mock filter_collection_by_coverage to return the same collection
        mock_filter.return_value = mock_collection

        result = downloads.count_images_in_ee_collection(polygon, start_date, end_date)

    # Each satellite processes both tiers by default [1, 2]
    # With 10 images per tier: 10 + 10 = 20 images per satellite
    assert result == {"L5": 20, "L7": 20, "L8": 20, "L9": 20, "S2": 20, "S1": 20}


def test_count_images_in_ee_collection_invalid_dates(mock_ee, polygon):
    """Test error handling for invalid date ranges."""
    # Mock EE initialization to pass
    mock_ee.ImageCollection.return_value = MagicMock()

    with pytest.raises(ValueError, match="End date.*must be after start date"):
        downloads.count_images_in_ee_collection(polygon, "2018-01-01", "2017-12-01")


def test_count_images_in_ee_collection_uninitialized_ee(
    mock_ee, polygon, start_date, end_date
):
    mock_ee.ImageCollection.side_effect = Exception("Earth Engine not initialized")

    with pytest.raises(Exception):
        downloads.count_images_in_ee_collection(polygon, start_date, end_date)


def test_count_images_in_ee_collection_custom_months(
    mock_ee, polygon, start_date, end_date
):
    """Test counting images with custom months filter."""
    # Mock EE initialization check
    mock_ee.ImageCollection.return_value = MagicMock()
    mock_ee.Geometry.Polygon.return_value = MagicMock()

    # Mock get_collection_by_tier to return a mock collection with 5 images
    with (
        patch("coastseg.downloads.get_collection_by_tier") as mock_get_collection,
        patch("coastseg.downloads.filter_collection_by_coverage") as mock_filter,
    ):
        mock_collection = MagicMock()
        mock_collection.size.return_value.getInfo.return_value = 5
        mock_get_collection.return_value = mock_collection

        # Mock filter_collection_by_coverage to return the same collection
        mock_filter.return_value = mock_collection

        result = downloads.count_images_in_ee_collection(
            polygon, start_date, end_date, months_list=[12, 1]
        )  # Each satellite processes both tiers by default [1, 2]
    # With 5 images per tier: 5 + 5 = 10 images per satellite
    assert result == {"L5": 10, "L7": 10, "L8": 10, "L9": 10, "S2": 10, "S1": 10}


def test_get_collection_by_tier(mock_ee, polygon, start_date, end_date):
    mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filterMetadata.return_value.filter.return_value = (
        MagicMock()
    )

    result = downloads.get_collection_by_tier(polygon, start_date, end_date, "L8", 1)

    assert result is not None
    mock_ee.ImageCollection.assert_called_with("LANDSAT/LC08/C02/T1_TOA")
    mock_ee.ImageCollection.return_value.filterBounds.assert_called()
    mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.assert_called()
    mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filterMetadata.assert_called()
    mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filterMetadata.return_value.filter.assert_called()


def test_get_collection_by_tier_invalid_tier(mock_ee, polygon, start_date, end_date):
    result = downloads.get_collection_by_tier(polygon, start_date, end_date, "L8", 3)
    assert result is None


def test_get_collection_by_tier_invalid_satellite(
    mock_ee, polygon, start_date, end_date
):
    result = downloads.get_collection_by_tier(polygon, start_date, end_date, "L10", 1)
    assert result is None


def test_get_collection_by_tier_custom_months(mock_ee, polygon, start_date, end_date):
    mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filterMetadata.return_value.filter.return_value = (
        MagicMock()
    )

    result = downloads.get_collection_by_tier(
        polygon, start_date, end_date, "L8", 1, months_list=[12, 1]
    )

    assert result is not None
    mock_ee.ImageCollection.assert_called_with("LANDSAT/LC08/C02/T1_TOA")
    mock_ee.ImageCollection.return_value.filterBounds.assert_called()
    mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.assert_called()
    mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filterMetadata.assert_called()
    mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filterMetadata.return_value.filter.assert_called()


def test_get_collection_by_tier_default_months(mock_ee, polygon, start_date, end_date):
    mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filterMetadata.return_value.filter.return_value = (
        MagicMock()
    )

    result = downloads.get_collection_by_tier(polygon, start_date, end_date, "L8", 1)

    assert result is not None
    mock_ee.ImageCollection.assert_called_with("LANDSAT/LC08/C02/T1_TOA")
    mock_ee.ImageCollection.return_value.filterBounds.assert_called()
    mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.assert_called()
    mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filterMetadata.assert_called()
    mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filterMetadata.return_value.filter.assert_called()


# ============================================================================
# COMPREHENSIVE TESTS FOR DOWNLOADS MODULE
# ============================================================================


class TestGetCollectionByTier:
    """Test get_collection_by_tier function comprehensively."""

    def test_get_collection_by_tier_all_satellites_tier1(
        self, mock_ee, polygon, start_date, end_date
    ):
        """Test collection retrieval for all satellites in tier 1."""
        mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filterMetadata.return_value.filter.return_value = (
            MagicMock()
        )

        satellites = ["L5", "L7", "L8", "L9", "S2", "S1"]
        for satellite in satellites:
            result = downloads.get_collection_by_tier(
                polygon, start_date, end_date, satellite, 1
            )
            assert result is not None, f"Failed for satellite {satellite}"

    def test_get_collection_by_tier_tier2_limited_satellites(
        self, mock_ee, polygon, start_date, end_date
    ):
        """Test that only L5, L7, L8 are available in tier 2."""
        mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filterMetadata.return_value.filter.return_value = (
            MagicMock()
        )

        # Available in tier 2
        tier2_satellites = ["L5", "L7", "L8"]
        for satellite in tier2_satellites:
            result = downloads.get_collection_by_tier(
                polygon, start_date, end_date, satellite, 2
            )
            assert result is not None, f"Tier 2 should be available for {satellite}"

        # Not available in tier 2
        tier2_unavailable = ["L9", "S2", "S1"]
        for satellite in tier2_unavailable:
            result = downloads.get_collection_by_tier(
                polygon, start_date, end_date, satellite, 2
            )
            assert result is None, f"Tier 2 should not be available for {satellite}"

    def test_get_collection_by_tier_datetime_conversion(self, mock_ee, polygon):
        """Test datetime object conversion to ISO format."""
        mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filterMetadata.return_value.filter.return_value = (
            MagicMock()
        )

        start_dt = datetime(2020, 1, 1)
        end_dt = datetime(2020, 12, 31)

        result = downloads.get_collection_by_tier(polygon, start_dt, end_dt, "L8", 1)

        assert result is not None
        # Verify the dates were converted properly
        mock_ee.Date.assert_called()

    def test_get_collection_by_tier_custom_cloud_cover(
        self, mock_ee, polygon, start_date, end_date
    ):
        """Test custom cloud cover filtering."""
        mock_collection = MagicMock()
        mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filterMetadata.return_value.filter.return_value = (
            mock_collection
        )

        result = downloads.get_collection_by_tier(
            polygon, start_date, end_date, "L8", 1, max_cloud_cover=50
        )

        assert result is not None
        # Verify cloud cover filter was applied
        mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filterMetadata.assert_called_with(
            "CLOUD_COVER", "less_than", 50
        )

    def test_get_collection_by_tier_sentinel1_no_cloud_filter(
        self, mock_ee, polygon, start_date, end_date
    ):
        """Test that Sentinel-1 doesn't apply cloud cover filter."""
        mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filter.return_value = (
            MagicMock()
        )

        result = downloads.get_collection_by_tier(
            polygon, start_date, end_date, "S1", 1
        )

        assert result is not None
        # For S1, filterMetadata should not be called for cloud cover
        mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filter.assert_called()


class TestCountImagesInEECollection:
    """Test count_images_in_ee_collection function comprehensively."""

    def test_count_images_string_date_parsing(self, mock_ee, polygon):
        """Test parsing of string dates."""
        mock_ee.ImageCollection.return_value = MagicMock()
        mock_ee.Geometry.Polygon.return_value = MagicMock()

        with (
            patch("coastseg.downloads.get_collection_by_tier") as mock_get_collection,
            patch("coastseg.downloads.filter_collection_by_coverage") as mock_filter,
        ):
            mock_collection = MagicMock()
            mock_collection.size.return_value.getInfo.return_value = 5
            mock_get_collection.return_value = mock_collection

            # Mock filter_collection_by_coverage to return the same collection
            mock_filter.return_value = mock_collection

            result = downloads.count_images_in_ee_collection(
                polygon, "2020-01-01", "2020-12-31"
            )

        assert isinstance(result, dict)
        assert len(result) == 6  # 6 satellites

    def test_count_images_datetime_objects(self, mock_ee, polygon):
        """Test handling of datetime objects as inputs."""
        mock_ee.ImageCollection.return_value = MagicMock()
        mock_ee.Geometry.Polygon.return_value = MagicMock()

        with (
            patch("coastseg.downloads.get_collection_by_tier") as mock_get_collection,
            patch("coastseg.downloads.filter_collection_by_coverage") as mock_filter,
        ):
            mock_collection = MagicMock()
            mock_collection.size.return_value.getInfo.return_value = 3
            mock_get_collection.return_value = mock_collection

            # Mock filter_collection_by_coverage to return the same collection
            mock_filter.return_value = mock_collection

            start_dt = datetime(2020, 1, 1)
            end_dt = datetime(2020, 12, 31)

            result = downloads.count_images_in_ee_collection(polygon, start_dt, end_dt)

        assert isinstance(result, dict)
        # Should have counted images for all satellites
        expected_satellites = ["L5", "L7", "L8", "L9", "S2", "S1"]
        assert all(sat in result for sat in expected_satellites)

    def test_count_images_custom_satellites_and_tiers(
        self, mock_ee, polygon, start_date, end_date
    ):
        """Test custom satellites and tiers parameters."""
        mock_ee.ImageCollection.return_value = MagicMock()
        mock_ee.Geometry.Polygon.return_value = MagicMock()

        with (
            patch("coastseg.downloads.get_collection_by_tier") as mock_get_collection,
            patch("coastseg.downloads.filter_collection_by_coverage") as mock_filter,
        ):
            mock_collection = MagicMock()
            mock_collection.size.return_value.getInfo.return_value = 8
            mock_get_collection.return_value = mock_collection

            # Mock filter_collection_by_coverage to return the same collection
            mock_filter.return_value = mock_collection

            result = downloads.count_images_in_ee_collection(
                polygon, start_date, end_date, satellites=["L8", "S2"], tiers=[1]
            )

        assert result == {"L8": 8, "S2": 8}

    def test_count_images_invalid_date_types(self, mock_ee, polygon):
        """Test error handling for invalid date types."""
        mock_ee.ImageCollection.return_value = MagicMock()

        with pytest.raises(
            ValueError, match="start_date must be a string or datetime object"
        ):
            downloads.count_images_in_ee_collection(polygon, 12345, "2020-12-31")  # type: ignore

        with pytest.raises(
            ValueError, match="end_date must be a string or datetime object"
        ):
            downloads.count_images_in_ee_collection(polygon, "2020-01-01", 12345)  # type: ignore

    def test_count_images_with_roi_coverage_filter(
        self, mock_ee, polygon, start_date, end_date
    ):
        """Test minimum ROI coverage filtering."""
        mock_ee.ImageCollection.return_value = MagicMock()

        with patch("coastseg.downloads.get_collection_by_tier") as mock_get_collection:
            with patch(
                "coastseg.downloads.filter_collection_by_coverage"
            ) as mock_filter:
                mock_collection = MagicMock()
                mock_collection.size.return_value.getInfo.return_value = 15
                mock_get_collection.return_value = mock_collection
                mock_filter.return_value = mock_collection

                result = downloads.count_images_in_ee_collection(
                    polygon, start_date, end_date, min_roi_coverage=0.8
                )

        # Verify coverage filter was applied
        assert mock_filter.called
        assert isinstance(result, dict)


class TestDownloadsInputValidation:
    """Test input validation for downloads module functions."""

    def test_count_images_empty_satellites_list(
        self, mock_ee, polygon, start_date, end_date
    ):
        """Test behavior with empty satellites list."""
        mock_ee.ImageCollection.return_value = MagicMock()

        result = downloads.count_images_in_ee_collection(
            polygon, start_date, end_date, satellites=[]
        )

        assert result == {}

    def test_count_images_empty_tiers_list(
        self, mock_ee, polygon, start_date, end_date
    ):
        """Test behavior with empty tiers list."""
        mock_ee.ImageCollection.return_value = MagicMock()

        result = downloads.count_images_in_ee_collection(
            polygon, start_date, end_date, tiers=[]
        )

        # All counts should be 0 since no tiers to check
        expected = {"L5": 0, "L7": 0, "L8": 0, "L9": 0, "S2": 0, "S1": 0}
        assert result == expected


class TestDownloadsEdgeCases:
    """Test edge cases and error scenarios."""

    def test_count_images_none_return_from_get_collection(
        self, mock_ee, polygon, start_date, end_date
    ):
        """Test handling when get_collection_by_tier returns None."""
        mock_ee.ImageCollection.return_value = MagicMock()

        with patch("coastseg.downloads.get_collection_by_tier") as mock_get_collection:
            mock_get_collection.return_value = None  # Simulate invalid satellite/tier

            result = downloads.count_images_in_ee_collection(
                polygon, start_date, end_date, satellites=["INVALID_SAT"]
            )

        assert result == {"INVALID_SAT": 0}

    def test_get_collection_extreme_cloud_cover_values(
        self, mock_ee, polygon, start_date, end_date
    ):
        """Test extreme cloud cover values."""
        mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filterMetadata.return_value.filter.return_value = (
            MagicMock()
        )

        # Test 0% cloud cover
        result_zero = downloads.get_collection_by_tier(
            polygon, start_date, end_date, "L8", 1, max_cloud_cover=0
        )
        assert result_zero is not None

        # Test 100% cloud cover
        result_hundred = downloads.get_collection_by_tier(
            polygon, start_date, end_date, "L8", 1, max_cloud_cover=100
        )
        assert result_hundred is not None

    def test_count_images_single_month_filter(
        self, mock_ee, polygon, start_date, end_date
    ):
        """Test filtering by a single month."""
        mock_ee.ImageCollection.return_value = MagicMock()
        mock_ee.Geometry.Polygon.return_value = MagicMock()

        with (
            patch("coastseg.downloads.get_collection_by_tier") as mock_get_collection,
            patch("coastseg.downloads.filter_collection_by_coverage") as mock_filter,
        ):
            mock_collection = MagicMock()
            mock_collection.size.return_value.getInfo.return_value = 2
            mock_get_collection.return_value = mock_collection

            # Mock filter_collection_by_coverage to return the same collection
            mock_filter.return_value = mock_collection

            result = downloads.count_images_in_ee_collection(
                polygon,
                start_date,
                end_date,
                months_list=[7],  # July only
            )

        assert isinstance(result, dict)
        # Should still return counts for all satellites
        assert len(result) == 6
