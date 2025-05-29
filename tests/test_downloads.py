from unittest.mock import patch, call
import pytest

from coastseg import downloads
from unittest.mock import patch, MagicMock
from datetime import datetime


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

    mock_chain = (
        mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value
    )
    mock_chain.filterMetadata.return_value.filter.return_value.size.return_value.getInfo.return_value = (
        10
    )
    mock_chain.filter.return_value.size.return_value.getInfo.return_value = (
        10  # for Sentinel-1 or any other path
    )

    result = downloads.count_images_in_ee_collection(polygon, start_date, end_date)

    # gets images from both tiers available so 10 images for each satellite from each tier (1,2)
    # Since L9 and S2 are not only available in tier 1, they only have 10 images
    assert result == {"L5": 20, "L7": 20, "L8": 20, "L9": 10, "S2": 10, "S1": 10}


def test_count_images_in_ee_collection_invalid_dates(polygon):
    with pytest.raises(ValueError):
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
    mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filterMetadata.return_value.filter.return_value.size.return_value.getInfo.return_value = (
        5
    )

    mock_chain = (
        mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value
    )
    # Set this for S1 because S1 does not use the filterMetadata like the other satellites becuase it doesn't filter by cloud cover
    mock_chain.filter.return_value.size.return_value.getInfo.return_value = (
        5  # for Sentinel-1
    )

    result = downloads.count_images_in_ee_collection(
        polygon, start_date, end_date, months_list=[12, 1]
    )

    # gets images from both tiers available so 5 images for each satellite from each tier (1,2), this is because in count_images_in_ee_collection if no tiers are passed then tiers 1,2 are used
    # Since L9 and S2 are not only available in tier 1, they only have 5 images
    assert result == {"L5": 10, "L7": 10, "L8": 10, "L9": 5, "S2": 5, "S1": 5}


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
