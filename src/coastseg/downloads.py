import logging
from datetime import datetime
from typing import Any, Collection, Dict, List, Optional, Union

import ee
from coastsat.SDS_download import filter_collection_by_coverage

logger = logging.getLogger(__name__)


def get_collection_by_tier(
    polygon: List[List[float]],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    satellite: str,
    tier: int,
    max_cloud_cover: float = 95,
    months_list: Optional[List[int]] = None,
) -> Optional[Any]:
    """
    Retrieves filtered Earth Engine ImageCollection for given satellite and tier.

    Args:
        polygon: Polygon coordinates as list of lists of floats.
        start_date: Start date as string or datetime.
        end_date: End date as string or datetime.
        satellite: Satellite name.
        tier: Data tier.
        max_cloud_cover: Maximum cloud cover percentage.
        months_list: List of months to include.

    Returns:
        Filtered ImageCollection or None if invalid.

    Example:
        >>> collection = get_collection_by_tier(polygon, '2020-01-01', '2020-12-31', 'L8', 1, 50, [6,7,8])
        >>> print(collection.size().getInfo())
        12
    """
    if months_list is None:
        months_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # Converting datetime objects to string if passed as datetime
    if isinstance(start_date, datetime):
        start_date = start_date.isoformat()
    if isinstance(end_date, datetime):
        end_date = end_date.isoformat()

    # Define collection names for tier 1 and tier 2
    col_names = {
        1: {
            "L5": "LANDSAT/LT05/C02/T1_TOA",
            "L7": "LANDSAT/LE07/C02/T1_TOA",
            "L8": "LANDSAT/LC08/C02/T1_TOA",
            "L9": "LANDSAT/LC09/C02/T1_TOA",
            "S2": "COPERNICUS/S2_HARMONIZED",
            "S1": "COPERNICUS/S1_GRD",
        },
        2: {
            "L5": "LANDSAT/LT05/C02/T2_TOA",
            "L7": "LANDSAT/LE07/C02/T2_TOA",
            "L8": "LANDSAT/LC08/C02/T2_TOA",
        },
    }

    # Validate inputs and get collection name
    if tier not in col_names:
        print(f"Invalid tier ({tier})")
        return None
    # not all satellites are in tier 2 return None for any that are not
    if satellite not in col_names[tier]:
        return None

    collection_name = col_names[tier][satellite]
    # Mapping satellite names to their respective cloud properties
    cloud_properties = {
        "L5": "CLOUD_COVER",
        "L7": "CLOUD_COVER",
        "L8": "CLOUD_COVER",
        "L9": "CLOUD_COVER",
        "S2": "CLOUDY_PIXEL_PERCENTAGE",
        # Note: S1 (Sentinel-1) does not use a cloud property
    }
    cloud_property = cloud_properties.get(satellite)

    # Create a filter to select images with system:time_start month in the monthsToKeep list
    def filter_by_month(month):
        return ee.Filter.calendarRange(month, month, "month")  # type: ignore

    month_filters = [filter_by_month(month) for month in months_list]
    month_filter = ee.Filter.Or(month_filters)  # type: ignore
    collection = (
        ee.ImageCollection(collection_name)  # type: ignore
        .filterBounds(ee.Geometry.Polygon(polygon))  # type: ignore
        .filterDate(ee.Date(start_date), ee.Date(end_date))  # type: ignore
    )

    # Apply cloud cover filter only if available
    if cloud_property:
        collection = collection.filterMetadata(
            cloud_property, "less_than", max_cloud_cover
        )

    # apply the month filter to only include images from the months in the months_list
    collection = collection.filter(month_filter)

    return collection


def count_images_in_ee_collection(
    polygon: List[List[float]],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    max_cloud_cover: float = 95,
    satellites: Collection[str] = ("L5", "L7", "L8", "L9", "S2", "S1"),
    tiers: Optional[List[int]] = None,
    months_list: Optional[List[int]] = None,
    min_roi_coverage: float = 0.0,
) -> Dict[str, int]:
    """
    Counts images in Earth Engine collections for given satellites and parameters.

    Args:
        polygon (list[list[float]]): Vertices of a polygon in [lon, lat] coordinates.
        start_date (Union[str, datetime]): Start date ('YYYY-MM-DD' or datetime).
        end_date (Union[str, datetime]): End date ('YYYY-MM-DD' or datetime).
        max_cloud_cover (float, optional): Maximum allowed cloud cover percentage. Defaults to 95.
        satellites (Collection[str], optional): Satellite names to count images for. Defaults to ("L5", "L7", "L8", "L9", "S2", "S1").
        tiers (list[int], optional): List of tiers to include. Defaults to [1, 2].
        months_list (list[int], optional): Months (1-12) to include. Defaults to all months.
        min_roi_coverage (float, optional): Minimum percentage of ROI that must be covered by imagery. Defaults to 0.0.

    Returns:
        Dictionary mapping satellite names to image counts.

    Raises:
        ValueError: If dates are invalid or end before start.
        Exception: If Earth Engine not initialized.

    Example:
        >>> polygon = [[[151.2957545, -33.7390216],
        ... [151.312234, -33.7390216],
        ... [151.312234, -33.7012561],
        ... [151.2957545, -33.7012561],
        ... [151.2957545, -33.7390216]]]
        >>> count_images_in_ee_collection(polygon, '2017-12-01', '2018-01-01')
        {'L5': 10, 'L7': 15, 'L8': 20, 'L9': 5, 'S2': 30, 'S1': 25}
    """
    if months_list is None:
        months_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # Check types of start_date and end_date
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    elif not isinstance(start_date, datetime):
        raise ValueError("start_date must be a string or datetime object")

    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    elif not isinstance(end_date, datetime):
        raise ValueError("end_date must be a string or datetime object")

    # Check that end_date is after start_date
    if end_date <= start_date:
        raise ValueError(
            f"End date: {end_date.strftime('%Y-%m-%d')} must be after start date: {start_date.strftime('%Y-%m-%d')}"
        )

    # Check if EE was initialized or not
    try:
        ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")  # type: ignore
    except Exception as e:
        raise Exception(
            f"Earth Engine not initialized. Please run ee.Initialize(project='gee project id') before calling this function.{e}"
        )

    if tiers is None:
        tiers = [1, 2]

    image_counts = {}
    images_in_tier_count = 0
    for satellite in satellites:
        images_in_tier_count = 0
        for tier in tiers:
            # this returns an ee.ImageCollection for the given polygon, date range, satellite, and tier
            collection = get_collection_by_tier(
                polygon,
                start_date,
                end_date,
                satellite,
                tier,
                max_cloud_cover,
                months_list=months_list,
            )
            if collection:
                collection = filter_collection_by_coverage(
                    collection,
                    ee.Geometry.Polygon(polygon),  # type: ignore
                    min_roi_coverage=min_roi_coverage,
                )
                images_in_tier_count += collection.size().getInfo()  # type: ignore
        image_counts[satellite] = images_in_tier_count

    return image_counts
