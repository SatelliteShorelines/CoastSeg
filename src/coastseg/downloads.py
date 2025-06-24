from datetime import datetime
import logging
import ee
from typing import Collection, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def get_collection_by_tier(
    polygon: List[List[float]],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    satellite: str,
    tier: int,
    max_cloud_cover: float = 95,
    months_list=None,
) -> Union[ee.ImageCollection, None]:
    """
    This function takes the required parameters and returns an ImageCollection from
    the specified satellite and tier filtered by the given polygon, date range, and cloud cover.

    Args:
    polygon (List[List[float]]): The polygon to filter the ImageCollection by.
    start_date (Union[str, datetime]): The start date to filter the ImageCollection by.
    end_date (Union[str, datetime]): The end date to filter the ImageCollection by.
    satellite (str): The satellite to select the ImageCollection from.
    tier (int): The tier of the satellite data.
    max_cloud_cover (float): The maximum cloud cover percentage for the entire scene (not just the roi) to filter the ImageCollection by.

    Returns:
    ee.ImageCollection or None: The filtered ImageCollection or None if the inputs are invalid.
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
        ee.ImageCollection(collection_name)
        .filterBounds(ee.Geometry.Polygon(polygon))
        .filterDate(ee.Date(start_date), ee.Date(end_date))
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
    polygon: list[list[float]],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    max_cloud_cover: float = 95,
    satellites: Collection[str] = ("L5", "L7", "L8", "L9", "S2", "S1"),
    tiers: list[int] = None,
    months_list: list[int] = None,
) -> dict:
    """
    Count the number of images in specified satellite collections over a certain area and time period.

    Parameters:
    polygon (list[list[float]]): A list of lists representing the vertices of a polygon in lon/lat coordinates.
    start_date (str or datetime): The start date of the time period. If a string, it should be in 'YYYY-MM-DD' format.
    end_date (str or datetime): The end date of the time period. If a string, it should be in 'YYYY-MM-DD' format.
    max_cloud_cover (float, optional): The maximum cloud cover percentage. Images with a cloud cover percentage higher than this will be excluded. Defaults to 99.
    satellites (Collection[str], optional): A collection of satellite names. The function will return image counts for these satellites. Defaults to ("L5","L7","L8","L9","S2").
    tiers (list[int], optional): A list of tiers. The function will return image counts for these tiers. Defaults to [1,2]
    months_list (list[int], optional): A list of months to filter the images by. Defaults to None meaning all the months will be included.
    Returns:
    dict: A dictionary where the keys are the satellite names and the values are the image counts.

    Raises:
    ValueError: If start_date or end_date are not strings or datetime objects.

    Example:
    >>> polygon = [[[151.2957545, -33.7390216],
    ... [151.312234, -33.7390216],
    ... [151.312234, -33.7012561],
    ... [151.2957545, -33.7012561],
    ... [151.2957545, -33.7390216]]]
    >>> start_date = '2017-12-01'
    >>> end_date = '2018-01-01'
    >>> count_images(polygon, start_date, end_date)
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
        ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
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
                images_in_tier_count += collection.size().getInfo()
        image_counts[satellite] = images_in_tier_count

    return image_counts
