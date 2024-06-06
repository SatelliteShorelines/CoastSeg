import geopandas as gpd
import json
from typing import List, Optional, Set, Dict, Union
from datetime import datetime
import ee
import os
from typing import Collection
from tqdm.auto import tqdm
import argparse


def get_roi_polygon(
    roi_gdf: gpd.GeoDataFrame, roi_id: Optional[int] = None, index: Optional[int] = None
) -> Optional[List[List[float]]]:
    """
    Get the polygon coordinates for a region of interest (ROI) in a GeoDataFrame.

    Args:
        roi_gdf (gpd.GeoDataFrame): The GeoDataFrame containing the ROI data.
        roi_id (Optional[int], optional): The ID of the ROI to retrieve. Defaults to None.
        index (Optional[int], optional): The index of the ROI to retrieve. Defaults to None.

    Returns:
        Optional[List[List[float]]]: The polygon coordinates of the ROI as a list of lists of floats,
        or None if the ROI is not found.

    """
    if roi_id is not None:
        geoseries = roi_gdf[roi_gdf["id"] == roi_id]["geometry"]
    elif index is not None:
        geoseries = roi_gdf.iloc[[index]]["geometry"]
    else:
        return None

    if not geoseries.empty:
        return [[[x, y] for x, y in list(geoseries.iloc[0].exterior.coords)]]
    return None


def get_collection_by_tier(
    polygon: List[List[float]],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    satellite: str,
    tier: int,
    max_cloud_cover: float = 95,
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
    max_cloud_cover (float): The maximum cloud cover percentage to filter the ImageCollection by.

    Returns:
    ee.ImageCollection or None: The filtered ImageCollection or None if the inputs are invalid.
    """

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
    }
    cloud_property = cloud_properties.get(satellite)

    collection = (
        ee.ImageCollection(collection_name)
        .filterBounds(ee.Geometry.Polygon(polygon))
        .filterDate(ee.Date(start_date), ee.Date(end_date))
        .filterMetadata(cloud_property, "less_than", max_cloud_cover)
    )
    return collection


def count_images_in_ee_collection(
    polygon: list[list[float]],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    max_cloud_cover: float = 95,
    satellites: Collection[str] = ("L5", "L7", "L8", "L9", "S2"),
) -> dict:
    """
    Count the number of images in specified satellite collections over a certain area and time period.

    Parameters:
    polygon (list[list[float]]): A list of lists representing the vertices of a polygon in lon/lat coordinates.
    start_date (str or datetime): The start date of the time period. If a string, it should be in 'YYYY-MM-DD' format.
    end_date (str or datetime): The end date of the time period. If a string, it should be in 'YYYY-MM-DD' format.
    max_cloud_cover (float, optional): The maximum cloud cover percentage. Images with a cloud cover percentage higher than this will be excluded. Defaults to 99.
    satellites (Collection[str], optional): A collection of satellite names. The function will return image counts for these satellites. Defaults to ("L5","L7","L8","L9","S2").

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
    # Check types of start_date and end_date
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    elif not isinstance(start_date, datetime):
        raise ValueError("start_date must be a string or datetime object")

    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    elif not isinstance(end_date, datetime):
        raise ValueError("end_date must be a string or datetime object")

    image_counts = {}
    images_in_tier_count = 0
    for satellite in satellites:
        images_in_tier_count = 0
        for tier in [1, 2]:
            collection = get_collection_by_tier(
                polygon, start_date, end_date, satellite, tier, max_cloud_cover
            )
            if collection:
                images_in_tier_count += collection.size().getInfo()
        image_counts[satellite] = images_in_tier_count

    return image_counts


def get_image_counts_for_roi(polygon:list, identifier:str, settings:dict):
    """
    Retrieves the image counts for a given region of interest (ROI) within a specified date range.
    Prints the image counts for each satellite in the specified date ranges for the ROI in the format:
    
        ROI ID: <ROI ID>
        L7: 0 images
        L8: 1 images
        L9: 3 images
        S2: 9 images

    Args:
        polygon (list): The coordinates of the polygon representing the ROI.
        identifier (str): The ID for the ROI.
        settings (dict): A dictionary containing the settings for image retrieval.
            The settings should contain the following keys:
            - dates: A list of date ranges to query.
            - sat_list: A list of satellite names to query.
            
    Returns:
        list: A list of dictionaries, where each dictionary contains the ROI ID, date range, image count, and polygon.

    """
    results = []
    for start_date, end_date in tqdm(
        settings["dates"],
        desc=f"Processing Date Ranges for ROI {identifier}",
        leave=False,
    ):
        images_count = count_images_in_ee_collection(
            polygon,
            start_date,
            end_date,
            satellites=settings["sat_list"],
        )
        result = {
            "roi_id": identifier,
            "date_range": [start_date, end_date],
            "images_count": images_count,
            "polygon": polygon,
        }
        results.append(result)
        print(json.dumps(result, indent=2))
    return results


def preview_images_for_rois(
    rois_gdf: gpd.GeoDataFrame, selected_ids: Set[int], settings: Dict
):
    """
    Preview the number images available for the given regions of interest (ROIs).
    
    Provides the number of images availabe for each satellite in the specified date ranges for each ROI.

    Args:
        rois_gdf (gpd.GeoDataFrame): A GeoDataFrame containing the ROIs.
        selected_ids (Set[int]): A set of selected ROI IDs. If empty, all ROI IDs will be used.
        settings (Dict): A dictionary containing the processing settings.

    Returns:
        results: The results of processing the ROIs.

    """
    if "id" in rois_gdf.columns:
        if selected_ids:
            roi_ids = selected_ids
        else:
            roi_ids = set(rois_gdf["id"].tolist())

        pbar = tqdm(roi_ids, desc="Querying API", leave=False, unit="roi")
        for roi_id in pbar:
            pbar.set_description(f"Querying API for ROI: {roi_id}")
            polygon = get_roi_polygon(rois_gdf, roi_id=roi_id)
            if polygon:
                results = get_image_counts_for_roi(polygon, roi_id, settings)
    else:
        pbar = tqdm(range(len(rois_gdf)), desc="Querying API", leave=False, unit="roi")
        for index in pbar:
            pbar.set_description(f"Querying API for ROI: {index}")
            # for index in tqdm(range(len(rois_gdf)), desc="Querying API", leave=False):
            polygon = get_roi_polygon(rois_gdf, index=index)
            if polygon:
                results = get_image_counts_for_roi(polygon, index, settings)

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query satellite image counts for regions of interest."
    )
    parser.add_argument(
        "geojson",
        type=str,
        help="Filepath to the GeoJSON file containing the regions of interest.",
    )
    parser.add_argument(
        "dates",
        type=str,
        nargs="+",
        help="Date ranges for the queries. Format: 'start_date,end_date'",
    )

    args = parser.parse_args()
    return args


def main():
    # ========================
    # === User Input Section ==
    # ========================
    args = parse_args()
    # Step 1 - Specify the date ranges for the queries.
    # --------------------------------------------------
    # Specify the date ranges for the queries.
    # Each date range is a list containing two strings: the start date and the end date.
    # Dates should be formatted as 'YYYY-MM-DD'.
    date_ranges = [date_range.split(",") for date_range in args.dates]

    # Step 2 - Specify the geographic regions of interest (ROIs).
    # --------------------------------------------------
    # Provide the filepath to your GeoJSON file containing the regions of interest (ROIs).
    # Make sure to put an 'r' in front of the filepath to make it a raw string, especially on Windows.
    rois_gdf = gpd.read_file(args.geojson)

    # Step 3 Optional - Specify the ROI ids you want to process.
    # --------------------------------------------------
    # Optional: Select specific ROIs by their IDs. If you want to process all ROIs, leave this set empty.
    # Example: selected_ids = set([1, 2, 3]) if you want to select ROIs with IDs 1, 2, and 3.
    selected_ids = set()

    # List of satellite names you want to query for. You can add or remove items as needed.
    sat_list = ["L5", "L7", "L8", "L9", "S2"]

    # Whether to save the query results to a JSON file. Set to True to save, False to not save.
    save_to_file = True

    # =========================
    # === Script Execution ====
    # =========================

    settings = {"dates": date_ranges, "sat_list": sat_list}

    # Check if EE was initialized or not
    try:
        ee.ImageCollection("LANDSAT/LT05/C01/T1_TOA")
    except:
        ee.Initialize()

    results = preview_images_for_rois(rois_gdf, selected_ids, settings)

    if save_to_file:
        # Save results to JSON file
        print(f"Saving results to results.json {os.path.abspath('results.json')}")
        with open(os.path.abspath("results.json"), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
