import copy
import logging
import os
from typing import Any, Callable, List

# Third-party imports
import geopandas as gpd

# Internal dependencies imports
from coastseg import shoreline, shoreline_extraction_area, transects

logger = logging.getLogger(__name__)


FEATURE_TYPE_MAP = {
    "transect": transects.Transects,
    "transects": transects.Transects,
    "shoreline": shoreline.Shoreline,
    "shorelines": shoreline.Shoreline,
    "reference_shoreline": shoreline.Shoreline,
    "reference_shorelines": shoreline.Shoreline,
    "shoreline_extraction_area": shoreline_extraction_area.Shoreline_Extraction_Area,
    "shoreline extraction area": shoreline_extraction_area.Shoreline_Extraction_Area,
}


def edit_geojson_files(
    filepaths: List[str],
    filter_function: Callable[..., gpd.GeoDataFrame],
    **kwargs: Any,
) -> None:
    """
    Apply filter function to GeoJSON files.

    Args:
        filepaths: List of GeoJSON file paths.
        filter_function: Function to apply to GeoDataFrames.
        **kwargs: Keyword arguments for filter function.
    """
    for filepath in filepaths:
        if os.path.exists(filepath):
            edit_gdf_file(filepath, filter_function, **kwargs)


def edit_gdf_file(
    filepath: str,
    filter_function: Callable[..., gpd.GeoDataFrame],
    **kwargs: Any,
) -> None:
    """
    Apply filter function to GeoDataFrame from file.

    Args:
        filepath: Path to GeoJSON file.
        filter_function: Function to apply to GeoDataFrame.
        **kwargs: Keyword arguments for filter function.
    """
    gdf = read_gpd_file(filepath)
    new_gdf = filter_function(gdf, **kwargs)
    new_gdf.to_file(filepath, driver="GeoJSON")


def create_geofeature_geodataframe(
    geofeature_path: str, roi_gdf: gpd.GeoDataFrame, epsg_code: str, feature_type: str
) -> gpd.GeoDataFrame:
    """
    Create geofeatures GeoDataFrame from file or ROI.

    Args:
        geofeature_path: Path to geofeature file.
        roi_gdf: ROI GeoDataFrame.
        epsg_code: EPSG code for CRS.
        feature_type: Type of geofeature.

    Returns:
        Geofeatures GeoDataFrame.

    Raises:
        Exception: If no features intersect ROI.
    """
    feature_type = (
        feature_type.lower()
    )  # Convert to lower case for case insensitive comparison

    if os.path.exists(geofeature_path):
        # Load features from file
        geofeature_gdf = load_feature_from_file(geofeature_path, feature_type)
    else:
        # if a geofeature file is not given load features from ROI
        geofeature_gdf = load_geofeatures_from_roi(roi_gdf, feature_type)

    logger.info(
        f"{feature_type}_gdf: length {len(geofeature_gdf)} sample {geofeature_gdf.head(1)}"
    )
    if geofeature_gdf.empty:
        raise Exception(
            f"None of the {feature_type}s intersected the ROI. Try a different {feature_type}"
        )
    geofeature_gdf = geofeature_gdf.loc[:, ["id", "geometry"]]
    geofeature_gdf = geofeature_gdf.to_crs(epsg_code)
    return geofeature_gdf


def load_geofeatures_from_roi(
    roi_gdf: gpd.GeoDataFrame, feature_type: str
) -> gpd.GeoDataFrame:
    """
    Load geofeatures from ROI.

    Args:
        roi_gdf: ROI GeoDataFrame.
        feature_type: Type of geofeature.

    Returns:
        Geofeatures GeoDataFrame.

    Raises:
        ValueError: If feature_type unsupported.
        Exception: If no features in ROI.
    """
    feature_type = (
        feature_type.lower()
    )  # Convert to lower case for case insensitive comparison

    if feature_type in FEATURE_TYPE_MAP:
        feature_object = FEATURE_TYPE_MAP[feature_type](bbox=roi_gdf)
    else:
        logger.error(f"Unsupported feature_type: {feature_type}")
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    if feature_object.gdf.empty:
        logger.error(
            f"CoastSeg currently does not have {feature_type}s available in this region. Try uploading your own {feature_type}s.geojson"
        )
        raise Exception(
            f"CoastSeg currently does not have {feature_type}s available in this region. Try uploading your own {feature_type}s.geojson"
        )

    feature_gdf = copy.deepcopy(feature_object.gdf)

    return feature_gdf


def read_gpd_file(filename: str) -> gpd.GeoDataFrame:
    """
    Read GeoDataFrame from file.

    Args:
        filename: Path to geospatial file.

    Returns:
        GeoDataFrame.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if os.path.exists(filename):
        logger.info(f"Opening \n {filename}")
        return gpd.read_file(filename)
    else:
        raise FileNotFoundError(filename)


def load_geodataframe_from_file(
    feature_path: str, feature_type: str
) -> gpd.GeoDataFrame:
    """
    Load GeoDataFrame from file.

    Args:
        feature_path: Path to feature file.
        feature_type: Type of feature. Example: 'shoreline', 'rois', 'transects', 'bbox'.

    Returns:
        GeoDataFrame.

    Raises:
        ValueError: If file is empty.
    """
    logger.info(f"Attempting to load {feature_type} from a file")
    original_feature_gdf = read_gpd_file(feature_path)
    try:
        # attempt to load features from a config file
        feature_gdf = extract_feature_from_geodataframe(
            original_feature_gdf, feature_type=feature_type
        )
    except ValueError as e:
        # if it isn't a config file then just ignore the error
        logger.info(f"This probably wasn't a config : {feature_path} \n {e}")
        feature_gdf = original_feature_gdf  # if this wasn't a config then just return the original file

    if feature_gdf.empty:
        raise ValueError(f"Empty {feature_type} file provided: {feature_path}")
    return feature_gdf


def load_feature_from_file(feature_path: str, feature_type: str) -> gpd.GeoDataFrame:
    """
    Load feature from file as GeoDataFrame.

    Args:
        feature_path: Path to feature file.
        feature_type: Type of feature.

    Returns:
        Feature GeoDataFrame.

    Raises:
        ValueError: If file empty or feature_type unsupported.
    """
    gdf = load_geodataframe_from_file(feature_path, feature_type)
    feature = create_feature(feature_type, gdf)
    feature_gdf = copy.deepcopy(feature.gdf)
    return feature_gdf


def create_feature(feature_type: str, gdf: gpd.GeoDataFrame) -> Any:
    """
    Create feature object from GeoDataFrame.

    Args:
        feature_type: Type of feature.
        gdf: GeoDataFrame.

    Returns:
        Feature object.

    Raises:
        ValueError: If feature_type unsupported.
    """
    feature_type = (
        feature_type.lower()
    )  # Convert to lower case for case insensitive comparison
    if feature_type in FEATURE_TYPE_MAP:
        if "transect" in feature_type:
            feature_object = FEATURE_TYPE_MAP[feature_type](transects=gdf)
        elif "shoreline_extraction_area" in feature_type:
            feature_object = FEATURE_TYPE_MAP[feature_type](gdf)
        elif "shoreline" in feature_type:
            feature_object = FEATURE_TYPE_MAP[feature_type](shoreline=gdf)
        elif "roi" in feature_type:
            feature_object = FEATURE_TYPE_MAP[feature_type](rois_gdf=gdf)
        else:
            feature_object = FEATURE_TYPE_MAP[feature_type](gdf)
    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    return feature_object


def extract_feature_from_geodataframe(
    gdf: gpd.GeoDataFrame, feature_type: str, type_column: str = "type"
) -> gpd.GeoDataFrame:
    """
    Extract features of type from GeoDataFrame.

    Args:
        gdf: GeoDataFrame to extract from.
        feature_type: Type of feature to extract.Typically one of the following:
            'shoreline', 'rois', 'transects', 'bbox'.
        type_column: Column name for types.

    Returns:
        Filtered GeoDataFrame.

    Raises:
        ValueError: If type_column missing or feature_type invalid.
    """
    # Convert column names to lower case for case-insensitive matching
    gdf.columns = gdf.columns.str.lower()
    type_column = type_column.lower()

    # Check if type_column exists in the GeoDataFrame
    if type_column not in gdf.columns:
        raise ValueError(
            f"Column '{type_column}' does not exist in the GeoDataFrame. Incorrect config_gdf.geojson loaded"
        )

    # Handling pluralization of feature_type
    feature_types = {
        feature_type.lower(),
        (feature_type + "s").lower(),
        (feature_type.rstrip("s")).lower(),
    }

    # Filter the GeoDataFrame for the specified types
    filtered_gdf = gdf[gdf[type_column].str.lower().isin(feature_types)]

    return filtered_gdf
