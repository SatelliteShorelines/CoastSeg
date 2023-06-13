import os
import copy
import logging
from typing import Callable, List
from typing import Union

# Internal dependencies imports
from coastseg import exceptions
from coastseg import roi
from coastseg import shoreline
from coastseg import transects
import geopandas as gpd

logger = logging.getLogger(__name__)


def create_geofeature_geodataframe(
    geofeature_path: str, roi_gdf: gpd.GeoDataFrame, epsg_code: str, feature_type: str
) -> gpd.GeoDataFrame:
    """
    Creates geofeature (shoreline or transects) as a GeoDataFrame.

    If the geofeature file exists at the provided path, it is read into a GeoDataFrame.
    If the GeoDataFrame is not empty, geofeature are extracted using the appropriate class
    (Shoreline or Transects) and a deep copy of the resulting GeoDataFrame is made.
    If the geofeature file does not exist, geofeatures are created based on the ROI GeoDataFrame.
    The resulting GeoDataFrame is then converted to the specified EPSG code.

    Args:
        geofeature_path (str): Path to the geofeature file. 
        roi_gdf (gpd.GeoDataFrame): GeoDataFrame representing the region of interest.
        epsg_code (str): EPSG code for the desired coordinate reference system.
        feature_type (str): Type of geofeature (e.g., 'shoreline' or 'transect')

    Returns:
        gpd.GeoDataFrame: Geofeatures as a GeoDataFrame in the specified EPSG code.

    Raises:
        ValueError: If the geofeature file is empty.
    """
    feature_type = feature_type.lower()  # Convert to lower case for case insensitive comparison

    if os.path.exists(geofeature_path):
        # Load features from file
        geofeature_gdf = load_feature_from_file(geofeature_path, feature_type)
    else: 
        # if a geofeature file is not given load features from ROI
        geofeature_gdf = load_geofeatures_from_roi(roi_gdf, feature_type)

    logger.info(f"{feature_type}_gdf: {geofeature_gdf}")
    if geofeature_gdf.empty:
        raise Exception(f"None of the {feature_type}s intersected the ROI. Try a different {feature_type}")
    geofeature_gdf = geofeature_gdf.loc[:, ["id", "geometry"]]
    geofeature_gdf = geofeature_gdf.to_crs(epsg_code)
    return geofeature_gdf

def load_geofeatures_from_roi(roi_gdf: gpd.GeoDataFrame, feature_type: str) -> gpd.GeoDataFrame:
    """
    Given a Region Of Interest (ROI), this function attempts to load any geographic features (transects or shorelines)
    that exist in that region. If none exist, the user is advised to upload their own file.

    Args:
        roi_gdf (gpd.GeoDataFrame): GeoDataFrame representing the region of interest.
        feature_type (str): Type of the geographic feature, e.g. 'shoreline', 'transect'.

    Returns:
        gpd.GeoDataFrame: Geographic features as a GeoDataFrame.

    Raises:
        ValueError: If no geographic features were found in the given ROI.
    """
    feature_type = feature_type.lower()  # Convert to lower case for case insensitive comparison

    if feature_type == 'transect' or feature_type == 'transects':
        feature_object = transects.Transects(bbox=roi_gdf)
    elif feature_type == 'shoreline' or feature_type == 'shorelines':
        feature_object = shoreline.Shoreline(bbox=roi_gdf)
    else:
        logger.error(f"Unsupported feature_type: {feature_type}")
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    if feature_object.gdf.empty:
        logger.error(f"CoastSeg currently does not have {feature_type}s available in this region. Try uploading your own {feature_type}s.geojson")
        raise Exception(f"CoastSeg currently does not have {feature_type}s available in this region. Try uploading your own {feature_type}s.geojson")

    feature_gdf = copy.deepcopy(feature_object.gdf)

    return feature_gdf

def read_gpd_file(filename: str) -> gpd.GeoDataFrame:
    """
    Returns geodataframe from geopandas geodataframe file
    """
    if os.path.exists(filename):
        logger.info(f"Opening \n {filename}")
        return gpd.read_file(filename)
    else:
        raise FileNotFoundError

def load_geodataframe_from_file(feature_path: str, feature_type: str) -> gpd.GeoDataFrame:
    """
    Load a geographic feature from a file. The file is read into a GeoDataFrame.
    Can read both geojson files and config_gdf.geojson files

    Args:
        feature_path (str): Path to the feature file.
        feature_type (str): Type of the geographic feature, e.g. 'shoreline', 'transect','rois','bbox'

    Returns:
        gpd.GeoDataFrame: Geographic feature as a GeoDataFrame.

    Raises:
        ValueError: If the feature file is empty.
    """
    logger.info(f"Attempting to load {feature_type} from a file")
    feature_gdf = read_gpd_file(feature_path)
    try:
        # attempt to load features from a config file
        feature_gdf = extract_feature_from_geodataframe(feature_gdf, feature_type=feature_type)
    except ValueError as e:
        # if it isn't a config file then just ignore the error
        logger.info(f"This probably wasn't a config : {feature_path} \n {e}")
    if feature_gdf.empty:
        raise ValueError(f"Empty {feature_type} file provided: {feature_path}")
    return feature_gdf

def load_feature_from_file(feature_path: str, feature_type: str):
    gdf=load_geodataframe_from_file(feature_path,feature_type)
    feature = create_feature(feature_type,gdf)
    feature_gdf = copy.deepcopy(feature.gdf)
    return  feature_gdf



def create_feature(feature_type:str,gdf):
    if feature_type == 'transect' or feature_type == 'transects':
        feature_object = transects.Transects(transects=gdf)
    elif feature_type == 'shoreline' or feature_type == 'shorelines':
        feature_object = shoreline.Shoreline(shoreline=gdf)
    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")
    
    return feature_object


def extract_feature_from_geodataframe(
     gdf: gpd.GeoDataFrame, feature_type: str, type_column: str = "type"
) -> gpd.GeoDataFrame:
    """
    Extracts a GeoDataFrame of features of a given type and specified columns from a larger GeoDataFrame.
    
    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the features to extract.
        feature_type (str): The type of feature to extract. Typically one of the following 'shoreline','rois','transects','bbox'
        type_column (str, optional): The name of the column containing feature types. Defaults to 'type'.
        
    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame containing only the features of the specified type and columns.
        
    Raises:
        ValueError: Raised when feature_type or any of the columns specified do not exist in the GeoDataFrame.
    """
    # Check if type_column exists in the GeoDataFrame
    if type_column not in gdf.columns:
        raise ValueError(f"Column '{type_column}' does not exist in the GeoDataFrame. Incorrect config_gdf.geojson loaded")
    
    # select only the features that are of the correct type and have the correct columns
    feature_gdf = gdf[gdf[type_column] == feature_type]
    
    return feature_gdf