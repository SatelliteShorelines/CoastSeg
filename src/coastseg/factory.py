import logging
from math import sqrt
from typing import Union, Optional, Callable, Dict
from enum import Enum

from coastseg.bbox import Bounding_Box
from coastseg.shoreline import Shoreline
from coastseg.transects import Transects
from coastseg.roi import ROI
from coastseg import exception_handler
from coastseg import coastseg_map
from geopandas import GeoDataFrame

logger = logging.getLogger(__name__)

__all__ = ["Factory"]


def create_shoreline(
    coastsegmap, gdf: Optional[GeoDataFrame] = None, **kwargs
) -> Shoreline:
    """
    Create a `Shoreline` object from a `GeoDataFrame` or a `coastsegmap`.

    Args:
        coastsegmap: A `coastsegmap` object.
        gdf: A `GeoDataFrame` of the shoreline, or `None` to load the shoreline from the `CoastlineMap`.
        **kwargs: Optional keyword arguments.

    Returns:
        A `Shoreline` object.

    Raises:
        Object_Not_Found: If the `coastsegmap` does not have a valid bounding box or shoreline.
    """
    if gdf is not None:
        shoreline = Shoreline(shoreline=gdf)
    else:
        exception_handler.check_if_None(coastsegmap.bbox, "bounding box")
        exception_handler.check_if_gdf_empty(coastsegmap.bbox.gdf, "bounding box")
        shoreline = Shoreline(coastsegmap.bbox.gdf)
        exception_handler.check_if_gdf_empty(shoreline.gdf, "shoreline")

    logger.info("Shoreline were loaded on map")
    coastsegmap.shoreline = shoreline
    return shoreline


def create_transects(
    coastsegmap, gdf: Optional[GeoDataFrame] = None, **kwargs
) -> Transects:
    if gdf is not None:
        transects = Transects(transects=gdf)
    else:
        exception_handler.check_if_None(coastsegmap.bbox, "bounding box")
        exception_handler.check_if_gdf_empty(coastsegmap.bbox.gdf, "bounding box")
        transects = Transects(coastsegmap.bbox.gdf)
        exception_handler.check_if_gdf_empty(
            transects.gdf,
            "transects",
            "Transects Not Found in this region. Draw a new bounding box",
        )

    logger.info("Transects were loaded on map")
    coastsegmap.transects = transects
    return transects


def create_bbox(
    coastsegmap, gdf: Optional[GeoDataFrame] = None, **kwargs
) -> Bounding_Box:
    if gdf is not None:
        bbox = Bounding_Box(gdf)
        exception_handler.check_if_gdf_empty(bbox.gdf, "bounding box")
    else:
        geometry = coastsegmap.draw_control.last_draw["geometry"]
        bbox = Bounding_Box(geometry)
        exception_handler.check_if_gdf_empty(bbox.gdf, "bounding box")
    coastsegmap.remove_bbox()
    coastsegmap.draw_control.clear()
    logger.info("Bounding Box was loaded on map")
    coastsegmap.bbox = bbox
    return bbox


def create_rois(
    coastsegmap: coastseg_map, gdf: Optional[GeoDataFrame] = None, **kwargs
) -> ROI:
    if gdf is not None:
        rois = ROI(rois_gdf=gdf)
        exception_handler.check_if_gdf_empty(rois.gdf, "rois")
    else:
        # to create an ROI a bounding box must exist
        exception_handler.check_if_None(coastsegmap.bbox, "bounding box")
        # generate a shoreline within the bounding box
        if coastsegmap.shoreline is None:
            coastsegmap.load_feature_on_map("shoreline")
        logger.info(
            f"coastsegmap.shoreline:{coastsegmap.shoreline}\ncoastsegmap.bbox:{coastsegmap.bbox}"
        )

        lg_area = kwargs.get("lg_area")
        sm_area = kwargs.get("sm_area")
        units = kwargs.get("units")
        if lg_area is None or sm_area is None or units is None:
            raise Exception("Must provide ROI area and units")

        # if units is kilometers convert to meters
        if units == "km²":
            sm_area = sm_area * (10**6)
            lg_area = lg_area * (10**6)

        # get length of ROI square by taking square root of area
        small_len = sqrt(sm_area)
        large_len = sqrt(lg_area)

        # create rois within the bbox that intersect shorelines
        rois = ROI(
            coastsegmap.bbox.gdf,
            coastsegmap.shoreline.gdf,
            square_len_lg=large_len,
            square_len_sm=small_len,
        )
        exception_handler.check_if_gdf_empty(rois.gdf, "rois")

    coastsegmap.remove_all_rois()
    logger.info("ROIs were loaded on map")
    coastsegmap.rois = rois
    return rois


class FeatureType(Enum):
    SHORELINE = "shoreline"
    TRANSECTS = "transects"
    BBOX = "bbox"
    ROIS = "rois"


class Factory:
    # this dictionarymaps the feature name to the function that creates it
    # ex. "shoreline" maps to create_shoreline function
    _feature_makers: Dict[str, Callable] = {
        "shoreline": create_shoreline,
        "shorelines": create_shoreline,
        "transects": create_transects,
        "transect": create_transects,
        "bbox": create_bbox,
        "rois": create_rois,
        "roi": create_rois,
    }

    @staticmethod
    def make_feature(
        coastsegmap: "CoastSeg_Map",
        feature_name: str,
        gdf: Optional[GeoDataFrame] = None,
        **kwargs,
    ) -> Union[Shoreline, Transects, Bounding_Box, ROI]:
        logger.info(
            f"feature_name {feature_name}\ncoastsegmap: {coastsegmap}\nGdf: {gdf}\nkwargs: {kwargs}"
        )
        # get the function that can be used to create the feature_name
        # ex. "shoreline" would get the create_shoreline function
        feature_maker = Factory._feature_makers.get(feature_name)
        # if a geodataframe is provided
        if gdf is not None:
            if gdf.empty:
                return None

        return feature_maker(coastsegmap, gdf, **kwargs)
