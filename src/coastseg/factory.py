import logging
from typing import Union

from coastseg.bbox import Bounding_Box
from coastseg import common
from coastseg.shoreline import Shoreline
from coastseg.transects import Transects
from coastseg.roi import ROI
from coastseg import exceptions
from coastseg import extracted_shoreline
from coastseg import exception_handler

logger = logging.getLogger(__name__)


class Factory:
    def _get_feature_maker(self, feature_type):
        if feature_type.lower() == "shoreline":
            return create_shoreline
        elif feature_type.lower() == "transects":
            return create_transects
        elif feature_type.lower() == "bbox":
            return create_bbox
        elif feature_type.lower() == "rois":
            return create_rois

    def make_feature(self, coastsegmap, feature, gdf=None, **kwargs):
        # get function to create feature based on feature requested
        feature_maker = self._get_feature_maker(feature)
        # create feature from geodataframe given by gdf
        if gdf is not None:
            return feature_maker(coastsegmap, gdf)
        # if geodataframe not provided then create feature from scratch
        return feature_maker(coastsegmap, **kwargs)


def create_shoreline(
    coastsegmap, gdf: "geopandas.GeoDataFrame" = None, **kwargs
) -> Shoreline:
    """Creates shorelines within bounding box on map

    Creates shorelines if no gdf(geodataframe). If a gdf(geodataframe) is
    provided a shoreline is created from it

    Sets coastsegmap.shoreline to created shoreline

    Raises:
        exceptions.Object_Not_Found: raised if bounding box is missing or empty
        exceptions.Object_Not_Found: raised if shoreline is empty
    """
    # if shoreline gdf is given make a shoreline from it
    if gdf is not None:
        shoreline = Shoreline(shoreline=gdf)
    else:
        # ensure valid bbox exists to create shorelines in
        exception_handler.check_if_None(coastsegmap.bbox, "bounding box")
        exception_handler.check_if_gdf_empty(coastsegmap.bbox.gdf, "bounding box")
        # if bounding box exists create shoreline within it
        shoreline = Shoreline(coastsegmap.bbox.gdf)
        exception_handler.check_if_gdf_empty(shoreline.gdf, "shoreline")

    # Save shoreline to coastseg_map
    logger.info("Shoreline were loaded on map")
    print("Shoreline were loaded on map")
    coastsegmap.shoreline = shoreline
    return shoreline


def create_transects(
    coastsegmap, gdf: "geopandas.GeoDataFrame" = None, **kwargs
) -> Transects:
    """Creates transects within bounding box on map

    Creates transects if no gdf(geodataframe). If a gdf(geodataframe) is
    provided a transect is created from it

    Sets coastsegmap.transect to created transect

    Raises:
        exceptions.Object_Not_Found: raised if bounding box is missing or empty
        exceptions.Object_Not_Found: raised if transect is empty
    """
    # if gdf is given make a transects from it
    if gdf is not None:
        transects = Transects(transects=gdf)
    else:
        # ensure valid bbox exists to create transects in
        exception_handler.check_if_None(coastsegmap.bbox, "bounding box")
        exception_handler.check_if_gdf_empty(coastsegmap.bbox.gdf, "bounding box")
        # if bounding box exists load transects within it
        transects = Transects(coastsegmap.bbox.gdf)
        exception_handler.check_if_gdf_empty(
            transects.gdf,
            "transects",
            "Transects Not Found in this region. Draw a new bounding box",
        )

    logger.info("Transects were loaded on map")
    print("Transects were loaded on map")
    coastsegmap.transects = transects
    return transects


def create_bbox(
    coastsegmap, gdf: "geopandas.GeoDataFrame" = None, **kwargs
) -> Bounding_Box:
    """Creates bounding box

    Creates bounding box if no gdf(geodataframe). If a gdf(geodataframe) is
    provided a bbox is created from it

    Sets coastsegmap.bbox to created bbox
    Raises:
        exceptions.Object_Not_Found: raised if bounding box is missing or empty
        exceptions.Object_Not_Found: raised if bbox is empty
    """
    # if gdf is given make a bounding box from it
    if gdf is not None:
        bbox = Bounding_Box(gdf)
        exception_handler.check_if_gdf_empty(bbox.gdf, "bounding box")
    else:
        # get last drawn polygon on map and create bounding box from it
        geometry = coastsegmap.draw_control.last_draw["geometry"]
        bbox = Bounding_Box(geometry)
        # make sure bounding box created is not empty
        exception_handler.check_if_gdf_empty(bbox.gdf, "bounding box")

    # clean drawn feature from map
    coastsegmap.remove_bbox()
    coastsegmap.draw_control.clear()
    logger.info("Bounding Box was loaded on map")
    print("Bounding Box was loaded on map")
    # save bbox to coastseg_map
    coastsegmap.bbox = bbox
    return bbox


def create_rois(
    coastsegmap, gdf: "geopandas.GeoDataFrame" = None, **kwargs
) -> Bounding_Box:
    """Creates rois (region of interest)

    Creates rois if no gdf(geodataframe). If a gdf(geodataframe) is
    provided rois are created from it

    Sets coastsegmap.rois to created rois
    Raises:
        exceptions.Object_Not_Found: raised if rois are missing or empty
        exceptions.Object_Not_Found: raised if rois are empty
    """
    # if gdf is given make a rois from it
    if gdf is not None:
        rois = ROI(rois_gdf=gdf)
        exception_handler.check_if_gdf_empty(rois.gdf, "rois")
    else:
        # create rois within bounding box
        large_len = None
        small_len = None
        if "large_len" in kwargs:
            large_len = kwargs["large_len"]
        if "small_len" in kwargs:
            small_len = kwargs["small_len"]
        if large_len is None or small_len is None:
            raise Exception("Must provide ROI lengths")
        rois = coastsegmap.generate_ROIS_fishnet(large_len, small_len)
        # make sure rois created is not empty
        exception_handler.check_if_gdf_empty(rois.gdf, "rois")

    # clean drawn feature from map
    coastsegmap.remove_all_rois()
    logger.info("ROIs was loaded on map")
    print("ROIs was loaded on map")
    # save rois to coastseg_map
    coastsegmap.rois = rois
    return rois
