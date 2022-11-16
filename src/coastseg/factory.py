import os
import json
import logging
import copy
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
    # def __init__(self):
    def _get_feature_maker(self, feature_type):
        if feature_type == "shoreline":
            return create_shoreline
        elif feature_type == "transects":
            return create_transects
        elif feature_type == "bbox":
            return create_bbox

    def make_feature(self, coastsegmap, feature, gdf=None):
        # get function to create feature based on feature requested
        feature_maker = self._get_feature_maker(feature)
        # create feature from geodataframe
        if gdf is not None:
            return feature_maker(coastsegmap, gdf)
        # create feature from scratch
        return feature_maker(coastsegmap)


def create_shoreline(coastsegmap, gdf=None):
    # if shoreline gdf is given make a shoreline from it
    if gdf is not None:
        shoreline = Shoreline(shoreline=gdf)
    else:
        exception_handler.check_if_None(coastsegmap.bbox, "bounding box")
        exception_handler.check_if_gdf_empty(coastsegmap.bbox.gdf, "bounding box")
        # if bounding box exists create shoreline within it
        shoreline = Shoreline(coastsegmap.bbox.gdf)
        exception_handler.check_if_gdf_empty(shoreline.gdf, "shoreline")

    # Save shoreline to coastseg_map
    logger.info("Shoreline were loaded on map")
    print("Shoreline were loaded on map")
    coastsegmap.shoreline = shoreline
    # clean up (this is always done)
    return shoreline


def create_transects(coastsegmap, gdf=None):
    # if shoreline gdf is given make a shoreline from it
    if gdf is not None:
        transects = Transects(transects=gdf)
    else:
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


def create_bbox(coastsegmap, gdf=None):
    # if shoreline gdf is given make a shoreline from it
    if gdf is not None:
        bbox = Bounding_Box(gdf)
        exception_handler.check_if_gdf_empty(bbox.gdf, "bounding box")
    else:
        exception_handler.check_if_None(coastsegmap.bbox, "bounding box")
        exception_handler.check_if_gdf_empty(coastsegmap.bbox.gdf, "bounding box")
        # if bounding box exists load transects within it
        geometry = coastsegmap.draw_control.last_draw["geometry"]
        bbox = Bounding_Box(geometry)
        exception_handler.check_if_gdf_empty(bbox.gdf, "bounding box")

    # clean drawn feature from map
    coastsegmap.draw_control.clear()
    coastsegmap.bbox = bbox
    return bbox
