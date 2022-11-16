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

class Factory:
    # def __init__(self):
    def _get_feature_maker(self,feature_type):
        if feature_type == 'shoreline':
            return create_shoreline
        elif feature_type == 'transects':
            return create_transects
        
        
    def make_feature(self,coastsegmap,feature,file=""):
        # get function to create feature based on feature requested
        feature_maker = self._get_feature_maker(feature)
        if file !="":
            gdf = common.read_gpd_file(file)
            return feature_maker(coastsegmap,gdf)
        
        return feature_maker(coastsegmap)

def create_shoreline(coastsegmap,gdf=None):
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
        coastsegmap.shoreline = shoreline

    # clean up (this is always done)
    # return shoreline object (this is always done)
    return shoreline


def create_transects(coastsegmap,gdf=None):
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
        coastsegmap.shoreline = shoreline

    # clean up (this is always done)
    # return shoreline object (this is always done)
        