# Standard library imports
import logging
# Internal dependencies imports
from src.coastseg import common
# External dependencies imports
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import geometry

logger = logging.getLogger(__name__)
logger.info("I am a log from %s",__name__)

class Roi():
    """Roi (Region of Interest) a square where data can be downloaded from
    """
   
    def __init__(self, bbox: gpd.GeoDataFrame,
                 shoreline: gpd.GeoDataFrame,
                 rois_gdf : gpd.GeoDataFrame,
                 square_len_lg : float=0,
                 square_len_sm : float=0,
                 filename:str=None):
        self.filename="rois.geojson"
        
        if rois_gdf:
            self.gdf = rois_gdf
        elif rois_gdf == None:
            self.gdf = self.create_geodataframe(bbox,shoreline,square_len_lg,square_len_sm)
        
        if square_len_sm == square_len_lg == 0:
            logger.error("Invalid square size for ROI")
            raise Exception("Invalid square size for ROI. Must be greater than 0")
        
        if filename:
            self.filename=filename
            
    def create_geodataframe(self, bbox:gpd.geodataframe, shoreline: gpd.GeoDataFrame, large_length:float=7500,small_length:float=5000,crs:str='EPSG:4326') -> gpd.GeoDataFrame:
        """Generates series of overlapping ROIS along shoreline on map using fishnet method
        with the crs specified by crs
        Args:
            
            crs (str, optional): coordinate reference system string. Defaults to 'EPSG:4326'.

        Returns:
            gpd.GeoDataFrame: 
        """
        # Create a single set of fishnets with square size = small and/or large side lengths that overlap each other
        if small_length or large_length == 0:
            # create a fishnet geodataframe with square size of either large_length or small_length
            fishnet_size = large_length if large_length!= 0 else small_length
            fishnet_intersect_gpd = self.fishnet_gpd(bbox, shoreline,fishnet_size)
        else:
            # Create two fishnets, one big (2000m) and one small(1500m) so they overlap each other
            fishnet_gpd_large = self.fishnet_gpd(bbox, shoreline, large_length)
            fishnet_gpd_small = self.fishnet_gpd(bbox, shoreline, small_length)
            # Concat the fishnets together to create one overlapping set of rois
            fishnet_intersect_gpd = gpd.GeoDataFrame(pd.concat([fishnet_gpd_large, fishnet_gpd_small], ignore_index=True))

        # Assign an id to each ROI square in the fishnet
        num_roi = len(fishnet_intersect_gpd)
        fishnet_intersect_gpd['id'] = np.arange(0, num_roi)
    
        return fishnet_intersect_gpd
    
    def fishnet_intersection(self, fishnet: gpd.GeoDataFrame,
                             data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Returns fishnet where it intersects with data
        Args:
            fishnet (geopandas.geodataframe.GeoDataFrame): geodataframe consisting of equal sized squares
            data (geopandas.geodataframe.GeoDataFrame): a vector or polygon for the fishnet to intersect

        Returns:
            geopandas.geodataframe.GeoDataFrame: intersection of fishnet and data
        """
        intersection_gpd = gpd.sjoin(fishnet, data, op='intersects')
        # intersection_gpd.drop(columns=['index_right', 'soc', 'exs', 'f_code', 'id', 'acc'], inplace=True)
        return intersection_gpd
    
    def create_rois(self, bbox:gpd.GeoDataFrame, square_size:float, input_espg="epsg:4326",output_espg="epsg:4326")->gpd.geodataframe:
        """Creates a fishnet of square shaped ROIs with side length= square_size

        Args:
            bbox (geopandas.geodataframe.GeoDataFrame): bounding box(bbox) where the ROIs will be generated
            square_size (float): side length of square ROI in meters
            input_espg (str, optional): espg code bbox is currently in. Defaults to "epsg:4326".
            output_espg (str, optional): espg code the ROIs will output to. Defaults to "epsg:4326".

        Returns:
            gpd.geodataframe: geodataframe containing all the ROIs
        """        
        # coords : list[tuple] first & last tuple are equal
        bbox_coords = list(bbox.iloc[0]['geometry'].exterior.coords)
        # get the utm_code for center of bbox
        center_x,center_y = common.get_center_rectangle(bbox_coords)
        utm_code = common.convert_wgs_to_utm(center_x,center_y)
        projected_espg=f'epsg:{utm_code}'
        logger.info(f"utm_code: {utm_code}")
        logger.info(f"projected_espg_code: {projected_espg}")
        # project geodataframe to new CRS specifed by utm_code
        projected_bbox_gdf = bbox.to_crs(projected_espg)
        # create fishnet of rois
        fishnet = self.create_fishnet(projected_bbox_gdf,projected_espg, output_espg,square_size)
        return fishnet 

    def create_fishnet(self, bbox_gdf:gpd.GeoDataFrame, input_espg:str, output_espg:str,square_size:float = 1000)->gpd.geodataframe:
        """Returns a fishnet of ROIs that intersects the bouding box specified by bbox_gdf where each ROI(square) has a side length = square size(meters)
        
        Args:
            bbox_gdf (gpd.geodataframe): Bounding box that fishnet intersects.
            input_espg (str): espg string that bbox_gdf is projected in
            output_espg (str): espg to convert the fishnet of ROIs to.
            square_size (int, optional): Size of each square in fishnet(meters). Defaults to 1000.

        Returns:
            gpd.geodataframe: fishnet of ROIs that intersects bbox_gdf. Each ROI has a side lenth = sqaure_size
        """
        minX, minY, maxX, maxY = bbox_gdf.total_bounds
        # Create a fishnet where each square has side length = square size
        x, y = (minX, minY)
        geom_array = []
        while y <= maxY:
            while x <= maxX:
                geom = geometry.Polygon([(x, y), (x, y + square_size), (x + square_size,
                                        y + square_size), (x + square_size, y), (x, y)])
                # add each square to geom_array
                geom_array.append(geom)
                x += square_size
            x = minX
            y += square_size
        
        # create geodataframe to hold all the (rois)squares
        fishnet = gpd.GeoDataFrame(geom_array, columns=['geometry']).set_crs(input_espg)
        logger.info(f"\n ROIs area before conversion to {output_espg}:\n {fishnet.area}")
        fishnet = fishnet.to_crs(output_espg)
        return fishnet

    def fishnet_gpd(
                self,
                gpd_bbox: gpd.GeoDataFrame,
                shoreline_gdf: gpd.GeoDataFrame,
                square_length: int = 1000) -> gpd.GeoDataFrame:
            """
            Returns fishnet where it intersects the shoreline
            
            Args:
                gpd_bbox (GeoDataFrame): bounding box (bbox) around shoreline
                shoreline_gdf (GeoDataFrame): shoreline in the bbox
                square_size (int, optional): size of each square in the fishnet. Defaults to 1000.

            Returns:
                GeoDataFrame: intersection of shoreline_gdf and fishnet. Only squares that intersect shoreline are kept
            """
            # Get the geodataframe for the fishnet within the bbox
            fishnet_gpd = self.create_rois(gpd_bbox, square_length)
            # Get the geodataframe for the fishnet intersecting the shoreline
            fishnet_intersect_gpd = self.fishnet_intersection(fishnet_gpd, shoreline_gdf)
            return fishnet_intersect_gpd
