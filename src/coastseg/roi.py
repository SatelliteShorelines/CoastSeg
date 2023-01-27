# Standard library imports
import logging
import os
import json

# Internal dependencies imports
from coastseg import common
from coastseg import exceptions

# External dependencies imports
import geopandas as gpd
import pandas as pd
from shapely import geometry
from ipyleaflet import GeoJSON

logger = logging.getLogger(__name__)


class ROI:
    """A class that controls all the ROIs on the map"""

    LAYER_NAME = "ROIs"
    SELECTED_LAYER_NAME = "Selected ROIs"

    def __init__(
        self,
        bbox: gpd.GeoDataFrame = None,
        shoreline: gpd.GeoDataFrame = None,
        rois_gdf: gpd.GeoDataFrame = None,
        square_len_lg: float = 0,
        square_len_sm: float = 0,
        filename: str = None,
    ):
        # roi_settings : after ROIs have been downloaded holds all download settings
        self.roi_settings = {}
        # extract_shorelines : dictionary with ROIs' ids as the keys holding the extracted shorelines
        self.extracted_shorelines = {}
        # cross_distance_transects : dictionary with ROIs' ids as the keys holding the transects cross distances
        self.cross_distance_transects = {}
        self.filename = "rois.geojson"
        if filename:
            self.filename = filename

        if rois_gdf is not None:
            # check if geodataframe column has 'id' column and add one if one doesn't exist
            if "id" not in rois_gdf.columns:
                rois_gdf["id"] = list(map(str, rois_gdf.index.tolist()))
            # get row ids of ROIs with area that's too large
            drop_ids = common.get_ids_with_invalid_area(rois_gdf)
            if len(drop_ids) > 0:
                print("Dropping ROIs that are an invalid size ")
                logger.info(f"Dropping ROIs that are an invalid size {drop_ids}")
                rois_gdf.drop(index=drop_ids, axis=0, inplace=True)
            # convert crs of ROIs to the map crs
            rois_gdf.to_crs("EPSG:4326")
            self.gdf = rois_gdf
            return

        elif rois_gdf is None:
            if shoreline is None:
                raise exceptions.Object_Not_Found("shorelines")
            if bbox is None:
                raise exceptions.Object_Not_Found("bounding box")
            if shoreline.empty:
                raise exceptions.Object_Not_Found("shorelines")
            if bbox.empty:
                raise exceptions.Object_Not_Found("bounding box")

        if square_len_sm == square_len_lg == 0:
            logger.error("Invalid square size for ROI")
            raise Exception("Invalid square size for ROI. Must be greater than 0")

        if rois_gdf is None:
            self.gdf = self.create_geodataframe(
                bbox, shoreline, square_len_lg, square_len_sm
            )

    def set_roi_settings(self, roi_settings: dict) -> None:
        logger.info(f"Saving roi_settings {roi_settings}")
        self.roi_settings = roi_settings

    def update_extracted_shorelines(self, extract_shoreline: dict):
        self.extracted_shorelines.update(extract_shoreline)
        logger.info(f"After self.extracted_shorelines : {self.extracted_shorelines}")

    def create_geodataframe(
        self,
        bbox: gpd.geodataframe,
        shoreline: gpd.GeoDataFrame,
        large_length: float = 7500,
        small_length: float = 5000,
        crs: str = "EPSG:4326",
    ) -> gpd.GeoDataFrame:
        """Generates series of overlapping ROIS along shoreline on map using fishnet method
        with the crs specified by crs
        Args:

            crs (str, optional): coordinate reference system string. Defaults to 'EPSG:4326'.

        Returns:
            gpd.GeoDataFrame: a series of ROIs along the shoreline. The ROIs are squares with side lengths of both
            large_length and small_length
        """
        # Create a single set of fishnets with square size = small and/or large side lengths that overlap each other
        logger.info(f"Small Length: {small_length}  Large Length: {large_length}")
        if small_length == 0 or large_length == 0:
            logger.info("Creating one fishnet")
            # create a fishnet geodataframe with square size of either large_length or small_length
            fishnet_size = large_length if large_length != 0 else small_length
            fishnet_intersect_gdf = self.get_fishnet_gdf(bbox, shoreline, fishnet_size)
        else:
            logger.info("Creating two fishnets")
            # Create two fishnets, one big (2000m) and one small(1500m) so they overlap each other
            fishnet_gpd_large = self.get_fishnet_gdf(bbox, shoreline, large_length)
            fishnet_gpd_small = self.get_fishnet_gdf(bbox, shoreline, small_length)
            logger.info(f"fishnet_gpd_large : {fishnet_gpd_large}")
            logger.info(f"fishnet_gpd_small : {fishnet_gpd_small}")
            # Concat the fishnets together to create one overlapping set of rois
            fishnet_intersect_gdf = gpd.GeoDataFrame(
                pd.concat([fishnet_gpd_large, fishnet_gpd_small], ignore_index=True)
            )

        # Assign an id to each ROI square in the fishnet
        ids = range(0, len(fishnet_intersect_gdf), 1)
        fishnet_intersect_gdf["id"] = list(map(lambda x: str(x), ids))
        del ids
        logger.info(f"Created fishnet_intersect_gdf: {fishnet_intersect_gdf}")
        return fishnet_intersect_gdf

    def style_layer(self, geojson: dict, layer_name: str) -> "ipyleaflet.GeoJSON":
        """Return styled GeoJson object with layer name

        Args:
            geojson (dict): geojson dictionary to be styled
            layer_name(str): name of the GeoJSON layer
        Returns:
            "ipyleaflet.GeoJSON": ROIs as GeoJson layer styled with yellow dashes
        """
        assert geojson != {}, "ERROR.\n Empty geojson cannot be drawn onto  map"
        return GeoJSON(
            data=geojson,
            name=layer_name,
            style={
                "color": "#555555",
                "fill_color": "#555555",
                "fillOpacity": 0.1,
                "weight": 1,
            },
            hover_style={"color": "red", "fillOpacity": 0.1, "color": "crimson"},
        )

    def fishnet_intersection(
        self, fishnet: gpd.GeoDataFrame, data: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Returns fishnet where it intersects with data
        Args:
            fishnet (geopandas.geodataframe.GeoDataFrame): geodataframe consisting of equal sized squares
            data (geopandas.geodataframe.GeoDataFrame): a vector or polygon for the fishnet to intersect

        Returns:
            geopandas.geodataframe.GeoDataFrame: intersection of fishnet and data
        """
        intersection_gdf = gpd.sjoin(
            left_df=fishnet, right_df=data, how="inner", predicate="intersects"
        )
        columns_to_drop = list(intersection_gdf.columns.difference(["geometry"]))
        intersection_gdf.drop(columns=columns_to_drop, inplace=True)
        intersection_gdf.drop_duplicates(inplace=True)
        return intersection_gdf

    def create_rois(
        self,
        bbox: gpd.GeoDataFrame,
        square_size: float,
        input_espg="epsg:4326",
        output_espg="epsg:4326",
    ) -> gpd.geodataframe:
        """Creates a fishnet of square shaped ROIs with side length= square_size

        Args:
            bbox (geopandas.geodataframe.GeoDataFrame): bounding box(bbox) where the ROIs will be generated
            square_size (float): side length of square ROI in meters
            input_espg (str, optional): espg code bbox is currently in. Defaults to "epsg:4326".
            output_espg (str, optional): espg code the ROIs will output to. Defaults to "epsg:4326".

        Returns:
            gpd.geodataframe: geodataframe containing all the ROIs
        """
        projected_espg = common.get_epsg_from_geometry(bbox.iloc[0]["geometry"])
        logger.info(f"projected_espg_code: {projected_espg}")
        # project geodataframe to new CRS specified by utm_code
        projected_bbox_gdf = bbox.to_crs(projected_espg)
        # create fishnet of rois
        fishnet = self.create_fishnet(
            projected_bbox_gdf, projected_espg, output_espg, square_size
        )
        return fishnet

    def create_fishnet(
        self,
        bbox_gdf: gpd.GeoDataFrame,
        input_espg: str,
        output_espg: str,
        square_size: float = 1000,
    ) -> gpd.geodataframe:
        """Returns a fishnet of ROIs that intersects the bounding box specified by bbox_gdf where each ROI(square) has a side length = square size(meters)

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
                geom = geometry.Polygon(
                    [
                        (x, y),
                        (x, y + square_size),
                        (x + square_size, y + square_size),
                        (x + square_size, y),
                        (x, y),
                    ]
                )
                # add each square to geom_array
                geom_array.append(geom)
                x += square_size
            x = minX
            y += square_size

        # create geodataframe to hold all the (rois)squares
        fishnet = gpd.GeoDataFrame(geom_array, columns=["geometry"]).set_crs(input_espg)
        logger.info(
            f"\n ROIs area before conversion to {output_espg}:\n {fishnet.area}"
        )
        fishnet = fishnet.to_crs(output_espg)
        return fishnet

    def get_fishnet_gdf(
        self,
        bbox_gdf: gpd.GeoDataFrame,
        shoreline_gdf: gpd.GeoDataFrame,
        square_length: int = 1000,
    ) -> gpd.GeoDataFrame:
        """
        Returns fishnet where it intersects the shoreline

        Args:
            bbox_gdf (GeoDataFrame): bounding box (bbox) around shoreline
            shoreline_gdf (GeoDataFrame): shoreline in the bbox
            square_size (int, optional): size of each square in the fishnet. Defaults to 1000.

        Returns:
            GeoDataFrame: intersection of shoreline_gdf and fishnet. Only squares that intersect shoreline are kept
        """
        # Get the geodataframe for the fishnet within the bbox
        fishnet = self.create_rois(bbox_gdf, square_length)
        # Get the geodataframe for the fishnet intersecting the shoreline
        fishnet_intersection = self.fishnet_intersection(fishnet, shoreline_gdf)
        return fishnet_intersection

    def save_transects_to_json(self, roi_id: int, cross_distance: dict):
        if cross_distance == 0:
            print(f"Did not save transects to json for ROI:{roi_id}")
            logger.info(f"Did not save transects to json for ROI:{roi_id}")
            return
        sitename = str(self.roi_settings[roi_id]["sitename"])
        logger.info(f"sitename: {sitename}")
        filename = f"transects_cross_distances" + str(roi_id) + ".json"
        logger.info(f"filename: {filename}")
        save_path = os.path.abspath(
            os.path.join(os.getcwd(), "data", sitename, filename)
        )
        logger.info(f"save_path: {save_path}")
        for key in cross_distance.keys():
            tmp = cross_distance[key].tolist()
            cross_distance[key] = tmp
        logger.info(f"cross_distance: {cross_distance}")

        with open(save_path, "w") as f:
            json.dump(cross_distance, f)
        print(f"\nSaved transects to json for ROI: {roi_id} {save_path}")
        logger.info(f"Saved transects to json for ROI: {roi_id} {save_path}")
