# Standard library imports
import logging
import os

# Internal dependencies imports
from coastseg.common import remove_z_axis, read_gpd_file

# External dependencies imports
import geopandas as gpd
import pandas as pd
from ipyleaflet import GeoJSON


logger = logging.getLogger(__name__)


class Transects:
    """Transects: contains the transects within a region specified by bbox (bounding box)"""

    LAYER_NAME = "transects"

    def __init__(
        self,
        bbox: gpd.GeoDataFrame = None,
        transects: gpd.GeoDataFrame = None,
        filename: str = None,
    ):
        self.gdf = gpd.GeoDataFrame()
        self.filename = "transects.geojson"
        # if a transects geodataframe provided then copy it
        if transects is not None:
            if not transects.empty:
                # remove z-axis from transects
                transects = remove_z_axis(transects)
                self.gdf = transects

        elif bbox is not None:
            if not bbox.empty:
                self.gdf = self.create_geodataframe(bbox)
        if filename:
            self.filename = filename

    def create_geodataframe(
        self, bbox: gpd.GeoDataFrame, crs: str = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        """Creates a geodataframe with the crs specified by crs
        Args:
            rectangle (dict): geojson dictionary
            crs (str, optional): coordinate reference system string. Defaults to 'EPSG:4326'.

        Returns:
            gpd.GeoDataFrame: geodataframe with geometry column = rectangle and given crs
        """
        # geodataframe to hold all transects in bbox
        all_transects_in_bbox_gdf = gpd.GeoDataFrame()
        intersecting_transect_files = self.get_intersecting_files(bbox)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # for each transect file clip it to the bbox and add to map
        for transect_file in intersecting_transect_files:
            print("Loading ", transect_file)
            transects_name = os.path.splitext(transect_file)[0]
            transect_path = os.path.abspath(
                os.path.join(script_dir, "transects", transect_file)
            )
            transects_gdf = read_gpd_file(transect_path)
            # Get all the transects that intersect with bbox
            transects_in_bbox = gpd.sjoin(
                left_df=transects_gdf,
                right_df=bbox,
                how="inner",
                predicate="intersects",
            )
            columns_to_drop = list(
                set(transects_in_bbox.columns) - set(["id", "geometry", "slope"])
            )
            transects_in_bbox.drop(columns_to_drop, axis=1, inplace=True)

            if transects_in_bbox.empty:
                print("Skipping ", transects_name)
            elif not transects_in_bbox.empty:
                # if any transects intersect with bbox add them to all_transects_in_bbox_gdf that holds all the transects
                if all_transects_in_bbox_gdf.empty:
                    all_transects_in_bbox_gdf = transects_in_bbox
                elif not all_transects_in_bbox_gdf.empty:
                    # Combine shorelines from different files into single geodataframe
                    all_transects_in_bbox_gdf = gpd.GeoDataFrame(
                        pd.concat(
                            [all_transects_in_bbox_gdf, transects_in_bbox],
                            ignore_index=True,
                        )
                    )
                    print("Adding transects from ", transects_name)

        if all_transects_in_bbox_gdf.empty:
            logger.warning("No transects found here.")

        # remove z-axis from transects
        all_transects_in_bbox_gdf = remove_z_axis(all_transects_in_bbox_gdf)
        return all_transects_in_bbox_gdf

    def style_layer(self, geojson: dict, layer_name: str) -> dict:
        """Return styled GeoJson object with layer name

        Args:
            geojson (dict): geojson dictionary to be styled
            layer_name(str): name of the GeoJSON layer

        Returns:
            "ipyleaflet.GeoJSON": transects as styled GeoJSON layer
        """
        assert geojson != {}, "ERROR.\n Empty geojson cannot be drawn onto  map"
        # Add style to each feature in the geojson
        return GeoJSON(
            data=geojson,
            name=layer_name,
            style={
                "color": "grey",
                "fill_color": "grey",
                "opacity": 1,
                "fillOpacity": 0.2,
                "weight": 2,
            },
            hover_style={"color": "blue", "fillOpacity": 0.7},
        )

    def get_intersecting_files(self, bbox_gdf: gpd.geodataframe) -> list:
        """Returns a list of filenames that intersect with bbox_gdf

        Args:
            gpd_bbox (geopandas.geodataframe.GeoDataFrame): bbox containing ROIs
            type (str): to be used later

        Returns:
            list: intersecting_files containing filenames whose contents intersect with bbox_gdf
        """
        # dataframe containing total bounding box for each transects file
        total_bounds_df = self.load_total_bounds_df()
        # filenames where transects/shoreline's bbox intersect bounding box drawn by user
        intersecting_files = []
        for filename in total_bounds_df.index:
            minx, miny, maxx, maxy = total_bounds_df.loc[filename]
            intersection_df = bbox_gdf.cx[minx:maxx, miny:maxy]
            # save filenames where gpd_bbox & bounding box for set of transects intersect
            if intersection_df.empty == False:
                intersecting_files.append(filename)
        return intersecting_files

    def load_total_bounds_df(self) -> pd.DataFrame:
        """Returns dataframe containing total bounds for each set of shorelines in the csv file specified by location

        Args:
            location (str, optional): determines whether usa or world shoreline bounding boxes are loaded. Defaults to 'usa'.
            can be either 'world' or 'usa'

        Returns:
            pd.DataFrame:  Returns dataframe containing total bounds for each set of shorelines
        """
        # Load in the total bounding box from csv
        # Create the directory to hold the downloaded shorelines from Zenodo
        script_dir = os.path.dirname(os.path.abspath(__file__))
        bounding_box_dir = os.path.abspath(os.path.join(script_dir, "bounding_boxes"))
        if not os.path.exists(bounding_box_dir):
            os.mkdir(bounding_box_dir)

        transects_csv = os.path.join(bounding_box_dir, "transects_bounding_boxes.csv")
        if not os.path.exists(transects_csv):
            print("Did not find transects csv at ", transects_csv)
            # @todo download transects csv from github
        else:
            total_bounds_df = pd.read_csv(transects_csv)

        total_bounds_df.index = total_bounds_df["filename"]
        if "filename" in total_bounds_df.columns:
            total_bounds_df.drop("filename", axis=1, inplace=True)
        return total_bounds_df
