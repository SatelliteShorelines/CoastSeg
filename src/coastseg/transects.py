# Standard library imports
import logging
import os
from typing import List, Optional
import uuid

# Internal dependencies imports
from coastseg.common import remove_z_axis, replace_column

# External dependencies imports
import geopandas as gpd
import pandas as pd
from ipyleaflet import GeoJSON


logger = logging.getLogger(__name__)

def load_intersecting_transects(
    rectangle: gpd.GeoDataFrame, transect_files: List[str], transect_dir: str
) -> gpd.GeoDataFrame:
    """
    Loads transects from a list of GeoJSON files in the transect directory, selects the transects that intersect with
    a rectangle defined by a GeoDataFrame, and returns a new GeoDataFrame with the selected columns ('id', 'geometry', 'slope').

    Args:
        rectangle (gpd.GeoDataFrame): A GeoDataFrame defining the rectangle to select transects within.
        transect_files (List[str]): A list of filenames of the GeoJSON transect files to load.
        transect_dir (str): The directory where the GeoJSON transect files are located.

    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame with the selected columns ('id', 'geometry', 'slope') containing the transects
        that intersect with the rectangle.
    """
    # Create an empty GeoDataFrame to hold the selected transects
    selected_transects = gpd.GeoDataFrame(columns=["id", "geometry", "slope"])

    # Get the bounding box of the rectangle
    bbox = rectangle.bounds.iloc[0].tolist()

    # Iterate over each transect file and select the transects that intersect with the rectangle
    for transect_file in transect_files:
        transects_name = os.path.splitext(transect_file)[0]
        transect_path = os.path.join(transect_dir, transect_file)
        transects = gpd.read_file(transect_path,bbox=bbox)
        if transects.empty:
            logger.info("Skipping %s", transects_name)
            continue
        elif not transects.empty:
            logger.info("Adding transects from %s", transects_name)
        if "id" not in [col.lower() for col in transects.columns]:
            transects["id"] = transects.apply(lambda row: uuid.uuid4().hex, axis=1)
        # Append the selected transects to the output GeoDataFrame
        selected_transects = pd.concat(
            [selected_transects, transects], ignore_index=True
        )
    return selected_transects


class Transects:
    """A class representing a collection of transects within a specified bounding box."""

    LAYER_NAME = "transects"

    def __init__(
        self,
        bbox: gpd.GeoDataFrame = None,
        transects: gpd.GeoDataFrame = None,
        filename: str = None,
    ):
        """
        Initialize a Transects object with either a bounding box GeoDataFrame, a transects GeoDataFrame, 
        or a filename string.
        """
        self.gdf = gpd.GeoDataFrame()
        self.filename = filename if filename else "transects.geojson"
        self.process_transects(bbox, transects)


    def __str__(self):
        return f"Transects: geodataframe {self.gdf}"

    def __repr__(self):
        return f"Transects: geodataframe {self.gdf}"

    def process_transects(self, bbox: Optional[gpd.GeoDataFrame] = None, transects: Optional[gpd.GeoDataFrame] = None):
        """
        Processes the transects or bounding box provided during initialization.
        """
        if transects is not None:
            self.process_provided_transects(transects)

        elif bbox is not None:
            self.process_bbox(bbox)


    def process_provided_transects(self, transects: gpd.GeoDataFrame):
        """
        Initalize transects with the provided transects in a geodataframe
        """
        if not transects.empty:
            # if 'id' column is not present and 'name' column is replace 'name' with 'id'
            # id neither exist create a new column named 'id' with row index
            if "ID" in transects.columns:
                logger.info(f"ID in transects.columns: {transects.columns}")
                transects.rename(columns={"ID": "id"}, inplace=True)
            replace_column(transects, new_name="id", replace_col="name")
            # remove z-axis from transects
            transects = remove_z_axis(transects)
            # if an id column does not exist create one
            if "id" not in [col.lower() for col in transects.columns]:
                transects["id"] = transects.apply(lambda row: uuid.uuid4().hex, axis=1)
            self.gdf = transects

    def process_bbox(self, bbox: gpd.GeoDataFrame):
        """
        Processes the provided bounding box GeoDataFrame.
        """
        if not bbox.empty:
            self.gdf = self.create_geodataframe(bbox)

    def create_geodataframe(
        self,
        bbox: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Creates a geodataframe with the crs specified by crs
        Args:
            rectangle (dict): geojson dictionary
        Returns:
            gpd.GeoDataFrame: geodataframe with geometry column = rectangle and given crs
        """
        # create a new dataframe that only contains the geometry column of the bbox
        bbox = bbox[["geometry"]]
        # get transect geosjson files that intersect with bounding box
        intersecting_transect_files = self.get_intersecting_files(bbox)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        transect_dir = os.path.abspath(os.path.join(script_dir, "transects"))
        # for each transect file clip it to the bbox and add to map
        transects_in_bbox = load_intersecting_transects(
            bbox, intersecting_transect_files, transect_dir
        )
        if transects_in_bbox.empty:
            logger.warning("No transects found here.")

        # remove z-axis from transects
        transects_in_bbox = remove_z_axis(transects_in_bbox)
        transects_in_bbox["id"] = transects_in_bbox.index.astype(str).tolist()

        return transects_in_bbox


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
        else:
            total_bounds_df = pd.read_csv(transects_csv)

        total_bounds_df.index = total_bounds_df["filename"]
        if "filename" in total_bounds_df.columns:
            total_bounds_df.drop("filename", axis=1, inplace=True)
        return total_bounds_df
