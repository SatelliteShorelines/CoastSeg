import os
from ipyleaflet import DrawControl, GeoJSON, LayersControl
import leafmap
from CoastSeg import download_roi
from CoastSeg import bbox
from ipywidgets import Layout
import ipywidgets as widgets

from typing import Tuple
import pandas as pd
import numpy as np
import json
import geopandas as gpd
from ipyleaflet import GeoJSON
from shapely import geometry

debug_map_view = widgets.Output(layout={'border': '1px solid black'})


class CoastSeg_Map:

    shoreline_file = os.getcwd() + os.sep + "third_party_data" + os.sep + "stanford-xv279yj9196-geojson.json"

    def __init__(self, map_settings: dict = None):
        # data : geojson data of the rois generated
        self.data = None
        # selected_set : ids of the selected rois
        self.selected_set = set()
        # geojson_layer : layer with all rois
        self.geojson_layer = None
        # selected layer :  layer containing all selected rois
        self.selected_layer = None
        # shapes_list : Empty list to hold all the polygons drawn by the user
        self.shapes_list = []
        # coastline_for_map : coastline vector geojson for map layer
        self.coastline_for_map = None
        # selected_ROI : Geojson for all the ROIs selected by the user
        self.selected_ROI = None
        # self.transect_names : list of transect names on map
        self.transect_names = []
        
        CoastSeg_Map.check_shoreline_file_exists()
        # If map_settings is not provided use default settings
        if not map_settings:
            map_settings = {
                "center_point": (36, -121.5),
                "zoom": 13,
                "draw_control": False,
                "measure_control": False,
                "fullscreen_control": False,
                "attribution_control": True,
                "Layout": Layout(width='100%', height='100px')
            }
        self.m = leafmap.Map(
            draw_control=map_settings["draw_control"],
            measure_control=map_settings["measure_control"],
            fullscreen_control=map_settings["fullscreen_control"],
            attribution_control=map_settings["attribution_control"],
            center=map_settings["center_point"],
            zoom=map_settings["zoom"],
            layout=map_settings["Layout"])
        # # Create drawing controls
        self.draw_control = self.create_DrawControl(DrawControl())
        self.draw_control.on_draw(self.handle_draw)
        self.m.add_control(self.draw_control)
        layer_control = LayersControl(position='topright')
        self.m.add_control(layer_control)
        
     
    def load_transects_bbox_df(self) -> "pandas.core.frame.DataFrame":
        """Returns dataframe containing total bounds for each set of transects in the csv file,
        transects_bounding_boxes.csv.
        
        Returns:
            pandas.core.frame.DataFrame : Returns dataframe containing total bounds for each set of transects
        """        
        # Load in the total bounding box for each transects file
        transects_df=pd.read_csv("transects_bounding_boxes.csv")
        transects_df.index=transects_df["filename"]
        if 'filename' in transects_df.columns:
            transects_df.drop("filename",axis=1, inplace=True)
        return transects_df 
    
    
    def get_transect_filenames(self, gpd_bbox):
        # dataframe containig total bounding box for each transects file
        transects_df=self.load_transects_bbox_df()
        # filenames where transects'bbox intersect bounding box drawn by user
        intersecting_transect_files=[]
        for transect_file in transects_df.index:
            minx, miny, maxx, maxy=transects_df.loc[transect_file]
            transects_intersection = gpd_bbox.cx[minx:maxx, miny:maxy]
            gpd_bbox.cx[minx:maxx, miny:maxy].plot()
            # save transect filenames where gpd_bbox & bounding box for set of transects intersect
            if transects_intersection.empty == False:
                intersecting_transect_files.append(transect_file)
        return intersecting_transect_files
    
    
    def load_transect(self, filename :str)->'geopandas.geodataframe.GeoDataFrame':
        """Load transect as a geodataframe from geojson file containing transects called filename

        Args:
            filename (str): geojson file containing transects

        Returns:
            'geopandas.geodataframe.GeoDataFrame': (transects_gpd) transects from geojson file in geodataframe
            str : (transects_layer_name) name of transect file to be used as layer name
        """        
        # full path to transect file
        transect_path=os.path.abspath(os.getcwd())+os.sep+"Coastseg"+os.sep+"transects"+os.sep+filename
        # Loads transect file
        transects_gpd=bbox.read_gpd_file(transect_path)
        transects_layer_name=os.path.splitext(os.path.basename(transect_path))[0]
        return transects_gpd,transects_layer_name
    

    def clip_transect(self, transects_gdf:'geopandas.geodataframe.GeoDataFrame', bbox_gdf:'geopandas.geodataframe.GeoDataFrame')->'geopandas.geodataframe.GeoDataFrame':        
        """Clip transects_gdf to bbox_gdf. Only transects within bbox will be kept.

        Args:
            transects_gdf (geopandas.geodataframe.GeoDataFrame): transects read from geojson file
            bbox_gdf (geopandas.geodataframe.GeoDataFrame): drawn bbox

        Returns:
            geopandas.geodataframe.GeoDataFrame: clipped transects within bbox
        """        
        transects_in_bbox = gpd.clip(transects_gdf, bbox_gdf)
        transects_in_bbox = transects_in_bbox.to_crs('EPSG:4326')
        return transects_in_bbox


    def style_transect(self, transect_gdf:'geopandas.geodataframe.GeoDataFrame')->dict:
        """Converts transect_gdf to json and adds style to its properties

        Converts transect_gdf to json and adds an object called 'style' containing the styled
        json for the map.
        Args:
            transect_gdf (geopandas.geodataframe.GeoDataFrame): clipped transects loaded from geojson file

        Returns:
            dict: clipped transects with style properties
        """        
        # Save the fishnet intersection with coastline to json
        transect_dict = json.loads(transect_gdf.to_json())
        # Add style to each feature in the geojson
        for feature in transect_dict["features"]:
            feature["properties"]["style"] = {
                "color": "grey",
                "weight": 1,
                "fillColor": "grey",
                "fillOpacity": 0.2,
            }
        return transect_dict
        

    def load_transects_on_map(self) -> None:
        """Adds transects within the drawn bbox onto the map
        """        
        # ensure drawn bbox(bounding box) within allowed size
        bbox.validate_bbox_size(self.shapes_list)
        # load user drawn bbox as gdf(geodataframe)
        gpd_bbox = bbox.create_geodataframe_from_bbox(self.shapes_list)
        # list of all the transects files bbox intersects
        intersecting_transect_files = self.get_transect_filenames(gpd_bbox)
        # for each transect file clip it to the bbox and add to map
        for transect_file in intersecting_transect_files:
            print("Loading ",transect_file)
            data,transects_layer_name=self.load_transect(transect_file)
            transects_in_bbox=self.clip_transect(data, gpd_bbox)
            if transects_in_bbox.empty:
                print("Skipping ",transects_layer_name)
            else:
                print("Adding transects from ",transects_layer_name)
                # add transect's name to array transect_name 
                self.transect_names.append(transects_layer_name)
                # style and add the transect to the map 
                transect_layer=self.style_transect(transects_in_bbox)
                self.m.add_geojson(
                transect_layer, layer_name=transects_layer_name)
        if self.transect_names == []:
            print("No transects were found in this region. Draw a new bounding box.")


    def check_shoreline_file_exists():
        """ Prints an error message if the shoreline file does not exist"""
        if not os.path.exists(CoastSeg_Map.shoreline_file):
            print("\n The geojson shoreline file does not exist.")
            print("Please ensure the shoreline file is the directory 'third_party_data' ")


    def remove_all(self):
        """Remove the bbox, coastline, all rois from the map"""
        self.remove_bbox()
        self.remove_transects()
        self.remove_coastline()
        self.remove_all_rois()
        self.remove_saved_roi()

    def remove_transects(self):
        """Removes all the transects from the map.
        Removes each layer with its name in self.transect_names"""
        for transect_name in self.transect_names:
            existing_layer=self.m.find_layer(transect_name)
            if existing_layer is not None:
                self.m.remove_layer(existing_layer)
        self.transect_names=[]

    def remove_bbox(self):
        """Remove all the bounding boxes from the map"""
        self.draw_control.clear()
        self.shapes_list = []

    def remove_coastline(self):
        """Removes the coastline from the map"""
        if self.coastline_for_map:
            self.m.remove_layer(self.coastline_for_map)
            self.coastline_for_map = None

    def remove_saved_roi(self):
        """Removes all the saved ROI"""
        self.selected_ROI = None

    def remove_all_rois(self):
        """Removes all the unselected rois from the map """
        # Remove the selected rois
        existing_layer = self.m.find_layer('Selected ROIs')
        if existing_layer is not None:
            self.m.remove_layer(existing_layer)
            self.selected_layer = None
        existing_layer = self.m.find_layer('GeoJSON data')
        if existing_layer is not None:
            # Remove the layer from the map
            self.m.remove_layer(existing_layer)
            self.geojson_layer = None
            # clear the stylized geojson
            self.data = None
        self.selected_set = set()

    def create_DrawControl(self, draw_control):
        """ modifies the given draw control so that only rectangles can be drawn

        Args:
            draw_control (ipyleaflet.leaflet.DrawControl): draw control to modify

        Returns:
            ipyleaflet.leaflet.DrawControl: modified draw control with only ability to draw rectangles
        """
        draw_control.polyline = {}
        draw_control.circlemarker = {}
        draw_control.polygon = {}
        draw_control.rectangle = {
            "shapeOptions": {
                "fillColor": "green",
                "color": "green",
                "fillOpacity": 0.1,
                "Opacity": 0.1
            },
            "drawError": {
                "color": "#dd253b",
                "message": "Ops!"
            },
            "allowIntersection": False,
            "transform": True
        }
        return draw_control
    

    @debug_map_view.capture(clear_output=True)
    def handle_draw(self, target: 'ipyleaflet.leaflet.DrawControl', action: str, geo_json: dict):
        """Adds or removes the bounding box from shapes_list when drawn/deleted from map
        Args:
            target (ipyleaflet.leaflet.DrawControl): draw control used
            action (str): name of the most recent action ex. 'created', 'deleted'
            geo_json (dict): geojson dictionary for bounding box
        """
        self.action = action
        self.geo_json = geo_json
        self.target = target
        if self.draw_control.last_action == 'created' and self.draw_control.last_draw['geometry']['type'] == 'Polygon':
            self.shapes_list.append(self.draw_control.last_draw['geometry'])
        if self.draw_control.last_action == 'deleted':
            self.shapes_list.remove(self.draw_control.last_draw['geometry'])

    def fishnet_intersection(self, fishnet: "geopandas.geodataframe.GeoDataFrame",
                             data: "geopandas.geodataframe.GeoDataFrame") -> "geopandas.geodataframe.GeoDataFrame":
        """Returns fishnet where it intersects with data
        Args:
            fishnet (geopandas.geodataframe.GeoDataFrame): geodataframe consisting of equal sized squares
            data (geopandas.geodataframe.GeoDataFrame): a vector or polygon for the fishnet to intersect

        Returns:
            geopandas.geodataframe.GeoDataFrame: intersection of fishnet and data
        """
        intersection_gpd = gpd.sjoin(fishnet, data, op='intersects')
        intersection_gpd.drop(columns=['index_right', 'soc', 'exs', 'f_code', 'id', 'acc'], inplace=True)
        return intersection_gpd

    def fishnet(self, data: "geopandas.geodataframe.GeoDataFrame",
                square_size: int = 1000) -> "geopandas.geodataframe.GeoDataFrame":
        """Returns a fishnet that intersects data where each square is square_size
        Args:
            data (geopandas.geodataframe.GeoDataFrame): Bounding box that fishnet intersects.
            square_size (int, optional): _description_. Size of each square in fishnet. Defaults to 1000.

        Returns:
            geopandas.geodataframe.GeoDataFrame: Fishnet that intersects data
        """
        # Reproject to projected coordinate system so that map is in meters
        data = data.to_crs('EPSG:3857')
        # Get minX, minY, maxX, maxY from the extent of the geodataframe
        minX, minY, maxX, maxY = data.total_bounds
        # Create a fishnet
        x, y = (minX, minY)
        geom_array = []
        while y <= maxY:
            while x <= maxX:
                geom = geometry.Polygon([(x, y), (x, y + square_size), (x + square_size,
                                        y + square_size), (x + square_size, y), (x, y)])
                geom_array.append(geom)
                x += square_size
            x = minX
            y += square_size

        fishnet = gpd.GeoDataFrame(geom_array, columns=['geometry']).set_crs('EPSG:3857')
        fishnet = fishnet.to_crs('EPSG:4326')
        return fishnet


    def fishnet_gpd(
            self,
            gpd_bbox: "GeoDataFrame",
            coastline_gpd: "GeoDataFrame",
            square_size: int = 1000) -> "GeoDataFrame":
        """
        Returns fishnet where it intersects the coastline
        
        Args:
            gpd_bbox (GeoDataFrame): bounding box (bbox) around coastline
            coastline_gpd (GeoDataFrame): coastline in the bbox
            square_size (int, optional): size of each square in the fishnet. Defaults to 1000.

        Returns:
            GeoDataFrame: intersection of coastline_gpd and fishnet. Only squares that intersect coastline are kept
        """
        # Get the geodataframe for the fishnet within the bbox
        fishnet_gpd = self.fishnet(gpd_bbox, square_size)
        # Get the geodataframe for the fishnet intersecting the coastline
        fishnet_intersect_gpd = self.fishnet_intersection(fishnet_gpd, coastline_gpd)
        return fishnet_intersect_gpd


    def generate_ROIS_fishnet(self):
        """Generates series of overlapping ROIS along coastline on map using fishnet method"""
        # Ensure drawn bbox(bounding box) within allowed size
        bbox.validate_bbox_size(self.shapes_list)
        # dictionary containing geojson coastline
        roi_coastline = bbox.get_coastline(CoastSeg_Map.shoreline_file, self.shapes_list)
        # coastline geojson styled for the map
        self.coastline_for_map = self.get_coastline_layer(roi_coastline)
        self.m.add_layer(self.coastline_for_map)

        # Get the geodataframe for the coastline
        coastline_gpd = bbox.get_coastline_gpd(self.shoreline_file, self.shapes_list)
        # Get the geodataframe for the bbox
        gpd_bbox = bbox.create_geodataframe_from_bbox(self.shapes_list)
        # Create two fishnets, one big (2000m) and one small(1500m) so they overlap each other
        fishnet_gpd_large = self.fishnet_gpd(gpd_bbox, coastline_gpd,2000)
        fishnet_gpd_small = self.fishnet_gpd(gpd_bbox, coastline_gpd, 1500)

        # Concat the fishnets together to create one overlapping set of rois
        fishnet_intersect_gpd = gpd.GeoDataFrame(pd.concat([fishnet_gpd_large, fishnet_gpd_small], ignore_index=True))

        # Add an id column
        num_roi = int(fishnet_intersect_gpd.count())
        fishnet_intersect_gpd['id'] = np.arange(0, num_roi)

        # Save the fishnet intersection with coastline to json
        fishnet_geojson = fishnet_intersect_gpd.to_json()
        fishnet_dict = json.loads(fishnet_geojson)

        # Add style to each feature in the geojson
        for feature in fishnet_dict["features"]:
            feature["properties"]["style"] = {
                "color": "grey",
                "weight": 1,
                "fillColor": "grey",
                "fillOpacity": 0.2,
            }
        # Save the data
        self.data = fishnet_dict


    def get_coastline_layer(self, roi_coastline: dict) -> "ipyleaflet.GeoJSON":
        """get_coastline_layer returns the  coastline as GeoJson object.

        Args:
            roi_coastline (dict): geojson dictionary for portion of coastline in bbox

        Returns:
            "ipyleaflet.GeoJSON": coastline as GeoJson object styled with yellow dashes
        """
        assert roi_coastline != {}, "ERROR.\n Empty geojson cannot be drawn onto  map"
        return GeoJSON(
            data=roi_coastline,
            name="Coastline",
            style={
                'color': 'yellow',
                'fill_color': 'yellow',
                'opacity': 1,
                'dashArray': '5',
                'fillOpacity': 0.5,
                'weight': 4},
            hover_style={
                'color': 'white',
                'dashArray': '4',
                'fillOpacity': 0.7},
        )


    def get_geojson_layer(self) -> "'ipyleaflet.leaflet.GeoJSON'":
        """Returns GeoJSON for generated ROIs
        Returns:
            GeoJSON: geojson object that can be added to the map
        """
        if self.geojson_layer is None and self.data:
            self.geojson_layer = GeoJSON(data=self.data, name="GeoJSON data", hover_style={"fillColor": "red"})
        return self.geojson_layer


    def geojson_onclick_handler(self, event: str = None, id: 'NoneType' = None, properties: dict = None, **args):
        """On click handler for when unselected geojson is clicked.

        Adds the geojson's id to the selected_set. Replaces current selected layer with a new one that includes
        the recently clicked geojson.

        Args:
            event (str, optional): event fired ('click'). Defaults to None.
            id (NoneType, optional):  Defaults to None.
            properties (dict, optional): geojson dict for clicked geojson. Defaults to None.
        """
        if properties is None:
            return
        cid = properties["id"]
        self.selected_set.add(cid)
        if self.selected_layer is not None:
            self.m.remove_layer(self.selected_layer)

        self.selected_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(self.selected_set),
            name="Selected ROIs",
            hover_style={"fillColor": "blue"},
        )
        self.selected_layer.on_click(self.selected_onclick_handler)
        self.m.add_layer(self.selected_layer)

    def selected_onclick_handler(self, event: str = None, id: 'NoneType' = None, properties: dict = None, **args):
        """On click handler for selected geojson layer.

        Removes clicked layer's cid from the selected_set and replaces the select layer with a new one with
        the clicked layer removed from select_layer.

        Args:
            event (str, optional): event fired ('click'). Defaults to None.
            id (NoneType, optional):  Defaults to None.
            properties (dict, optional): geojson dict for clicked selected geojson. Defaults to None.
        """
        if properties is None:
            return
        # Remove the current layers cid from selected set
        cid = properties["id"]
        self.selected_set.remove(cid)
        if self.selected_layer is not None:
            self.m.remove_layer(self.selected_layer)
        # Recreate the selected layers wihout the layer that was removed
        self.selected_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(self.selected_set),
            name="Selected ROIs",
            hover_style={"fillColor": "blue"},
        )
        # Recreate the onclick handler for the selected layers
        self.selected_layer.on_click(self.selected_onclick_handler)
        # Add selected layer to the map
        self.m.add_layer(self.selected_layer)

    def add_geojson_layer_to_map(self):
        """ Add the geojson for the generated roi as a styled layer to the map"""
        geojson_layer = self.get_geojson_layer()
        geojson_layer.on_click(self.geojson_onclick_handler)
        self.m.add_layer(geojson_layer)

    def save_roi_fishnet(self, filename: str) -> None:
        """ Saves the selected roi to a geojson file called ""
        """
        # Saves the selected roi to a geojson file
        # returns the selected set as geojson
        selected_geojson = self.convert_selected_set_to_geojson(self.selected_set)
        filepath = os.path.join(os.getcwd(), filename)
        self.save_to_geojson_file(filepath, selected_geojson)
        self.selected_ROI = selected_geojson

    def save_to_geojson_file(self, out_file: str, geojson: dict, **kwargs) -> None:
        """save_to_geojson_file Saves given geojson to a geojson file at outfile

        Args:
            out_file (str): The output file path
            geojson (dict): geojson dict containing FeatureCollection for all geojson objects in selected_set
        """
        # Save the geojson to a file
        out_file = leafmap.check_file_path(out_file)
        ext = os.path.splitext(out_file)[1].lower()
        if ext == ".geojson":
            out_geojson = out_file
        else:
            out_geojson = os.path.splitext(out_file)[1] + ".geojson"

        with open(out_geojson, "w") as f:
            json.dump(geojson, f, **kwargs)

    def convert_selected_set_to_geojson(self, selected_set: set) -> dict:
        """Returns a geojson dict containing a FeatureCollection for all the geojson objects in the
        selected_set
        Args:
            selected_set (set): ids of the selected geojson

        Returns:
           dict: geojson dict containing FeatureCollection for all geojson objects in selected_set
        """
        geojson = {"type": "FeatureCollection", "features": []}
        # Select the geojson in the selected layer
        geojson["features"] = [
            feature
            for feature in self.data["features"]
            if feature["properties"]["id"] in selected_set
        ]
        # Modify geojson style for each polygon in the selected layer
        for feature in self.data["features"]:
            feature["properties"]["style"] = {
                "color": "blue",
                "weight": 2,
                "fillColor": "grey",
                "fillOpacity": 0.2,
            }
        return geojson
