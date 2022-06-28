import os
from ipyleaflet import DrawControl, GeoJSON, LayersControl
import leafmap
from CoastSeg import download_roi
from CoastSeg import bbox
from CoastSeg import make_overlapping_roi
from ipywidgets import Layout
import ipywidgets as widgets

import json
import geopandas as gpd
from ipyleaflet import GeoJSON
from shapely import geometry

debug_map_view = widgets.Output(layout={'border': '1px solid black'})
class CoastSeg_Map:
    
    shoreline_file=os.getcwd()+os.sep+"third_party_data"+os.sep+"stanford-xv279yj9196-geojson.json"
    
    
    def __init__(self, map_settings: dict=None):
        # data : geojson data of the rois generated
        self.data=None
        # selected_set : ids of the selected rois
        self.selected_set=set()
        # geojson_layer : layer with all rois
        self.geojson_layer = None
        # selected layer :  layer containing all selected rois
        self.selected_layer = None
        # shapes_list : Empty list to hold all the polygons drawn by the user
        self.shapes_list=[]
        # coastline_for_map : coastline vector geojson for map layer
        self.coastline_for_map=None
        # selected_ROI : Geojson for all the ROIs selected by the user
        self.selected_ROI=None
        self.collection=None
        CoastSeg_Map.check_shoreline_file_exists()
        # If map_settings is not provided use default settings
        if not map_settings:
            map_settings={
            "center_point": ( 36, -121.5),
            "zoom":13,
            "draw_control":False,
            "measure_control":False, 
            "fullscreen_control":False, 
            "attribution_control":True,
            "Layout":Layout(width='100%', height='100px')
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
        self.draw_control=self.create_DrawControl(DrawControl())
        self.draw_control.on_draw(self.handle_draw)
        self.m.add_control(self.draw_control)
        layer_control = LayersControl(position='topright')
        self.m.add_control(layer_control)
    
    
    def check_shoreline_file_exists():
        """ Prints an error message if the shoreline file does not exist"""
        if not os.path.exists(CoastSeg_Map.shoreline_file):
            print("\n The geojson shoreline file does not exist.")
            print("Please ensure the shoreline file is the directory 'third_party_data' ")
    
    
    def remove_all(self):
        """Remove the bbox, coastline, all rois from the map"""
        self.remove_bbox()
        self.remove_coastline()
        self.remove_all_rois()
        self.remove_saved_roi()
    
    
    def remove_bbox(self):
        """Remove all the bounding boxes from the map"""
        self.draw_control.clear()
        self.shapes_list=[]
    
    
    def remove_coastline(self):
        """Removes the coastline from the map"""
        if self.coastline_for_map:  
            self.m.remove_layer(self.coastline_for_map)
            self.coastline_for_map = None


    def remove_saved_roi(self):
        """Removes all the saved ROI"""
        self.selected_ROI=None


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
   
    
    def create_DrawControl(self,draw_control):
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
            "transform":True
        }
        return  draw_control 
    

    @debug_map_view.capture(clear_output=True)
    def handle_draw(self,target: 'ipyleaflet.leaflet.DrawControl', action :str, geo_json :dict):
        """Adds or removes the bounding box from shapes_list when drawn/deleted from map
        Args:
            target (ipyleaflet.leaflet.DrawControl): draw control used
            action (str): name of the most recent action ex. 'created', 'deleted'
            geo_json (dict): geojson dictionary for bounding box
        """
        self.action = action
        self.geo_json = geo_json
        self.target = target
        if self.draw_control.last_action == 'created'and self.draw_control.last_draw['geometry']['type']=='Polygon' :
            self.shapes_list.append( self.draw_control.last_draw['geometry'])
        if self.draw_control.last_action == 'deleted':
            self.shapes_list.remove(self.draw_control.last_draw['geometry'])
    

    def fishnet_intersection(fishnet:"geopandas.geodataframe.GeoDataFrame",data: "geopandas.geodataframe.GeoDataFrame")->"geopandas.geodataframe.GeoDataFrame":
        """Returns fishnet where it intersects with data
        Args:
            fishnet (geopandas.geodataframe.GeoDataFrame): geodataframe consisting of equal sized squares
            data (geopandas.geodataframe.GeoDataFrame): a vector or polygon for the fishnet to intersect

        Returns:
            geopandas.geodataframe.GeoDataFrame: intersection of fishnet and data
        """    
        intersection_gpd=gpd.sjoin(fishnet, data, op='intersects')
        intersection_gpd.drop(columns=['index_right','soc','exs','f_code','id','acc'],inplace=True)
        return intersection_gpd
    

    def fishnet(data:"geopandas.geodataframe.GeoDataFrame", square_size :int=1000) -> "geopandas.geodataframe.GeoDataFrame":
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
                geom = geometry.Polygon([(x,y), (x, y+square_size), (x+square_size, y+square_size), (x+square_size, y), (x, y)])
                geom_array.append(geom)
                x += square_size
            x = minX
            y += square_size
    
        fishnet = gpd.GeoDataFrame(geom_array, columns=['geometry']).set_crs('EPSG:3857')
        fishnet=fishnet.to_crs('EPSG:4326')
        return fishnet 


    def set_data(self, roi_filename:str):
        """Creates styled geojson for rois generated based on geojson read in from roi_filename  
        Args:
            roi_filename (str): name of the geojson file containing the geojson
            of all the rois generated
        """
        # Read the geojson for all the ROIs generated
        self.data=download_roi.read_geojson_file(roi_filename)
        # Add style to each feature in the geojson
        for feature in self.data["features"]:
            feature["properties"]["style"] = {
                "color": "grey",
                "weight": 1,
                "fillColor": "grey",
                "fillOpacity": 0.2,
            }


    def generate_ROIS_fishnet(self ):
        """Generates  series of overlapping ROIS along the coastline on the map using the fishnet method
        """
        # Make sure your bounding box is within the allowed size
        bbox.validate_bbox_size(self.shapes_list)
        #dictionary containing geojson coastline
        roi_coastline=bbox.get_coastline(CoastSeg_Map.shoreline_file, self.shapes_list)
        # coastline geojson styled for the map
        self.coastline_for_map=self.get_coastline_layer(roi_coastline)
        self.m.add_layer(self.coastline_for_map)
        # Get the geodataframe for the coastline
        roi_coastline=bbox.get_coastline_gpd(self.shoreline_file, self.shapes_list)
        # Get the geodataframe for the bbox
        gpd_bbox=bbox.create_geodataframe_from_bbox(self.shapes_list)
        # Get the geodataframe for the fishnet within the bbox
        fishnet_gpd=bbox.fishnet(gpd_bbox)
        # Get the geodataframe for the fishnet intersecting the coastline 
        fishnet_intersect_gpd=bbox.fishnet_intersection(fishnet_gpd,roi_coastline)
        # Save the fishnet intersection with coastline to json
        fishnet_geojson=fishnet_intersect_gpd.to_json()
        fishnet_dict=json.loads(fishnet_geojson)
        # Convert the json to stylized geojson
        # fishnet_for_map =GeoJSON(data=json.loads(fishnet_geojson), name="fishnet", hover_style={"fillColor": "red"})
        # Add style to each feature in the geojson
        for feature in fishnet_dict["features"]:
            feature["properties"]["style"] = {
                "color": "grey",
                "weight": 1,
                "fillColor": "grey",
                "fillOpacity": 0.2,
            }
        # Save the data
        self.data=fishnet_dict

    
    def generate_ROIS(self, roi_filename :str, csv_filename: str, overlap_percent:float = None ):
        """Generates  series of overlapping ROIS along the coastline on the map
        Args:
            roi_filename (str): name of the geojson file containing the geojson
            of all the rois generated
            csv_filename (str): name of csv file containing overlap
        """
        # Make sure your bounding box is within the allowed size
        bbox.validate_bbox_size(self.shapes_list)
        #dictionary containing geojson coastline
        roi_coastline=bbox.get_coastline(CoastSeg_Map.shoreline_file, self.shapes_list)
        # #coastline styled for the map
        self.coastline_for_map=self.get_coastline_layer(roi_coastline)
        self.m.add_layer(self.coastline_for_map)
        # #Get the rois using the coastline  within bounding box
        make_overlapping_roi.get_ROIs(roi_coastline,roi_filename,csv_filename)
        # Save the data from the ROI file to data
        self.set_data(roi_filename)
        # overlap_btw_vectors_df=make_overlapping_roi.min_overlap_btw_vectors(roi_filename,csv_filename, overlap_percent)


    def save_roi_to_file(self, selected_roi_file : str, roi_filename :str):
        """saves the selected roi to a geojson file with the name selected_roi_file

        Args:
            selected_roi_file (str):  The name of the geojson file to save the ROI selected
            roi_filename (str):The filename of the geojson file containing all the ROI
        """

        self.selected_ROI=download_roi.save_roi(roi_filename, selected_roi_file, self.selected_set)
  
 
    def get_coastline_layer(self,roi_coastline: dict):
        """get_coastline_layer _summary_

        _extended_summary_

        Args:
            roi_coastline (dict): _description_

        Returns:
            _type_: _description_
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
             self.geojson_layer=GeoJSON(data=self.data, name="GeoJSON data", hover_style={"fillColor": "red"})
        return self.geojson_layer
    
    
    def geojson_onclick_handler(self, event: str=None , id : 'NoneType'=None , properties :dict = None, **args):
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
        

    def selected_onclick_handler(self,event: str=None , id : 'NoneType'=None , properties :dict = None, **args):
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
            data = self.convert_selected_set_to_geojson(self.selected_set),
            name="Selected ROIs",
            hover_style={"fillColor": "blue"},
        )
        # Recreate the onclick handler for the selected layers
        self.selected_layer.on_click(self.selected_onclick_handler)
        # Add selected layer to the map
        self.m.add_layer(self.selected_layer)
    
    
    def add_geojson_layer_to_map(self):
        """ Add the geojson for the generated roi as a styled layer to the map"""
        geojson_layer =self.get_geojson_layer()
        geojson_layer.on_click(self.geojson_onclick_handler)
        self.m.add_layer(geojson_layer)
        
            
    def convert_selected_set_to_geojson(self,selected_set: set) -> dict:
        """Returns a geojson dict containing a FeatureCollection for all the geojson objects in the
        selected_set
        Args:
            selected_set (set): ids of the selected geojson

        Returns:
            dict: geojson dict containing a FeatureCollection for all the geojson objects in the
        selected_set
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
