import os
from ipyleaflet import DrawControl, GeoJSON, LayersControl,  WidgetControl, GeoJSON
from shapely import geometry
from leafmap import Map, check_file_path
from CoastSeg import bbox
from ipywidgets import Layout, HTML, Accordion
from  requests import get
from pandas import read_csv, concat
from numpy import arange
import json
import geopandas as gpd
# new imports
from skimage.io import imread, imsave
import numpy as np
 
class CoastSeg_Map:

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
        # selected_ROI : Geojson for all the ROIs selected by the user
        self.selected_ROI = None
        # self.transect_names : list of transect names on map
        self.transect_names = []
        # self.shoreline_names
        self.shoreline_names = []
        # self.shorelines_gdf
        self.shorelines_gdf = gpd.GeoDataFrame()
        
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
        self.m = Map(
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
        
        # html accordion containing shoreline data on hover
        html = HTML("Hover over shoreline")
        html.layout.margin = "0px 20px 20px 20px"

        self.main_accordion = Accordion(children=[html], titles=('Shoreline Data'))
        self.main_accordion.set_title(0,'Shoreline Data')
        
        hover_shoreline_control = WidgetControl(widget=self.main_accordion, position="topright")
        self.m.add_control(hover_shoreline_control)

    def update_shoreline_html(self, feature, **kwargs):
        # Modifies main html body of Shoreline Data Accordion
        self.main_accordion.children[0].value="""
        <p>Mean Sig Waveheight: {}</p>
        <p>Tidal Range: {}</p>
        <p>Erodibility: {}</p>
        <p>River: {}</p>
        <p>Sinuosity: {}</p>
        <p>Slope: {}</p>
        <p>Turbid: {}</p>
        <p>CSU_ID: {}</p>
        """.format(
            feature['properties']['MEAN_SIG_WAVEHEIGHT'],
            feature['properties']['TIDAL_RANGE'],
            feature['properties']['ERODIBILITY'],
            feature['properties']['river_label'],
            feature['properties']['sinuosity_label'],
            feature['properties']['slope_label'],
            feature['properties']['turbid_label'],
            feature['properties']['CSU_ID'])        
    
    def rescale_array(self, dat, mn, mx):
        """
        rescales an input dat between mn and mx
        Code from doodleverse_utils by Daniel Buscombe
        source: https://github.com/Doodleverse/doodleverse_utils
        """
        m = min(dat.flatten())
        M = max(dat.flatten())
        return (mx - mn) * (dat - m) / (M - m) + mn

# Work in progress
    def RGB_to_MNDWI(self, files):
        """
        Converts RGB imagery to MNDWI imagery
        Original Code from doodleverse_utils by Daniel Buscombe
        source: https://github.com/Doodleverse/doodleverse_utils
        """
        # files : 
        # output_path: directory to store MNDWI imagery
        output_path = os.getcwd()
        for counter,f in enumerate(files):
            datadict={}
            # Read green band from RGB image and cast to float
            green_band = imread(f[0])[:,:,1].astype('float')
            # Read SWIR and cast to float
            swir = imread(f[1]).astype('float')
            # Transform 0 to np.NAN 
            green_band[green_band==0]=np.nan
            swir[swir==0]=np.nan
            # Mask out the NaNs
            green_band = np.ma.filled(green_band)
            swir = np.ma.filled(swir)
            
            # MNDWI imagery formula (Green – SWIR) / (Green + SWIR)
            mndwi = np.divide(swir - green_band, swir + green_band)
            # Convert the NaNs to -1
            mndwi[np.isnan(mndwi)]=-1
            # Rescale to be between 0 - 255
            mndwi = self.rescale_array(mndwi,0,255)
            # Save meta data for savez_compressed()
            datadict['arr_0'] = mndwi.astype(np.uint8)
            datadict['num_bands'] = 1
            datadict['files'] = [fi.split(os.sep)[-1] for fi in f]
            # Remove the file extension from the name
            ROOT_STRING = f[0].split(os.sep)[-1].split('.')[0]
            #print(ROOT_STRING)
            segfile = output_path+os.sep+ROOT_STRING+'_noaug_nd_data_000000'+str(counter)+'.npz'
            np.savez_compressed(segfile, **datadict)
            del datadict, mndwi, green_band, swir
     
    def load_total_bounds_df(self,type:str) -> "pandas.core.frame.DataFrame":
        """Returns dataframe containing total bounds for each set of either shorelines or transects in the csv file.
        
        Args:
            type: Either "transects" or "shorelines" determines which csv is loaded
        
        Returns:
            pandas.core.frame.DataFrame : Returns dataframe containing total bounds for each set of shorelines or transects
        """ 
        # Load in the total bounding box from csv
        if type.lower() == 'transects':
           total_bounds_df=read_csv("transects_bounding_boxes.csv")
        elif type.lower() == 'shorelines':
            total_bounds_df=read_csv("shorelines_bounding_boxes.csv")
        total_bounds_df.index=total_bounds_df["filename"]
        if 'filename' in total_bounds_df.columns:
            total_bounds_df.drop("filename",axis=1, inplace=True)
        return total_bounds_df 
    
    
    def get_intersecting_files(self, gpd_bbox : 'geopandas.geodataframe.GeoDataFrame', type : str):
        # dataframe containing total bounding box for each transects or shoreline file
        total_bounds_df=self.load_total_bounds_df(type)
        # filenames where transects/shoreline'bbox intersect bounding box drawn by user
        intersecting_files=[]
        for filename in total_bounds_df.index:
            minx, miny, maxx, maxy=total_bounds_df.loc[filename]
            intersection_df = gpd_bbox.cx[minx:maxx, miny:maxy]
            gpd_bbox.cx[minx:maxx, miny:maxy].plot()
            # save filenames where gpd_bbox & bounding box for set of transects or shorelines intersect
            if intersection_df.empty == False:
                intersecting_files.append(filename)
        return intersecting_files
    
    
    def get_layer_name(self, filename :str)->str:
        """Returns layer name derived from the filename without the extension
            Ex. "shoreline.geojson" -> shoreline

        Args:
            filename (str): geojson file associated with layer name 

        Returns:
            str : (layer_name) name of transect file to be used as layer name
        """        
        layer_name=os.path.splitext(filename)[0]
        return layer_name
    

    def style_transect(self, transect_gdf:'geopandas.geodataframe.GeoDataFrame')->dict:
        """Converts transect_gdf to json and adds style to its properties

        Converts transect_gdf to json and adds an object called 'style' containing the styled
        json for the map.
        Args:
            transect_gdf (geopandas.geodataframe.GeoDataFrame): clipped transects loaded from geojson file

        Returns:
            dict: clipped transects with style properties
        """        
        # Save the fishnet intersection with shoreline to json
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
        intersecting_transect_files = self.get_intersecting_files(gpd_bbox,'transects')
        # for each transect file clip it to the bbox and add to map
        for transect_file in intersecting_transect_files:
            print("Loading ",transect_file)
            transects_layer_name=self.get_layer_name(transect_file)
            transect_path=os.path.abspath(os.getcwd())+os.sep+"Coastseg"+os.sep+"transects"+os.sep+transect_file
            data=bbox.read_gpd_file(transect_path)
            transects_in_bbox=bbox.clip_to_bbox(data, gpd_bbox)
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


    def download_shoreline(self, filename='NE_USA_Delaware_Maine_ref_shoreline.geojson',dataset_id: str = '6824429'):
        zenodo_id = dataset_id.split('_')[-1]
        root_url = 'https://zenodo.org/record/' + zenodo_id + '/files/'
        # Create the directory to hold the downloaded shorelines from Zenodo
        shoreline_direc = './CoastSeg/shorelines/'
        if not os.path.exists('./CoastSeg/shorelines'):
            os.mkdir('./CoastSeg/shorelines')
        # outfile : location where  model id saved
        outfile = shoreline_direc + os.sep + filename
        # Download the model from Zenodo
        if not os.path.exists(outfile):
            url = (root_url + filename)
            print('Retrieving model {} ...'.format(url))
            print("Saving to {}".format(outfile))
            self.download_url(url, outfile)
                
                
    def download_url(self, url: str, save_path: str, chunk_size: int = 128):
        """Downloads the model from the given url to the save_path location.
        Args:
            url (str): url to model to download
            save_path (str): directory to save model
            chunk_size (int, optional):  Defaults to 128.
        """
        r = get(url, stream=True)
        with open(save_path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)


    def remove_all(self):
        """Remove the bbox, shoreline, all rois from the map"""
        self.remove_bbox()
        self.remove_transects()
        self.remove_shoreline()
        self.remove_all_rois()
        self.remove_saved_roi()
        self.remove_shoreline_html()

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


    def remove_shoreline(self):
        """Removes the shoreline from the map"""
        for shoreline_name in self.shoreline_names:
            existing_layer=self.m.find_layer(shoreline_name)
            if existing_layer is not None:
                self.m.remove_layer(existing_layer)
        self.shoreline_names = []
        self.shorelines_gdf = gpd.GeoDataFrame()

    def remove_shoreline_html(self):
        """Clear the shoreline html accoridon """
        self.main_accordion.children[0].value ="Hover over the shoreline."

    def remove_saved_roi(self):
        """Removes all the saved ROI"""
        self.selected_ROI = None
         # Remove the selected rois
        existing_layer = self.m.find_layer('Selected ROIs')
        if existing_layer is not None:
            self.m.remove_layer(existing_layer)
            self.selected_layer = None
            self.selected_set = set()

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
        # intersection_gpd.drop(columns=['index_right', 'soc', 'exs', 'f_code', 'id', 'acc'], inplace=True)
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
            shoreline_gdf: "GeoDataFrame",
            square_size: int = 1000) -> "GeoDataFrame":
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
        fishnet_gpd = self.fishnet(gpd_bbox, square_size)
        # Get the geodataframe for the fishnet intersecting the shoreline
        fishnet_intersect_gpd = self.fishnet_intersection(fishnet_gpd, shoreline_gdf)
        return fishnet_intersect_gpd


    def load_shoreline_on_map(self) -> None:
        """Adds shoreline within the drawn bbox onto the map"""
        # geodataframe to hold all shorelines in bbox
        shorelines_in_bbox_gdf = gpd.GeoDataFrame()
        # ensure drawn bbox(bounding box) within allowed size
        bbox.validate_bbox_size(self.shapes_list)
        # load user drawn bbox as gdf(geodataframe)
        gpd_bbox = bbox.create_geodataframe_from_bbox(self.shapes_list)
        # list of all the shoreline files that bbox intersects
        intersecting_shoreline_files = self.get_intersecting_files(gpd_bbox,'shorelines')
        # for each transect file clip it to the bbox and add to map
        for file in intersecting_shoreline_files:
            shoreline_path=os.path.abspath(os.getcwd())+os.sep+"Coastseg"+os.sep+"shorelines"+os.sep+file
            # Check if the shoreline exists if it doesn't download it
            if  os.path.exists(shoreline_path):
                print("\n Loading the file now.")
                shoreline=bbox.read_gpd_file(shoreline_path)
            else:
                print("\n The geojson shoreline file does not exist. Downloading it now.")
                # Download shoreline geojson from Zenodo
                #@todo replace this with download function
                # ADD filename parameter and list of zendodo ids
                self.download_shoreline()
                
            # Create a single dataframe to hold all shorelines from all files
            shoreline_in_bbox=bbox.clip_to_bbox(shoreline, gpd_bbox)
            if shorelines_in_bbox_gdf.empty:
                shorelines_in_bbox_gdf = shoreline_in_bbox
            elif not shorelines_in_bbox_gdf.empty:
                # Combine shorelines from different files into single geodataframe 
                shorelines_in_bbox_gdf = gpd.GeoDataFrame(concat([shorelines_in_bbox_gdf, shoreline_in_bbox], ignore_index=True))
            
        if shorelines_in_bbox_gdf.empty:
            print("No shoreline found here.")
        else:
            # Create layer name from  shoreline geojson filenames
            filenames=list(map(self.get_layer_name,intersecting_shoreline_files))
            layer_name=""
            for file in filenames:
                layer_name+=file+'_'
            layer_name = layer_name[:-1]
            
            # Save new shoreline name and gdf  
            self.remove_shoreline()
            self.shoreline_names.append(layer_name)
            self.shorelines_gdf = shorelines_in_bbox_gdf
            
            # style and add the shoreline to the map 
            shorelines_gdf_geojson = self.shorelines_gdf.to_json()
            shorelines_gdf_geojson_dict = json.loads(shorelines_gdf_geojson)
            shoreline_layer=self.get_shoreline_layer(shorelines_gdf_geojson_dict, layer_name)
            shoreline_layer.on_hover(self.update_shoreline_html)
            self.m.add_layer(shoreline_layer)
        if self.shoreline_names == []:
            print("No shorelines were found in this region. Draw a new bounding box.")


    def generate_ROIS_fishnet(self):
        """Generates series of overlapping ROIS along shoreline on map using fishnet method"""
        # Ensure drawn bbox(bounding box) within allowed size
        bbox.validate_bbox_size(self.shapes_list)
        # Get the geodataframe for the bbox
        gpd_bbox = bbox.create_geodataframe_from_bbox(self.shapes_list)
        if self.shorelines_gdf.empty:
            self.load_shoreline_on_map()
        # Create two fishnets, one big (2000m) and one small(1500m) so they overlap each other
        fishnet_gpd_large = self.fishnet_gpd(gpd_bbox, self.shorelines_gdf,2000)
        fishnet_gpd_small = self.fishnet_gpd(gpd_bbox, self.shorelines_gdf, 1500)

        # Concat the fishnets together to create one overlapping set of rois
        fishnet_intersect_gpd = gpd.GeoDataFrame(concat([fishnet_gpd_large, fishnet_gpd_small], ignore_index=True))

        # Add an id column
        num_roi = len(fishnet_intersect_gpd)
        fishnet_intersect_gpd['id'] = arange(0, num_roi)

        # Save the fishnet intersection with shoreline to json
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
           

    def get_shoreline_layer(self, roi_shoreline: dict, layer_name :str) -> "ipyleaflet.GeoJSON":
        """get_shoreline_layer returns the  shoreline as GeoJson object.

        Args:
            roi_shoreline (dict): geojson dictionary for portion of shoreline in bbox

        Returns:
            "ipyleaflet.GeoJSON": shoreline as GeoJson object styled with yellow dashes
        """
        assert roi_shoreline != {}, "ERROR.\n Empty geojson cannot be drawn onto  map"
        return GeoJSON(
            data=roi_shoreline,
            name=layer_name,
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
        """ Saves the selected roi to a geojson file called """
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
        out_file = check_file_path(out_file)
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
