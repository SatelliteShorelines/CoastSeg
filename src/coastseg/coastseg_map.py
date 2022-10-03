import os
import json
import logging
from glob import glob

from src.coastseg import bbox
from src.coastseg import bbox_class
from src.coastseg import common
from src.coastseg.shoreline import Shoreline
from src.coastseg.transects import Transects
from src.coastseg import exceptions

import requests
import geopandas as gpd
import numpy as np
from pandas import read_csv, concat
from shapely import geometry
from skimage.io import imread
from fiona.errors import DriverError
from shapely.geometry import Polygon
from pyproj import Proj, transform
from coastsat import SDS_tools, SDS_download, SDS_tools,SDS_transects,SDS_shoreline
from ipyleaflet import DrawControl, LayersControl,  WidgetControl, GeoJSON
from leafmap import Map, check_file_path
from ipywidgets import Layout, HTML, Accordion
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Qt5Agg")


logger = logging.getLogger(__name__)
logger.info("Loaded module:  %s",__name__)


class CoastSeg_Map:

    def __init__(self, map_settings: dict = None):
        # settings:  used to select data to download and preprocess settings
        self.settings = {}
        # data : geojson data of the rois generated
        self.data = None
        # selected_set : ids of the selected rois
        self.selected_set = set()
        # geojson_layer : layer with all rois
        self.geojson_layer = None
        # selected layer :  layer containing all selected rois
        self.selected_layer = None
        # selected_ROI : Geojson for all the ROIs selected by the user
        self.selected_ROI = None
        # self.transect : transect object containing transects loaded on map
        self.transects = None
        # self.shoreline : shoreline object containing shoreline loaded on map
        self.shoreline = None
        # self.shorelines_gdf : all the shorelines within the bbox
        self.shorelines_gdf = gpd.GeoDataFrame()
        # Stores all the transects on the map
        self.transects_in_bbox_list=[]
        # Bbox saved by the user
        self.bbox = None
        # preprocess_settings : dictionary of settings used by coastsat to download imagery
        self.preprocess_settings = {}
        # create map if map_settings not provided else use default settings
        self.map = self.create_map(map_settings)
        # create controls and add to map
        self.draw_control = self.create_DrawControl(DrawControl())
        self.draw_control.on_draw(self.handle_draw)
        self.map.add_control(self.draw_control)
        layer_control = LayersControl(position='topright')
        self.map.add_control(layer_control)
        hover_shoreline_control = self.create_shoreline_widget()
        self.map.add_control(hover_shoreline_control)

    def create_map(self, map_settings:dict):
        """create an interactive map object using the map_settings

        Args:
            map_settings (dict): settings to control how map is created

        Returns:
           ipyleaflet.Map: ipyleaflet interactive Map object 
        """        
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
        return Map(
            draw_control=map_settings["draw_control"],
            measure_control=map_settings["measure_control"],
            fullscreen_control=map_settings["fullscreen_control"],
            attribution_control=map_settings["attribution_control"],
            center=map_settings["center_point"],
            zoom=map_settings["zoom"],
            layout=map_settings["Layout"])

    def create_shoreline_widget(self):
        """creates a accordion style widget controller to hold shoreline data.

        Returns:
           ipyleaflet.WidgetControl: an widget control for an accordion widget
        """        
        html = HTML("Hover over shoreline")
        html.layout.margin = "0px 20px 20px 20px"

        self.shoreline_accordion = Accordion(children=[html], titles=('Shoreline Data'))
        self.shoreline_accordion.set_title(0,'Shoreline Data')
        
        return WidgetControl(widget=self.shoreline_accordion, position="topright")

    def save_settings(self, **kwargs):
        """Saves the settings for downloading data in a dictionary
        Pass in data in the form of 
        save_settings(sat_list=sat_list, collection='C01',dates=dates,**preprocess_settings)
        *You must use the names sat_list, collection, and dates 
        """
        for key,value in kwargs.items():
            self.settings[key]=value

    def update_shoreline_html(self, feature, **kwargs):
        # Modifies main html body of Shoreline Data Accordion
        self.shoreline_accordion.children[0].value="""
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
    
    
    def get_rois_gdf(self,selected_roi:dict) -> gpd.geodataframe:
        """ Returns rois as a geodataframe in the espg 4326

        Args:
           selected_rois (dict): rois selected by the user. Must contain the following fields:
                {'features': [
                    'id': (str) roi_id 
                    ''geometry':{
                        'type':Polygon
                        'coordinates': list of coordinates that make up polygon
                    }
                ],...}
        Returns:
            gpd.geodataframe: all selected roi data such as geometery, id and more
        """
        polygons=[]
        for roi in selected_roi["features"]:
            polygons.append( Polygon(roi['geometry']['coordinates'][0]))
        gdf = gpd.GeoDataFrame(selected_roi["features"],geometry= polygons, crs="EPSG:4326")
        return gdf 

    def make_coastsat_compatible(self, shoreline_in_roi:gpd.geodataframe)->np.ndarray:
        """Return the shoreline_in_roi as an np.array in the form: 
            array([[lat,lon,0],[lat,lon,0],[lat,lon,0]....])

        Args:
            shoreline_in_roi (gpd.geodataframe): clipped portion of shoreline withinv a roi

        Returns:
            np.ndarray: shorelines in the form: 
                array([[lat,lon,0],[lat,lon,0],[lat,lon,0]....])
        """
        # Then convert the shoreline to lat,lon tuples for CoastSat
        shorelines = []
        for k in shoreline_in_roi['geometry'].keys():
            #For each linestring portion of shoreline convert to lat,lon tuples
            shorelines.append(tuple(np.array(shoreline_in_roi['geometry'][k]).tolist()))
        # shorelines = [([lat,lon],[lat,lon],[lat,lon]),([lat,lon],[lat,lon],[lat,lon])...]
        # Stack all the tuples into a single list of n rows X 2 columns
        shorelines = np.vstack(shorelines)
        # Add third column of 0s to represent mean sea level
        shorelines = np.insert(shorelines, 2, np.zeros(len(shorelines)), axis=1)
        # shorelines = array([[lat,lon,0],[lat,lon,0],[lat,lon,0]....])
        return shorelines

    def extract_shorelines_from_rois(self, selected_rois:dict,inputs_dict:dict, pre_process_settings: dict, shorelines_gdf: gpd.geodataframe)->dict:
        """Returns a dictionary containing the extracted shorelines for each roi
        Args:
            selected_rois (dict): rois selected by the user. Must contain the following fields:
                {'features': [
                    'id': (str) roi_id 
                    ''geometry':{
                        'type':Polygon
                        'coordinates': list of coordinates that make up polygon
                    }
                ]}
            inputs_dict (dict): dictionary containing inputs dict for each roi.
                {roi_id: 'inputs': dict
                        input parameters (sitename, filepath, polygon, dates, sat_list, collection)
                }
            pre_process_settings (dict): settings used for CoastSat. Must have the following fields:
                'inputs': dict
                    input parameters (sitename, filepath, polygon, dates, sat_list)
                'output_epsg': int
                     output spatial reference system as EPSG code
                'check_detection': bool
                    if True, lets user manually accept/reject the mapped shorelines
                'max_dist_ref': int
                    alongshore distance considered caluclate the intersection
                'adjust_detection': bool
                    if True, allows user to manually adjust the detected shoreline


            shorelines_gdf (gpd.geodataframe): shorelines as a geodataframe. Contains columns:
                'OBJECTID', 'MasterKey', 'RandomSort', 'MEAN_SIG_WAVEHEIGHT',
                'TIDAL_RANGE', 'CHLOROPHYLL', 'TURBIDITY', 'TEMP_MOISTURE',
                'EMU_PHYSICAL', 'REGIONAL_SINUOSITY', 'GHM', 'MAX_SLOPE',
                'OUTFLOW_DENSITY', 'ERODIBILITY', 'Cluster', 'LENGTH_GEO', 'chl_label',
                'river_label', 'sinuosity_label', 'slope_label', 'tidal_label',
                'turbid_label', 'wave_label', 'CSU_Descriptor', 'CSU_ID',
                'OUTFLOW_DENSITY_RESCALED', 'Shape_Length', 'geometry'

        Returns:
            extracted_shorelines (dict): dictionary with roi_id keys that identify roi associates with shorelines
            {
                roi_id:{
                    dates: [datetime.datetime,datetime.datetime], ...
                    shorelines: [array(),array()]    }  
            }
        """
        rois_gdf=self.get_rois_gdf(selected_rois)
        extracted_shorelines={}
        # Extract shorelines after you downloaded them
        for roi in selected_rois["features"]:
            # Choose a SINGLE specific roi by id to start with
            roi_id=roi["id"]
            # Get the specific roi by id
            rois_gdf[rois_gdf['id']==roi_id]
            # Clip shoreline to specific roi
            shoreline_in_roi = gpd.clip(shorelines_gdf, rois_gdf[rois_gdf['id']==roi_id])
    #         shoreline_in_roi = shoreline_in_roi.to_crs('EPSG:4326')
            shorelines=self.make_coastsat_compatible(shoreline_in_roi)
            print("\nshorelines: ",shorelines)
            # Make a copy of preprocess settings to modify
            tmp_setting=pre_process_settings
            s_proj=self.convert_espg(4326,tmp_setting['output_epsg'],shorelines)
            # individual Roi polygon
            polygon=roi['geometry']['coordinates'][0]
            sitename=inputs_dict[int(roi_id)]['sitename']
            filepath=os.path.join(os.getcwd(), 'data')
            sat_list=inputs_dict[int(roi_id)]['sat_list']
            dates=inputs_dict[int(roi_id)]['dates']
            collection=inputs_dict[int(roi_id)]['landsat_collection']
    #         inputs_tmp=inputs_dict[int(roi_id)]
            inputs = {'polygon': polygon, 'dates': dates, 'sat_list': sat_list, 'sitename': sitename,
                    'filepath':filepath,
                    'landsat_collection': collection}
            # Add refernce shoreline and the shoreline buffer distance for this specific ROI
            tmp_setting['reference_shoreline'] = s_proj
            tmp_setting['max_dist_ref']=25
            # DO NOT have user adjust shorelines manually
            tmp_setting['adjust_detection']=False
            # DO NOT have user check for valid shorelines
            tmp_setting['check_detection']=False
            tmp_setting['inputs']=inputs
            metadata = SDS_download.get_metadata(inputs) 

            output = SDS_shoreline.extract_shorelines(metadata, tmp_setting)
            output = SDS_tools.remove_duplicates(output) # removes duplicates (images taken on the same date by the same satellite)
            output = SDS_tools.remove_inaccurate_georef(output, 10) # remove inaccurate georeferencing (set threshold to 10 m)
            # Add the extracted shoreline to the extracted shorelines dictionary 
            extracted_shorelines[int(roi_id)]=output
            geomtype = 'lines' # choose 'points' or 'lines' for the layer geometry
            gdf = SDS_tools.output_to_gdf(output, geomtype)
            if gdf is None:
                print("No shorelines for ROI ",roi_id)
            else:
                gdf.crs = {'init':'epsg:'+str(tmp_setting['output_epsg'])} # set layer projection
                # save GEOJSON layer to file
                sitename=inputs['sitename']
                filepath=inputs['filepath']
                gdf.to_file(os.path.join(filepath, sitename, '%s_output_%s_staticref.geojson'%(sitename,geomtype)),
                                                driver='GeoJSON', encoding='utf-8')
        return extracted_shorelines

    def convert_espg(self,input_epsg:int,output_epsg:int,coastsat_array:np.ndarray, is_transects:bool = False)->np.ndarray:
        """Convert the coastsat_array espg to the output_espg 
        @ todo add funcionality for is_transects
        Args:
            input_epsg (int): input espg
            output_epsg (int): output espg
            coastsat_array (np.ndarray): array of coordiinates as [[lat,lon],[lat,lon]....]

        Returns:
            np.ndarray: array with output espg in the form [[[lat,lon,0.]]]
        """
        if input_epsg is None:
            input_epsg=4326
        inProj = Proj(init='epsg:'+str(input_epsg))
        outProj = Proj(init='epsg:'+str(output_epsg))
        s_proj = []
        # Convert all the lat,ln coords to new espg (operation takes some time....)
        for coord in coastsat_array:
            x2,y2 = transform(inProj,outProj,coord[0],coord[1])
            s_proj.append([x2,y2,0.])
        return np.array(s_proj)

    def is_shoreline_present(self,extracted_shorelines:dict,roi_id:int) -> bool:
        """Returns true if shoreline array exists for roi_id

        Args:
            extracted_shorelines (dict): dict in form of :
                {   roi_id : {dates: [datetime.datetime,datetime.datetime], 
                    shorelines: [array(),array()]}  } 
            roi_id (int): id of the roi

        Returns:
            bool: false if shoreline does not exist at roi_id
        """
        for shoreline in extracted_shorelines[roi_id]['shorelines']:
            if shoreline.size != 0:
                return True
        return False

    def get_intersecting_transects(self,rois_gdf:gpd.geodataframe,transect_data :gpd.geodataframe,id :str) -> gpd.geodataframe:
        """Returns a transects that intersect with the roi with id provided
        Args:
            rois_gdf (gpd.geodataframe): rois with geometery, ids and more
            transect_data (gpd.geodataframe): transects geomemtry
            id (str): id of roi

        Returns:
            gpd.geodataframe: _description_
        """
        poly_roi=self.convert_roi_to_polygon(rois_gdf,id)
        transect_mask=transect_data.intersects(poly_roi,align=False)
        return transect_data[transect_mask]

    def compute_transects(self,selected_rois:dict,extracted_shorelines:dict,settings:dict) -> dict:
        """Returns a dict of cross distances for each roi's transects 

        Args:
            selected_rois (dict): rois selected by the user. Must contain the following fields:
                {'features': [
                    'id': (str) roi_id 
                    ''geometry':{
                        'type':Polygon
                        'coordinates': list of coordinates that make up polygon
                    }
                ],...}
            extracted_shorelines (dict): dictionary with roi_id keys that identify roi associates with shorelines
            {
                roi_id:{
                    dates: [datetime.datetime,datetime.datetime], ...
                    shorelines: [array(),array()]    }  
            }

            settings (dict): settings used for CoastSat. Must have the following fields:
               'output_epsg': int
                    output spatial reference system as EPSG code
                'along_dist': int
                    alongshore distance considered caluclate the intersection

        Returns:
            dict: cross_distances_rois with format:
            { roi_id :  dict
                time-series of cross-shore distance along each of the transects. Not tidally corrected. }
        """
        cross_distances_rois={}
        transect_in_roi=None
        # Input and output projections
        inProj = Proj(init='epsg:4326')
        outProj = Proj(init='epsg:'+str(settings['output_epsg']))
        if self.transect_names == []:
            print("No transects were found in this region.")
            return None
        if self.transects_in_bbox_list != []:
            rois_gdf=self.get_rois_gdf(selected_rois)
            for roi in selected_rois["features"]:
                # Choose a SINGLE specific roi by id to start with
                roi_id=roi["id"]
                # Get transects intersecting with single roi
                for transect in self.transects_in_bbox_list:
                    transect_in_roi=self.get_intersecting_transects(rois_gdf,transect,roi_id)
                # Do not compute the transect if no shoreline exists
                if self.is_shoreline_present(extracted_shorelines,int(roi_id)):
                    print("Shoreline present at ROI: ",roi_id)
                    trans = []
                    for k in transect_in_roi['geometry'].keys():
                        trans.append(tuple(np.array(transect_in_roi['geometry'][k]).tolist()))
                    # convert to dict of numpy arrays of start and end points
                    transects = {}
                    for counter,i in enumerate(trans):
                        x0,y0 = transform(inProj,outProj,i[0][0],i[0][1])
                        x1,y1 = transform(inProj,outProj,i[1][0],i[1][1])    
                        transects['NA'+str(counter)] = np.array([[x0,y0],[x1,y1]])
                    # defines along-shore distance over which to consider shoreline points to compute median intersection (robust to outliers)
                    if 'along_dist' not in settings.keys():
                        settings['along_dist'] = 25 
                    output=extracted_shorelines[int(roi_id)]
                    cross_distance = SDS_transects.compute_intersection(output, transects, settings) 
                    cross_distances_rois[roi_id]= cross_distance
                
            return cross_distances_rois

    def load_bbox_from_file(self, filename):
        bbox_gdf = gpd.read_file(bbox_file)
        bbox_dict = json.loads(bbox_gdf.to_json())
        bbox_layer = GeoJSON(
            data=bbox_dict,
            name="Bbox",
            style={
                'color': '#75b671',
                'fill_color': '#75b671',
                'opacity': 1,
                'fillOpacity': 0.2,
                'weight': 4},
        )
        self.coastseg_map.map.add_layer(bbox_layer)
        print(f"Loaded the rois from the file :\n{bbox_file}")
        logger.info(f"Loaded the rois from the file :\n{bbox_file}")

    def load_transects_on_map(self) -> None:
        """Adds transects within the drawn bbox onto the map"""        
        if self.bbox is None:
            raise exceptions.Object_Not_Found()
        elif self.bbox.gdf.empty:
            raise exceptions.Object_Not_Found()
        # if a bounding box exists create a shoreline within it
        transects = Transects(self.bbox.gdf)
        if transects.gdf.empty:
            raise exceptions.Object_Not_Found("Transects Not Found in this region. Draw a new bounding box")
        else:
            layer_name = 'transects'
            # Replace old transect with new one
            if self.transects:
                self.remove_layer_by_name(layer_name)
                del self.transects
                self.transects=None
            # style and add the transect to the map 
            new_layer = self.create_layer(transects, layer_name)
            self.map.add_layer(new_layer)
            logger.info(f"Add layer to map: {new_layer}")
            # Save transect to coastseg_map
            self.transects=transects
              
    def download_url(self, url: str, save_path: str, filename:str=None, chunk_size: int = 128):
        """Downloads the data from the given url to the save_path location.
        Args:
            url (str): url to data to download
            save_path (str): directory to save data
            chunk_size (int, optional):  Defaults to 128.
        """
        with requests.get(url, stream=True) as r:
            if r.status_code == 404:
                raise exceptions.DownloadError(os.path.basename(save_path))

            # check header to get content length, in bytes
            total_length = int(r.headers.get("Content-Length"))
            with open(save_path, 'wb') as fd:
                with tqdm(total=total_length, unit='B', unit_scale=True,unit_divisor=1024,desc=f"Downloading {filename}",initial=0, ascii=True) as pbar:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            fd.write(chunk)
                            pbar.update(len(chunk))

    def remove_all(self):
        """Remove the bbox, shoreline, all rois from the map"""
        self.remove_bbox()
        self.remove_layer_by_name('transects')
        self.remove_layer_by_name('shoreline')
        self.remove_all_rois()
        self.remove_shoreline_html()

    def remove_bbox(self):
        """Remove all the bounding boxes from the map"""
        if self.bbox:
            del self.bbox
            self.bbox = None
        self.draw_control.clear()
        existing_layer=self.map.find_layer('Bbox')
        if existing_layer is not None:
                self.map.remove_layer(existing_layer)

    def remove_layer_by_name(self, layer_name):
        existing_layer=self.map.find_layer(layer_name)
        if existing_layer is not None:
            self.map.remove_layer(existing_layer)
        logger.info(f"Removed layer {layer_name}")


    def remove_shoreline_html(self):
        """Clear the shoreline html accoridon """
        self.shoreline_accordion.children[0].value ="Hover over the shoreline."

    def remove_all_rois(self):
        """Removes all the unselected rois from the map """
        self.selected_ROI = None
        # Remove the selected rois
        existing_layer = self.map.find_layer('Selected ROIs')
        if existing_layer is not None:
            self.map.remove_layer(existing_layer)
            self.selected_layer = None
        existing_layer = self.map.find_layer('GeoJSON data')
        if existing_layer is not None:
            # Remove the layer from the map
            self.map.remove_layer(existing_layer)
            self.geojson_layer = None
            # clear the stylized geojson
            self.data = None
        self.selected_set = set()

    def create_DrawControl(self, draw_control : "ipyleaflet.leaflet.DrawControl"):
        """ Modifies given draw control so that only rectangles can be drawn

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
        """Adds or removes the bounding box  when drawn/deleted from map
        Args:
            target (ipyleaflet.leaflet.DrawControl): draw control used
            action (str): name of the most recent action ex. 'created', 'deleted'
            geo_json (dict): geojson dictionary
        """
        self.action = action
        self.geo_json = geo_json
        self.target = target
        if self.draw_control.last_action == 'created' and self.draw_control.last_draw['geometry']['type'] == 'Polygon':
            # validate the bbox size
            bbbox_area = common.get_area(self.draw_control.last_draw['geometry'])
            bbox_class.Bounding_Box.check_bbox_size(bbbox_area)
            # if a bbox already exists delete it
            if self.bbox:
                del self.bbox
                self.bbox = None
            # Save new bbox to coastseg_map
            self.bbox = bbox_class.Bounding_Box(self.draw_control.last_draw['geometry'])
        if self.draw_control.last_action == 'deleted':
            if self.bbox:
                del self.bbox
                self.bbox = None
             
    def load_shoreline_on_map(self) -> None:
        """Adds shoreline within the drawn bbox onto the map"""
        if self.bbox is None:
            raise exceptions.BBox_Not_Found()
        elif self.bbox.gdf.empty:
            raise exceptions.BBox_Not_Found()
        # if a bounding box exists create a shoreline within it
        shoreline = Shoreline(self.bbox.gdf)
        if shoreline.gdf.empty:
             raise exceptions.Shoreline_Not_Found()
        else:
            layer_name = 'shoreline'
            # Replace old shoreline with new one
            if self.shoreline:
                self.remove_layer_by_name(layer_name)
                del self.shoreline
                self.shoreline=None
            # style and add the shoreline to the map 
            new_layer = self.create_layer(shoreline, layer_name)
            # add on_hover handler to update shoreline widget when user hovers over shoreline
            new_layer.on_hover(self.update_shoreline_html)
            self.map.add_layer(new_layer)
            logger.info(f"Add layer to map: {new_layer}")
            # Save shoreline to coastseg_map
            self.shoreline=shoreline

    def create_layer(self,feature, layer_name:str):
        layer_geojson = json.loads(feature.gdf.to_json())
        # convert layer to GeoJson and style it accordingly
        styled_layer = feature.style_layer(layer_geojson, layer_name)
        return styled_layer

    def save_bbox_to_file(self):
        # Ensure drawn bbox(bounding box) within allowed size
        bbox.validate_bbox_size(self.shapes_list)
        # Get the geodataframe for the bbox
        self.bbox = bbox.create_geodataframe_from_bbox(self.shapes_list)
        self.bbox.to_file("bbox.geojson", driver='GeoJSON')
        print("Saved bbox to bbox.geojson")
    
    def style_rois(self, df: gpd.geodataframe):
        """converts a geodataframe to a json dict and styles it as a fishnet grid

        Args:
            df (gpd.geodataframe): fishnet dataframe

        Returns:
            dict: fishnet grid styled as a black grid
        """
        fishnet_geojson = df.to_json()
        fishnet_dict = json.loads(fishnet_geojson)

        # Add style to each feature in the geojson
        for feature in fishnet_dict["features"]:
            feature["properties"]["style"] = {
                "color": "black",
                "weight": 3,
                "fillColor": "grey",
                "fillOpacity": 0.1,
            }
        return fishnet_dict

    def generate_ROIS_fishnet(self,large_fishnet=7500,small_fishnet=5000):
        """Generates series of overlapping ROIS along shoreline on map using fishnet method"""
        # Ensure drawn bbox(bounding box) within allowed size
        bbox.validate_bbox_size(self.shapes_list)
        # Get the geodataframe for the bbox
        gpd_bbox = bbox.create_geodataframe_from_bbox(self.shapes_list)
        # save bbox 
        if self.shorelines_gdf.empty:
            self.load_shoreline_on_map()
        # Large fishnet cannot be 0. Throw an error
        if large_fishnet == 0:
            raise Exception("Large fishnet size must be greater than 0")
        # Create two fishnets, one big (2000m) and one small(1500m) so they overlap each other
        fishnet_gpd_large = self.fishnet_gpd(gpd_bbox, self.shorelines_gdf,large_fishnet)

        if small_fishnet == 0:
            # If small fishnet is 0 it means only the large fishnet should exist
            fishnet_intersect_gpd=fishnet_gpd_large
        else:
            fishnet_gpd_small = self.fishnet_gpd(gpd_bbox, self.shorelines_gdf, small_fishnet)
            # Concat the fishnets together to create one overlapping set of rois
            fishnet_intersect_gpd = gpd.GeoDataFrame(concat([fishnet_gpd_large, fishnet_gpd_small], ignore_index=True))

        # Add an id column to fishnet dataframe
        num_roi = len(fishnet_intersect_gpd)
        fishnet_intersect_gpd['id'] = np.arange(0, num_roi)
        
        # style fishnet and convert to dictionary to be added to map
        fishnet_dict = self.style_rois(fishnet_intersect_gpd)

        # Save the styled fishnet to data for interactivity to be added later
        self.data = fishnet_dict

    def get_geojson_layer(self) -> "'ipyleaflet.leaflet.GeoJSON'":
        """Returns GeoJSON for generated ROIs
        Returns:
            GeoJSON: geojson object that can be added to the map
        """
        if self.geojson_layer is None and self.data:
            self.geojson_layer = GeoJSON(data=self.data, name="GeoJSON data", hover_style={"fillColor": "red","color":"crimson"})
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
            self.map.remove_layer(self.selected_layer)

        self.selected_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(self.selected_set),
            name="Selected ROIs",
            hover_style={"fillColor": "blue","color": "aqua"},
        )
        self.selected_layer.on_click(self.selected_onclick_handler)
        self.map.add_layer(self.selected_layer)

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
            self.map.remove_layer(self.selected_layer)
        # Recreate the selected layers wihout the layer that was removed
        self.selected_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(self.selected_set),
            name="Selected ROIs",
            hover_style={"fillColor": "blue","color": "aqua"},
        )
        # Recreate the onclick handler for the selected layers
        self.selected_layer.on_click(self.selected_onclick_handler)
        # Add selected layer to the map
        self.map.add_layer(self.selected_layer)

    def add_geojson_layer_to_map(self):
        """ Add the geojson for the generated roi as a styled layer to the map"""
        geojson_layer = self.get_geojson_layer()
        geojson_layer.on_click(self.geojson_onclick_handler)
        self.map.add_layer(geojson_layer)

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
                "weight": 3,
                "fillColor": "grey",
                "fillOpacity": 0.1,
            }
        return geojson
