import os
import json
import logging
from glob import glob
from typing import Union

from src.coastseg.bbox import Bounding_Box
from src.coastseg import common
from src.coastseg.shoreline import Shoreline
from src.coastseg.transects import Transects
from src.coastseg.roi import ROI
from src.coastseg import exceptions

import requests
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from coastsat import SDS_tools, SDS_download, SDS_tools,SDS_transects,SDS_shoreline, SDS_preprocess
from ipyleaflet import DrawControl, LayersControl,  WidgetControl, GeoJSON
from leafmap import Map, check_file_path
from ipywidgets import Layout, HTML, Accordion
from tqdm.auto import tqdm
from pyproj import Proj, transform
import matplotlib
matplotlib.use("Qt5Agg")


logger = logging.getLogger(__name__)
logger.info("Loaded module:  %s",__name__)


class CoastSeg_Map:

    def __init__(self, map_settings: dict = None):
        # settings:  used to select data to download and preprocess settings
        self.settings = {}
        # selected_set : ids of the selected rois
        self.selected_set = set()
        # ROI_layer : layer containing all rois
        self.ROI_layer = None
        # rois : ROI(Region of Interest)
        self.rois = None
        # selected_ROI_layer :  layer containing all selected rois
        self.selected_ROI_layer = None
        # self.transect : transect object containing transects loaded on map
        self.transects = None
        # self.shoreline : shoreline object containing shoreline loaded on map
        self.shoreline = None
        # Bbox saved by the user
        self.bbox = None
         # inputs_dict: dictionary of inputs used to download data with CoastSat
        self.inputs_dict={}
        # preprocess_settings : dictionary of settings used by coastsat to download imagery
        self.preprocess_settings = {}
         #-----------------------------------------
        # @todo remove these
        # self.shorelines_gdf : all the shorelines within the bbox
        self.shorelines_gdf = gpd.GeoDataFrame()
        # Stores all the transects on the map
        self.transects_in_bbox_list=[]
        #-----------------------------------------
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


    def download_imagery(self) -> None:
        """download_imagery  downloads selected rois as jpgs

        Raises:
            Exception: raised if settings is missing
            Exception: raised if 'dates','sat_list', and 'collection' are not in settings
            Exception: raised if no ROIs have been selected
        """        
        # 1. Check imagery available and check for ee credentials
        print("Download_imagery called")
        logger.info("Download_imagery called")
        if self.settings is None:
            logger.error("No settings found.")
            raise Exception("No settings found. Create settings before downloading")
        if not set(['dates','sat_list','collection']).issubset(set(self.settings.keys())):
            logger.error(f"Missing keys from settings: {set(['dates','sat_list','collection'])-set(self.settings.keys())}")
            raise Exception(f"Missing keys from settings: {set(['dates','sat_list','collection'])-set(self.settings.keys())}")
        if self.selected_ROI_layer is None:
            logger.error("No ROIs were selected.")
            raise Exception("No ROIs were selected. Make sure to click save roi before downloading.")
        
        logger.info(f"self.settings: {self.settings}")
        logger.info(f"self.selected_ROI_layer: {self.selected_ROI_layer}")
        logger.info(f"dates: {self.settings['dates']}")
        logger.info(f"sat_list: {self.settings['sat_list']}")
        logger.info(f"collection: {self.settings['collection']}")
        logger.info(f"selected_roi_geojson: {self.settings}")

        dates=self.settings['dates']
        sat_list = self.settings['sat_list']
        collection = self.settings['collection']
        selected_roi_geojson = self.selected_ROI_layer.data
        inputs_list=common.get_inputs_list(selected_roi_geojson,dates,sat_list, collection)
        logger.info(f"inputs_list {inputs_list}")
        print("Download in process")
        tmp_settings =  self.settings
        for inputs in tqdm(inputs_list, desc="Downloading ROIs"):
            metadata = SDS_download.retrieve_images(inputs)
            print(f"inputs: {inputs}")
            logger.info(f"inputs: {inputs}")
            tmp_settings['inputs'] = inputs
            SDS_preprocess.save_jpg(metadata, tmp_settings)
        
        # Return the inputs used to download data with CoastSat
        self.inputs_dict={}
        for input in inputs_list:
            self.inputs_dict[int(input['roi_id'])]=input
        
        # Save the json of the inputs to each sitename directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for inputs in inputs_list:
            sitename=str(input['sitename'])
            filename=sitename+".json"
            shoreline_path=os.path.abspath(os.path.join(script_dir,"data",sitename,filename))
            common.write_to_json(shoreline_path,input)
        return inputs_dict
    
    
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
    
    def make_coastsat_compatible(shoreline_in_roi:gpd.geodataframe)->np.ndarray:
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

    def get_inputs(settings:dict):
        # individual Roi polygon
        polygon=settings['geometry']['coordinates'][0]
        sitename=settings['sitename']
        filepath=settings['file_path']
        sat_list=settings['sat_list']
        dates=settings['dates']
        collection=settings['landsat_collection']
        return {'polygon': polygon, 'dates': dates, 'sat_list': sat_list, 'sitename': sitename,
                'filepath':filepath,
                'landsat_collection': collection}

    def extract_shorelines_from_roi(self, rois_gdf:gpd.GeoDataFrame,inputs:dict, settings: dict, shorelines_gdf: gpd.geodataframe, id:int=None)->dict:
        """Returns a dictionary containing the extracted shorelines for roi specified by rois_gdf
        """
        extracted_shorelines={}
        
        if id is None:
            single_roi=rois_gdf
        else:
            # Select a single roi by id
            single_roi = rois_gdf[rois_gdf['id']==id]
        # if the id was not found in the geodataframe raise an exception
        if single_roi.empty:
            logger.error(f"Id: {id} was not found in {rois_gdf}")
            raise Exception(f"Id: {id} was not found in {rois_gdf}")
        
        if not set(['dates','sat_list','collection']).issubset(set(settings.keys())):
            logger.error(f"Missing keys from settings: {set(['dates','sat_list','collection'])-set(self.settings.keys())}")
            raise Exception(f"Missing keys from settings: {set(['dates','sat_list','collection'])-set(self.settings.keys())}")
        
        # Clip shoreline to specific roi
        shoreline_in_roi = gpd.clip(shorelines_gdf, single_roi)
        logger.info(f"clipped shorelines{shoreline_in_roi}")
        shorelines=self.make_coastsat_compatible(shoreline_in_roi)
        logger.info(f"coastsat shorelines{shorelines}")
        
        s_proj=common.convert_espg(4326,settings['output_epsg'],shorelines)
        logger.info(f"s_proj: {s_proj}")
        # copy settings to shoreline_settings so it can be modified
        shoreline_settings=settings
        # Add reference shoreline and shoreline buffer distance for this specific ROI
        shoreline_settings['reference_shoreline'] = s_proj
        # DO NOT have user adjust shorelines manually
        shoreline_settings['adjust_detection']=False
        # DO NOT have user check for valid shorelines
        shoreline_settings['check_detection']=False
        # copy inputs for this specific roi
        shoreline_settings['inputs']=inputs
        # get the metadata used to extract the shoreline
        metadata = SDS_download.get_metadata(inputs) 
        output = SDS_shoreline.extract_shorelines(metadata, shoreline_settings)
        # postprocessing by removing duplicates and removing in inaccurate georeferencing (set threshold to 10 m)
        output = SDS_tools.remove_duplicates(output) # removes duplicates (images taken on the same date by the same satellite)
        output = SDS_tools.remove_inaccurate_georef(output, 10) # remove inaccurate georeferencing (set threshold to 10 m)
        
        # Add the extracted shoreline to the extracted shorelines dictionary 
        # use the roi's id to identify which roi the extract shoreline is associated with
        extracted_shorelines[settings['roi_id']]=output
        geomtype = 'lines' # choose 'points' or 'lines' for the layer geometry
        extract_shoreline_gdf = SDS_tools.output_to_gdf(output, geomtype)
        # 
        if extract_shoreline_gdf is None:
            logger.warn("No shorelines for ROI {roi_id}")
            raise Exception("No shorelines for ROI {roi_id}")
        else:
            extract_shoreline_gdf.to_crs = "epsg:" +str(settings['output_epsg']) # set layer projection
            # save GEOJSON layer to file
            sitename=inputs['sitename']
            filepath=inputs['filepath']
            extract_shoreline_gdf.to_file(os.path.join(filepath, sitename, '%s_output_%s_staticref.geojson'%(sitename,geomtype)),
                                            driver='GeoJSON', encoding='utf-8')
        return extracted_shorelines

# og func
    def extract_shorelines_from_rois(self, rois_gdf:gpd.GeoDataFrame,inputs_dict:dict, pre_process_settings: dict, shorelines_gdf: gpd.geodataframe)->dict:
        """Returns a dictionary containing the extracted shorelines for each roi
        Args:
            selected_rois (dict): rois selected by the user. Must contain the following fields:
                {'features': [
                    'id': (str) roi_id 
                    ''geometry':{
                        'type':Polygon
                        'coordinates': list of coordinates that make up polygon }
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
        extracted_shorelines={}
        if not set(['dates','sat_list','collection']).issubset(set(self.settings.keys())):
            logger.error(f"Missing keys from settings: {set(['dates','sat_list','collection'])-set(self.settings.keys())}")
            raise Exception(f"Missing keys from settings: {set(['dates','sat_list','collection'])-set(self.settings.keys())}")
        
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
            s_proj=common.convert_espg(4326,tmp_setting['output_epsg'],shorelines)
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
              
    def remove_all(self):
        """Remove the bbox, shoreline, all rois from the map"""
        self.remove_bbox()
        self.remove_shoreline()
        self.remove_transects()
        self.remove_all_rois()
        self.remove_shoreline_html()

    def remove_bbox(self):
        """Remove all the bounding boxes from the map"""
        if self.bbox is not None:
            del self.bbox
            self.bbox = None
        self.draw_control.clear()
        existing_layer=self.map.find_layer(Bounding_Box.LAYER_NAME)
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

    def remove_shoreline(self):

        del self.shoreline
        self.remove_layer_by_name(Shoreline.LAYER_NAME)
        self.shoreline = None
        
    def remove_transects(self):
        del self.transects
        self.transects = None
        self.remove_layer_by_name(Transects.LAYER_NAME)

    def remove_all_rois(self):
        """Removes all the unselected rois from the map """
        # Remove the selected rois
        self.selected_ROI_layer = None
        # if self.rois is not None:
        del self.rois
        self.rois =None
        # if self.ROI_layer is not None:
        del self.ROI_layer
        self.ROI_layer = None
        # remove both roi layers from map
        existing_layer = self.map.find_layer('Selected ROIs')
        if existing_layer is not None:
            self.map.remove_layer(existing_layer)
            
        existing_layer = self.map.find_layer(ROI.LAYER_NAME)
        if existing_layer is not None:
            # Remove the layer from the map
            self.map.remove_layer(existing_layer)
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
            bbox_area = common.get_area(self.draw_control.last_draw['geometry'])
            Bounding_Box.check_bbox_size(bbox_area)
            # if a bbox already exists delete it
            self.bbox=None
            # Save new bbox to coastseg_map
            self.bbox = Bounding_Box(self.draw_control.last_draw['geometry'])
        if self.draw_control.last_action == 'deleted':
            self.remove_bbox()
    
    def load_bbox_on_map(self, file=None):
        # remove old bbox if it exists
        self.remove_bbox()
        if file is not None:
            gdf = gpd.read_file(file)
            self.bbox = Bounding_Box(gdf)
            new_layer = self.create_layer(self.bbox,self.bbox.LAYER_NAME)
            self.map.add_layer(new_layer)
            print(f"Loaded the bbox from the file :\n{file}")
            logger.info(f"Loaded the bbox from the file :\n{file}")
                 
    def load_shoreline_on_map(self) -> None:
        """Adds shoreline within the drawn bbox onto the map"""
        if self.bbox is None:
            raise exceptions.Object_Not_Found('bounding box')
        elif self.bbox.gdf.empty:
            raise exceptions.Object_Not_Found('bounding box')
        # if a bounding box exists create a shoreline within it
        shoreline = Shoreline(self.bbox.gdf)
        if shoreline.gdf.empty:
            raise exceptions.Object_Not_Found('shorelines')
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
    
    def load_rois_on_map(self, large_len=7500, small_len=5000,file:str=None):
        # Remove old ROI_layers
        self.remove_all_rois()
        if file is not None:
            gdf = gpd.read_file(file)
            self.rois = ROI(rois_gdf = gdf)
            # Create new ROI layer
            self.ROI_layer = self.create_layer(self.rois,ROI.LAYER_NAME)
            self.ROI_layer.on_click(self.geojson_onclick_handler)
            self.map.add_layer(self.ROI_layer)
            print(f"Loaded the rois from the file :\n{file}")
            logger.info(f"Loaded the rois from the file :\n{file}")
        else:
            self.generate_ROIS_fishnet(large_len, small_len)

    def generate_ROIS_fishnet(self,large_len=7500,small_len=5000):
        """Generates series of overlapping ROIS along shoreline on map using fishnet method"""
        if self.bbox is None:
            raise exceptions.Object_Not_Found('bounding box')
        logger.info("bbox for ROIs: {self.bbox.gdf}")
        logger.info("self.shoreline before check: {self.shoreline}")
        # If no shoreline exists on map then load one in
        if self.shoreline is None:
            self.load_shoreline_on_map()
        logger.info("self.shoreline used for ROIs:{ self.shoreline}")
        self.rois = None
        self.rois = ROI(self.bbox.gdf,self.shoreline.gdf,
                   square_len_lg=large_len,
                   square_len_sm=small_len)
        self.ROI_layer = self.create_layer(self.rois,ROI.LAYER_NAME)
        self.ROI_layer.on_click(self.geojson_onclick_handler)
        self.map.add_layer(self.ROI_layer)

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
        ROI_id = properties["id"]
        # Add the id of the clicked ROI to selected_set
        self.selected_set.add(ROI_id)
        if self.selected_ROI_layer is not None:
            self.map.remove_layer(self.selected_ROI_layer)

        self.selected_ROI_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(self.selected_set),
            name="Selected ROIs",
            hover_style={"fillColor": "blue",'fillOpacity': 0.1,"color": "aqua"},
        )
        
        self.selected_ROI_layer.on_click(self.selected_onclick_handler)
        self.map.add_layer(self.selected_ROI_layer)

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
        if self.selected_ROI_layer is not None:
            self.map.remove_layer(self.selected_ROI_layer)
        # Recreate the selected layers wihout the layer that was removed
        self.selected_ROI_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(self.selected_set),
            name="Selected ROIs",
            hover_style={"fillColor": "blue",'fillOpacity': 0.1,"color": "aqua"},
        )
        # Recreate the onclick handler for the selected layers
        self.selected_ROI_layer.on_click(self.selected_onclick_handler)
        # Add selected layer to the map
        self.map.add_layer(self.selected_ROI_layer)

    def check_selected_set(self):
        if self.selected_set is None:
            raise Exception ("Must select at least 1 ROI first before you can save ROIs.")
        if len(self.selected_set) == 0:
            raise Exception ("Must select at least 1 ROI first before you can save ROIs.")

    def save_feature_to_file(self,feature:Union[Bounding_Box, Shoreline,Transects, ROI]):
        if feature is None:
            raise exceptions.Object_Not_Found(feature.LAYER_NAME)
        elif isinstance(feature,ROI):
            # check if any ROIs were selected by making sure the selected set isn't empty 
            self.check_selected_set()
            print(f"Saved {feature.LAYER_NAME} to file: {feature.filename}")
            logger.info(f"Saved feature to file: {feature.filename}")
            logger.info(f"feature: {feature.gdf[feature.gdf['id'].isin(self.selected_set)]}")
            feature.gdf[feature.gdf['id'].isin(self.selected_set)].to_file(feature.filename, driver='GeoJSON')
        else:
            logger.info(f"type( {feature})")
            feature.gdf.to_file(feature.filename, driver='GeoJSON')
        print(f"Save {feature.LAYER_NAME} to {feature.filename}")
        logger.info(f"Save {feature.LAYER_NAME} to {feature.filename}")
    
    def convert_selected_set_to_geojson(self, selected_set: set) -> dict:
        """Returns a geojson dict containing a FeatureCollection for all the geojson objects in the
        selected_set
        Args:
            selected_set (set): ids of the selected geojson

        Returns:
           dict: geojson dict containing FeatureCollection for all geojson objects in selected_set
        """
        # create a new geojson dictionary to hold the selected ROIs 
        selected_rois = {"type": "FeatureCollection", "features": []}
        # Copy only the selected ROIs based on the ids in selected_set from ROI_geojson 
        selected_rois["features"] = [
            feature
            for feature in self.ROI_layer.data["features"]
            if feature["properties"]["id"] in selected_set
        ]
        # Modify change the style of each ROI in the new selected
        # selected rois will appear blue and unselected rois will appear grey
        for feature in selected_rois["features"]:
            feature["properties"]["style"] = {
                "color": "blue",
                "weight": 3,
                "fillColor": "blue",
                "fillOpacity": 0.1,
            }
        return selected_rois
