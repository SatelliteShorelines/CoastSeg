import os
import json
import logging
from typing import Union

from src.coastseg.bbox import Bounding_Box
from src.coastseg import common
from src.coastseg.shoreline import Shoreline
from src.coastseg.transects import Transects
from src.coastseg.roi import ROI
from src.coastseg import exceptions

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from coastsat import SDS_tools, SDS_download, SDS_tools,SDS_transects,SDS_shoreline, SDS_preprocess
from ipyleaflet import DrawControl, LayersControl,  WidgetControl, GeoJSON
from leafmap import Map
from ipywidgets import Layout, HTML, Accordion
from tqdm.auto import tqdm
from pyproj import Proj, transform
import matplotlib
matplotlib.use("Qt5Agg")

logger = logging.getLogger(__name__)
logger.info(f"Loaded module:  {__name__}")

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
                "center_point": (40.8630302395, -124.166267),
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
            logger.info(f"Saving to jpg. Metadata: {metadata}")
            SDS_preprocess.save_jpg(metadata, tmp_settings)
        
        # Return the inputs used to download data with CoastSat
        inputs_dict={}
        for inputs in inputs_list:
            inputs_dict[int(inputs['roi_id'])]=inputs
        logger.info(f"inputs_dict: {inputs_dict}")
        # Save inputs_dict to ROI class
        self.rois.set_inputs_dict(inputs_dict)
        # Save the json of the inputs to each sitename directory
        for inputs in inputs_list:
            sitename=str(inputs['sitename'])
            filename=sitename+".json"
            save_path=os.path.abspath(os.path.join(os.getcwd(),"data",sitename,filename))
            logger.info(f"inputs written to json: {inputs} \n Saved to {save_path}")
            common.write_to_json(save_path,inputs)
        
        logger.info("Done downloading")
        
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
    
    def make_coastsat_compatible(self,shoreline_in_roi:gpd.geodataframe)->np.ndarray:
        """Return the shoreline_in_roi as an np.array in the form: 
            array([[lat,lon,0],[lat,lon,0],[lat,lon,0]....])

        Args:
            shoreline_in_roi (gpd.geodataframe): clipped portion of shoreline within a roi

        Returns:
            np.ndarray: shorelines in the form: 
                array([[lat,lon,0],[lat,lon,0],[lat,lon,0]....])
        """
        # Then convert the shoreline to lat,lon tuples for CoastSat
        shorelines = []
        # Use explode to break multilinestrings in linestrings
        shoreline_in_roi_exploded = shoreline_in_roi.explode()
        for k in shoreline_in_roi_exploded['geometry'].keys():
            #For each linestring portion of shoreline convert to lat,lon tuples
            shorelines.append(tuple(np.array(shoreline_in_roi_exploded['geometry'][k]).tolist()))
        # shorelines = [([lat,lon],[lat,lon],[lat,lon]),([lat,lon],[lat,lon],[lat,lon])...]
        # Stack all the tuples into a single list of n rows X 2 columns
        shorelines = np.vstack(shorelines)
        # Add third column of 0s to represent mean sea level
        shorelines = np.insert(shorelines, 2, np.zeros(len(shorelines)), axis=1)
        # shorelines = array([[lat,lon,0],[lat,lon,0],[lat,lon,0]....])
        return shorelines

    def extract_all_shorelines(self)->None:
        """ Use this function when the user interactively downloads rois
        Iterates through all the ROIS downloaded by the user as indicated by the inputs_dict generated by
        download_imagery() and extracts a shoreline for each of them
        """        
        if self.rois is None:
            raise Exception("No Rois on map")
        if self.settings is None:
            raise Exception("No settings found. Please load settings")
        if self.rois.inputs_dict == {}:
            raise Exception("No inputs settings found. Please click download ROIs first")
        if self.shoreline is None:
            raise Exception("No Shoreline found. Please load a shoreline on the map first.")
        
        # Use the roi_ids in inputs_dict to select the downloaded rois from the ROI geodataframe
        roi_ids=self.rois.inputs_dict.keys()
        print(f"Extracting shorelines from ROIs: {roi_ids}")
        selected_rois_gdf = self.rois.gdf[self.rois.gdf['id'].isin(roi_ids)]
        # if none of the ids in inputs_dict are in the ROI geodataframe raise an Exception
        if selected_rois_gdf.empty:
            raise Exception("The ROIs did not match the ids in input settings")
        
        # holds all the extracted shoreline geodataframes associated with each ROI
        extracted_shoreline_dict={}
        for id in roi_ids:
            try:
                print(f"Extracting shorelines from ROI with the id:{id}")
                extracted_shoreline_dict[id]=self.extract_shorelines_from_roi(selected_rois_gdf,self.shoreline.gdf,self.rois.inputs_dict[id],self.settings,id=id)
                logger.info(f"extract_all_shorelines : extracted_shoreline_dict[id]: {extracted_shoreline_dict[id]}")
            except exceptions.Id_Not_Found as id_error:
                logger.error(f"{id_error}")
                print(f"{id_error}. \n Skipping to next ROI")
        
        # Save all the extracted_shorelines to ROI
        self.rois.update_extracted_shorelines(extracted_shoreline_dict) 
        logger.info(f"extract_all_shorelines : self.rois.extracted_shorelines {self.rois.extracted_shorelines}")                 
        
        
    def extract_shorelines_from_roi(self, rois_gdf:gpd.GeoDataFrame, shorelines_gdf: gpd.geodataframe, inputs:dict, settings: dict, id:int=None)->dict:
        """Returns a dictionary containing the extracted shorelines for roi specified by rois_gdf
        """
        if id is None:
            single_roi=rois_gdf
        else:
            # Select a single roi by id
            single_roi = rois_gdf[rois_gdf['id']==id]
        # if the id was not found in the geodataframe raise an exception
        if single_roi.empty:
            logger.error(f"Id: {id} was not found in {rois_gdf}")
            raise exceptions.Id_Not_Found(id)
        
        if not set(['dates','sat_list','collection']).issubset(set(settings.keys())):
            logger.error(f"Missing keys from settings: {set(['dates','sat_list','collection'])-set(self.settings.keys())}")
            raise Exception(f"Missing keys from settings: {set(['dates','sat_list','collection'])-set(self.settings.keys())}")
        
        # Clip shoreline to specific roi
        shoreline_in_roi = gpd.clip(shorelines_gdf, single_roi)
        # if no shorelines exist within the roi return an empty dictionary
        if shoreline_in_roi.empty:
            logger.warn(f"No shorelines could be clipped to ROI: {id}")
            return {}
        logger.info(f"clipped shorelines{shoreline_in_roi}")
        #@todo move this to common or to shoreline
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
        
        logger.info(f"shoreline_settings: {shoreline_settings}")
        
        # get the metadata used to extract the shoreline
        metadata = SDS_download.get_metadata(inputs) 
        logger.info(f"metadata: {metadata}")
        extracted_shoreline_dict = SDS_shoreline.extract_shorelines(metadata, shoreline_settings)
        print(f"extracted_shoreline_dict: {extracted_shoreline_dict}")
        logger.info(f"extracted_shoreline_dict: {extracted_shoreline_dict}")
        # postprocessing by removing duplicates and removing in inaccurate georeferencing (set threshold to 10 m)
        extracted_shoreline_dict = SDS_tools.remove_duplicates(extracted_shoreline_dict) # removes duplicates (images taken on the same date by the same satellite)
        logger.info(f"after remove_duplicates : extracted_shoreline_dict: {extracted_shoreline_dict}")
        extracted_shoreline_dict = SDS_tools.remove_inaccurate_georef(extracted_shoreline_dict, 10) # remove inaccurate georeferencing (set threshold to 10 m)
        logger.info(f"after remove_inaccurate_georef : extracted_shoreline_dict: {extracted_shoreline_dict}")
        
        logger.info(f"extracted_shoreline_dict: {extracted_shoreline_dict}")
        geomtype = 'lines' # choose 'points' or 'lines' for the layer geometry
        
        extract_shoreline_gdf = SDS_tools.output_to_gdf(extracted_shoreline_dict, geomtype)
        logger.info(f"extract_shoreline_gdf: {extract_shoreline_gdf}")
        # if extracted shorelines is None then return an empty geodataframe
        if extract_shoreline_gdf is None:
            logger.warn(f"No shorelines could be extracted for for ROI {id}")
            print(f"No shorelines could be extracted for for ROI {id}")
        else:
            extract_shoreline_gdf.crs = "epsg:" +str(settings['output_epsg'])
            # save GEOJSON layer to file
            sitename=inputs['sitename']
            filepath=inputs['filepath']
            logger.info(f'Saving shoreline to file{filepath}. \n Extracted Shoreline: {extract_shoreline_gdf}')
            extract_shoreline_gdf.to_file(os.path.join(filepath, sitename, '%s_output_%s_staticref.geojson'%(sitename,geomtype)),
                                            driver='GeoJSON', encoding='utf-8')
        
        logger.info(f"Returning extracted shoreline {extracted_shoreline_dict}")
        print(f"Extracted shoreline: {extracted_shoreline_dict}")
        return extracted_shoreline_dict

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
        logger.info(f"is_shoreline_present() : extracted_shorelines[roi_id]['shorelines']: {extracted_shorelines[roi_id]['shorelines']}")
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
        poly_roi=common.convert_gdf_to_polygon(rois_gdf,id)
        transect_mask=transect_data.intersects(poly_roi,align=False)
        return transect_data[transect_mask]

    def compute_transects_for_roi(self,roi_id:int,inProj:Proj, outProj: Proj)-> dict:
        cross_distance = 0
        try:
            if roi_id is None:
                single_roi=self.rois.gdf
            else:
                # Select a single roi by id
                single_roi = self.rois.gdf[self.rois.gdf['id']==roi_id]
            # if the id was not found in the geodataframe raise an exception
            if single_roi.empty:
                logger.error(f"Id: {id} was not found in {self.rois.gdf}")
                raise exceptions.Id_Not_Found(id)
            
            extracted_shoreline=self.rois.extracted_shorelines[int(roi_id)]
            # if no extracted shoreline exists for the roi's id then return cross distance = 0  
            if extracted_shoreline == {} :
                return 0
            
            # Get transects intersecting this specific ROI as a geodataframe
            transect_in_roi=self.get_intersecting_transects(self.rois.gdf,self.transects.gdf,roi_id)
            # Convert the extracted shorelines to the same crs as transects geodataframe
            # Find which transects intersect the shorelines
            logger.info(f"compute_transects_for_roi() :transect_in_roi: {transect_in_roi}")
            # Do not compute the transect if no shoreline exists
            if not self.is_shoreline_present(self.rois.extracted_shorelines,int(roi_id)):
                raise exceptions.No_Extracted_Shoreline(f"No extracted shoreline at this roi {roi_id}")
            else:
                print("Shoreline present at ROI: ",roi_id)
                print("Creating transects...")
                logger.info(f"compute_transects_for_roi() :Shoreline present at ROI: {roi_id}")
                # convert transects to lan,lon tuples 
                transects_coords = []
                for k in transect_in_roi['geometry'].keys():
                    transects_coords.append(tuple(np.array(transect_in_roi['geometry'][k]).tolist()))
                print(f"transects_coords:{transects_coords}")
                logger.info(f"compute_transects_for_roi() : transects_coords: {transects_coords}")
                # convert to dict of numpy arrays of start and end points
                transects = {}
                for counter,i in enumerate(transects_coords):
                    x0,y0 = transform(inProj,outProj,i[0][0],i[0][1])
                    x1,y1 = transform(inProj,outProj,i[1][0],i[1][1])    
                    transects['NA'+str(counter)] = np.array([[x0,y0],[x1,y1]])
                logger.info(f"compute_transects_for_roi():: transects: {transects}")
                # defines along-shore distance over which to consider shoreline points to compute median intersection (robust to outliers)
                if 'along_dist' not in self.settings.keys():
                    self.settings['along_dist'] = 25 
                cross_distance = SDS_transects.compute_intersection(extracted_shoreline, transects, self.settings) 
                print(f"transects cross_distance:{cross_distance}")
        except Exception as err:
            logger.exception(f"Compute transects")
        return cross_distance

    def compute_transects(self) -> dict:
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
        cross_distance_transects={}
        if self.transects is None:
            logger.error("No transects were loaded onto the map.")
            raise Exception("No transects were loaded onto the map.")
        if self.rois is None:
            logger.error("No ROIs have been loaded")
            raise Exception("No ROIs have been loaded")
        if self.settings is None:
            logger.error("No settings have been loaded")
            raise Exception("No settings have been loaded")
        if self.rois.extracted_shorelines == {}:
            logger.error("No shorelines have been extracted. Extract shorelines first.")
            raise Exception("No shorelines have been extracted. Extract shorelines first.") 
        
        # Input and output projections
        inProj = Proj(init='epsg:4326')
        outProj = Proj(init='epsg:'+str(self.settings['output_epsg']))
        
        print(f"self.rois.extracted_shorelines.keys() : {self.rois.extracted_shorelines.keys()}")
        logger.info(f"compute_transects: self.rois.extracted_shorelines.keys() : {self.rois.extracted_shorelines.keys()}")
        
        cross_distance_transects ={}
        cross_distance = 0
        
        for roi_id in self.rois.extracted_shorelines.keys():
            try:
                cross_distance = self.compute_transects_for_roi(roi_id,inProj, outProj)
                if cross_distance == 0:
                    print(f"No transects existed for ROI {roi_id}")
                cross_distance_transects[roi_id] = cross_distance
                self.rois.save_transects_to_json(roi_id,cross_distance)
                print(f"cross_distance_transects[roi_id]: {cross_distance_transects[roi_id]}")
                logger.info(f"compute_transects:: cross_distance_transects[roi_id]: {cross_distance_transects[roi_id]}")
            except exceptions.No_Extracted_Shoreline as no_extracted_shoreline:
                logger.info(f"ROI id:{roi_id} has no extracted shoreline. No transects computed")
                print(f"ROI id:{roi_id} has no extracted shoreline. No transects computed")
        
        
        self.rois.cross_distance_transects = cross_distance_transects

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
        self.remove_layer_by_name('geodataframe')

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
  
    def load_gdf_on_map(self, file=None):
        # assumes the geodataframe has a crs
        layer_name = 'geodataframe'
        self.remove_layer_by_name(layer_name)
        if file is not None:
            gdf = gpd.read_file(file)
            new_gdf = gdf.to_crs('EPSG:4326')
            new_gdf.drop(new_gdf.columns.difference(['geometry']), 'columns',inplace=True)
            layer_geojson = json.loads(new_gdf.to_json())
            new_layer =  GeoJSON(
                data=layer_geojson,
                name=layer_name,
                style={
                    'color': '#9933ff',
                    'fill_color': '#9933ff',
                    'opacity': 1,
                    'fillOpacity': 0.5,
                    'weight': 2},
                hover_style={
                    'color': 'white',
                    'fillOpacity': 0.7},
                )
            self.map.add_layer(new_layer)
            print(f"Loaded the geodataframe from the file :\n{file}")
            logger.info(f"Loaded the geodataframe from the file :\n{file}") 
  
    
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
            logger.info(f"Loading ROIs from file {file}")
            gdf = gpd.read_file(file)
            self.rois = ROI(rois_gdf = gdf)
            logger.info(f"ROIs: {self.rois}")
            # Create new ROI layer
            self.ROI_layer = self.create_layer(self.rois,ROI.LAYER_NAME)
            logger.info(f"ROI_layer: {self.ROI_layer}")
            self.ROI_layer.on_click(self.geojson_onclick_handler)
            self.map.add_layer(self.ROI_layer)
            print(f"Loaded the rois from the file :\n{file}")
            logger.info(f"Loaded the rois from the file :\n{file}")
        else:
            logger.info(f"No file provided. Generating ROIs")
            self.generate_ROIS_fishnet(large_len, small_len)

    def generate_ROIS_fishnet(self,large_len=7500,small_len=5000):
        """Generates series of overlapping ROIS along shoreline on map using fishnet method"""
        if self.bbox is None:
            raise exceptions.Object_Not_Found('bounding box')
        logger.info(f"bbox for ROIs: {self.bbox.gdf}")
        logger.info(f"self.shoreline before check: {self.shoreline}")
        # If no shoreline exists on map then load one in
        if self.shoreline is None:
            self.load_shoreline_on_map()
        logger.info(f"self.shoreline used for ROIs:{ self.shoreline}")
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
        logger.info(f"geojson_onclick_handler: properties : {properties}")
        logger.info(f"geojson_onclick_handler: ROI_id : {properties['id']}")
        ROI_id = properties["id"]
        # Add the id of the clicked ROI to selected_set
        self.selected_set.add(ROI_id)
        logger.info(f"Added ID to selected set: {self.selected_set}")
        if self.selected_ROI_layer is not None:
            self.map.remove_layer(self.selected_ROI_layer)

        self.selected_ROI_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(self.selected_set),
            name="Selected ROIs",
            hover_style={"fillColor": "blue",'fillOpacity': 0.1,"color": "aqua"},
        )
        logger.info(f"selected_ROI_layer: {self.selected_ROI_layer}")
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
        logger.info(f"selected_onclick_handler: properties : {properties}")
        logger.info(f"selected_onclick_handler: ROI_id to remove : {properties['id']}")
        cid = properties["id"]
        self.selected_set.remove(cid)
        logger.info(f"selected set after ID removal: {self.selected_set}")
        if self.selected_ROI_layer is not None:
            self.map.remove_layer(self.selected_ROI_layer)
        # Recreate the selected layers without the layer that was removed
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
                "weight": 2,
                "fillColor": "blue",
                "fillOpacity": 0.1,
            }
        return selected_rois
