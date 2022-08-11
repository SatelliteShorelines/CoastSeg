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
from glob import glob
import geopandas as gpd
# new imports
from skimage.io import imread, imsave
import numpy as np

from CoastSeg import SDS_tools, SDS_download, SDS_tools,SDS_transects,SDS_shoreline
from CoastSeg.file_functions import mk_new_dir
from shapely.geometry import Polygon
from pyproj import Proj, transform
import matplotlib
matplotlib.use("Qt5Agg")


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
        # self.shoreline_names : names shoreline layers
        self.shoreline_names = []
        # self.shorelines_gdf : all the shorelines within the bbox
        self.shorelines_gdf = gpd.GeoDataFrame()
        # Stores all the transects on the map
        self.transects_in_bbox_list=[]
        
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

    def RGB_to_MNDWI(self, RGB_dir_path:str, NIR_dir_path :str, output_path:str)->None:
        """Converts two directories of RGB and NIR imagery to MNDWI imagery in a directory named
         'MNDWI_outputs'.

        Args:
            RGB_dir_path (str): full path to directory containing RGB images
            NIR_dir_path (str): full path to directory containing NIR images
        
        Original Code from doodleverse_utils by Daniel Buscombe
        source: https://github.com/Doodleverse/doodleverse_utils
        """        
        paths=[RGB_dir_path,NIR_dir_path]
        files=[]
        for data_path in paths:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"{data_path} not found")
            f = sorted(glob(data_path+os.sep+'*.jpg'))
            if len(f)<1:
                f = sorted(glob(data_path+os.sep+'images'+os.sep+'*.jpg'))
            files.append(f)

        # creates matrix:  bands(RGB) x number of samples(NIR)
        # files=[['full_RGB_path.jpg','full_NIR_path.jpg'],
        # ['full_jpg_path.jpg','full_NIR_path.jpg']....]
        files = np.vstack(files).T

        # output_path: directory to store MNDWI outputs
        output_path += os.sep+ 'MNDWI_outputs'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        # Create subfolder to hold MNDWI ouputs in
        output_path=mk_new_dir('MNDWI_ouputs',output_path )
        

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
            # save MNDWI as .npz file 
            segfile = output_path+os.sep+ROOT_STRING+'_noaug_nd_data_000000'+str(counter)+'.npz'
            np.savez_compressed(segfile, **datadict)
            del datadict, mndwi, green_band, swir
            
        return output_path
     
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
    
    
    def get_intersecting_files(self, bbox_gdf : gpd.geodataframe, type : str)-> list:
        """Returns a list of filesnames that intersect with bbox_gdf

        Args:
            gpd_bbox (geopandas.geodataframe.GeoDataFrame): bbox containing ROIs
            type (str): to be used later

        Returns:
            list: intersecting_files containing filenames whose contents intersect with bbox_gdf
        """
        # dataframe containing total bounding box for each transects or shoreline file
        total_bounds_df=self.load_total_bounds_df(type)
        # filenames where transects/shoreline'bbox intersect bounding box drawn by user
        intersecting_files=[]
        for filename in total_bounds_df.index:
            minx, miny, maxx, maxy=total_bounds_df.loc[filename]
            intersection_df = bbox_gdf.cx[minx:maxx, miny:maxy]
            bbox_gdf.cx[minx:maxx, miny:maxy].plot()
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
    

    def style_transect(self, transect_gdf:gpd.geodataframe)->dict:
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

    def convert_roi_to_polygon(self,rois_gdf:gpd.geodataframe,id:str)->Polygon:
        """Returns the roi with given id as Shapely.Polygon

        Args:
            rois_gdf (gpd.geodataframe): geodataframe with all rois
            id (str): roi_id

        Returns:
            Polygon: roi with the id converted to Shapely.Polygon
        """
        if id is None:
            single_roi=rois_gdf
        else:
            # Select a single roi
            single_roi = rois_gdf[rois_gdf['id']==id]
        single_roi=single_roi["geometry"].to_json()
        single_roi = json.loads(single_roi)
        poly_roi = Polygon(single_roi["features"][0]["geometry"]['coordinates'][0])
        return poly_roi
    
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
            # Get all the transects that intersect with bbox
            # Replace clip with intersects
            transects_in_bbox=self.get_intersecting_transects(gpd_bbox,data,None)
            # transects_in_bbox=bbox.clip_to_bbox(data, gpd_bbox)
            if transects_in_bbox.empty:
                print("Skipping ",transects_layer_name)
            else:
                # Add the transect to list of all transects
                self.transects_in_bbox_list.append(transects_in_bbox)
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
        self.transects_in_bbox_list=[]

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
                    zenodo_id_mapping={
                        'E_USA_SouthCarolina_NorthCarolina_ref_shoreline.geojson':'6824465',
                        'NE_USA_Delaware_Maine_ref_shoreline.geojson':'6824429',
                        'SE_USA_Louisiana_Georgia_ref_shoreline.geojson':'6824473',
                        'S_USA_Texas_Louisiana_ref_shoreline.geojson':'6824487',
                        'W_USA_California_ref_shoreline.geojson':'6824504',
                        'W_USA_Oregon_Washington_ref_shoreline.geojson':'6824510',
                        'USA_Alaska_ref_shoreline.geojson':'6836629'
                    }
                    self.download_shoreline(file, zenodo_id_mapping[file])
                    shoreline_path=os.path.abspath(os.getcwd())+os.sep+"Coastseg"+os.sep+"shorelines"+os.sep+file
                    shoreline=bbox.read_gpd_file(shoreline_path)
                    
                # Create a single dataframe to hold all shodwonlrelines from all files
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


    def generate_ROIS_fishnet(self,large_fishnet=7500,small_fishnet=5000):
        """Generates series of overlapping ROIS along shoreline on map using fishnet method"""
        # Ensure drawn bbox(bounding box) within allowed size
        bbox.validate_bbox_size(self.shapes_list)
        # Get the geodataframe for the bbox
        gpd_bbox = bbox.create_geodataframe_from_bbox(self.shapes_list)
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

        # Add an id column
        num_roi = len(fishnet_intersect_gpd)
        fishnet_intersect_gpd['id'] = arange(0, num_roi)
        # @todo remove this test code
        self.fishnet_intersect_gpd=fishnet_intersect_gpd
        # Save the fishnet intersection with shoreline to json
        fishnet_geojson = fishnet_intersect_gpd.to_json()
        fishnet_dict = json.loads(fishnet_geojson)

        # Add style to each feature in the geojson
        for feature in fishnet_dict["features"]:
            feature["properties"]["style"] = {
                "color": "black",
                "weight": 3,
                "fillColor": "grey",
                "fillOpacity": 0.1,
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
            self.m.remove_layer(self.selected_layer)

        self.selected_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(self.selected_set),
            name="Selected ROIs",
            hover_style={"fillColor": "blue","color": "aqua"},
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
            hover_style={"fillColor": "blue","color": "aqua"},
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
                "weight": 3,
                "fillColor": "grey",
                "fillOpacity": 0.1,
            }
        return geojson
