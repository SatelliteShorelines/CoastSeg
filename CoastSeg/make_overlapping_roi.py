# External imports
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import shape, LineString
from geojson import Feature, FeatureCollection, dump
import geojson
from tqdm.notebook import tqdm_notebook

# Internal imports
from CoastSeg import make_rois

class ROI:

    def __init__(self,coastline:dict):
        assert bool(coastline) != False
        self._coastline=coastline
     
    @property
    def coastline(self):
        return self._coastline

    @coastline.setter
    def coastline(self,new_coastline):
        #DO NOT ALLOW THE USER TO MODIFY THE COASTLINE
        raise Exception("ERROR: Not allowed to modify the coastline geojson after it has been added to the map")

    def get_overlap_csv(csv_name:"str"):
        """Creates an empty geodataframe to hold the overlapping ROIs information"""
        df_overlap=gpd.GeoDataFrame({"id":[],'primary_id':[],'geometry':[],'intersection_area':[], '%_overlap':[]})
        df_overlap=df_overlap.astype({'id': 'int32','primary_id': 'int32'})
        #Add a check if this file exists then delete it if it does
        df_overlap.to_csv(csv_name, mode='w',index=False)
        return df_overlap 
        
    def get_ROIs(self,roi_filename,csv_filename):
        """Writes the ROIs to a geojson file and the overlap between those ROIs to a csv file."""
        coastline=self._coastline
        lines_list=self._create_linestring_list()
        #temp_filename file where each segment of the coastline's rois will be written
        temp_filename="temp.geojson"
        # Start_id is used as the starting number for writing the ROI ids
        start_id=0
        # df_master_overlap=ROI.get_overlap_csv('overlap.csv')
        df_master_overlap=ROI.get_overlap_csv(csv_filename)
        finalized_roi={'type': 'FeatureCollection','features': []}

        for line in tqdm_notebook(lines_list,desc="Calculating Overlap"):
            Overlapping_ROIs=make_rois.ROIGroup(line,start_id)
            Overlapping_ROIs.calculate_overlap()
#           Once all the overlapping ROIs have been created update the start_id for the next set of ROIs
            start_id=Overlapping_ROIs.end_id   
            #Get the dataframe that overlapped and add it to the master overlap df
            overlap_df=Overlapping_ROIs.overlap_df
            df_master_overlap=df_master_overlap.append(overlap_df, ignore_index=False)
            #Read the geojson data for the ROIs and add it to the geojson list
            rois_geojson=ROI.read_geojson_from_file(temp_filename)
            for single_roi in rois_geojson["features"]:
                finalized_roi["features"].append(single_roi)
         
        # Write to the permanent geojson and csv files
        ROI.write_to_geojson_file(roi_filename,finalized_roi,perserve_id=True)
        df_master_overlap.to_csv(csv_filename, mode='a',header=False, index=False)

    def get_selected_roi(selected_set:tuple,roi_data:dict)-> dict:
        """
        Returns the geojson of the ROIs selected by the user

        Arguments:
        -----------
        selected_set:tuple
            A tuple containing the ids of the ROIs selected by the user
            
        roi_data:dict
            A geojson dict containing all the rois currently on the map

        Returns:
        -----------
        geojson_polygons: dict
           geojson dictionary containing all the ROIs selected  
        """
        # Check if selected_set is empty
        assert len(selected_set) != 0, "\n Please select at least one ROI from the map before continuing."
        # Create a dictionary for the selected ROIs and add the user's selected ROIs to it  
        selected_ROI={}
        selected_ROI["features"] = [
            feature
            for feature in roi_data["features"]
            if feature["properties"]["id"] in selected_set
        ]
        return selected_ROI                
       
    @staticmethod
    def min_overlap_btw_vectors(geojsonfile, csv_filename,overlap_percent: float =.65):
        overlap_btw_vectors_df=ROI.get_overlap_dataframe(geojsonfile)
        # Set of IDs where the overlap >= 65%
        drop_list=list(set(overlap_btw_vectors_df[overlap_btw_vectors_df["%_overlap"]>overlap_percent]["primary_id"]))
        
        # load the geojson for all the rois across all parts of the coastline
        geojson=ROI.read_geojson_from_file(geojsonfile)
        all_ids=[]
        ids_in_features = [] 
        features=[]
        # Remove the features overlapping more than 65% from the geojson 
        for feature in tqdm_notebook(geojson["features"],desc="Removing ROI with Excessive Overlap"):
            all_ids.append(feature["properties"]["id"])
            if feature["properties"]["id"]  not in drop_list:
                features.append(Feature(geometry=feature["geometry"], properties=feature["properties"]))
                ids_in_features.append(feature["properties"]["id"])
        
        ids_not_in_features=set(all_ids)-set(ids_in_features)
#       Checks if all the ROIS were removed if this was the case then we want to return the original data
        if len(ids_in_features)==0:
            # print("ALL ROIS were removed by overlap check")
            return overlap_btw_vectors_df
        else:
            feature_collection = FeatureCollection(features)
            with open(geojsonfile, 'w') as f:
                 dump(feature_collection, f)
    #         Drop the any rows with PID in the drop_list
            min_overlap_df=overlap_btw_vectors_df[~overlap_btw_vectors_df["primary_id"].isin(drop_list)]
            min_overlap_df=min_overlap_df[~min_overlap_df["id"].isin(drop_list)]
            min_overlap_df.to_csv(csv_filename)
            return min_overlap_df

    @staticmethod
    def read_geojson_from_file(selected_roi_file: str)-> dict:
        """
         Returns the geojson of the selected ROIs from the file specified by selected_roi_file

        Arguments:
        -----------
        selected_roi_file: str
            The filename of the geojson file containing all the ROI selected by the user

        Returns:
        -----------
        data: dict
            geojson of the selected ROIs
        """
        with open(selected_roi_file) as f:
            data = geojson.load(f)
        return data

    @staticmethod
    def get_overlap_dataframe(filename):
    #     Read the geojson into a dataframe for the related ROIs along this portion of the coastline
        df = gpd.read_file(filename) 
        # Make dataframe to hold all the overlays 
        df_master=gpd.GeoDataFrame({"id":[],'primary_id':[],'geometry':[],'intersection_area':[], '%_overlap':[]})
        df_master=df_master.astype({'id': 'int32','primary_id': 'int32'})
    #     Iterate through all the polygons in the dataframe
        for index in df.index:
            polygon=df.iloc[index]
            df_intersection=ROI.compute_overlap(polygon,df)
            if not df_intersection.empty:
                df_master=df_master.append(df_intersection)
        return df_master
    
    @staticmethod
    def compute_overlap(polygon: 'pandas.core.series.Series',df:'geopandas.geodataframe.GeoDataFrame' ):
        """
       Create a geopandas.geodataframe containing the overlap between the current polygon and the other ROIs in df
       The geodataframe contains data about which ROIs in df intersected with the polygon as indicated by the column "id",
       the id of polygon is in the column named "primary_id", the percentage of the polygon's area that was overlapping
       is in the column "%_overlap", and lastly, the amount of polygon's area being intersected is in the "intersection_area" 
       column.
       ""

        Arguments:
        -----------
        polygon: 'pandas.core.series.Series'
            Polygon that is being checked for overlap with the other ROI polygons in df

        df:'geopandas.geodataframe.GeoDataFrame'
            A geodataframe containing all the ROI polygons along a particular vector

        Returns:
        -----------
        res_intersection: 'geopandas.geodataframe.GeoDataFrame'
           A geodataframe containing the ROIs that intersected with each other as well as the %_overlap calculated.
        """
    #     Create a dataframe for polygon currently being checked for overlap
        poly_series=gpd.GeoSeries(polygon["geometry"])
        df1 = gpd.GeoDataFrame({'geometry': poly_series,"primary_id":polygon["id"]})
        df1=df1.set_crs(df.crs)
    #     Overlay the current ROI's geometry onto the rest of ROIs within the same region. See if any intersect
        res_intersection = df.overlay(df1, how='intersection')
    #     Drop the rows where the ROIs are overlapping themselves
        res_intersection.drop(res_intersection[res_intersection['id'] == res_intersection['primary_id'] ].index, inplace = True)
        res_intersection=res_intersection.set_crs(df.crs)
        # Append the intersection area to the dataframe
        intersection_area=res_intersection.area
        intersection_area=intersection_area.rename("intersection_area")
        res_intersection=res_intersection.merge(intersection_area,left_index=True, right_index=True)
        
        # Get the area of any given ROI 
        total_area=df.area[0]

        # Compute % overlap
        df1_percent_overlap=res_intersection["intersection_area"]/total_area
        df1_percent_overlap=df1_percent_overlap.rename("%_overlap")
    #     Add the % overlap to the dataframe
        res_intersection=res_intersection.merge(df1_percent_overlap,left_index=True, right_index=True) 
        res_intersection
        return res_intersection

    
    @staticmethod
    def write_to_geojson_file(filename : str,geojson_polygons : dict, perserve_id: bool =False, start_id: int =0):
        """Make a filename.geojson file from dictionary geojson_polygons 

        Arguments:
        -----------

        filename : str
            The name of the geojson file to be written to. MUST end in .geojson

        geojson_polygons : dict
            The dictionary containing the geojson data. Must contain all geojson within the features section of geojson.

        perserve_id: boolean   default=False
            Boolean that if True perserves the property id and writes it to the geojson file

        start_id: int   default=0
           Starts writing the ids of the geojson at this index 

        Returns:
        --------
        end_id : int
            The (last id written to geojson). Intended to be used as the starting id for next iteration.
        """
        features = []
        count=0
        end_id=start_id
        if not perserve_id:
            for geoObj in geojson_polygons["features"]:
                features.append(Feature(properties={"id":count+start_id},geometry=geoObj["geometry"]))
                count=count+1
        elif perserve_id:
            for geoObj in geojson_polygons["features"]:
                features.append(Feature(properties={"id": geoObj["properties"]["id"]},geometry=geoObj["geometry"]))

        feature_collection = FeatureCollection(features)
        with open(f'{filename}', 'w') as f:
            dump(feature_collection, f)
        end_id+=count
        return end_id
    
    
    def _create_linestring_list(self)->list:
        """
        Create a list of linestrings from the multilinestrings and linestrings that compose the vector

        Arguments:
        -----------
        vector_within_bounding_box_geojson: dict
            geojson vector 
        Returns:
        -----------
        lines_list: list
            list of multiple shapely.geometry.linestring.LineString that represent each segment of the vector
        """
        lines_list=[]
        vector_within_bounding_box_geojson=self._coastline
        length_vector_bbox_features=len(vector_within_bounding_box_geojson['features'])
        length_vector_bbox_features

        if(length_vector_bbox_features != 1):
            for i in range(0,length_vector_bbox_features):
                if  vector_within_bounding_box_geojson['features'][i]['geometry']['type'] == 'MultiLineString':
                    for y in range(len(vector_within_bounding_box_geojson['features'][i]['geometry']['coordinates'])):
                        line=LineString(vector_within_bounding_box_geojson['features'][i]['geometry']['coordinates'][y])
                        lines_list.append(line)
                elif  vector_within_bounding_box_geojson['features'][i]['geometry']['type'] == 'LineString':
                    line=LineString(vector_within_bounding_box_geojson['features'][i]['geometry']['coordinates'])
                    lines_list.append(line)
        else:
            for i in range(0,len(vector_within_bounding_box_geojson['features'])):
                if  vector_within_bounding_box_geojson['features'][0]['geometry']['type'] == 'MultiLineString':
                    for y in range(len(vector_within_bounding_box_geojson['features'][0]['geometry']['coordinates'])):
                        line=LineString(vector_within_bounding_box_geojson['features'][0]['geometry']['coordinates'][y])
                        lines_list.append(line)
                elif  vector_within_bounding_box_geojson['features'][i]['geometry']['type'] == 'LineString':
                    line=LineString(vector_within_bounding_box_geojson['features'][i]['geometry']['coordinates'])
                    lines_list.append(line)
        return lines_list
        

    def _convert_to_geojson(self,upper_right_y : float, upper_right_x: float,upper_left_y: float, upper_left_x: float,lower_left_y: float,  lower_left_x: float,lower_right_y: float,lower_right_x: float) -> dict:
        """Convert the 4 corners of the rectangle into geojson  """
        geojson_feature={}
        geojson_feature["type"]="Feature"
        geojson_feature["properties"]={}
        geojson_feature["geometry"]={}

        geojson_polygon={}
        geojson_polygon["type"]="Polygon"
        geojson_polygon["coordinates"]=[]
    #     The coordinates(which are 1,2 arrays) are nested within a parent array
        nested_array=[]
        nested_array.append([upper_right_x, upper_right_y])
        nested_array.append([upper_left_x, upper_left_y])
        nested_array.append([lower_left_x, lower_left_y])
        nested_array.append([lower_right_x, lower_right_y])
        #GeoJson rectangles have the first point repeated again as the last point
        nested_array.append([upper_right_x, upper_right_y])

        geojson_polygon["coordinates"].append(nested_array)

        geojson_feature["geometry"]=geojson_polygon
        return geojson_feature
    
    def _create_reactangles(self,points_list_collection: list,size:int=0.04)-> dict:
        """
        Create the geojson rectangles for each point in the points_list_collection

        Arguments:
        -----------
        points_list_collection: list
            list of tuples containing all the interpolated points along the given vector

        size: float
            A float that will be used as the multiplier for the ROI sizes

        Returns:
        -----------
        geojson_polygons: dict
           geojson dictionary contains all the rectangles generated
        """                                         
        geojson_polygons={"type": "FeatureCollection","features":[]}

        # Create a rectangle at each point on the line
        # Swap the x and y for each point because ipyleaflet swaps them for draw methods
        for points_list in points_list_collection:
            for point in points_list:
                upper_right_x=point[0]-(size/2)
                upper_right_y=point[1]-(size/2)
                upper_left_x=point[0]+(size/2)
                upper_left_y=point[1]-(size/2)
                lower_left_x=point[0]+(size/2)
                lower_left_y=point[1]+(size/2)
                lower_right_x=point[0]-(size/2)
                lower_right_y=point[1]+(size/2)
                #Convert each set of points to geojson (DONT swap x and y this time)
                geojson_polygon=self._convert_to_geojson(upper_right_x, upper_right_y,upper_left_x, upper_left_y,lower_left_x,lower_left_y,lower_right_x,lower_right_y)
                geojson_polygons["features"].append(geojson_polygon)
        return geojson_polygons
    
    