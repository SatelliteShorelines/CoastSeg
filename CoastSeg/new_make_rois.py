# External dependencies imports
import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import shape, LineString
from shapely.ops import unary_union

# Internal imports
from CoastSeg import new_make_overlapping_roi
TEMP_FILENAME="temp.geojson"

def get_geojson_polygons(linestring):
    """ Returns the ROI rectangles in geojson"""
    multipoint_list=interpolate_points(linestring)
    tuples_list=convert_multipoints_to_tuples(multipoint_list)
    geojson_polygons=create_reactangles(tuples_list)
    return geojson_polygons

def check_all_ROI_overlap(df_all_ROIs,df_overlap):
    """Compares the IDs of the ROIs in df_overlap(contains only the ids of the overlapping ROIs), to df_all_rois(contains the ids of all ROIs)
    Returns
    True: If all the IDs in df_all_ROIs are also in df_overlap
    False: If NOT all the IDs in df_all_ROIs are also in df_overlap"""
    all_ids_list=list(df_all_ROIs["id"])
    # print(f"\n all_ids_list:{all_ids_list}\n")
    overlapping_ids=df_overlap["primary_id"]
    # print(f"\n overlapping_ids:\n{overlapping_ids}\n")
    missing_list=list(set(all_ids_list) - set(overlapping_ids))
    if  missing_list ==[]:
        return True
    return False
            
def adjust_num_pts(new_num_pts):
    """Rounds the number of points to a whole number 1<=x<=100"""
    new_num_pts=int(round(new_num_pts,0))
    if new_num_pts<1:
        new_num_pts=1
    elif new_num_pts > 100:
        new_num_pts =100
    else:
        return new_num_pts 
           
def check_average_ROI_overlap(df_overlap,percentage):
    """" Returns True if the mean overlap in the df is greater than percentage"""
    mean_overlap=df_overlap["%_overlap"].mean()
    if mean_overlap > percentage:
        return True
    return False
     
def create_overlap(filename: str, line:"shapely.geometry.linestring.LineString",start_id:int,end_id_list:list):
    """
    Check if all the ROI overlap with at both of its neighbors. If it doesn't increase the number of ROIs up to 26. If the number of ROIs exceed 26,
    then no more will be drawn even if there is not enough overlap.
    Arguments:
    -----------
    filename: str
        The name of the file containing all the geojson data for all the ROIs generated.
    line:  shapely.geometry.linestring.LineString 
        represents each segment of the vector
    start_id:  int 
        the id number for the first geojson in the line
    Returns:
    -----------
    Updated df_overlap.
    """         
    #Initial number of points interpolated along the line
    num_pts=5
    # Boolean indicates whether every single ROI generated overlaps at least one other ROI
    do_all_ROI_overlap=False
    #Boolean indicates whether mean overlap was over 80% and if the number points is greater than 2
    is_overlap_excessive=True
    df_overlap=new_make_overlapping_roi.get_overlap_dataframe(filename)
    # If df_overlap is means not a single ROI overlapped each other 
    if not df_overlap.empty:
        df_all_ROIs = gpd.read_file(filename)
        do_all_ROI_overlap=check_all_ROI_overlap(df_all_ROIs,df_overlap)
        
        if do_all_ROI_overlap==True:
            if check_average_ROI_overlap(df_overlap,.35)== True:
            # If the average overlap is over 35% decrease number of rois by 1
                num_pts=adjust_num_pts(num_pts-1)
                is_overlap_excessive=True
                # print(f"num_pts decreased to: {num_pts}")
        if do_all_ROI_overlap==False:
             # If not all the rois overlap increase number of rois by 1
                num_pts=adjust_num_pts(num_pts+1)
                # print(f"num_pts increased to: {num_pts}")
# Keep looping while not all the rois overlap and the average overlap is more than 80%
    while do_all_ROI_overlap== False and  is_overlap_excessive== True:
        multipoint_list=interpolate_points(line,num_pts)
        tuples_list=convert_multipoints_to_tuples(multipoint_list)
        geojson_polygons=create_reactangles(tuples_list)
        end_id=new_make_overlapping_roi.write_to_geojson_file(filename,geojson_polygons,perserve_id=False,start_id=start_id)
        end_id_list.append(end_id)
        print(f"\nend_id_list {end_id_list}\n")
        df_overlap=new_make_overlapping_roi.get_overlap_dataframe(filename)
        #If df_overlap is empty means not a single ROI overlapped each other 
        if not df_overlap.empty:
            df_all_ROIs = gpd.read_file(filename)
            do_all_ROI_overlap=check_all_ROI_overlap(df_all_ROIs,df_overlap)
        else:
             do_all_ROI_overlap=False
        if do_all_ROI_overlap== False:
            if num_pts == 1 or num_pts>25:
                    # print(f"IN LOOP: num_pts is 1. BREAKING")
                    break    #if the num_pts == 1 means no more ROIs should be removed
            num_pts=adjust_num_pts(num_pts+1)
        else:   # all ROIs overlap 
                if num_pts == 1 or num_pts>25:
                    break    #if the num_pts == 1 means no more ROIs should be removed         
                is_overlap_excessive=check_average_ROI_overlap(df_overlap,.35)
                if is_overlap_excessive== True:
                # If the average overlap is over 35% decrease number of rois by 1
                    num_pts=adjust_num_pts(num_pts-1)
                    is_overlap_excessive=True
                    # print(f"IN LOOP: num_pts decreased to: {num_pts}")
    return df_overlap

def interpolate_points(line:"shapely.geometry.linestring.LineString",num_pts=5)->list:
    """
    Create a list of multipoints for the interpolated points along each linestring
    Arguments:
    -----------
    line: "shapely.geometry.linestring.LineString"
        shapely.geometry.linestring.LineString that represents each segment of the coastline vector
    num_pts: int
        integer value representing the number of interpolated points created along the LineString
    Returns:
    -----------
    multipoint_list: list
    A list of multiple shapely.geometry.multipoint.MultiPoint 
    """
    # multipoint_list holds the multipoint for each feature of the coastline within the bbox
    multipoint_list=[]
    distance_delta=line.length/num_pts
    distances = np.arange(0, line.length, distance_delta)
    if line.is_closed:
        #Its a closed shape so its boundary points are NULL
        boundary=shapely.geometry.Point(line.coords[0])
    else: 
        boundary=line.boundary.geoms[0]
            
    points = [line.interpolate(distance) for distance in distances] + [boundary]
    multipoint = unary_union(points) 
    multipoint_list.append(multipoint)
    return multipoint_list

def convert_multipoints_to_tuples(multipoint_list:list)-> list:
    """
    Create a list of tuples for the points in multipoint_list
    Arguments:
    -----------
    multipoint_list: list
        A list of multiple shapely.geometry.multipoint.MultiPoint
    Returns:
    -----------
    tuples_list: list
        A list of tuples each tuple represents a single point
    """
    tuples_list=[]
    for multipoint in  multipoint_list:
       # Create an empty array to hold all the points as tuples
        points_list=[]
        if isinstance(multipoint,shapely.geometry.Point):
            point= multipoint
            point_tuple=(point.coords[0][1],point.coords[0][0])
            points_list.append(point_tuple)
        else:
            # First get each point from the multipoint object
            points_array=[point for point in multipoint.geoms]
            # For each point swap lat and lng because ipyleaflet swaps them
            for point in points_array:
                point_tuple=(point.coords[0][1],point.coords[0][0])
                points_list.append(point_tuple)
        tuples_list.append(points_list)
    return tuples_list

def convert_corners_to_geojson(upper_right_y : float, upper_right_x: float,upper_left_y: float, upper_left_x: float,lower_left_y: float,  lower_left_x: float,lower_right_y: float,lower_right_x: float) -> dict:
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

def create_reactangles(tuples_list: list,size:int=0.04)-> dict:
    """
    Create the geojson rectangles for each point in the tuples_list
    Arguments:
    -----------
    tuples_list: list
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
    for points_list in tuples_list:
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
            geojson_polygon=convert_corners_to_geojson(upper_right_x, upper_right_y,upper_left_x, upper_left_y,lower_left_x,lower_left_y,lower_right_x,lower_right_y)
            geojson_polygons["features"].append(geojson_polygon)
    return geojson_polygons