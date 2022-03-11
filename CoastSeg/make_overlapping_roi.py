# External imports
import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import LineString
from shapely.ops import unary_union
from geojson import Feature, FeatureCollection, dump
import geojson
from tqdm.notebook import tqdm_notebook
import os

# Global vars
TEMP_FILENAME = "temp.geojson"


def get_empty_overlap_df():
    """Creates an empty geodataframe to hold the overlapping ROIs information"""
    df_overlap = gpd.GeoDataFrame({"id": [],
                                   'primary_id': [],
                                   'geometry': [],
                                   'intersection_area': [],
                                   '%_overlap': []})
    df_overlap = df_overlap.astype({'id': 'int32', 'primary_id': 'int32'})
    return df_overlap


def get_ROIs(coastline: dict, roi_filename: str, csv_filename: str):
    """Writes the ROIs to a geojson file and the overlap between those ROIs to a csv file.
    Arguments:
    -----------
    coastline : dict
        Geojson containing the portion of the coastline clipped within the bbox
    roi_filename:str
        File name of the json file to hold the roi data
    csv_filename:str
        File name of the csv file to hold the overlap between the rois
    """
    lines_list = get_linestring_list(coastline)
    # TEMP_FILENAME: file where each segment of coastline's rois written to
    # master_overlap_df: geopandas dataframe containing all the overlap data
    # for the rois
    # Start_id is used as the starting number for writing the ROI ids
    start_id = 0
    # list to hold all the end_ids created in create_overlap()
    end_id_list = []
    master_overlap_df = get_empty_overlap_df()
    finalized_roi = {'type': 'FeatureCollection', 'features': []}
    for line in tqdm_notebook(lines_list, desc="Calculating Overlap"):
        geojson_polygons = get_geojson_polygons(line)
        end_id = write_to_geojson_file(
            TEMP_FILENAME,
            geojson_polygons,
            perserve_id=False,
            start_id=start_id)
        overlap_df = create_overlap(TEMP_FILENAME, line, start_id, end_id_list)
        if len(end_id_list) != 0:
            # Get the most recent end_id and clear the list of end_ids
            start_id = end_id_list.pop()
            end_id_list = []
        else: 
            # Once all the overlapping ROIs have been created update the
            # start_id for the next set of ROIs
            start_id = end_id
        master_overlap_df = master_overlap_df.append(
            overlap_df, ignore_index=False)
        # Read the geojson data for the ROIs and add it to the geojson list
        rois_geojson = read_geojson_from_file(TEMP_FILENAME)
        for single_roi in rois_geojson["features"]:
            finalized_roi["features"].append(single_roi)

    # Write to the permanent geojson and csv files
    write_to_geojson_file(roi_filename, finalized_roi, perserve_id=True)
    master_overlap_df.to_csv(csv_filename, mode='a', header=False, index=False)


def min_overlap_btw_vectors(
        geojsonfile,
        csv_filename,
        overlap_percent: float = .65):
    overlap_btw_vectors_df = get_overlap_dataframe(geojsonfile)
    # Set of IDs where the overlap >= 65%
    drop_list = list(set(
        overlap_btw_vectors_df[overlap_btw_vectors_df["%_overlap"] > overlap_percent]["primary_id"]))

    # load the geojson for all the rois across all parts of the coastline
    geojson = read_geojson_from_file(geojsonfile)
    all_ids = []
    ids_in_features = []
    features = []
    # Remove the features overlapping more than 65% from the geojson
    for feature in tqdm_notebook(
            geojson["features"],
            desc="Removing ROI with Excessive Overlap"):
        all_ids.append(feature["properties"]["id"])
        if feature["properties"]["id"] not in drop_list:
            features.append(
                Feature(
                    geometry=feature["geometry"],
                    properties=feature["properties"]))
            ids_in_features.append(feature["properties"]["id"])

    ids_not_in_features = set(all_ids) - set(ids_in_features)
# Checks if all the ROIS were removed if this was the case then we want to
# return the original data
    if len(ids_in_features) == 0:
        return overlap_btw_vectors_df
    else:
        feature_collection = FeatureCollection(features)
        with open(geojsonfile, 'w') as f:
            dump(feature_collection, f)
#         Drop the any rows with PID in the drop_list
        min_overlap_df = overlap_btw_vectors_df[~overlap_btw_vectors_df["primary_id"].isin(
            drop_list)]
        min_overlap_df = min_overlap_df[~min_overlap_df["id"].isin(drop_list)]
        min_overlap_df.to_csv(csv_filename)
        return min_overlap_df


def read_geojson_from_file(selected_roi_file: str) -> dict:
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
    assert os.path.exists(selected_roi_file), f"ERROR: {selected_roi_file} does not exist to read selected rois from"
    with open(selected_roi_file) as f:
        data = geojson.load(f)
    return data


def get_overlap_dataframe(filename):
    # Read the geojson into a dataframe for the related ROIs along this
    # portion of the coastline
    df = gpd.read_file(filename)
    # Make dataframe to hold all the overlays
    df_master = get_empty_overlap_df()
#     Iterate through all the polygons in the dataframe
    for index in df.index:
        polygon = df.iloc[index]
        df_intersection = compute_overlap(polygon, df)
        if not df_intersection.empty:
            df_master = df_master.append(df_intersection)
    return df_master


def compute_overlap(
        polygon: 'pandas.core.series.Series',
        df: 'geopandas.geodataframe.GeoDataFrame'):
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
    poly_series = gpd.GeoSeries(polygon["geometry"])
    df1 = gpd.GeoDataFrame(
        {'geometry': poly_series, "primary_id": polygon["id"]})
    df1 = df1.set_crs(df.crs)
# Overlay the current ROI's geometry onto the rest of ROIs within the same
# region. See if any intersect
    res_intersection = df.overlay(df1, how='intersection')
#     Drop the rows where the ROIs are overlapping themselves
    res_intersection.drop(
        res_intersection[res_intersection['id'] == res_intersection['primary_id']].index, inplace=True)
    res_intersection = res_intersection.set_crs(df.crs)
    # Append the intersection area to the dataframe
    intersection_area = res_intersection.area
    intersection_area = intersection_area.rename("intersection_area")
    res_intersection = res_intersection.merge(
        intersection_area, left_index=True, right_index=True)

    # Get the area of any given ROI
    total_area = df.area[0]
    # Compute % overlap
    df1_percent_overlap = res_intersection["intersection_area"] / total_area
    df1_percent_overlap = df1_percent_overlap.rename("%_overlap")
#     Add the % overlap to the dataframe
    res_intersection = res_intersection.merge(
        df1_percent_overlap, left_index=True, right_index=True)
    res_intersection
    return res_intersection


def write_to_geojson_file(
        filename: str,
        geojson_polygons: dict,
        perserve_id: bool = False,
        start_id: int = 0):
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
    count = 0
    end_id = start_id
    if not perserve_id:
        for geoObj in geojson_polygons["features"]:
            features.append(
                Feature(
                    properties={
                        "id": count +
                        start_id},
                    geometry=geoObj["geometry"]))
            count = count + 1
    elif perserve_id:
        for geoObj in geojson_polygons["features"]:
            features.append(
                Feature(
                    properties={
                        "id": geoObj["properties"]["id"]},
                    geometry=geoObj["geometry"]))
    feature_collection = FeatureCollection(features)
    with open(f'{filename}', 'w') as f:
        dump(feature_collection, f)
    end_id += count
    return end_id


def get_linestring_list(vector_in_bbox_geojson: dict) -> list:
    """
    Create a list of linestrings from the multilinestrings and linestrings that compose the vector
    Arguments:
    -----------
    vector_in_bbox_geojson: dict
        geojson vector
    Returns:
    -----------
    lines_list: list
        list of multiple shapely.geometry.linestring.LineString that represent each segment of the vector
    """
    lines_list = []
    length_vector_bbox_features = len(vector_in_bbox_geojson['features'])
    assert length_vector_bbox_features != 0, "ERROR: There  must be at least 1 feature in bounding box."
    for i in range(0, length_vector_bbox_features):
        if vector_in_bbox_geojson['features'][i]['geometry']['type'] == 'MultiLineString':
            for y in range(
                    len(vector_in_bbox_geojson['features'][i]['geometry']['coordinates'])):
                line = LineString(
                    vector_in_bbox_geojson['features'][i]['geometry']['coordinates'][y])
                lines_list.append(line)
        elif vector_in_bbox_geojson['features'][i]['geometry']['type'] == 'LineString':
            line = LineString(
                vector_in_bbox_geojson['features'][i]['geometry']['coordinates'])
            lines_list.append(line)
        else:
            raise AssertionError("Error: Only features of types LineString or MultiLineString are allowed.")
    return lines_list


def get_geojson_polygons(linestring):
    """ Returns the ROI rectangles in geojson"""
    multipoint_list = interpolate_points(linestring)
    tuples_list = convert_multipoints_to_tuples(multipoint_list)
    geojson_polygons = create_reactangles(tuples_list)
    return geojson_polygons


def interpolate_points(
        line: "shapely.geometry.linestring.LineString",
        num_pts=5) -> list:
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
    # multipoint_list holds the multipoint for each feature of the coastline
    # within the bbox
    multipoint_list = []
    distance_delta = line.length / num_pts
    distances = np.arange(0, line.length, distance_delta)
    if line.is_closed:
        # Its a closed shape so its boundary points are NULL
        boundary = shapely.geometry.Point(line.coords[0])
    else:
        boundary = line.boundary.geoms[0]

    points = [line.interpolate(distance)
              for distance in distances] + [boundary]
    multipoint = unary_union(points)
    multipoint_list.append(multipoint)
    return multipoint_list


def convert_multipoints_to_tuples(multipoint_list: list) -> list:
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
    tuples_list = []
    for multipoint in multipoint_list:
       # Create an empty array to hold all the points as tuples
        points_list = []
        if isinstance(multipoint, shapely.geometry.Point):
            point = multipoint
            point_tuple = (point.coords[0][1], point.coords[0][0])
            points_list.append(point_tuple)
        else:
            # First get each point from the multipoint object
            points_array = [point for point in multipoint.geoms]
            # For each point swap lat and lng because ipyleaflet swaps them
            for point in points_array:
                point_tuple = (point.coords[0][1], point.coords[0][0])
                points_list.append(point_tuple)
        tuples_list.append(points_list)
    return tuples_list


def convert_corners_to_geojson(
        upper_right_y: float,
        upper_right_x: float,
        upper_left_y: float,
        upper_left_x: float,
        lower_left_y: float,
        lower_left_x: float,
        lower_right_y: float,
        lower_right_x: float) -> dict:
    """Convert the 4 corners of the rectangle into geojson  """
    geojson_feature = {}
    geojson_feature["type"] = "Feature"
    geojson_feature["properties"] = {}
    geojson_feature["geometry"] = {}
    geojson_polygon = {}
    geojson_polygon["type"] = "Polygon"
    geojson_polygon["coordinates"] = []
#     The coordinates(which are 1,2 arrays) are nested within a parent array
    nested_array = []
    nested_array.append([upper_right_x, upper_right_y])
    nested_array.append([upper_left_x, upper_left_y])
    nested_array.append([lower_left_x, lower_left_y])
    nested_array.append([lower_right_x, lower_right_y])
    # GeoJson rectangles have the first point repeated again as the last point
    nested_array.append([upper_right_x, upper_right_y])
    geojson_polygon["coordinates"].append(nested_array)
    geojson_feature["geometry"] = geojson_polygon
    return geojson_feature


def create_reactangles(tuples_list: list, size: int = 0.04) -> dict:
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
    geojson_polygons = {"type": "FeatureCollection", "features": []}
    # Create a rectangle at each point on the line
    # Swap the x and y for each point because ipyleaflet swaps them for draw
    # methods
    for points_list in tuples_list:
        for point in points_list:
            upper_right_x = point[0] - (size / 2)
            upper_right_y = point[1] - (size / 2)
            upper_left_x = point[0] + (size / 2)
            upper_left_y = point[1] - (size / 2)
            lower_left_x = point[0] + (size / 2)
            lower_left_y = point[1] + (size / 2)
            lower_right_x = point[0] - (size / 2)
            lower_right_y = point[1] + (size / 2)
            # Convert each set of points to geojson (DONT swap x and y this
            # time)
            geojson_polygon = convert_corners_to_geojson(
                upper_right_x,
                upper_right_y,
                upper_left_x,
                upper_left_y,
                lower_left_x,
                lower_left_y,
                lower_right_x,
                lower_right_y)
            geojson_polygons["features"].append(geojson_polygon)
    return geojson_polygons

# ROI OVERLAP RELATED FUNCTIONS
# -------------------------------------


def create_overlap(
        filename: str,
        line: "shapely.geometry.linestring.LineString",
        start_id: int,
        end_id_list: list):
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
    # Initial number of points interpolated along the line
    num_pts = 5
    # Boolean indicates whether every single ROI generated overlaps at least
    # one other ROI
    do_all_ROI_overlap = False
    # Boolean indicates whether mean overlap was over 80% and if the number
    # points is greater than 2
    is_overlap_excessive = True
    df_overlap = get_overlap_dataframe(filename)
    # If df_overlap is means not a single ROI overlapped each other
    if not df_overlap.empty:
        df_all_ROIs = gpd.read_file(filename)
        do_all_ROI_overlap = check_all_ROI_overlap(df_all_ROIs, df_overlap)

        if do_all_ROI_overlap:
            if check_average_ROI_overlap(df_overlap, .35):
                # If the average overlap is over 35% decrease number of rois by
                # 1
                num_pts = adjust_num_pts(num_pts - 1)
                is_overlap_excessive = True
                # print(f"num_pts decreased to: {num_pts}")
        if not do_all_ROI_overlap:
            # If not all the rois overlap increase number of rois by 1
            num_pts = adjust_num_pts(num_pts + 1)
            # print(f"num_pts increased to: {num_pts}")
# Keep looping while not all the rois overlap and the average overlap is
# more than 80%
    while do_all_ROI_overlap is False and is_overlap_excessive:
        multipoint_list = interpolate_points(line, num_pts)
        tuples_list = convert_multipoints_to_tuples(multipoint_list)
        geojson_polygons = create_reactangles(tuples_list)
        end_id = write_to_geojson_file(
            filename,
            geojson_polygons,
            perserve_id=False,
            start_id=start_id)
        end_id_list.append(end_id)
        # print(f"\nend_id_list {end_id_list}\n")
        df_overlap = get_overlap_dataframe(filename)
        # If df_overlap is empty means not a single ROI overlapped each other
        if not df_overlap.empty:
            df_all_ROIs = gpd.read_file(filename)
            do_all_ROI_overlap = check_all_ROI_overlap(df_all_ROIs, df_overlap)
        else:
            do_all_ROI_overlap = False
        if not do_all_ROI_overlap:
            if num_pts == 1 or num_pts > 25:
                break  # means no more ROIs should be removed or added
            # This executes if not all the roi overlap so another roi needs to be added
            num_pts = adjust_num_pts(num_pts + 1)
        else:   # some ROIs overlap
            if num_pts == 1 or num_pts > 25:
                break  # means no more ROIs should be removed or added
            is_overlap_excessive = check_average_ROI_overlap(df_overlap, .35)
            if is_overlap_excessive:
                # If the average overlap is over 35% decrease number of rois by
                num_pts = adjust_num_pts(num_pts - 1)
                is_overlap_excessive = True
                # print(f"IN LOOP: num_pts decreased to: {num_pts}")
    return df_overlap


def check_average_ROI_overlap(df_overlap, percentage):
    """" Returns True if the mean overlap in the df is greater than percentage"""
    mean_overlap = df_overlap["%_overlap"].mean()
    if mean_overlap > percentage:
        return True
    return False


def adjust_num_pts(new_num_pts):
    """Rounds the number of points to a whole number 1<=x<=100"""
    new_num_pts = int(round(new_num_pts, 0))
    if new_num_pts < 1:
        new_num_pts = 1
    elif new_num_pts > 100:
        new_num_pts = 100
    return new_num_pts


def check_all_ROI_overlap(df_all_ROIs, df_overlap):
    """Compares the IDs of the ROIs in df_overlap(contains only the ids of the overlapping ROIs), to df_all_rois(contains the ids of all ROIs)
    Returns
    True: If all the IDs in df_all_ROIs are also in df_overlap
    False: If NOT all the IDs in df_all_ROIs are also in df_overlap"""
    all_ids_list = list(df_all_ROIs["id"])
    overlapping_ids = df_overlap["primary_id"]
    missing_list = list(set(all_ids_list) - set(overlapping_ids))
    if missing_list == []:
        return True
    return False
