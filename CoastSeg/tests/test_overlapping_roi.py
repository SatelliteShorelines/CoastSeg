from CoastSeg import make_overlapping_roi as overlap_roi
import geopandas as gpd
import pickle
import pytest

# Tests for overlap_roi.get_empty_overlap_df()
# -------------------------------------------------------------
# 1. test if it returns a dataframe correct columns with correct datatypes
# - ensure overlap_df is empty
# - ensure correct columns exists
# - ensure columns are correct 
def test_get_empty_overlap_df():
    test_df = overlap_roi.get_empty_overlap_df()
    assert test_df.empty

    assert test_df.columns.values[0] == 'id'
    assert test_df.columns.values[1] == 'primary_id'
    assert test_df.columns.values[2] == 'geometry'
    assert test_df.columns.values[3] == 'intersection_area'
    assert test_df.columns.values[4] == '%_overlap'

    # assert isinstance((test_df["id"].dtypes) 'int32')
    assert test_df["id"].dtypes == 'int32'
    assert test_df["primary_id"].dtypes == 'int32'


# Tests for overlap_roi.adjust_num_pts()
# -------------------------------------------------------------
# 1. test if it returns a valid int number of points (1<=x<=100)
# - ensure new_num_pts cannot be < 1 and sets new_num_pts=1
# - ensure new_num_pts cannot be > 100 and sets new_num_pts=100
# - ensure new_num_pts is a rounded int when returned
def test_adjust_num_pts():
    num_pts_less_1 = -1.3
    num_pts_greater_100 = 101.2
    num_pts_valid = 50.10
    # ensure new_num_pts cannot be < 1 and sets new_num_pts=1
    num_pts = overlap_roi.adjust_num_pts(num_pts_less_1)
    assert num_pts == 1
    # - ensure new_num_pts cannot be > 100 and sets new_num_pts=100
    num_pts = overlap_roi.adjust_num_pts(num_pts_greater_100)
    assert num_pts == 100
    # ensure new_num_pts is a rounded int when returned
    num_pts = overlap_roi.adjust_num_pts(num_pts_valid)
    assert num_pts == 50


# Tests for overlap_roi.get_linestring_list()
# -------------------------------------------------------------
# 1. test if it raises assertions for incorrect inputs
# - ensure assertionerror is raised if there are no features
# - ensure assertionerror is raised if there the features' geometry != MultiLineString or LineString
def test_get_linestring_list():
    # Read in a valid coastline
    # Load polygon from disc
    roi_coastline_pkl = './CoastSeg/tests/test_data/ca_simple_coastline.pkl'
    with open(roi_coastline_pkl, "rb") as file:
        out_roi_coastline_pkl = pickle.load(file)

    with pytest.raises(AssertionError):
        copy_roi_coastline = out_roi_coastline_pkl.copy()
        copy_roi_coastline['features'] = []
        overlap_roi.get_linestring_list(copy_roi_coastline)
    with pytest.raises(AssertionError):
        copy_roi_coastline = out_roi_coastline_pkl.copy()
        copy_roi_coastline['features'][0]['geometry']['type'] = 'Point'
        overlap_roi.get_linestring_list(copy_roi_coastline)

# 2. test if it returns a valid list linestrings
# - ensure the correct number of lines are returned for MultiLineString
# - ensure the at least 1 line is returned for LineString
def test_valid_get_linestring_list(expected_coastline_list, expected_lines_list):
    coastline_list = expected_coastline_list.copy()
    expected_lines_list = expected_lines_list.copy()
    
    for index, valid_roi_coastline in enumerate(coastline_list):
        copy_roi_coastline = valid_roi_coastline.copy()
        lines_list = overlap_roi.get_linestring_list(copy_roi_coastline)
        assert len(lines_list) >= 0
        if len(copy_roi_coastline["features"]) == 1:
            if copy_roi_coastline["features"][0]['geometry']['type'] == 'LineString':
                assert len(lines_list) == 1
            elif copy_roi_coastline["features"][0]['geometry']['type'] == 'MultiLineString':
                assert len(lines_list) >= 1
        assert lines_list == expected_lines_list[index]


# Tests for overlap_roi.read_geojson_from_file()
# -------------------------------------------------------------
# 1. test if it raises FileNotFound when the filename given does not exist
def test_read_geojson_from_file():
    with pytest.raises(FileNotFoundError):
        overlap_roi.read_geojson_from_file("badfilename.json")


# Tests for overlap_roi.interpolate_points
# -------------------------------------------------------------
# 1. test if it returns a multipoint for the island coastline
# 2. test if it returns  valid multipoints for the duck coastline
def test_interpolate_points(expected_lines_list):
    # The expected result for the island lines list is the 2nd item in the list
    for line in expected_lines_list[1]:
        multipoint_list = overlap_roi.interpolate_points(line)
        # Ensures multipoint exists
        assert len(multipoint_list)!=0


def test_interpolate_points_duck(expected_lines_list, expected_multipoint_list):
    #  Expected_multipoint_list are the  multipoints for the duck coastline
    expected_multipoint_list_copy = expected_multipoint_list.copy()
    # The expected result for the duck lines list is the 3rd item in the list
    for index,line in enumerate(expected_lines_list[2]):
        multipoint_list = overlap_roi.interpolate_points(line)
        # Ensures multipoint exists
        assert multipoint_list ==  expected_multipoint_list_copy[index]

# Tests for overlap_roi.read_geojson_from_file
# -------------------------------------------------------------
# Test if it raises assertion error when invalid geojson file is given
def test_read_geojson_from_file():
    with pytest.raises(AssertionError):
        overlap_roi.read_geojson_from_file("fake_selection.geojson")


# Tests for overlap_roi.get_geojson_polygons()
# -------------------------------------------------------------
# 1. test if it raises FileNotFound when the filename given does not exist
# def test_get_geojson_polygons():
#     assert 


# Tests for overlap_roi.check_average_ROI_overlap()
# -------------------------------------------------------------
# 1. test if it raises KeyError when columns do not exist
def test_invlaid_check_average_ROI_overlap():
    # Make an invalid geodataframe with incorrect column names
    test_df = gpd.GeoDataFrame({"id": []})
    with pytest.raises(KeyError):
        overlap_roi.check_average_ROI_overlap(test_df,.35)

    # test_df = gpd.GeoDataFrame({"id": [], '%_overlap': []})

# 2. test if returns True if the average overlap is more than 0.35
def test_greater_overlap_check_average_ROI_overlap():
    # Make geodataframe with % overlap column whose average is greater than 0.35
    test_df = gpd.GeoDataFrame({'%_overlap': [0.35,0.45,0.67,0.32,0.55]})
    result = overlap_roi.check_average_ROI_overlap(test_df, 0.35)
    assert result == True


# 3. test if returns False if the average overlap is less than 0.35
def test_less_overlap_check_average_ROI_overlap():
    # Make geodataframe with % overlap column whose average is less than 0.35
    test_df = gpd.GeoDataFrame({'%_overlap': [0.35,0.10,0.20,0.1,0.09]})
    result = overlap_roi.check_average_ROI_overlap(test_df, 0.35)
    assert result == False


# Tests for overlap_roi.check_average_ROI_overlap()
# -------------------------------------------------------------
# 1. test if returns True if df_overlap contains all the ids in df_all_ROIs
def test_true_all_roi_check_all_ROI_overlap():
    # Make an invalid geodataframe with incorrect column names
    df_all_ROIs = gpd.GeoDataFrame({"id" : [1, 2, 3]})
    df_overlap = gpd.GeoDataFrame({"primary_id" : [1,  2, 3]})
    result = overlap_roi.check_all_ROI_overlap(df_all_ROIs, df_overlap)
    assert result == True

# 2. test if returns False if df_overlap does NOT contain all the ids in df_all_ROIs
def  test_false_all_roi_check_all_ROI_overlap():
    df_all_ROIs = gpd.GeoDataFrame({"id" :  [1, 2, 3]})
    df_overlap = gpd.GeoDataFrame({"primary_id" : [1]})
    result = overlap_roi.check_all_ROI_overlap(df_all_ROIs, df_overlap)
    assert result == False


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
#   GeoJson rectangles have the first point repeated again as the last point
    nested_array = [[[upper_right_x, upper_right_y], [upper_left_x, upper_left_y], [lower_left_x, lower_left_y], [lower_right_x, lower_right_y], [upper_right_x, upper_right_y]]]
    geojson_polygon = {"type" : "Polygon", "coordinates" : nested_array}
    geojson_feature = {"type" : "Feature","properties" : {},"geometry" : geojson_polygon}
    return geojson_feature


def test_convert_corners_to_geojson():
    upper_right_y= 1.0
    upper_right_x= 2.0
    upper_left_y= 3.0
    upper_left_x= 4.0
    lower_left_y= 5.0
    lower_left_x= 6.0
    lower_right_y= 7.0
    lower_right_x= 9.0
    result=overlap_roi.convert_corners_to_geojson(upper_right_y,upper_right_x,upper_left_y,upper_left_x,lower_left_y,lower_left_x,lower_right_y,lower_right_x)
    assert isinstance(result, dict)
    nested_array = [[[upper_right_x, upper_right_y], [upper_left_x, upper_left_y], [lower_left_x, lower_left_y], [lower_right_x, lower_right_y], [upper_right_x, upper_right_y]]]
    expected_geojson_polygon = {"type" : "Polygon", "coordinates" : nested_array}
    expected_geojson_feature = {"type" : "Feature","properties" : {},"geometry" : expected_geojson_polygon}
    assert result == expected_geojson_feature

