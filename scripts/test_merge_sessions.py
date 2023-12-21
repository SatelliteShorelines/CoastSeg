import os
import pandas as pd
from coastseg import merge_utils, file_utilities
from functools import reduce
import geopandas as gpd
from merge_sessions import main, parse_arguments
import pytest
import sys
import argparse
import shutil

TEST_DATA_LOCATION = r"C:\development\doodleverse\coastseg\CoastSeg\test_data"
SAVE_LOCATION = os.path.join(TEST_DATA_LOCATION, "merged_sessions")

# helper functions
# --------------
def get_unique_dates_from_geojson_files(gdfs:list[gpd.GeoDataFrame]):
    unique_dates = set()
    for gdf in gdfs:
        unique_dates.update(gdf.date)
    return unique_dates

def check_geojson_files(gdf1, gdf2, columns: list[str]):
    """
    Check if the specified columns exist in both GeoDataFrame objects and if their values match.

    Args:
        gdf1 (GeoDataFrame): The first GeoDataFrame object.
        gdf2 (GeoDataFrame): The second GeoDataFrame object.
        columns (list[str]): A list of column names to check.

    Raises:
        AssertionError: If any of the specified columns are missing in either GeoDataFrame or if their values do not match.
    """
    if isinstance(columns, str):
        columns = [columns]
    for column in columns:
        assert (column in gdf1.columns) and (column in gdf2.columns), f"{column} column missing in one of the files"
        assert set(gdf1[column]) == set(gdf2[column]), f"{column} do not match"

def assert_all_files_exist(dest:str):
    """
    Check if all the required files exist in the specified destination directory.

    Parameters:
    dest (str): The destination directory path.

    Raises:
    AssertionError: If any of the required files does not exist.
    """
    assert os.path.exists(dest)
    assert os.path.exists(os.path.join(dest, "extracted_shorelines_dict.json"))
    assert os.path.exists(os.path.join(dest, "extracted_shorelines_lines.geojson"))
    assert os.path.exists(os.path.join(dest, "extracted_shorelines_points.geojson"))
    assert os.path.exists(os.path.join(dest, "merged_config.geojson"))
    assert os.path.exists(os.path.join(dest, "transect_time_series.csv"))

def verify_merged_session(dest:str):
    """
    Verify the consistency of a merged session.

    Args:
        dest (str): The destination directory of the merged session.

    Raises:
        AssertionError: If any of the verification checks fail.

    Returns:
        None
    """
    # 1. read in extracted_shorelines_points.geojson from merged session
    shoreline_points_gdf = merge_utils.read_first_geojson_file(dest, "extracted_shorelines_points.geojson")
    # 2. read in extracted_shorelines_lines.geojson from merged session
    shoreline_lines_gdf = merge_utils.read_first_geojson_file(dest, "extracted_shorelines_lines.geojson")
    # 3.verify that the 'date' and 'satname' columns are present and consistent in the geojson files
    check_geojson_files(shoreline_points_gdf, shoreline_lines_gdf, ['date', 'satname'])
    # 4. read in the extracted_shoreline_dict.json from merged session
    extracted_shorelines_dict=file_utilities.load_data_from_json(os.path.join(dest, "extracted_shorelines_dict.json"))
    # 5. Check if all the dates & satellites in the geojson files are present in the dictionary
    columns = ['dates', 'satname']
    for column in columns:
        if column == 'dates':
            assert shoreline_points_gdf['date'].isin(extracted_shorelines_dict.get(column)).all()
        else:
            assert shoreline_points_gdf[column].isin(extracted_shorelines_dict.get(column)).all()
    # 6. Read in the transects_time_series.csv from merged session
    transect_time_series= pd.read_csv(os.path.join(dest, "transect_time_series.csv"))
    # 8. Check if all the dates in the geojson files are present in the csv file
    assert shoreline_points_gdf['date'].isin(transect_time_series['dates']).all()
    # 9. Check if the length of dates in the dictionary is the same as the number of dates in the csv file
    assert len(extracted_shorelines_dict.get('dates')) == len(transect_time_series['dates'])
    # 10. Check if the length of all the values in the dictionary is the same as the number of dates in the csv file
    for key in extracted_shorelines_dict.keys():
        assert len(extracted_shorelines_dict.get(key)) == len(transect_time_series['dates'])
        
def validate_dates(session_locations:list[str], dest:str):
    """
    Validates that the dates before and after merging shoreline geojson files are the same.

    Args:
        session_locations (list[str]): List of file paths to the original sessions.
        dest (str): File path to the new merged session.

    Raises:
        AssertionError: If the dates before and after merging are not the same.
    """
    # get the dates from the shoreline geojson files located in the original sessions 
    gdfs = merge_utils.process_geojson_files(
        session_locations,
        ["extracted_shorelines_points.geojson", "extracted_shorelines.geojson"],
        merge_utils.convert_lines_to_multipoints,
        merge_utils.read_first_geojson_file,
    )
    # get all the dates before merging
    unique_dates = get_unique_dates_from_geojson_files(gdfs)
    # apply the merge and average functions that's applied in merge_sessions.py
    merged_shorelines = reduce(merge_utils.merge_and_average, gdfs)
    merged_dates = set(merged_shorelines["date"])
    # check that the dates are the same before and after merging
    assert unique_dates == merged_dates
    
    # read the merged geojson file and check if all the dates are present
    # get the dates from the shoreline geojson files located in the new merged session
    merged_gdfs = merge_utils.process_geojson_files(
        dest,
        ["extracted_shorelines_points.geojson", "extracted_shorelines.geojson"],
        merge_utils.convert_lines_to_multipoints,
        merge_utils.read_first_geojson_file,
    )
    # get all the unique dates after merging
    result_unique_dates = get_unique_dates_from_geojson_files(merged_gdfs)
    # check that the dates are the same before and after merging
    assert unique_dates == result_unique_dates
    assert merged_dates == result_unique_dates

def clear_directory(directory):
    """
    Deletes all the contents of the specified directory.

    :param directory: The path to the directory to be cleared.
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove files or links
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directories
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# ----------------------------

def test_with_all_arguments(monkeypatch):
    # Test case 1: Required arguments provided
    test_args = [
        "program_name",
        "-i",
        "session1",
        "session2",
        "-n",
        "merged_session",
        "-s",
        "save_location",
        "-ad",
        "30",
        "-mp",
        "5",
        "-ms",
        "20",
        "-mr",
        "20",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    args = parse_arguments()
    assert args.session_locations == ["session1", "session2"]
    assert args.merged_session_name == "merged_session"
    assert args.save_location == "save_location"
    assert args.along_dist == 30
    assert args.min_points == 5
    assert args.max_std == 20
    assert args.max_range == 20
    assert args.min_chainage == -100
    assert args.multiple_inter == "auto"
    assert args.prc_multiple == 0.1


def test_with_mandatory_arguments_only(monkeypatch):
    test_args = ["program_name", "-i", "session1", "session2", "-n", "merged_session"]
    monkeypatch.setattr(sys, "argv", test_args)
    args = parse_arguments()
    assert args.session_locations == ["session1", "session2"]
    assert args.merged_session_name == "merged_session"
    # Check if defaults are correctly set for optional arguments
    assert args.save_location == os.path.join(os.getcwd(), "merged_sessions")
    assert args.along_dist == 25
    assert args.min_points == 3
    assert args.max_std == 15
    assert args.max_range == 30
    assert args.min_chainage == -100
    assert args.multiple_inter == "auto"
    assert args.prc_multiple == 0.1


def test_main_with_overlapping():
    # Create a Namespace object with your arguments
    source_dest = os.path.join(TEST_DATA_LOCATION, "test_case4_overlapping")
    session_locations = [
        os.path.join(source_dest, session) for session in os.listdir(source_dest)
    ]
    if not all([os.path.exists(session) for session in session_locations]):
        raise Exception("Test data not found. Please download the test data")

    merged_session_name = "merged_session"
    merged_session_name = "merged_session_test_case4"
    dest = os.path.join(SAVE_LOCATION, merged_session_name)
    if os.path.exists(dest):
        clear_directory(dest)

    mock_args = argparse.Namespace(
        session_locations=session_locations,
        save_location=SAVE_LOCATION,
        merged_session_name=merged_session_name,
        along_dist=25,
        min_points=3,
        max_std=15,
        max_range=30,
        min_chainage=-100,
        multiple_inter="auto",
        prc_multiple=0.1,
    )

    main(mock_args)
    # after the merge sessions script is run, check that the merged session location exists
    assert_all_files_exist(dest)
    # check that the dates before and after merging are the same
    validate_dates(session_locations, dest)
    # check that the dates before and after merging are the same
    verify_merged_session(dest)


def test_main_with_same_rois():
    # Create a Namespace object with your arguments
    source_dest = os.path.join(TEST_DATA_LOCATION, "test_case1_same_rois")
    session_locations = [
        os.path.join(source_dest, session) for session in os.listdir(source_dest)
    ]
    if not all([os.path.exists(session) for session in session_locations]):
        raise Exception("Test data not found. Please download the test data")

    merged_session_name = "merged_session"
    merged_session_name = "merged_session_test_case1"
    dest = os.path.join(SAVE_LOCATION, merged_session_name)
    if os.path.exists(dest):
        clear_directory(dest)

    mock_args = argparse.Namespace(
        session_locations=session_locations,
        save_location=SAVE_LOCATION,
        merged_session_name=merged_session_name,
        along_dist=25,
        min_points=3,
        max_std=15,
        max_range=30,
        min_chainage=-100,
        multiple_inter="auto",
        prc_multiple=0.1,
    )

    main(mock_args)
    # after the merge sessions script is run, check that the merged session location exists
    assert_all_files_exist(dest)
    # check that the dates before and after merging are the same
    validate_dates(session_locations, dest)
    # check that the dates before and after merging are the same
    verify_merged_session(dest)


def test_main_with_different_rois():
    # Create a Namespace object with your arguments
    source_dest = os.path.join(TEST_DATA_LOCATION, "test_case2_different_rois")
    session_locations = [
        os.path.join(source_dest, session) for session in os.listdir(source_dest)
    ]
    if not all([os.path.exists(session) for session in session_locations]):
        raise Exception("Test data not found. Please download the test data")

    merged_session_name = "merged_session"
    merged_session_name = "merged_session_test_case2"
    dest = os.path.join(SAVE_LOCATION, merged_session_name)
    if os.path.exists(dest):
        clear_directory(dest)

    mock_args = argparse.Namespace(
        session_locations=session_locations,
        save_location=SAVE_LOCATION,
        merged_session_name=merged_session_name,
        along_dist=25,
        min_points=3,
        max_std=15,
        max_range=30,
        min_chainage=-100,
        multiple_inter="auto",
        prc_multiple=0.1,
    )

    main(mock_args)
    # after the merge sessions script is run, check that the merged session location exists
    assert_all_files_exist(dest)
    # check that the dates before and after merging are the same
    validate_dates(session_locations, dest)
    # check that the dates before and after merging are the same
    verify_merged_session(dest)


def test_main_with_overlapping_dates():
    # Create a Namespace object with your arguments
    source_dest = os.path.join(TEST_DATA_LOCATION, "test_case3_rois_overlapping_dates")
    session_locations = [
        os.path.join(source_dest, session) for session in os.listdir(source_dest)
    ]
    if not all([os.path.exists(session) for session in session_locations]):
        raise Exception("Test data not found. Please download the test data")

    merged_session_name = "merged_session"
    merged_session_name = "merged_session_test_case3"
    dest = os.path.join(SAVE_LOCATION, merged_session_name)
    if os.path.exists(dest):
        clear_directory(dest)

    mock_args = argparse.Namespace(
        session_locations=session_locations,
        save_location=SAVE_LOCATION,
        merged_session_name=merged_session_name,
        along_dist=25,
        min_points=3,
        max_std=15,
        max_range=30,
        min_chainage=-100,
        multiple_inter="auto",
        prc_multiple=0.1,
    )
    # call the main function in the merge_sessions.py script
    main(mock_args)
    # after the merge sessions script is run, check that the merged session location exists
    assert_all_files_exist(dest)
    # check that the dates before and after merging are the same
    validate_dates(session_locations, dest)
    # check that the dates before and after merging are the same
    verify_merged_session(dest)
