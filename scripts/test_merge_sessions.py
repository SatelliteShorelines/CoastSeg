import os
from coastseg import merge_utils, file_utilities
from coastseg.common import (
    convert_linestrings_to_multipoints,
    stringify_datetime_columns,
    get_cross_distance_df,
)
from functools import reduce
import geopandas as gpd
from coastsat import SDS_transects
import numpy as np
from merge_sessions import main, parse_arguments
import pytest
import sys
import argparse
import os
import shutil

TEST_DATA_LOCATION = r"C:\development\doodleverse\coastseg\CoastSeg\test_data"
SAVE_LOCATION = (
    r"C:\development\doodleverse\coastseg\CoastSeg\test_data\merged_sessions"
)


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
    assert os.path.exists(dest)
    assert os.path.exists(os.path.join(dest, "extracted_shorelines_dict.json"))
    assert os.path.exists(os.path.join(dest, "extracted_shorelines_lines.geojson"))
    assert os.path.exists(os.path.join(dest, "extracted_shorelines_points.geojson"))
    assert os.path.exists(os.path.join(dest, "merged_config.geojson"))
    assert os.path.exists(os.path.join(dest, "transect_time_series.csv"))

    # read all the shoreline geojson files to get the dates
    gdfs = merge_utils.process_geojson_files(
        session_locations,
        ["extracted_shorelines_points.geojson", "extracted_shorelines.geojson"],
        merge_utils.convert_lines_to_multipoints,
        merge_utils.read_first_geojson_file,
    )
    # get all the dates before merging
    total_set = set()
    for gdf in gdfs:
        total_set.update(gdf.date)
    # get the merged shorelines
    merged_shorelines = reduce(merge_utils.merge_and_average, gdfs)
    merged_set = set(merged_shorelines["date"])
    assert total_set == merged_set


def test_main_with_same_rois():
    # Create a Namespace object with your arguments
    source_dest = os.path.join(TEST_DATA_LOCATION, "test_case1_same_rois")
    session_locations = [
        os.path.join(source_dest, session) for session in os.listdir(source_dest)
    ]
    if not all([os.path.exists(session) for session in session_locations]):
        raise Exception("Test data not found. Please download the test data")

    merged_session_name = "merged_session"
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
    assert os.path.exists(dest)
    assert os.path.exists(os.path.join(dest, "extracted_shorelines_dict.json"))
    assert os.path.exists(os.path.join(dest, "extracted_shorelines_lines.geojson"))
    assert os.path.exists(os.path.join(dest, "extracted_shorelines_points.geojson"))
    assert os.path.exists(os.path.join(dest, "merged_config.geojson"))
    assert os.path.exists(os.path.join(dest, "transect_time_series.csv"))

    # read all the shoreline geojson files to get the dates
    gdfs = merge_utils.process_geojson_files(
        session_locations,
        ["extracted_shorelines_points.geojson", "extracted_shorelines.geojson"],
        merge_utils.convert_lines_to_multipoints,
        merge_utils.read_first_geojson_file,
    )
    # get all the dates before merging
    total_set = set()
    for gdf in gdfs:
        total_set.update(gdf.date)
    # get the merged shorelines
    merged_shorelines = reduce(merge_utils.merge_and_average, gdfs)
    merged_set = set(merged_shorelines["date"])
    assert total_set == merged_set


def test_main_with_different_rois():
    # Create a Namespace object with your arguments
    source_dest = os.path.join(TEST_DATA_LOCATION, "test_case2_different_rois")
    session_locations = [
        os.path.join(source_dest, session) for session in os.listdir(source_dest)
    ]
    if not all([os.path.exists(session) for session in session_locations]):
        raise Exception("Test data not found. Please download the test data")

    merged_session_name = "merged_session"
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
    assert os.path.exists(dest)
    assert os.path.exists(os.path.join(dest, "extracted_shorelines_dict.json"))
    assert os.path.exists(os.path.join(dest, "extracted_shorelines_lines.geojson"))
    assert os.path.exists(os.path.join(dest, "extracted_shorelines_points.geojson"))
    assert os.path.exists(os.path.join(dest, "merged_config.geojson"))
    assert os.path.exists(os.path.join(dest, "transect_time_series.csv"))

    # read all the shoreline geojson files to get the dates
    gdfs = merge_utils.process_geojson_files(
        session_locations,
        ["extracted_shorelines_points.geojson", "extracted_shorelines.geojson"],
        merge_utils.convert_lines_to_multipoints,
        merge_utils.read_first_geojson_file,
    )
    # get all the dates before merging
    total_set = set()
    for gdf in gdfs:
        total_set.update(gdf.date)
    # get the merged shorelines
    merged_shorelines = reduce(merge_utils.merge_and_average, gdfs)
    merged_set = set(merged_shorelines["date"])
    assert total_set == merged_set


def test_main_with_overlapping_dates():
    # Create a Namespace object with your arguments
    source_dest = os.path.join(TEST_DATA_LOCATION, "test_case3_rois_overlapping_dates")
    session_locations = [
        os.path.join(source_dest, session) for session in os.listdir(source_dest)
    ]
    if not all([os.path.exists(session) for session in session_locations]):
        raise Exception("Test data not found. Please download the test data")

    merged_session_name = "merged_session"
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
    assert os.path.exists(dest)
    assert os.path.exists(os.path.join(dest, "extracted_shorelines_dict.json"))
    assert os.path.exists(os.path.join(dest, "extracted_shorelines_lines.geojson"))
    assert os.path.exists(os.path.join(dest, "extracted_shorelines_points.geojson"))
    assert os.path.exists(os.path.join(dest, "merged_config.geojson"))
    assert os.path.exists(os.path.join(dest, "transect_time_series.csv"))

    # read all the shoreline geojson files to get the dates
    gdfs = merge_utils.process_geojson_files(
        session_locations,
        ["extracted_shorelines_points.geojson", "extracted_shorelines.geojson"],
        merge_utils.convert_lines_to_multipoints,
        merge_utils.read_first_geojson_file,
    )
    # get all the dates before merging
    total_set = set()
    for gdf in gdfs:
        total_set.update(gdf.date)
    # get the merged shorelines
    merged_shorelines = reduce(merge_utils.merge_and_average, gdfs)
    merged_set = set(merged_shorelines["date"])
    assert total_set == merged_set
