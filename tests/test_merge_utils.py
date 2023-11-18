# Standard library imports
from collections import defaultdict

# Related third party imports
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, Polygon
from coastseg import merge_utils
from functools import reduce

# Local application/library specific imports
from coastseg.merge_utils import (
    calculate_overlap,
    convert_lines_to_multipoints,
    merge_geometries,
)


@pytest.fixture
def gdf_empty():
    return gpd.gpd.GeoDataFrame()


@pytest.fixture
def gdf_with_crs():
    # Create an empty GeoSeries with the specified CRS
    geoseries = gpd.GeoSeries(crs="EPSG:4326")
    # Create the gpd.GeoDataFrame using the empty GeoSeries
    return gpd.gpd.GeoDataFrame(geometry=geoseries)


@pytest.fixture
def gdf_overlap():
    data = {
        "geometry": [
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
        ]
    }
    return gpd.gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def gdf_no_overlap():
    data = {
        "geometry": [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
        ]
    }
    return gpd.gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:4326")


@pytest.fixture
def empty_extracted_gdf():
    data = {
        "date": [],
        "geometry": [],
        "geoaccuracy": [],
        "satname": [],
        "cloud_cover": [],
    }
    return gpd.gpd.GeoDataFrame(data, crs="epsg:4326")


@pytest.fixture
def extracted_gdf1():
    data = {
        "date": [
            pd.Timestamp("2018-12-30 18:22:25"),
            pd.Timestamp("2019-1-28 05:12:28"),
            pd.Timestamp("2020-5-23 19:24:27"),
        ],
        "geometry": [
            MultiPoint([(-117.45892, 33.28226), (-118.45892, 35.28226)]),
            MultiPoint([(-117.45881, 33.28239), (-120.45892, 40.28226)]),
            MultiPoint([(-117.45875, 33.28242)]),
        ],
        "geoaccuracy": [
            5.088,
            5.802,
            6.596,
        ],
        "satname": ["L8", "L8", "L8"],
        "cloud_cover": [0.0, 0.23, 0.263967],
    }
    return gpd.gpd.GeoDataFrame(data, crs="epsg:4326")


@pytest.fixture
def extracted_gdf2():
    # this is the gdf shares pd.Timestamp('2018-12-30 18:22:25') and  pd.Timestamp('2020-5-23 19:24:27') with extracted_gdf1
    data = {
        "date": [
            pd.Timestamp("2018-12-30 18:22:25"),
            pd.Timestamp("2020-1-28 05:12:28"),
            pd.Timestamp("2020-5-23 19:24:27"),
        ],
        "geometry": [
            MultiPoint([(-117.44480, 33.26540)]),
            MultiPoint([(-117.45899, 33.28226)]),
            MultiPoint([(-117.45896, 33.28226)]),
        ],
        "geoaccuracy": [
            5.088,
            5.802,
            6.596,
        ],
        "satname": ["L8", "L8", "L8"],
        "cloud_cover": [0.0, 0.0, 0.263967],
    }
    return gpd.gpd.GeoDataFrame(data, crs="epsg:4326")


@pytest.fixture
def extracted_gdf3():
    # this is the gdf shares pd.Timestamp('2018-12-30 18:22:25') and  pd.Timestamp('2020-5-23 19:24:27') with extracted_gdf1
    data = {
        "date": [
            pd.Timestamp("2015-12-30 18:22:25"),
            pd.Timestamp("2019-1-28 05:12:28"),
            pd.Timestamp("2020-5-23 19:24:27"),
        ],
        "geometry": [
            MultiPoint([(-117.45896, 33.28226)]),
            MultiPoint([(-117.45894, 33.28226)]),
            MultiPoint([(-117.45891, 33.28232)]),
        ],
        "geoaccuracy": [
            5.088,
            5.802,
            6.596,
        ],
        "satname": ["L9", "L9", "L8"],
        "cloud_cover": [0.0, 0.1, 0.263967],
    }
    return gpd.gpd.GeoDataFrame(data, crs="epsg:4326")


def test_empty_gdf(gdf_empty):
    result = calculate_overlap(gdf_empty)
    assert result.empty


def test_empty_gdf_with_crs(gdf_with_crs):
    result = calculate_overlap(gdf_with_crs)
    assert result.empty
    assert result.crs == gdf_with_crs.crs


def test_overlap(gdf_overlap):
    result = calculate_overlap(gdf_overlap)
    assert not result.empty
    assert result.crs == gdf_overlap.crs
    assert len(result) == 1
    assert result.iloc[0].geometry.equals(Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]))


def test_no_overlap(gdf_no_overlap):
    result = calculate_overlap(gdf_no_overlap)
    assert result.empty


def test_convert_multipoints_to_linestrings_with_linestrings():
    """
    Test function to check if the convert_multipoints_to_linestrings function
    correctly converts a gpd.GeoDataFrame with LineString geometries to the same
    gpd.GeoDataFrame.
    """
    # Create a gpd.GeoDataFrame with LineString geometries
    data = {"geometry": [LineString([(0, 0), (1, 1)]), LineString([(2, 2), (3, 3)])]}
    gdf = gpd.gpd.GeoDataFrame(data)
    result = merge_utils.convert_multipoints_to_linestrings(gdf)
    assert result.equals(gdf)


def test_convert_multipoints_to_linestrings_with_multipoints():
    """
    Test function to check if the function `convert_multipoints_to_linestrings` correctly converts
    MultiPoint geometries to LineString geometries.
    """
    # Create a gpd.GeoDataFrame with MultiPoint geometries
    data = {"geometry": [MultiPoint([(0, 0), (1, 1)]), MultiPoint([(2, 2), (3, 3)])]}
    gdf = gpd.gpd.GeoDataFrame(data)
    result = merge_utils.convert_multipoints_to_linestrings(gdf)
    expected = gpd.gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 0), (1, 1)]), LineString([(2, 2), (3, 3)])]}
    )
    assert result.equals(expected)


def test_convert_multipoints_to_linestrings_with_mixed_geometries():
    """
    Test function to check if the function `convert_multipoints_to_linestrings` correctly
    converts a gpd.GeoDataFrame with mixed geometries (MultiPoint and LineString) to a gpd.GeoDataFrame
    with only LineString geometries.
    """
    # Create a gpd.GeoDataFrame with mixed geometries
    data = {
        "geometry": [
            MultiPoint([(0, 0), (1, 1)]),
            LineString([(2, 2), (3, 3)]),
            MultiPoint([(4, 4), (5, 5)]),
        ]
    }
    gdf = gpd.gpd.GeoDataFrame(data)
    result = merge_utils.convert_multipoints_to_linestrings(gdf)
    expected = gpd.gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(0, 0), (1, 1)]),
                LineString([(2, 2), (3, 3)]),
                LineString([(4, 4), (5, 5)]),
            ]
        }
    )
    assert result.equals(expected)


def test_dataframe_to_dict():
    """
    Test function to check if the `dataframe_to_dict` function correctly converts a DataFrame to a dictionary
    with specific mapping between dictionary keys and DataFrame columns.
    """
    # create a list of geometries
    geometries = [
        MultiPoint([(0, 0), (1, 1)]),
        MultiPoint([(2, 2), (3, 3)]),
        MultiPoint([(4, 4), (5, 5)]),
    ]

    # create a dictionary with the other columns
    data = {
        "geoaccuracy": [1, 2, 3],
        "cloud_cover": [0.1, 0.2, 0.3],
        "satname": ["L8", "L8", "L8"],
        "date": [
            pd.Timestamp("2018-12-30 18:22:25"),
            pd.Timestamp("2018-1-30 19:22:25"),
            pd.Timestamp("2022-01-03 19:22:25"),
        ],
        "geometry": geometries,
    }

    # create a gpd.GeoDataFrame from the dictionary
    df = gpd.gpd.GeoDataFrame(data, geometry="geometry", crs="epsg:4326")
    df.set_crs("epsg:4326", inplace=True)

    # Define the key mapping
    key_map = {
        "shorelines": "geometry",
        "dates": "date",
        "satname": "satname",
        "cloud_cover": "cloud_cover",
        "geoaccuracy": "geoaccuracy",
    }
    # Convert the DataFrame to a dictionary using the `dataframe_to_dict` function
    result = merge_utils.dataframe_to_dict(df, key_map)

    # Define the expected dictionary
    expected = {
        "geoaccuracy": [1, 2, 3],
        "cloud_cover": [0.1, 0.2, 0.3],
        "satname": ["L8", "L8", "L8"],
        "dates": [
            "2018-12-30 18:22:25",
            "2018-01-30 19:22:25",
            "2022-01-03 19:22:25",
        ],
        "shorelines": [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([[2.0, 2.0], [3.0, 3.0]]),
            np.array([[4.0, 4.0], [5.0, 5.0]]),
        ],
    }
    # Check if the resulting dictionary is equal to the expected dictionary
    assert result["geoaccuracy"] == expected["geoaccuracy"]
    assert result["cloud_cover"] == expected["cloud_cover"]
    assert result["satname"] == expected["satname"]
    assert result["dates"] == expected["dates"]
    assert all(
        np.array_equal(a, b)
        for a, b in zip(result["shorelines"], expected["shorelines"])
    )


def test_convert_lines_to_multipoints_with_linestrings():
    """
    Test function to check if the convert_lines_to_multipoints function
    correctly converts a gpd.GeoDataFrame with LineString geometries to a new
    gpd.GeoDataFrame with MultiPoint geometries.
    """
    # Create a gpd.GeoDataFrame with LineString geometries
    data = {"geometry": [LineString([(0, 0), (1, 1)]), LineString([(2, 2), (3, 3)])]}
    gdf = gpd.GeoDataFrame(data)
    result = convert_lines_to_multipoints(gdf)
    expected = gpd.GeoDataFrame(
        {"geometry": [MultiPoint([(0, 0), (1, 1)]), MultiPoint([(2, 2), (3, 3)])]}
    )
    assert result.equals(expected)


def test_convert_lines_to_multipoints_with_multilinestrings():
    """
    Test function to check if the convert_lines_to_multipoints function
    correctly converts a gpd.GeoDataFrame with MultiLineString geometries to a new
    gpd.GeoDataFrame with MultiPoint geometries.
    """
    # Create a gpd.GeoDataFrame with MultiLineString geometries
    data = {
        "geometry": [
            MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
            MultiLineString([[(4, 4), (5, 5)], [(6, 6), (7, 7)]]),
        ]
    }
    gdf = gpd.GeoDataFrame(data)
    result = convert_lines_to_multipoints(gdf)
    expected = gpd.GeoDataFrame(
        {
            "geometry": [
                MultiPoint([(0, 0), (1, 1), (2, 2), (3, 3)]),
                MultiPoint([(4, 4), (5, 5), (6, 6), (7, 7)]),
            ]
        }
    )
    assert result.equals(expected)


def test_convert_lines_to_multipoints_with_mixed_geometries():
    """
    Test function to check if the convert_lines_to_multipoints function
    correctly converts a gpd.GeoDataFrame with mixed geometries (LineString and MultiLineString)
    to a new gpd.GeoDataFrame with MultiPoint geometries.
    """
    # Create a gpd.GeoDataFrame with mixed geometries
    data = {
        "geometry": [
            LineString([(0, 0), (1, 1)]),
            MultiLineString([[(2, 2), (3, 3)], [(4, 4), (5, 5)]]),
        ]
    }
    gdf = gpd.GeoDataFrame(data)
    result = convert_lines_to_multipoints(gdf)
    expected = gpd.GeoDataFrame(
        {
            "geometry": [
                MultiPoint([(0, 0), (1, 1)]),
                MultiPoint([(2, 2), (3, 3), (4, 4), (5, 5)]),
            ]
        }
    )
    assert result.equals(expected)


def test_convert_lines_to_multipoints_with_points():
    """
    Test function to check if the convert_lines_to_multipoints function
    correctly handles a gpd.GeoDataFrame with Point geometries.
    """
    # Create a gpd.GeoDataFrame with Point geometries
    data = {"geometry": [Point(0, 0), Point(1, 1)]}
    gdf = gpd.GeoDataFrame(data, geometry="geometry")
    result = convert_lines_to_multipoints(gdf)
    expected = gpd.GeoDataFrame(
        {"geometry": [MultiPoint([(0, 0)]), MultiPoint([(1, 1)])]}
    )
    assert result.equals(expected)


def test_merge_geometries_with_default_columns_and_operation():
    """
    Test function to check if the merge_geometries function correctly merges geometries
    with the same date and satname using the default columns and operation.
    """
    # Create a gpd.GeoDataFrame with two rows of Point geometries with the same date and satname
    data = {
        "date": [
            "2022-01-01",
        ],
        "satname": [
            "sat1",
        ],
        "geometry_df1": [
            Point(0, 0),
        ],
        "geometry": [Point(1, 1)],
    }
    gdf = gpd.GeoDataFrame(data)

    # Merge the geometries using the merge_geometries function
    result = merge_geometries(gdf)

    # Define the expected gpd.GeoDataFrame with one row of a MultiPoint geometry
    expected = gpd.GeoDataFrame(
        {
            "date": ["2022-01-01"],
            "satname": ["sat1"],
            "geometry": [
                MultiPoint([(0, 0), (1, 1)]),
            ],
        }
    )

    # Check if the resulting gpd.GeoDataFrame is equal to the expected gpd.GeoDataFrame
    assert result.equals(expected)


def test_merge_geometries_with_standard_input():
    """
    Test function to check if the merge_geometries function correctly merges geometries
    with the same date and satname using the default columns and operation.
    """
    # Create a gpd.GeoDataFrame with two rows of Point geometries with the same date and satname
    data = {
        "date": [
            "2022-01-01",
        ],
        "satname": [
            "sat1",
        ],
        "geometry_df1": [
            Point(0, 0),
        ],
        "geometry_df2": [
            MultiPoint(
                [
                    (1, 1),
                ]
            )
        ],
    }
    gdf = gpd.GeoDataFrame(data)

    # Merge the geometries using the merge_geometries function
    result = merge_geometries(gdf)

    # Define the expected gpd.GeoDataFrame with one row of a MultiPoint geometry
    expected = gpd.GeoDataFrame(
        {
            "date": ["2022-01-01"],
            "satname": ["sat1"],
            "geometry": [
                MultiPoint([(0, 0), (1, 1)]),
            ],
        }
    )

    # Check if the resulting gpd.GeoDataFrame is equal to the expected gpd.GeoDataFrame
    assert result.equals(expected)


def test_merge_and_average(extracted_gdf1, extracted_gdf2, extracted_gdf3):
    # List of GeoDataFrames
    gdfs = [extracted_gdf1, extracted_gdf2, extracted_gdf3]

    # Perform a full outer join and average the numeric columns across all GeoDataFrames
    result = reduce(merge_utils.merge_and_average, gdfs)

    result.sort_values(by="date", inplace=True)
    result.reset_index(drop=True, inplace=True)

    assert len(result) == 6
    assert (
        result[["date", "satname"]].duplicated().sum() == 0
    ), "The combination of 'date' and 'satname' is not unique."
    # assert np.all(result['cloud_cover'] == [0.0, 0.115, 0.263967, 0.0, 0.0, 0.1])

    result
