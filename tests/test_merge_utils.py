# Standard library imports
from collections import defaultdict
import os

# Related third party imports
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, Polygon
from coastseg import merge_utils
from functools import reduce
import tempfile

# Local application/library specific imports
from coastseg.merge_utils import (
    calculate_overlap,
    convert_lines_to_multipoints,
    merge_geometries,
)


@pytest.fixture(scope="session")
def overlapping_roi_gdf_fixture():
    data = {
        "id": ["gac1", "gac6"],
        "type": ["roi", "roi"],
        "dummy": ["dummy1", "dummy2"],
        "geometry": [
            Polygon(
                [
                    (-121.89294822609095, 36.87805982149002),
                    (-121.892296987737, 36.923124285162714),
                    (-121.83616982432659, 36.922587577051516),
                    (-121.83685404313914, 36.87752398639186),
                    (-121.89294822609095, 36.87805982149002),
                ]
            ),
            Polygon(
                [
                    (-121.89236580918869, 36.91836671978996),
                    (-121.89178223277345, 36.95867333604402),
                    (-121.8415571844796, 36.95819392646526),
                    (-121.8421671948108, 36.917888007480336),
                    (-121.89236580918869, 36.91836671978996),
                ]
            ),
        ],
    }

    overlapping_roi_gdf = gpd.GeoDataFrame(data, geometry="geometry")
    return overlapping_roi_gdf


@pytest.fixture(scope="session")
def temp_geojson_file(overlapping_roi_gdf_fixture):
    # Create a temporary file and immediately close it to avoid lock issues on Windows
    tmpfile = tempfile.NamedTemporaryFile(suffix=".geojson", mode="w+", delete=False)
    tmpfile.close()

    # Now open the file again to write the data
    overlapping_roi_gdf_fixture.to_file(tmpfile.name, driver="GeoJSON")

    yield tmpfile.name  # This will provide the file path to the test function
    # Teardown code: delete the temporary file after the test session
    os.remove(tmpfile.name)


@pytest.fixture
def gdf_empty():
    data = {
        "date": [],
        "geometry": [],
        "geoaccuracy": [],
        "satname": [],
        "cloud_cover": [],
    }
    empty_gdf = gpd.GeoDataFrame(data, crs="epsg:4326")
    return empty_gdf


# @pytest.fixture
# def gdf_empty():
#     return gpd.gpd.GeoDataFrame()


@pytest.fixture
def gdf_with_crs():
    # Create an empty GeoSeries with the specified CRS
    geoseries = gpd.GeoSeries(crs="EPSG:4326")
    # Create the gpd.GeoDataFrame using the empty GeoSeries
    return gpd.gpd.GeoDataFrame(geometry=geoseries)


# gdf_overlap fixture replaced with shared overlapping_polygons_gdf from conftest.py
# gdf_no_overlap fixture replaced with shared non_overlapping_polygons_gdf from conftest.py


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


def test_calculate_overlap_empty_gdf(gdf_empty):
    result = calculate_overlap(gdf_empty)
    assert result.empty


def test_calculate_overlap_empty_gdf_with_crs(gdf_with_crs):
    result = calculate_overlap(gdf_with_crs)
    assert result.empty
    assert result.crs == gdf_with_crs.crs


def test_calculate_overlap(overlapping_polygons_gdf):
    result = calculate_overlap(overlapping_polygons_gdf)
    assert not result.empty
    assert result.crs == overlapping_polygons_gdf.crs
    assert len(result) == 1
    assert result.iloc[0].geometry.equals(Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]))


def test_calculate_overlap_no_overlap(non_overlapping_polygons_gdf):
    result = calculate_overlap(non_overlapping_polygons_gdf)
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
    # Concatenate the 'geoaccuracy' values from all GeoDataFrames
    expected_geoaccuracy = pd.concat([gdf["geoaccuracy"] for gdf in gdfs])

    # Check if the values in expected_geoaccuracy are present in the 'geoaccuracy' column of the result DataFrame
    assert expected_geoaccuracy.isin(result["geoaccuracy"]).all()


def test_merge_and_average_empty_gdf_and_non_empty(
    gdf_empty,
    extracted_gdf2,
):
    # List of GeoDataFrames
    gdfs = [gdf_empty, extracted_gdf2]

    # Perform a full outer join and average the numeric columns across all GeoDataFrames
    result = reduce(merge_utils.merge_and_average, gdfs)
    # this merge should not have any common dates and should contain 1+2+3 = 6 rows
    result.sort_values(by="date", inplace=True)
    result.reset_index(drop=True, inplace=True)

    assert len(result) == len(extracted_gdf2)
    assert result["date"].equals(extracted_gdf2["date"])
    assert result["satname"].equals(extracted_gdf2["satname"])
    assert result["cloud_cover"].equals(extracted_gdf2["cloud_cover"])
    assert result["geoaccuracy"].equals(extracted_gdf2["geoaccuracy"])
    # convert the result geometry to multipoint
    new_result = convert_lines_to_multipoints(result)
    assert new_result["geometry"].equals(extracted_gdf2["geometry"])


def test_merge_and_average_different_sized_gdfs(extracted_gdf1, extracted_gdf3):
    # make the geodataframes different sizes
    # Create a new GeoDataFrame with just the top row
    extracted_gdf1_1_row = extracted_gdf1.head(1)

    # this is the gdf shares pd.Timestamp('2018-12-30 18:22:25') and  pd.Timestamp('2020-5-23 19:24:27') with extracted_gdf1
    data = {
        "date": [
            pd.Timestamp("2015-12-30 18:22:25"),
            pd.Timestamp("2020-1-28 05:12:28"),
        ],
        "geometry": [
            MultiPoint([(-117.44480, 33.26540)]),
            MultiPoint([(-117.45899, 33.28226)]),
        ],
        "geoaccuracy": [
            5.088,
            6.02,
        ],
        "satname": [
            "L8",
            "L8",
        ],
        "cloud_cover": [0.0, 0.263967],
    }
    extracted_gdf2_2_row = gpd.GeoDataFrame(data, crs="epsg:4326")

    # List of GeoDataFrames
    gdfs = [extracted_gdf1_1_row, extracted_gdf2_2_row, extracted_gdf3]

    # Perform a full outer join and average the numeric columns across all GeoDataFrames
    result = reduce(merge_utils.merge_and_average, gdfs)
    # this merge should not have any common dates and should contain 1+2+3 = 6 rows
    result.sort_values(by="date", inplace=True)
    result.reset_index(drop=True, inplace=True)

    # Concatenate the 'geoaccuracy' values from all GeoDataFrames
    concated_gdf = pd.concat([gdf for gdf in gdfs])

    # Check if the values in expected_geoaccuracy are present in the 'geoaccuracy' column of the result DataFrame
    assert concated_gdf["geoaccuracy"].isin(result["geoaccuracy"]).all()

    assert len(result) == 6
    # Check if the values in expected_geoaccuracy are present in the 'geoaccuracy' column of the result DataFrame
    assert concated_gdf["geoaccuracy"].isin(result["geoaccuracy"]).all()
    assert concated_gdf["cloud_cover"].isin(result["cloud_cover"]).all()
    assert concated_gdf["date"].isin(result["date"]).all()
    assert concated_gdf["satname"].isin(result["satname"]).all()
    # this test should not have merged any geometries because they were all on different dates
    assert concated_gdf["date"].isin(result["date"]).all()


def test_merge_and_average_2_overlapping_gdfs(extracted_gdf1, extracted_gdf2):
    # List of GeoDataFrames
    # these gdfs have 2 dates with the same satellite in common
    gdfs = [extracted_gdf1, extracted_gdf2]

    # Perform a full outer join and average the numeric columns across all GeoDataFrames
    result = reduce(merge_utils.merge_and_average, gdfs)
    # this merge should not have any common dates and should contain 1+2+3 = 6 rows
    result.sort_values(by="date", inplace=True)
    result.reset_index(drop=True, inplace=True)

    # Concatenate the 'geoaccuracy' values from all GeoDataFrames
    concated_gdf = pd.concat([gdf for gdf in gdfs])

    # Check if the values in expected_geoaccuracy are present in the 'geoaccuracy' column of the result DataFrame
    assert concated_gdf["geoaccuracy"].isin(result["geoaccuracy"]).all()

    # from the original 6 rows with 2 overlapping dates, the result should have 4 rows
    assert len(result) == 4
    assert (
        result[["date", "satname"]].duplicated().sum() == 0
    ), "The combination of 'date' and 'satname' is not unique."
    # Concatenate the 'geoaccuracy' values from all GeoDataFrames
    expected_geoaccuracy = pd.concat([gdf["geoaccuracy"] for gdf in gdfs])

    # Check if the values in expected_geoaccuracy are present in the 'geoaccuracy' column of the result DataFrame
    assert expected_geoaccuracy.isin(result["geoaccuracy"]).all()
    assert isinstance(
        result[result["date"] == pd.Timestamp("2018-12-30 18:22:25")]["geometry"].iloc[
            0
        ],
        MultiPoint,
    )
    assert (
        len(
            result[result["date"] == pd.Timestamp("2018-12-30 18:22:25")]["geometry"]
            .iloc[0]
            .geoms
        )
        == 3
    )
    assert np.isin(["L8"], result["satname"]).all()


def test_merge_and_average_1_gdf(extracted_gdf1):
    # List of GeoDataFrames
    # these gdfs have 2 dates with the same satellite in common
    gdfs = [
        extracted_gdf1,
    ]

    # Perform a full outer join and average the numeric columns across all GeoDataFrames
    result = reduce(merge_utils.merge_and_average, gdfs)

    result.sort_values(by="date", inplace=True)
    result.reset_index(drop=True, inplace=True)

    assert len(result) == 3
    assert (
        result[["date", "satname"]].duplicated().sum() == 0
    ), "The combination of 'date' and 'satname' is not unique."
    assert isinstance(
        result[result["date"] == pd.Timestamp("2018-12-30 18:22:25")]["geometry"].iloc[
            0
        ],
        MultiPoint,
    )
    assert extracted_gdf1["geoaccuracy"].isin(result["geoaccuracy"]).all()
    assert result["date"].equals(extracted_gdf1["date"])
    assert result["satname"].equals(extracted_gdf1["satname"])
    assert result["cloud_cover"].equals(extracted_gdf1["cloud_cover"])
    # convert the result geometry to multipoint
    from coastseg.merge_utils import convert_lines_to_multipoints

    new_result = convert_lines_to_multipoints(result)
    assert new_result["geometry"].equals(extracted_gdf1["geometry"])


@pytest.fixture
def merged_config_no_nulls_no_index_right():
    data = {
        "type": ["bbox", "bbox", "roi", "roi", "shoreline", "shoreline"],
        "id": ["1", "1", "B", "B", "D", "C"],
        "geometry": [
            Point(0, 0),
            Point(0, 0),
            Polygon([(0, 0), (1, 1), (2, 2), (0, 0)]),
            Polygon([(0, 0), (1, 1), (2, 2), (0, 0)]),
            LineString([(0, 0), (1, 1), (2, 2)]),
            LineString([(0, 0), (1, 1), (2, 2)]),
        ],
    }
    return gpd.GeoDataFrame(data)


@pytest.fixture
def merged_config_nulls():
    data = {
        "type": ["bbox", "bbox", "roi", "roi", "shoreline", "shoreline"],
        "id": [None, np.nan, "B", "B", "D", "C"],
        "geometry": [
            Point(0, 0),
            Point(0, 0),
            Polygon([(0, 0), (1, 1), (2, 2), (0, 0)]),
            Polygon([(0, 0), (1, 1), (2, 2), (0, 0)]),
            LineString([(0, 0), (1, 1), (2, 2)]),
            LineString([(0, 0), (1, 1), (2, 2)]),
        ],
        "index_right": [0, 1, 2, 3, 4, 5],
    }
    return gpd.GeoDataFrame(data)


@pytest.fixture
def merged_config_nulls_all_unique():
    data = {
        "type": ["bbox", "bbox", "roi", "roi", "shoreline", "shoreline"],
        "id": [None, np.nan, "Z", "B", "D", "C"],
        "geometry": [
            Point(0, 0),
            Point(1, 1),
            Polygon([(0, 0), (1, 1), (2, 2), (0, 0)]),
            Polygon([(2, 2), (3, 4), (6, 5), (7, 8)]),
            LineString([(0, 0), (1, 1), (2, 2)]),
            LineString([(8, 8), (8, 5), (9, 4)]),
        ],
        "index_right": [0, 1, 2, 3, 4, 5],
    }
    return gpd.GeoDataFrame(data)


def test_aggregate_gdf_merged_config_with_nulls(merged_config_nulls):
    group_fields = ["type", "geometry"]
    result = merge_utils.aggregate_gdf(merged_config_nulls, group_fields)

    # Check if null values are filtered out
    assert result["id"].isnull().sum() == 0
    assert len(result) == 3
    # very the ids got combined for rows with the same type and geometry
    assert result[result["type"] == "shoreline"]["id"].values[0] == "D, C"
    assert result[result["type"] == "roi"]["id"].values[0] == "B"
    assert result[result["type"] == "bbox"]["id"].values[0] == ""


def test_aggregate_gdf_merged_config_no_nulls(merged_config_no_nulls_no_index_right):
    group_fields = ["type", "geometry"]
    result = merge_utils.aggregate_gdf(
        merged_config_no_nulls_no_index_right, group_fields
    )

    # Check if null values are filtered out
    assert result["id"].isnull().sum() == 0
    assert len(result) == 3
    # very the ids got combined for rows with the same type and geometry
    assert result[result["type"] == "shoreline"]["id"].values[0] == "D, C"
    assert result[result["type"] == "roi"]["id"].values[0] == "B"
    assert result[result["type"] == "bbox"]["id"].values[0] == "1"


def test_aggregate_gdf_merged_config_all_unique(merged_config_nulls_all_unique):
    group_fields = ["type", "geometry"]
    result = merge_utils.aggregate_gdf(merged_config_nulls_all_unique, group_fields)

    # Check if null values are filtered out
    assert result["id"].isnull().sum() == 0
    assert len(result) == 6
    # very the ids got combined for rows with the same type and geometry
    assert len(result[result["type"] == "shoreline"]) == 2
    assert len(result[result["type"] == "roi"]) == 2
    assert len(result[result["type"] == "bbox"]) == 2
    assert result[result["type"] == "shoreline"]["id"].isin(["D", "C"]).all()
    assert result[result["type"] == "roi"]["id"].isin(["B", "Z"]).all()


def test_filter_and_join_gdfs():
    # Create a sample GeoDataFrame
    data = {
        "type": ["roi", "shoreline", "roi", "shoreline"],
        "geometry": [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

    # Define the feature type to filter by
    feature_type = "roi"

    # Call the function with the sample data
    result = merge_utils.filter_and_join_gdfs(gdf, feature_type)

    # # Check that the result is a GeoDataFrame
    assert isinstance(result, gpd.GeoDataFrame), "The result should be a GeoDataFrame"

    # # Check that the result only contains 'roi' type features
    assert (
        result["type"].eq(feature_type).all()
    ), "The result should only contain 'roi' type features"

    # # Check that the spatial join keeps only the intersecting geometries
    # # For this, we'll need to make sure the original 'roi' points intersect with themselves
    assert (
        len(result) == 2
    ), "The result should only contain intersecting 'roi' geometries"


def test_read_geojson_files(
    temp_geojson_file,
):
    # Test reading a single GeoJSON file
    filepaths = [temp_geojson_file]
    result = merge_utils.read_geojson_files(filepaths)
    assert len(result) == 1
    assert isinstance(result[0], gpd.GeoDataFrame)
    assert result[0].shape[0] > 0

    # # Test reading multiple GeoJSON files
    # filepaths = ['/path/to/file1.geojson', '/path/to/file2.geojson']
    # result = merge_utils.read_geojson_files(filepaths)
    # assert len(result) == 2
    # assert isinstance(result[0], gpd.GeoDataFrame)
    # assert isinstance(result[1], gpd.GeoDataFrame)
    # assert result[0].shape[0] > 0
    # assert result[1].shape[0] > 0

    # Test filtering by column value
    filepaths = [temp_geojson_file]
    column = "type"
    value = "roi"
    result = merge_utils.read_geojson_files(filepaths, column, value)
    assert len(result) == 1
    assert isinstance(result[0], gpd.GeoDataFrame)
    assert result[0].shape[0] > 0
    assert all(result[0][column] == value)

    # Test keeping specific columns
    filepaths = [temp_geojson_file]
    keep_columns = ["geometry", "id", "type"]
    result = merge_utils.read_geojson_files(filepaths, keep_columns=keep_columns)
    assert len(result) == 1
    assert isinstance(result[0], gpd.GeoDataFrame)
    assert result[0].shape[0] > 0
    assert set(result[0].columns) == set(keep_columns)
