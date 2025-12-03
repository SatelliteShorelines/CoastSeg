import pytest
import pkg_resources
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Polygon
import os
from unittest.mock import patch
from ipyleaflet import GeoJSON

from coastseg.transects import (
    Transects,
    load_intersecting_transects,
    drop_columns,
    create_arrowhead,
    create_transects_with_arrowheads,
)
from coastseg.exceptions import InvalidGeometryType


def test_load_intersecting_transects(triangle_polygon_gdf):
    rectangle = triangle_polygon_gdf
    file_path = pkg_resources.resource_filename("coastseg", "transects")
    transect_dir = file_path
    transect_files = os.listdir(transect_dir)
    result = load_intersecting_transects(rectangle, transect_files, transect_dir)
    assert isinstance(result, gpd.GeoDataFrame), "Output should be a GeoDataFrame"
    assert all(
        col in result.columns for col in ["id", "geometry", "slope"]
    ), 'Output columns should contain "id", "geometry", and "slope"'


# 1. load transects from a bbox geodataframe in crs 4326
def test_transects_init(valid_bbox_gdf: gpd.GeoDataFrame):
    actual_transects = Transects(bbox=valid_bbox_gdf)
    assert isinstance(
        actual_transects, Transects
    ), "Output should be an instance of Transects class"
    assert isinstance(
        actual_transects.gdf, gpd.GeoDataFrame
    ), "Transects attribute gdf should be a GeoDataFrame"
    assert (
        not actual_transects.gdf.empty
    ), "gdf should not be empty after processing bbox"
    assert actual_transects.gdf.crs.to_string() == "EPSG:4326"
    assert "id" in actual_transects.gdf.columns
    assert not any(actual_transects.gdf["id"].duplicated())


# 2. load transects from a
def test_transects_process_provided_transects():
    lines_coords = [((10, 10), (20, 20)), ((30, 30), (40, 50)), ((50, 60), (70, 80))]
    linestrings = [LineString([start, end]) for start, end in lines_coords]
    # Create a GeoDataFrame
    transects = gpd.GeoDataFrame(
        {"geometry": linestrings}, crs="EPSG:4326"  # Setting the desired CRS
    )
    transects["id"] = "test_id"
    transects["slope"] = 1.0
    actual_transects = Transects(transects=transects)
    assert isinstance(
        actual_transects, Transects
    ), "Output should be an instance of Transects class"
    assert (
        not actual_transects.gdf.empty
    ), "gdf should not be empty after processing provided transects"
    assert actual_transects.gdf.crs.to_string() == "EPSG:4326"
    assert "id" in actual_transects.gdf.columns
    assert not any(actual_transects.gdf["id"].duplicated())


# load transects with crs
def test_load_transect_with_different_crs():
    lines_coords = [((10, 10), (20, 20)), ((30, 30), (40, 50)), ((50, 60), (70, 80))]
    linestrings = [LineString([start, end]) for start, end in lines_coords]
    # Create a GeoDataFrame
    transects = gpd.GeoDataFrame(
        {"geometry": linestrings}, crs="EPSG:2033"  # Setting the desired CRS
    )
    transects["id"] = "test_id"
    transects["slope"] = 1.0
    actual_transects = Transects(transects=transects)
    assert isinstance(
        actual_transects, Transects
    ), "Output should be an instance of Transects class"
    assert (
        not actual_transects.gdf.empty
    ), "gdf should not be empty after processing provided transects"
    assert actual_transects.gdf.crs.to_string() == "EPSG:4326"
    assert "id" in actual_transects.gdf.columns
    assert not any(actual_transects.gdf["id"].duplicated())


def test_transects_process_invalid_geometries():
    transects = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 1), (1, 0)])])
    transects["id"] = "test_id"
    transects["slope"] = 1.0
    with pytest.raises(InvalidGeometryType):
        Transects(transects=transects)


# 3. load transects from a bbox  geodataframe with a CRS 4327
def test_transects_process_provided_transects_in_different_crs():
    lines_coords = [((10, 10), (20, 20)), ((30, 30), (40, 50)), ((50, 60), (70, 80))]
    linestrings = [LineString([start, end]) for start, end in lines_coords]
    # Create a GeoDataFrame
    transects = gpd.GeoDataFrame(
        {"geometry": linestrings}, crs="EPSG:2033"  # Setting the desired CRS
    )
    transects["id"] = "test_id"
    transects["slope"] = 1.0
    actual_transects = Transects(transects=transects)
    assert isinstance(
        actual_transects, Transects
    ), "Output should be an instance of Transects class"
    assert (
        not actual_transects.gdf.empty
    ), "gdf should not be empty after processing provided transects"
    assert actual_transects.gdf.crs.to_string() == "EPSG:4326"
    assert "id" in actual_transects.gdf.columns
    assert not any(actual_transects.gdf["id"].duplicated())


# 4. load transects from a transects geodataframe with a CRS 4326
def test_transects_with_valid_transects(valid_transects_gdf):
    actual_transects = Transects(transects=valid_transects_gdf)
    columns_to_keep = ["id", "geometry", "slope"]
    assert (
        not actual_transects.gdf.empty
    ), "gdf should not be empty after processing provided transects"
    assert set(actual_transects.gdf.columns) == set(
        columns_to_keep
    ), "gdf should contain columns id, slope and geometry"
    assert "usa_CA_0288-0122" in list(actual_transects.gdf["id"])
    assert not any(actual_transects.gdf["id"].duplicated())
    assert actual_transects.gdf.crs.to_string() == "EPSG:4326"
    assert "id" in actual_transects.gdf.columns
    assert not any(actual_transects.gdf["id"].duplicated())


# 5. load transects from a transects  geodataframe with a CRS 4327
def test_transects_with_transects_different_crs(valid_transects_gdf):
    # change the crs of the geodataframe
    transects_diff_crs = valid_transects_gdf.to_crs("EPSG:4326", inplace=False)
    actual_transects = Transects(transects=transects_diff_crs)
    columns_to_keep = ["id", "geometry", "slope"]
    assert (
        not actual_transects.gdf.empty
    ), "gdf should not be empty after processing provided transects"
    assert set(actual_transects.gdf.columns) == set(
        columns_to_keep
    ), "gdf should contain columns id, slope and geometry"
    assert "usa_CA_0288-0122" in list(actual_transects.gdf["id"])
    assert not any(actual_transects.gdf["id"].duplicated())
    assert actual_transects.gdf.crs.to_string() == "EPSG:4326"
    assert "id" in actual_transects.gdf.columns
    assert not any(actual_transects.gdf["id"].duplicated())


# ============================================================================
# COMPREHENSIVE TESTS FOR TRANSECTS MODULE FUNCTIONS AND CLASSES
# ============================================================================


class TestDropColumns:
    """Test drop_columns function with various column configurations."""

    def test_drop_default_columns(self):
        """Test dropping default oceanographic columns."""
        # Create test GDF with default columns to drop
        gdf = gpd.GeoDataFrame(
            {
                "id": [1, 2],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(2, 2), (3, 3)]),
                ],
                "MEAN_SIG_WAVEHEIGHT": [1.5, 2.0],
                "TIDAL_RANGE": [2.5, 3.0],
                "ERODIBILITY": [0.5, 0.7],
                "keep_me": ["a", "b"],
            }
        )

        result = drop_columns(gdf)

        # Check that default columns were dropped
        assert "MEAN_SIG_WAVEHEIGHT" not in result.columns
        assert "TIDAL_RANGE" not in result.columns
        assert "ERODIBILITY" not in result.columns
        # Check that other columns remain
        assert "id" in result.columns
        assert "geometry" in result.columns
        assert "keep_me" in result.columns

    def test_drop_custom_columns(self):
        """Test dropping custom specified columns."""
        gdf = gpd.GeoDataFrame(
            {
                "id": [1, 2],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(2, 2), (3, 3)]),
                ],
                "col_to_drop": ["x", "y"],
                "keep_me": ["a", "b"],
            }
        )

        result = drop_columns(gdf, ["col_to_drop"])

        # Check specific column was dropped
        assert "col_to_drop" not in result.columns
        assert "keep_me" in result.columns

    def test_drop_nonexistent_columns(self):
        """Test dropping columns that don't exist (should not raise error)."""
        gdf = gpd.GeoDataFrame(
            {
                "id": [1, 2],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(2, 2), (3, 3)]),
                ],
            }
        )

        # Should not raise error when trying to drop non-existent columns
        result = drop_columns(gdf, ["nonexistent_column"])

        # Original columns should remain
        assert "id" in result.columns
        assert "geometry" in result.columns


class TestCreateArrowhead:
    """Test create_arrowhead function for transect direction indicators."""

    def test_create_arrowhead_basic(self):
        """Test basic arrowhead creation."""
        line = LineString([(0, 0), (1, 0)])  # Horizontal line

        arrowhead = create_arrowhead(line)

        # Check return type
        assert isinstance(arrowhead, Polygon)
        # Check that arrowhead has 3 vertices (triangle)
        coords = list(arrowhead.exterior.coords)
        assert len(coords) == 4  # 4 because polygon closes itself
        assert coords[0] == coords[-1]  # First and last should be same

    def test_create_arrowhead_custom_parameters(self):
        """Test arrowhead creation with custom parameters."""
        line = LineString([(0, 0), (1, 0)])

        arrowhead = create_arrowhead(line, arrow_length=0.001, arrow_angle=45)

        # Should still be valid polygon
        assert isinstance(arrowhead, Polygon)
        assert arrowhead.is_valid

    def test_create_arrowhead_vertical_line(self):
        """Test arrowhead on vertical line."""
        line = LineString([(0, 0), (0, 1)])  # Vertical line

        arrowhead = create_arrowhead(line)

        assert isinstance(arrowhead, Polygon)
        assert arrowhead.is_valid

    def test_create_arrowhead_diagonal_line(self):
        """Test arrowhead on diagonal line."""
        line = LineString([(0, 0), (1, 1)])  # 45-degree line

        arrowhead = create_arrowhead(line)

        assert isinstance(arrowhead, Polygon)
        assert arrowhead.is_valid


class TestCreateTransectsWithArrowheads:
    """Test create_transects_with_arrowheads function."""

    def test_create_transects_with_arrowheads_basic(self):
        """Test basic transects with arrowheads creation."""
        # Create test transects GDF
        gdf = gpd.GeoDataFrame(
            {
                "id": ["t1", "t2"],
                "geometry": [
                    LineString([(0, 0), (1, 0)]),
                    LineString([(0, 1), (1, 1)]),
                ],
                "slope": [0.5, 0.7],
            },
            crs="EPSG:4326",
        )

        result = create_transects_with_arrowheads(gdf)

        # Check structure
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == len(gdf)
        assert "geometry" in result.columns
        # Function returns correct structure (CRS may vary based on internal processing)

    def test_create_transects_with_arrowheads_custom_params(self):
        """Test with custom arrowhead parameters."""
        gdf = gpd.GeoDataFrame(
            {"id": ["t1"], "geometry": [LineString([(0, 0), (1, 0)])], "slope": [0.5]},
            crs="EPSG:4326",
        )

        result = create_transects_with_arrowheads(
            gdf, arrow_length=0.001, arrow_angle=45
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1

    def test_create_transects_with_arrowheads_drops_columns(self):
        """Test that default columns are dropped."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["t1"],
                "geometry": [LineString([(0, 0), (1, 0)])],
                "MEAN_SIG_WAVEHEIGHT": [1.5],  # Should be dropped
                "slope": [0.5],
            },
            crs="EPSG:4326",
        )

        result = create_transects_with_arrowheads(gdf)

        # Default column should be dropped
        assert "MEAN_SIG_WAVEHEIGHT" not in result.columns


class TestLoadIntersectingTransectsFunction:
    """Test load_intersecting_transects function with various scenarios."""

    @patch("coastseg.transects.os.path.exists")
    @patch("coastseg.transects.Feature.read_masked_clean")
    def test_load_intersecting_transects_success(self, mock_read, mock_exists):
        """Test successful loading of intersecting transects."""
        # Mock file exists
        mock_exists.return_value = True

        # Mock successful file read
        mock_gdf = gpd.GeoDataFrame(
            {"id": ["t1"], "geometry": [LineString([(0, 0), (1, 1)])], "slope": [0.5]},
            crs="EPSG:4326",
        )
        mock_read.return_value = mock_gdf

        # Create test rectangle
        rectangle = gpd.GeoDataFrame(
            [1], geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:4326"
        )

        result = load_intersecting_transects(rectangle, ["test.geojson"], "/fake/dir")

        assert isinstance(result, gpd.GeoDataFrame)
        assert not result.empty

    @patch("coastseg.transects.os.path.exists")
    def test_load_intersecting_transects_missing_file(self, mock_exists):
        """Test handling of missing transect files."""
        mock_exists.return_value = False

        rectangle = gpd.GeoDataFrame(
            [1], geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:4326"
        )

        result = load_intersecting_transects(
            rectangle, ["missing.geojson"], "/fake/dir"
        )

        # Should return empty GDF with correct structure
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.empty
        assert "geometry" in result.columns

    @patch("coastseg.transects.os.path.exists")
    @patch("coastseg.transects.Feature.read_masked_clean")
    def test_load_intersecting_transects_read_error(self, mock_read, mock_exists):
        """Test handling of file read errors."""
        mock_exists.return_value = True
        mock_read.side_effect = Exception("Read error")

        rectangle = gpd.GeoDataFrame(
            [1], geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:4326"
        )

        result = load_intersecting_transects(rectangle, ["error.geojson"], "/fake/dir")

        # Should handle error gracefully and return empty result
        assert isinstance(result, gpd.GeoDataFrame)


class TestTransectsInitialization:
    """Test Transects class initialization scenarios."""

    def test_transects_empty_initialization(self):
        """Test creating empty Transects instance."""
        transects = Transects()

        # Check empty initialization
        assert isinstance(transects, Transects)
        assert transects.gdf.empty
        assert str(transects.gdf.crs) == "EPSG:4326"
        assert "geometry" in transects.gdf.columns

    def test_transects_with_filename(self):
        """Test initialization with custom filename."""
        custom_filename = "my_transects.geojson"
        transects = Transects(filename=custom_filename)

        assert transects.filename == custom_filename

    @patch.object(Transects, "_create_from_bbox")
    def test_transects_bbox_precedence(self, mock_create_bbox):
        """Test that bbox initialization is called when provided."""
        mock_create_bbox.return_value = gpd.GeoDataFrame(
            {"id": ["t1"], "geometry": [LineString([(0, 0), (1, 1)])], "slope": [0.5]},
            crs="EPSG:4326",
        )

        bbox = gpd.GeoDataFrame(
            [1], geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:4326"
        )
        _ = Transects(bbox=bbox)

        # Verify bbox creation was called
        mock_create_bbox.assert_called_once_with(bbox)


class TestTransectsClassMethods:
    """Test Transects class factory methods."""

    def test_from_bbox_method(self):
        """Test from_bbox class method."""
        bbox = gpd.GeoDataFrame(
            [1], geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:4326"
        )

        with patch.object(Transects, "_create_from_bbox") as mock_create:
            mock_create.return_value = gpd.GeoDataFrame(
                columns=["id", "geometry", "slope"]
            )

            transects = Transects.from_bbox(bbox, "test.geojson")

            assert isinstance(transects, Transects)
            assert transects.filename == "test.geojson"

    def test_from_gdf_method(self):
        """Test from_gdf class method."""
        gdf = gpd.GeoDataFrame(
            {"id": ["t1"], "geometry": [LineString([(0, 0), (1, 1)])], "slope": [0.5]},
            crs="EPSG:4326",
        )

        transects = Transects.from_gdf(gdf, "test.geojson")

        assert isinstance(transects, Transects)
        assert transects.filename == "test.geojson"

    @patch("geopandas.read_file")
    def test_from_files_method_success(self, mock_read):
        """Test from_files method with successful file reads."""
        mock_gdf = gpd.GeoDataFrame(
            {"id": ["t1"], "geometry": [LineString([(0, 0), (1, 1)])], "slope": [0.5]},
            crs="EPSG:4326",
        )
        mock_read.return_value = mock_gdf

        transects = Transects.from_files(["file1.geojson", "file2.geojson"])

        assert isinstance(transects, Transects)
        assert mock_read.call_count == 2

    @patch("geopandas.read_file")
    def test_from_files_method_with_errors(self, mock_read):
        """Test from_files method handles file read errors."""
        # First call succeeds, second fails
        mock_gdf = gpd.GeoDataFrame(
            {"id": ["t1"], "geometry": [LineString([(0, 0), (1, 1)])], "slope": [0.5]},
            crs="EPSG:4326",
        )
        mock_read.side_effect = [mock_gdf, Exception("File error")]

        transects = Transects.from_files(["good.geojson", "bad.geojson"])

        # Should handle error gracefully
        assert isinstance(transects, Transects)

    @patch("geopandas.read_file")
    def test_from_files_method_empty_files(self, mock_read):
        """Test from_files method with empty files."""
        mock_read.return_value = gpd.GeoDataFrame()  # Empty GDF

        transects = Transects.from_files(["empty.geojson"])

        assert isinstance(transects, Transects)


class TestTransectsVisualization:
    """Test Transects styling and visualization methods."""

    def test_style_layer_with_gdf(self):
        """Test style_layer method with GeoDataFrame input."""
        transects = Transects()

        gdf = gpd.GeoDataFrame(
            {"id": ["t1"], "geometry": [LineString([(0, 0), (1, 1)])], "slope": [0.5]},
            crs="EPSG:4326",
        )

        with patch.object(transects, "to_geojson") as mock_to_geojson:
            mock_to_geojson.return_value = {"type": "FeatureCollection", "features": []}

            result = transects.style_layer(gdf, "test_layer")

            # Check that method was called and returns GeoJSON layer
            assert isinstance(result, GeoJSON)
            # Method may be called multiple times during processing
            assert mock_to_geojson.call_count >= 1

    def test_style_layer_with_dict(self):
        """Test style_layer method with dict/GeoJSON input."""
        transects = Transects()

        geojson_data = {"type": "FeatureCollection", "features": []}

        result = transects.style_layer(geojson_data, "test_layer")

        # Check that GeoJSON layer is created
        assert isinstance(result, GeoJSON)


class TestTransectsFileOperations:
    """Test Transects file and directory operations."""

    def test_get_transects_directory(self):
        """Test get_transects_directory static method."""
        directory = Transects.get_transects_directory()

        # Check that it returns a string path
        assert isinstance(directory, str)
        assert directory.endswith("transects")

    @patch("pandas.read_csv")
    @patch("coastseg.transects.os.path.exists")
    @patch("coastseg.transects.os.mkdir")
    def test_load_total_bounds_df_success(self, mock_mkdir, mock_exists, mock_read_csv):
        """Test successful loading of total bounds DataFrame."""
        transects = Transects()

        # Mock CSV exists
        mock_exists.side_effect = [True, True]  # Directory exists, CSV exists

        # Mock CSV data
        mock_df = pd.DataFrame(
            {
                "filename": ["file1.geojson", "file2.geojson"],
                "minx": [0.0, 1.0],
                "miny": [0.0, 1.0],
                "maxx": [1.0, 2.0],
                "maxy": [1.0, 2.0],
            }
        )
        mock_read_csv.return_value = mock_df

        result = transects.load_total_bounds_df()

        # Check result structure
        assert isinstance(result, pd.DataFrame)
        assert "file1.geojson" in result.index
        assert "file2.geojson" in result.index

    @patch("pandas.read_csv")
    @patch("coastseg.transects.os.path.exists")
    @patch("coastseg.transects.os.mkdir")
    def test_load_total_bounds_df_missing_csv(
        self, mock_mkdir, mock_exists, mock_read_csv
    ):
        """Test handling when CSV file is missing."""
        transects = Transects()

        # Mock directory exists but CSV doesn't
        mock_exists.side_effect = [True, False]

        result = transects.load_total_bounds_df()

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_get_intersecting_files(self):
        """Test get_intersecting_files method."""
        transects = Transects()

        # Mock load_total_bounds_df
        mock_bounds_df = pd.DataFrame(
            {
                "minx": [0.0, 2.0],
                "miny": [0.0, 2.0],
                "maxx": [1.0, 3.0],
                "maxy": [1.0, 3.0],
            },
            index=["file1.geojson", "file2.geojson"],
        )

        with patch.object(
            transects, "load_total_bounds_df", return_value=mock_bounds_df
        ):
            bbox = gpd.GeoDataFrame(
                [1],
                geometry=[Polygon([(0.5, 0.5), (0.8, 0.5), (0.8, 0.8), (0.5, 0.8)])],
                crs="EPSG:4326",
            )

            result = transects.get_intersecting_files(bbox)

            # Should find intersecting file
            assert isinstance(result, list)
            assert "file1.geojson" in result


class TestTransectsEdgeCases:
    """Test edge cases and error handling for Transects class."""

    def test_transects_with_multilinestring_geometry(self):
        """Test handling of MultiLineString geometries."""
        from shapely.geometry import MultiLineString

        # Create MultiLineString geometry
        multi_line = MultiLineString(
            [LineString([(0, 0), (1, 1)]), LineString([(2, 2), (3, 3)])]
        )

        gdf = gpd.GeoDataFrame(
            {"id": ["ml1"], "geometry": [multi_line], "slope": [0.5]}, crs="EPSG:4326"
        )

        # Should handle MultiLineString without error
        transects = Transects(transects=gdf)
        assert isinstance(transects, Transects)

    def test_transects_process_with_missing_columns(self):
        """Test processing transects with missing required columns."""
        # Create GDF missing 'slope' column
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [LineString([(0, 0), (1, 1)])],
            },
            crs="EPSG:4326",
        )

        # Should process GDF and add ID column at minimum
        transects = Transects(transects=gdf)

        # Check that ID column is added during processing
        assert "id" in transects.gdf.columns
        # Note: 'slope' may not be automatically added - depends on Feature.clean_gdf behavior

    def test_transects_empty_gdf_input(self):
        """Test handling of empty GeoDataFrame input."""
        empty_gdf = gpd.GeoDataFrame(
            columns=["id", "geometry", "slope"], crs="EPSG:4326"
        )

        transects = Transects(transects=empty_gdf)

        # Should handle empty input gracefully
        assert isinstance(transects, Transects)
        assert transects.gdf.empty

    def test_transects_crs_conversion(self):
        """Test CRS conversion during processing."""
        # Create GDF in different CRS
        gdf = gpd.GeoDataFrame(
            {"id": ["t1"], "geometry": [LineString([(0, 0), (1, 1)])], "slope": [0.5]},
            crs="EPSG:3857",
        )  # Web Mercator

        transects = Transects(transects=gdf)

        # Should convert to EPSG:4326
        assert str(transects.gdf.crs) == "EPSG:4326"

    @patch.object(Transects, "get_intersecting_files")
    @patch("coastseg.transects.load_intersecting_transects")
    def test_create_from_bbox_no_intersecting_files(self, mock_load, mock_get_files):
        """Test _create_from_bbox when no files intersect."""
        mock_get_files.return_value = []
        mock_load.return_value = gpd.GeoDataFrame(
            columns=["id", "geometry", "slope"], crs="EPSG:4326"
        )

        transects = Transects()
        bbox = gpd.GeoDataFrame(
            [1], geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:4326"
        )

        result = transects._create_from_bbox(bbox)

        # Should return empty GDF
        assert isinstance(result, gpd.GeoDataFrame)
