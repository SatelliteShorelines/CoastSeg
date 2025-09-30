import geopandas as gpd
import pytest
from unittest.mock import Mock, patch
from ipyleaflet import GeoJSON

# Geometry classes now used from shared fixtures in conftest.py

from coastseg import exceptions
from coastseg.shoreline import Shoreline, construct_download_url
from coastseg.exceptions import DownloadError


def test_shoreline_initialization():
    shoreline = Shoreline()
    assert isinstance(shoreline, Shoreline)
    assert isinstance(shoreline.gdf, gpd.GeoDataFrame)


# Test that when shoreline is not a linestring an error is thrown
def test_shoreline_wrong_geometry(large_polygon_gdf):
    """Test that creating Shoreline with Polygon geometry raises InvalidGeometryType error."""
    with pytest.raises(exceptions.InvalidGeometryType):
        Shoreline(shoreline=large_polygon_gdf)


# 1. load shorelines from a shorelines geodataframe with a CRS 4326 with no id
def test_initialize_shorelines_with_provided_shorelines(valid_shoreline_gdf):
    actual_shoreline = Shoreline(shoreline=valid_shoreline_gdf)
    assert not actual_shoreline.gdf.empty
    assert "id" in actual_shoreline.gdf.columns
    columns_to_keep = [
        "id",
        "geometry",
        "river_label",
        "ERODIBILITY",
        "CSU_ID",
        "turbid_label",
        "slope_label",
        "sinuosity_label",
        "TIDAL_RANGE",
        "MEAN_SIG_WAVEHEIGHT",
    ]
    assert all(
        col in actual_shoreline.gdf.columns for col in columns_to_keep
    ), "Not all columns are present in shoreline.gdf.columns"
    assert not any(actual_shoreline.gdf["id"].duplicated()) == True
    assert actual_shoreline.gdf.crs.to_string() == "EPSG:4326"


# 2. load shorelines from a shorelines geodataframe with a CRS 4327 with no id
def test_initialize_shorelines_with_wrong_CRS(valid_shoreline_gdf):
    # change the crs of the geodataframe
    shorelines_diff_crs = valid_shoreline_gdf.to_crs("EPSG:4327", inplace=False)
    actual_shoreline = Shoreline(shoreline=shorelines_diff_crs)
    assert not actual_shoreline.gdf.empty
    assert "id" in actual_shoreline.gdf.columns
    columns_to_keep = [
        "id",
        "geometry",
        "river_label",
        "ERODIBILITY",
        "CSU_ID",
        "turbid_label",
        "slope_label",
        "sinuosity_label",
        "TIDAL_RANGE",
        "MEAN_SIG_WAVEHEIGHT",
    ]
    assert all(
        col in actual_shoreline.gdf.columns for col in columns_to_keep
    ), "Not all columns are present in shoreline.gdf.columns"
    assert not any(actual_shoreline.gdf["id"].duplicated()) == True
    assert actual_shoreline.gdf.crs.to_string() == "EPSG:4326"


def test_intersecting_files(box_no_shorelines_transects):
    """
    Test case to verify the behavior of get_intersecting_shoreline_files
    when there are no intersecting shoreline files.

    Args:
        box_no_shorelines_transects: The box with no default shoreline or transects.

    """
    sl = Shoreline()
    assert sl.get_intersecting_shoreline_files(box_no_shorelines_transects) == []


def test_intersecting_files_valid_bbox(valid_bbox_gdf):
    """
    Test case to check if the get_intersecting_shoreline_files method returns a non-empty list
    when provided with a valid bounding box GeoDataFrame.
    """
    sl = Shoreline()
    assert sl.get_intersecting_shoreline_files(valid_bbox_gdf) != []


# 3. load shorelines from a shorelines geodataframe with empty ids
def test_initialize_shorelines_with_empty_id_column(valid_shoreline_gdf):
    """
    Test case to verify the initialization of shorelines with an empty 'id' column.

    Args:
        valid_shoreline_gdf (geopandas.GeoDataFrame): A valid GeoDataFrame containing shoreline data.
    """
    # change the crs of the geodataframe
    shorelines_diff_crs = valid_shoreline_gdf.to_crs("EPSG:4326", inplace=False)
    # make id column empty
    shorelines_diff_crs = shorelines_diff_crs.assign(id=None)
    actual_shoreline = Shoreline(shoreline=shorelines_diff_crs)
    assert not actual_shoreline.gdf.empty
    assert "id" in actual_shoreline.gdf.columns
    columns_to_keep = [
        "id",
        "geometry",
        "river_label",
        "ERODIBILITY",
        "CSU_ID",
        "turbid_label",
        "slope_label",
        "sinuosity_label",
        "TIDAL_RANGE",
        "MEAN_SIG_WAVEHEIGHT",
    ]
    assert all(
        col in actual_shoreline.gdf.columns for col in columns_to_keep
    ), "Not all columns are present in shoreline.gdf.columns"
    assert not any(actual_shoreline.gdf["id"].duplicated()) == True
    assert actual_shoreline.gdf.crs.to_string() == "EPSG:4326"


# 4. load shorelines from a shorelines geodataframe with identical ids
def test_initialize_shorelines_with_identical_ids(valid_shoreline_gdf):
    # change the crs of the geodataframe
    shorelines_diff_crs = valid_shoreline_gdf.to_crs("EPSG:4326", inplace=False)
    # make id column empty
    shorelines_diff_crs = shorelines_diff_crs.assign(id="bad_id")
    actual_shoreline = Shoreline(shoreline=shorelines_diff_crs)
    assert not actual_shoreline.gdf.empty
    assert "id" in actual_shoreline.gdf.columns
    columns_to_keep = [
        "id",
        "geometry",
        "river_label",
        "ERODIBILITY",
        "CSU_ID",
        "turbid_label",
        "slope_label",
        "sinuosity_label",
        "TIDAL_RANGE",
        "MEAN_SIG_WAVEHEIGHT",
    ]
    assert all(
        col in actual_shoreline.gdf.columns for col in columns_to_keep
    ), "Not all columns are present in shoreline.gdf.columns"
    assert not any(actual_shoreline.gdf["id"].duplicated()) == True
    assert actual_shoreline.gdf.crs.to_string() == "EPSG:4326"


def test_initialize_shorelines_with_bbox(valid_bbox_gdf):
    shoreline = Shoreline(bbox=valid_bbox_gdf)

    assert not shoreline.gdf.empty
    assert "id" in shoreline.gdf.columns
    columns_to_keep = [
        "id",
        "geometry",
        "river_label",
        "ERODIBILITY",
        "CSU_ID",
        "turbid_label",
        "slope_label",
        "sinuosity_label",
        "TIDAL_RANGE",
        "MEAN_SIG_WAVEHEIGHT",
    ]
    assert all(
        col in shoreline.gdf.columns for col in columns_to_keep
    ), "Not all columns are present in shoreline.gdf.columns"
    assert not any(shoreline.gdf["id"].duplicated()) == True


def test_style_layer():
    layer_name = "test_layer"
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [125.6, 10.1]},
                "properties": {"name": "test"},
            }
        ],
    }
    shoreline = Shoreline()
    layer = shoreline.style_layer(geojson_data, layer_name)

    assert layer.name == layer_name
    assert (
        layer.data["features"][0]["geometry"] == geojson_data["features"][0]["geometry"]
    )
    assert layer.style


# ============================================================================
# Unit Tests for Shoreline Class (Comprehensive with Mocking)
# ============================================================================


class TestShorelineInit:
    """Test Shoreline initialization and basic properties."""

    def test_default_initialization(self):
        """Test Shoreline initializes with default values."""
        # Create instance with defaults
        shoreline = Shoreline()

        # Check default state
        assert isinstance(shoreline, Shoreline)
        assert isinstance(shoreline.gdf, gpd.GeoDataFrame)
        assert shoreline.gdf.empty
        assert shoreline.filename == "shoreline.geojson"
        assert hasattr(shoreline, "base_dir")
        assert hasattr(shoreline, "bounds_file")
        assert hasattr(shoreline, "shoreline_dir")

    def test_initialization_with_filename(self):
        """Test Shoreline initialization with custom filename."""
        custom_name = "custom_shoreline.geojson"
        shoreline = Shoreline(filename=custom_name)

        assert shoreline.filename == custom_name

    def test_initialization_with_shoreline_data(self, valid_shoreline_gdf):
        """Test Shoreline initialization with provided shoreline data."""
        # Initialize with shoreline data
        shoreline = Shoreline(shoreline=valid_shoreline_gdf)

        # Check shoreline data was processed
        assert not shoreline.gdf.empty
        assert "id" in shoreline.gdf.columns
        assert "geometry" in shoreline.gdf.columns
        assert shoreline.gdf.crs == "EPSG:4326"

    def test_initialization_with_bbox(self, valid_bbox_gdf):
        """Test Shoreline initialization with bounding box (mocked download)."""
        # Mock the full chain to avoid FileNotFoundError
        mock_gdf = gpd.GeoDataFrame(
            {"geometry": [None], "id": ["test"]}, crs="EPSG:4326"
        )
        with patch.object(
            Shoreline,
            "get_intersecting_shoreline_files",
            return_value=["test_file.geojson"],
        ):
            with patch.object(Shoreline, "create_geodataframe", return_value=mock_gdf):
                shoreline = Shoreline(bbox=valid_bbox_gdf)
                assert isinstance(shoreline, Shoreline)
                assert not shoreline.gdf.empty

    def test_repr_with_data(self, valid_shoreline_gdf):
        """Test string representation with data."""
        shoreline = Shoreline(shoreline=valid_shoreline_gdf)

        # Check repr includes key information
        repr_str = repr(shoreline)
        assert "Shoreline" in repr_str
        assert "ids=" in repr_str

    def test_class_constants(self):
        """Test that class constants are properly defined."""
        assert Shoreline.DATASET_ID == "7814755"
        assert Shoreline.LAYER_NAME == "shoreline"
        assert Shoreline.SELECTED_LAYER_NAME == "Selected Shorelines"
        assert isinstance(Shoreline.COLUMNS_TO_KEEP, list)
        assert "geometry" in Shoreline.COLUMNS_TO_KEEP
        assert "id" in Shoreline.COLUMNS_TO_KEEP


class TestShorelineInitializeMethods:
    """Test shoreline initialization methods."""

    def test_initialize_shorelines_with_shorelines_valid_data(
        self, valid_shoreline_gdf
    ):
        """Test initialization with valid shoreline GeoDataFrame."""
        shoreline = Shoreline()

        # Initialize with valid data
        result = shoreline.initialize_shorelines_with_shorelines(valid_shoreline_gdf)

        # Check result
        assert isinstance(result, gpd.GeoDataFrame)
        assert not result.empty
        assert result.crs == "EPSG:4326"
        assert "id" in result.columns

    def test_initialize_shorelines_with_invalid_type(self):
        """Test initialization with invalid data type."""
        shoreline = Shoreline()

        # Should raise ValueError for non-GeoDataFrame - using type: ignore to test runtime behavior
        with pytest.raises(ValueError, match="Shorelines must be a GeoDataFrame"):
            shoreline.initialize_shorelines_with_shorelines("not a geodataframe")  # type: ignore

    def test_initialize_shorelines_with_empty_gdf(self):
        """Test initialization with empty GeoDataFrame."""
        shoreline = Shoreline()
        empty_gdf = gpd.GeoDataFrame()

        # Should handle empty GeoDataFrame gracefully
        result = shoreline.initialize_shorelines_with_shorelines(empty_gdf)
        assert result.empty

    def test_initialize_shorelines_with_wrong_geometry_type(self, standard_polygon_gdf):
        """Test initialization with invalid geometry types."""
        shoreline = Shoreline()

        # Should raise error for Polygon geometries
        with pytest.raises(exceptions.InvalidGeometryType):
            shoreline.initialize_shorelines_with_shorelines(standard_polygon_gdf)

    def test_initialize_shorelines_with_bbox_none_or_empty(self):
        """Test bbox initialization with None or empty bbox."""
        shoreline = Shoreline()

        # Test with None - using type: ignore to test runtime behavior
        with pytest.raises(ValueError, match="Bounding box cannot be None or empty"):
            shoreline.initialize_shorelines_with_bbox(None)  # type: ignore

        # Test with empty GeoDataFrame
        empty_bbox = gpd.GeoDataFrame()
        with pytest.raises(ValueError, match="Bounding box cannot be None or empty"):
            shoreline.initialize_shorelines_with_bbox(empty_bbox)


class TestShorelineFileOperations:
    """Test file operations and downloads."""

    def test_get_intersecting_shoreline_files_no_bounds_file(self, valid_bbox_gdf):
        """Test file intersection when bounds file doesn't exist."""
        shoreline = Shoreline(bounds_file="nonexistent.geojson")

        # Should return empty list when bounds file doesn't exist
        result = shoreline.get_intersecting_shoreline_files(valid_bbox_gdf)
        assert result == []

    def test_get_intersecting_shoreline_files_no_intersections(
        self, box_no_shorelines_transects
    ):
        """Test file intersection when no files intersect."""
        shoreline = Shoreline()

        # Should return empty list when no files intersect
        result = shoreline.get_intersecting_shoreline_files(box_no_shorelines_transects)
        assert result == []

    @patch("coastseg.shoreline.gpd.read_file")
    def test_get_intersecting_shoreline_files_with_existing_files(
        self, mock_read_file, valid_bbox_gdf
    ):
        """Test file intersection with existing local files."""
        # Mock bounds file data
        mock_bounds_gdf = gpd.GeoDataFrame(
            {
                "filename": ["test_shoreline.geojson"],
                "geometry": [valid_bbox_gdf.iloc[0].geometry],
            },
            crs="EPSG:4326",
        ).set_index("filename")
        mock_read_file.return_value = mock_bounds_gdf

        shoreline = Shoreline()

        with patch("os.path.exists", return_value=True):
            result = shoreline.get_intersecting_shoreline_files(valid_bbox_gdf)
            assert len(result) == 1
            assert result[0].endswith("test_shoreline.geojson")

    @patch("coastseg.shoreline.gpd.read_file")
    def test_get_intersecting_shoreline_files_with_download(
        self, mock_read_file, valid_bbox_gdf
    ):
        """Test file intersection with file download."""
        # Mock bounds file data
        mock_bounds_gdf = gpd.GeoDataFrame(
            {
                "filename": ["test_shoreline.geojson"],
                "geometry": [valid_bbox_gdf.iloc[0].geometry],
            },
            crs="EPSG:4326",
        ).set_index("filename")
        mock_read_file.return_value = mock_bounds_gdf

        shoreline = Shoreline()

        with patch("os.path.exists", return_value=False):
            with patch.object(shoreline, "download_shoreline") as mock_download:
                result = shoreline.get_intersecting_shoreline_files(valid_bbox_gdf)
                assert len(result) == 1
                mock_download.assert_called_once()

    @patch("coastseg.shoreline.gpd.read_file")
    def test_get_intersecting_shoreline_files_download_failure(
        self, mock_read_file, valid_bbox_gdf
    ):
        """Test handling of download failures."""
        # Mock bounds file data
        mock_bounds_gdf = gpd.GeoDataFrame(
            {
                "filename": ["test_shoreline.geojson"],
                "geometry": [valid_bbox_gdf.iloc[0].geometry],
            },
            crs="EPSG:4326",
        ).set_index("filename")
        mock_read_file.return_value = mock_bounds_gdf

        shoreline = Shoreline()

        with patch("os.path.exists", return_value=False):
            with patch.object(
                shoreline,
                "download_shoreline",
                side_effect=DownloadError("Download failed"),
            ):
                # Should raise FileNotFoundError when no files can be obtained
                with pytest.raises(
                    FileNotFoundError, match="No shoreline files could be obtained"
                ):
                    shoreline.get_intersecting_shoreline_files(valid_bbox_gdf)

    def test_download_shoreline(self):
        """Test shoreline download functionality."""
        shoreline = Shoreline()

        # Mock the download function to avoid network calls
        mock_download = Mock()
        shoreline.download_shoreline(
            "test.geojson", "/path/to/save", "12345", download_function=mock_download
        )

        # Check download function was called with correct parameters
        mock_download.assert_called_once()
        args = mock_download.call_args[0]
        assert "test.geojson" in args[0]  # URL should contain filename
        assert args[1] == "/path/to/save"  # Save location
        assert args[2] == "test.geojson"  # Filename parameter

    def test_download_shoreline_custom_function(self):
        """Test shoreline download with custom download function."""
        shoreline = Shoreline()
        mock_function = Mock()

        shoreline.download_shoreline(
            "test.geojson", "/path/to/save", "12345", download_function=mock_function
        )

        # Check custom function was called
        mock_function.assert_called_once()


class TestShorelineDataProcessing:
    """Test data processing and manipulation methods."""

    def test_get_clipped_shoreline(self, valid_bbox_gdf):
        """Test shoreline clipping functionality."""
        shoreline = Shoreline()

        # Mock the read_masked_clean method from parent class
        with patch.object(shoreline, "read_masked_clean") as mock_read:
            mock_result = gpd.GeoDataFrame({"geometry": [None]}, crs="EPSG:4326")
            mock_read.return_value = mock_result

            result = shoreline.get_clipped_shoreline(
                "test_file.geojson", valid_bbox_gdf, ["geometry", "id"]
            )

            # Check result and method was called with correct parameters
            assert result is mock_result
            mock_read.assert_called_once()
            call_kwargs = mock_read.call_args[1]
            assert call_kwargs["mask"] is valid_bbox_gdf
            assert "LineString" in call_kwargs["geometry_types"]
            assert "MultiLineString" in call_kwargs["geometry_types"]

    def test_create_geodataframe_no_files(self, valid_bbox_gdf):
        """Test create_geodataframe with no shoreline files."""
        shoreline = Shoreline()

        # Should raise FileNotFoundError when no files provided
        with pytest.raises(FileNotFoundError, match="No shoreline files were provided"):
            shoreline.create_geodataframe(valid_bbox_gdf, [])

    def test_create_geodataframe_with_files(self, valid_bbox_gdf):
        """Test create_geodataframe with shoreline files."""
        shoreline = Shoreline()

        # Mock get_clipped_shoreline to return valid data
        mock_shoreline_gdf = gpd.GeoDataFrame(
            {"geometry": [None], "id": ["test_id"]}, crs="EPSG:4326"
        )

        with patch.object(
            shoreline, "get_clipped_shoreline", return_value=mock_shoreline_gdf
        ):
            with patch.object(
                shoreline, "concat_clean", return_value=mock_shoreline_gdf
            ):
                with patch.object(
                    shoreline, "clean_gdf", return_value=mock_shoreline_gdf
                ):
                    result = shoreline.create_geodataframe(
                        valid_bbox_gdf, ["test_file.geojson"]
                    )

                    assert isinstance(result, gpd.GeoDataFrame)

    def test_create_geodataframe_with_invalid_files(self, valid_bbox_gdf):
        """Test create_geodataframe when all files fail processing."""
        shoreline = Shoreline()

        # Mock get_clipped_shoreline to raise exception
        with patch.object(
            shoreline,
            "get_clipped_shoreline",
            side_effect=Exception("File processing failed"),
        ):
            with pytest.raises(
                FileNotFoundError, match="No valid shoreline data found"
            ):
                shoreline.create_geodataframe(valid_bbox_gdf, ["bad_file.geojson"])


class TestShorelineStyleLayer:
    """Test shoreline styling functionality."""

    def test_style_layer_basic(self):
        """Test basic layer styling."""
        shoreline = Shoreline()
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
                    "properties": {},
                }
            ],
        }

        # Create styled layer
        result = shoreline.style_layer(geojson_data, "test_shoreline")

        # Check result is GeoJSON layer with shoreline-specific styling
        assert isinstance(result, GeoJSON)
        assert result.name == "test_shoreline"
        assert result.style["color"] == "black"
        assert result.style["dashArray"] == "5"
        assert result.hover_style["color"] == "white"

    def test_style_layer_empty_geojson(self):
        """Test styling with empty GeoJSON."""
        shoreline = Shoreline()

        # Should raise error for empty GeoJSON
        with pytest.raises(
            ValueError, match="Empty test_shoreline geojson cannot be drawn"
        ):
            shoreline.style_layer({}, "test_shoreline")


class TestConstructDownloadUrl:
    """Test the construct_download_url helper function."""

    def test_construct_download_url_basic(self):
        """Test basic URL construction."""
        result = construct_download_url(
            "https://zenodo.org/record/", "7814755", "shoreline.geojson"
        )

        expected = (
            "https://zenodo.org/record/7814755/files/shoreline.geojson?download=1"
        )
        assert result == expected

    def test_construct_download_url_different_params(self):
        """Test URL construction with different parameters."""
        result = construct_download_url(
            "https://example.com/", "12345", "test_file.json"
        )

        expected = "https://example.com/12345/files/test_file.json?download=1"
        assert result == expected


class TestShorelineEdgeCases:
    """Test edge cases and error conditions."""

    def test_initialize_shorelines_precedence(
        self, valid_shoreline_gdf, valid_bbox_gdf
    ):
        """Test that shoreline data takes precedence over bbox."""
        # When both shoreline and bbox provided, shoreline should be used
        with patch.object(
            Shoreline, "initialize_shorelines_with_bbox"
        ) as mock_bbox_init:
            shoreline = Shoreline(shoreline=valid_shoreline_gdf, bbox=valid_bbox_gdf)

            # bbox initialization should not be called
            mock_bbox_init.assert_not_called()
            assert not shoreline.gdf.empty

    def test_initialize_shorelines_neither_provided(self):
        """Test initialization when neither shoreline nor bbox provided."""
        shoreline = Shoreline()

        # Should have empty GeoDataFrame
        assert shoreline.gdf.empty

    def test_custom_bounds_and_shoreline_directories(self, tmp_path):
        """Test initialization with custom directory paths."""
        bounds_file = str(tmp_path / "custom_bounds.geojson")
        shorelines_dir = str(tmp_path / "custom_shorelines")

        shoreline = Shoreline(bounds_file=bounds_file, shorelines_dir=shorelines_dir)

        assert shoreline.bounds_file == bounds_file
        assert shoreline.shoreline_dir == shorelines_dir

    def test_file_operations_with_missing_directories(self):
        """Test behavior when shoreline directories don't exist."""
        shoreline = Shoreline(
            bounds_file="/nonexistent/bounds.geojson",
            shorelines_dir="/nonexistent/shorelines",
        )

        # Should handle missing directories gracefully
        assert isinstance(shoreline, Shoreline)

    def test_create_geodataframe_mixed_file_success_failure(self, valid_bbox_gdf):
        """Test create_geodataframe when some files succeed and others fail."""
        shoreline = Shoreline()

        # Mock mixed success/failure scenario
        def mock_get_clipped_side_effect(file_path, bbox, columns):
            if "good" in file_path:
                return gpd.GeoDataFrame(
                    {"geometry": [None], "id": ["test"]}, crs="EPSG:4326"
                )
            else:
                raise Exception("File processing failed")

        mock_good_gdf = gpd.GeoDataFrame(
            {"geometry": [None], "id": ["test"]}, crs="EPSG:4326"
        )

        with patch.object(
            shoreline, "get_clipped_shoreline", side_effect=mock_get_clipped_side_effect
        ):
            with patch.object(shoreline, "concat_clean", return_value=mock_good_gdf):
                with patch.object(shoreline, "clean_gdf", return_value=mock_good_gdf):
                    # Should succeed with at least one good file
                    result = shoreline.create_geodataframe(
                        valid_bbox_gdf, ["good_file.geojson", "bad_file.geojson"]
                    )

                    assert isinstance(result, gpd.GeoDataFrame)
