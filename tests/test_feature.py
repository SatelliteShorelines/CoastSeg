"""Tests for the Feature base class."""

import geopandas as gpd
import pytest
from ipyleaflet import GeoJSON
from shapely.geometry import Point

from coastseg.feature import Feature


class FeatureForTests(Feature):
    """Concrete implementation of Feature for testing."""

    pass


class TestFeatureInit:
    """Test Feature initialization and basic properties."""

    def test_default_initialization(self):
        """Test Feature initializes with default values."""
        # Create instance with defaults
        feature = FeatureForTests()

        # Check default state
        assert isinstance(feature.gdf, gpd.GeoDataFrame)
        assert feature.gdf.empty

    def test_custom_filename_initialization(self):
        """Test Feature initializes with custom filename."""
        # Create instance with custom filename
        feature = FeatureForTests(filename="custom.geojson")

        # Check custom filename
        assert feature.filename == "custom.geojson"

    def test_filename_validation_invalid_type(self):
        """Test filename setter raises error for non-string input."""
        feature = FeatureForTests()

        # Test invalid type - using type: ignore to test runtime behavior
        with pytest.raises(ValueError, match="Filename must be a string"):
            feature.filename = 123  # type: ignore

    def test_filename_validation_invalid_extension(self):
        """Test filename setter requires .geojson extension."""
        feature = FeatureForTests()

        # Test invalid extension
        with pytest.raises(ValueError, match="Filename must end with '.geojson'"):
            feature.filename = "test.txt"

    def test_repr_empty_feature(self):
        """Test string representation of empty Feature."""
        feature = FeatureForTests()

        # Check repr includes key information
        repr_str = repr(feature)
        assert "FeatureForTests" in repr_str
        assert "CRS=" in repr_str
        assert "columns=[]" in repr_str
        assert "<empty>" in repr_str

    def test_repr_with_data(self, standard_polygon_gdf):
        """Test string representation with data."""
        feature = FeatureForTests()
        feature.gdf = standard_polygon_gdf

        # Check repr includes data information
        repr_str = repr(feature)
        assert "FeatureForTests" in repr_str
        assert "geometry" in repr_str
        assert "EPSG:4326" in repr_str


class TestFeatureStaticMethods:
    """Test static utility methods."""

    def test_gdf_from_mapping(self):
        """Test creating GeoDataFrame from GeoJSON-like mapping."""
        # Create simple polygon mapping
        mapping = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        }

        # Convert to GeoDataFrame
        result = Feature.gdf_from_mapping(mapping)

        # Check result
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1
        assert result.crs == "EPSG:4326"
        assert result.iloc[0].geometry.geom_type == "Polygon"

    def test_gdf_from_mapping_custom_crs(self):
        """Test creating GeoDataFrame with custom CRS."""
        mapping = {"type": "Point", "coordinates": [0, 1]}

        # Convert with custom CRS
        result = Feature.gdf_from_mapping(mapping, crs="EPSG:3857")

        # Check CRS
        assert result.crs == "EPSG:3857"

    def test_drop_all_na_columns(self):
        """Test dropping columns that are all NaN."""
        # Create GeoDataFrame with all-NaN column
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [Point(0, 0), Point(1, 1)],
                "good_col": [1, 2],
                "all_na": [None, None],
            },
            crs="EPSG:4326",
        )

        # Drop all-NaN columns
        result = Feature.drop_all_na_columns(gdf)

        # Check all-NaN column removed
        assert "all_na" not in result.columns
        assert "good_col" in result.columns
        assert "geometry" in result.columns

    def test_ensure_crs_with_existing_crs(self, standard_polygon_gdf):
        """Test ensure_crs when GeoDataFrame already has CRS."""
        # GeoDataFrame already has EPSG:4326
        result = Feature.ensure_crs(standard_polygon_gdf, "EPSG:3857")

        # Should be reprojected to target CRS
        assert result.crs == "EPSG:3857"

    def test_ensure_crs_without_crs(self):
        """Test ensure_crs when GeoDataFrame has no CRS."""
        # Create GeoDataFrame without CRS
        gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]})

        # Set CRS
        result = Feature.ensure_crs(gdf, "EPSG:4326")

        # Check CRS was set
        assert result.crs == "EPSG:4326"

    def test_to_geojson_from_dict(self):
        """Test converting dictionary to GeoJSON (passthrough)."""
        geojson_dict = {"type": "FeatureCollection", "features": []}

        # Should return same dictionary
        result = Feature.to_geojson(geojson_dict)

        assert result is geojson_dict

    def test_to_geojson_from_geodataframe(self, standard_polygon_gdf):
        """Test converting GeoDataFrame to GeoJSON."""
        # Convert to GeoJSON
        result = Feature.to_geojson(standard_polygon_gdf)

        # Check result is valid GeoJSON structure
        assert isinstance(result, dict)
        assert result["type"] == "FeatureCollection"
        assert "features" in result


class TestFeatureMethods:
    """Test instance methods of Feature."""

    def test_ids_empty_gdf(self):
        """Test getting IDs from empty GeoDataFrame."""
        feature = FeatureForTests()

        # Empty GeoDataFrame should return empty list
        assert feature.ids() == []

    def test_ids_no_id_column(self, standard_polygon_gdf):
        """Test getting IDs when no 'id' column exists."""
        feature = FeatureForTests()
        feature.gdf = standard_polygon_gdf  # Has no 'id' column

        # Should return empty list
        assert feature.ids() == []

    def test_ids_with_id_column(self):
        """Test getting IDs when 'id' column exists."""
        # Create GeoDataFrame with ID column
        gdf = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0), Point(1, 1)], "id": ["id1", "id2"]},
            crs="EPSG:4326",
        )

        feature = FeatureForTests()
        feature.gdf = gdf

        # Should return list of string IDs
        result = feature.ids()
        assert result == ["id1", "id2"]

    def test_ids_converts_to_string(self):
        """Test ID conversion to string."""
        # Create GeoDataFrame with numeric IDs
        gdf = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0)], "id": [123]}, crs="EPSG:4326"
        )

        feature = FeatureForTests()
        feature.gdf = gdf

        # Should return string IDs
        result = feature.ids()
        assert result == ["123"]

    def test_remove_by_id_empty_gdf(self):
        """Test removing IDs from empty GeoDataFrame."""
        feature = FeatureForTests()

        # Should return unchanged empty GeoDataFrame
        result = feature.remove_by_id(["id1"])
        assert result.empty

    def test_remove_by_id_no_id_column(self, standard_polygon_gdf):
        """Test removing IDs when no 'id' column exists."""
        feature = FeatureForTests()
        feature.gdf = standard_polygon_gdf

        # Should return unchanged GeoDataFrame
        result = feature.remove_by_id(["id1"])
        assert len(result) == len(standard_polygon_gdf)

    def test_remove_by_id_none_input(self):
        """Test removing IDs with None input."""
        # Create GeoDataFrame with IDs
        gdf = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0), Point(1, 1)], "id": ["id1", "id2"]},
            crs="EPSG:4326",
        )

        feature = FeatureForTests()
        feature.gdf = gdf

        # Should return unchanged GeoDataFrame - using type: ignore to test runtime behavior
        result = feature.remove_by_id(None)  # type: ignore
        assert len(result) == 2

    def test_remove_by_id_single_string(self):
        """Test removing single ID as string."""
        # Create GeoDataFrame with IDs
        gdf = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0), Point(1, 1)], "id": ["id1", "id2"]},
            crs="EPSG:4326",
        )

        feature = FeatureForTests()
        feature.gdf = gdf

        # Remove single ID
        result = feature.remove_by_id("id1")

        # Check ID was removed
        assert len(result) == 1
        assert result.iloc[0]["id"] == "id2"

    def test_remove_by_id_single_int(self):
        """Test removing single ID as integer."""
        # Create GeoDataFrame with numeric IDs
        gdf = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0), Point(1, 1)], "id": [1, 2]}, crs="EPSG:4326"
        )

        feature = FeatureForTests()
        feature.gdf = gdf

        # Remove single numeric ID
        result = feature.remove_by_id(1)

        # Check ID was removed
        assert len(result) == 1
        assert result.iloc[0]["id"] == 2

    def test_remove_by_id_multiple_ids(self):
        """Test removing multiple IDs."""
        # Create GeoDataFrame with IDs
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
                "id": ["id1", "id2", "id3"],
            },
            crs="EPSG:4326",
        )

        feature = FeatureForTests()
        feature.gdf = gdf

        # Remove multiple IDs
        result = feature.remove_by_id(["id1", "id3"])

        # Check only id2 remains
        assert len(result) == 1
        assert result.iloc[0]["id"] == "id2"


class TestFeatureUtilityMethods:
    """Test utility and helper methods."""

    def test_concat_clean_empty_list(self):
        """Test concatenating empty list of GeoDataFrames."""
        # Empty list should return empty GeoDataFrame
        result = Feature.concat_clean([])

        assert isinstance(result, gpd.GeoDataFrame)
        assert result.empty

    def test_concat_clean_single_gdf(self, standard_polygon_gdf):
        """Test concatenating single GeoDataFrame."""
        # Single GeoDataFrame should be returned as-is
        result = Feature.concat_clean([standard_polygon_gdf])

        assert len(result) == len(standard_polygon_gdf)
        assert result.crs == standard_polygon_gdf.crs

    def test_concat_clean_multiple_gdfs(
        self, standard_polygon_gdf, triangle_polygon_gdf
    ):
        """Test concatenating multiple GeoDataFrames."""
        # Concatenate two GeoDataFrames
        result = Feature.concat_clean([standard_polygon_gdf, triangle_polygon_gdf])

        # Should have combined length
        assert len(result) == len(standard_polygon_gdf) + len(triangle_polygon_gdf)

    def test_concat_clean_drop_all_na(self):
        """Test concatenation with all-NaN column dropping."""
        # Create GeoDataFrames with all-NaN columns
        gdf1 = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0)], "good_col": [1], "all_na": [None]},
            crs="EPSG:4326",
        )

        gdf2 = gpd.GeoDataFrame(
            {"geometry": [Point(1, 1)], "good_col": [2], "all_na": [None]},
            crs="EPSG:4326",
        )

        # Concatenate with drop_all_na=True (default)
        result = Feature.concat_clean([gdf1, gdf2])

        # All-NaN column should be dropped
        assert "all_na" not in result.columns
        assert "good_col" in result.columns

    def test_concat_clean_keep_all_na(self):
        """Test concatenation keeping all-NaN columns."""
        # Create GeoDataFrames with all-NaN columns
        gdf1 = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0)], "all_na": [None]}, crs="EPSG:4326"
        )

        gdf2 = gpd.GeoDataFrame(
            {"geometry": [Point(1, 1)], "all_na": [None]}, crs="EPSG:4326"
        )

        # Concatenate with drop_all_na=False
        result = Feature.concat_clean([gdf1, gdf2], drop_all_na=False)

        # All-NaN column should be kept
        assert "all_na" in result.columns


class TestFeatureStyleLayer:
    """Test GeoJSON layer styling."""

    def test_style_layer_with_geodataframe(self, standard_polygon_gdf):
        """Test creating styled layer from GeoDataFrame."""
        feature = FeatureForTests()

        # Create styled layer
        result = feature.style_layer(standard_polygon_gdf, "test_layer")

        # Check result is GeoJSON layer
        assert isinstance(result, GeoJSON)
        assert result.name == "test_layer"

    def test_style_layer_with_dict(self):
        """Test creating styled layer from GeoJSON dict."""
        feature = FeatureForTests()
        geojson_dict = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {},
                }
            ],
        }

        # Create styled layer
        result = feature.style_layer(geojson_dict, "test_layer")

        # Check result
        assert isinstance(result, GeoJSON)
        assert result.name == "test_layer"

    def test_style_layer_custom_style(self, standard_polygon_gdf):
        """Test creating layer with custom style."""
        feature = FeatureForTests()
        custom_style = {"color": "red", "weight": 3}

        # Create styled layer with custom style
        result = feature.style_layer(
            standard_polygon_gdf, "test_layer", style=custom_style
        )

        # Check custom style applied
        assert result.style == custom_style

    def test_style_layer_empty_geojson_error(self):
        """Test error when creating layer from empty GeoJSON."""
        feature = FeatureForTests()

        # Empty GeoJSON should raise error
        with pytest.raises(
            ValueError, match="Empty test_layer geojson cannot be drawn"
        ):
            feature.style_layer({}, "test_layer")


class TestFeatureCheckSize:
    """Test area size validation."""

    def test_check_size_within_limits(self):
        """Test area within acceptable limits."""
        # Should not raise any exception
        Feature.check_size(
            500.0,
            min_area=100.0,
            max_area=1000.0,
            too_small_exc=ValueError,
            too_large_exc=RuntimeError,
        )

    def test_check_size_too_small(self):
        """Test area below minimum threshold."""
        # Should raise too_small_exc
        with pytest.raises(ValueError):
            Feature.check_size(50.0, min_area=100.0, too_small_exc=ValueError)

    def test_check_size_too_large(self):
        """Test area above maximum threshold."""
        # Should raise too_large_exc
        with pytest.raises(RuntimeError):
            Feature.check_size(1500.0, max_area=1000.0, too_large_exc=RuntimeError)

    def test_check_size_no_limits(self):
        """Test size check with no limits set."""
        # Should not raise any exception
        Feature.check_size(999999.0)

    def test_check_size_no_exceptions(self):
        """Test size check with limits but no exception classes."""
        # Should not raise any exception even if limits exceeded
        Feature.check_size(50.0, min_area=100.0)
        Feature.check_size(1500.0, max_area=1000.0)


class TestFeatureCleanGdf:
    """Test GeoDataFrame cleaning and preprocessing."""

    def test_clean_gdf_basic(self, standard_polygon_gdf):
        """Test basic GeoDataFrame cleaning."""
        # Clean with default parameters
        result = Feature.clean_gdf(standard_polygon_gdf)

        # Check result
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs == "EPSG:4326"
        assert "geometry" in result.columns

    def test_clean_gdf_create_ids(self):
        """Test cleaning with ID creation."""
        # Create GeoDataFrame without ID column
        gdf = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0), Point(1, 1)]}, crs="EPSG:4326"
        )

        # Clean with create_ids_flag=True and include 'id' in columns_to_keep
        result = Feature.clean_gdf(
            gdf, create_ids_flag=True, columns_to_keep=["geometry", "id"]
        )

        # Check ID column was created
        assert "id" in result.columns
        assert len(result["id"]) == 2

    def test_clean_gdf_unique_ids(self):
        """Test cleaning with unique ID creation."""
        # Create GeoDataFrame with duplicate IDs
        gdf = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0), Point(1, 1)], "id": ["dup", "dup"]},
            crs="EPSG:4326",
        )

        # Clean with unique_ids=True
        result = Feature.clean_gdf(gdf, unique_ids=True)

        # Check IDs are unique
        assert len(result["id"].unique()) == len(result)

    def test_clean_gdf_ids_as_str(self):
        """Test converting IDs to strings."""
        # Create GeoDataFrame with numeric IDs
        gdf = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0)], "id": [123]}, crs="EPSG:4326"
        )

        # Clean with ids_as_str=True, columns_to_keep includes 'id'
        result = Feature.clean_gdf(
            gdf, columns_to_keep=["geometry", "id"], ids_as_str=True
        )

        # Check ID is string
        assert isinstance(result.iloc[0]["id"], str)
        assert result.iloc[0]["id"] == "123"


class TestFeatureReadMaskedClean:
    """Test file reading with preprocessing."""

    @pytest.fixture
    def temp_geojson_file(self, tmp_path, standard_polygon_gdf):
        """Create temporary GeoJSON file for testing."""
        file_path = tmp_path / "test.geojson"
        standard_polygon_gdf.to_file(file_path, driver="GeoJSON")
        return str(file_path)

    def test_read_masked_clean_basic(self, temp_geojson_file):
        """Test basic file reading and cleaning."""
        # Read and clean file
        result = Feature.read_masked_clean(temp_geojson_file)

        # Check result
        assert isinstance(result, gpd.GeoDataFrame)
        assert not result.empty
        assert result.crs == "EPSG:4326"

    def test_read_masked_clean_with_mask(self, temp_geojson_file, triangle_polygon_gdf):
        """Test file reading with mask for clipping."""
        # Read with mask (should clip to intersection)
        result = Feature.read_masked_clean(temp_geojson_file, mask=triangle_polygon_gdf)

        # Check result is clipped
        assert isinstance(result, gpd.GeoDataFrame)
        # Note: Specific clipping behavior depends on geometry overlap


class TestFeatureEdgeCases:
    """Test edge cases and error conditions."""

    def test_remove_by_id_mixed_types(self):
        """Test removing IDs with mixed string/int types."""
        # Create GeoDataFrame with mixed ID types as strings
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
                "id": ["abc", "123", "xyz"],
            },
            crs="EPSG:4326",
        )

        feature = FeatureForTests()
        feature.gdf = gdf

        # Remove both string and numeric ID (as string) - using type: ignore for mixed types
        result = feature.remove_by_id(["abc", 123])  # type: ignore # 123 will be converted to '123'

        # Check both IDs were removed
        assert len(result) == 1
        assert result.iloc[0]["id"] == "xyz"

    def test_concat_clean_with_different_columns(self):
        """Test concatenating GeoDataFrames with different column sets."""
        # Create GeoDataFrames with different columns
        gdf1 = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0)], "col_a": [1], "common": ["x"]}, crs="EPSG:4326"
        )

        gdf2 = gpd.GeoDataFrame(
            {"geometry": [Point(1, 1)], "col_b": [2], "common": ["y"]}, crs="EPSG:4326"
        )

        # Concatenate - should handle missing columns gracefully
        result = Feature.concat_clean([gdf1, gdf2])

        # Check result has all geometries and handles NaN appropriately
        assert len(result) == 2
        assert "common" in result.columns

    def test_ensure_crs_with_none_crs(self):
        """Test ensure_crs behavior with None CRS on GeoDataFrame."""
        # Create GeoDataFrame without CRS explicitly set to None
        gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]})
        gdf.crs = None

        # Should set the target CRS
        result = Feature.ensure_crs(gdf, "EPSG:4326")

        assert result.crs == "EPSG:4326"

    def test_remove_by_id_empty_list(self):
        """Test removing empty list of IDs."""
        # Create GeoDataFrame with IDs
        gdf = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0)], "id": ["id1"]}, crs="EPSG:4326"
        )

        feature = FeatureForTests()
        feature.gdf = gdf

        # Remove empty list - should return unchanged
        result = feature.remove_by_id([])

        assert len(result) == 1
        assert result.iloc[0]["id"] == "id1"

    def test_clean_gdf_with_geometry_validation(self):
        """Test clean_gdf with geometry type validation."""
        # Create GeoDataFrame with Point geometry
        gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")

        # Clean with Point geometry validation - should pass
        result = Feature.clean_gdf(
            gdf, geometry_types=["Point"], feature_type="test_point"
        )

        assert len(result) == 1

    def test_style_layer_with_hover_style(self, standard_polygon_gdf):
        """Test creating layer with custom hover style."""
        feature = FeatureForTests()
        hover_style = {"fillOpacity": 0.7, "weight": 3}

        # Create layer with hover style
        result = feature.style_layer(
            standard_polygon_gdf, "test_layer", hover_style=hover_style
        )

        # Check hover style applied
        assert result.hover_style == hover_style
