import pytest
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point
from ipyleaflet import GeoJSON

from coastseg.shoreline_extraction_area import Shoreline_Extraction_Area
from coastseg.exceptions import InvalidGeometryType


class TestShorelineExtractionAreaInit:
    """Test initialization of Shoreline_Extraction_Area."""

    def test_init_with_default_parameters(self):
        """Test creating instance with default parameters."""
        area = Shoreline_Extraction_Area()

        # Check that it's an instance of the class
        assert isinstance(area, Shoreline_Extraction_Area)
        # Check that gdf is empty but properly initialized
        assert area.gdf.empty
        assert area.gdf.crs.to_string() == "EPSG:4326"
        assert list(area.gdf.columns) == ["geometry"]
        # Check default filename
        assert area.filename == "shoreline_extraction_area.geojson"

    def test_init_with_valid_gdf(self, valid_bbox_gdf):
        """Test creating instance with valid polygon GeoDataFrame."""
        area = Shoreline_Extraction_Area(gdf=valid_bbox_gdf)

        assert isinstance(area, Shoreline_Extraction_Area)
        assert not area.gdf.empty
        assert len(area.gdf) == 1
        # Check CRS conversion to default EPSG:4326
        assert area.gdf.crs.to_string() == "EPSG:4326"
        # Check geometry is preserved
        assert area.gdf.geometry.iloc[0].geom_type == "Polygon"

    def test_init_with_custom_filename(self):
        """Test creating instance with custom filename."""
        custom_filename = "custom_extraction_area.geojson"
        area = Shoreline_Extraction_Area(filename=custom_filename)

        assert area.filename == custom_filename
        assert area.gdf.empty

    def test_init_with_gdf_and_filename(self, valid_bbox_gdf):
        """Test creating instance with both GDF and custom filename."""
        custom_filename = "test_area.geojson"
        area = Shoreline_Extraction_Area(gdf=valid_bbox_gdf, filename=custom_filename)

        assert area.filename == custom_filename
        assert not area.gdf.empty
        assert len(area.gdf) == 1

    def test_init_with_different_crs_gdf(self, different_crs_polygon_gdf):
        """Test creating instance with GDF in different CRS gets converted."""
        area = Shoreline_Extraction_Area(gdf=different_crs_polygon_gdf)

        # Check CRS was converted to default EPSG:4326
        assert area.gdf.crs.to_string() == "EPSG:4326"
        assert not area.gdf.empty
        # Geometry should be preserved but reprojected
        assert area.gdf.geometry.iloc[0].geom_type == "Polygon"

    def test_init_with_multiple_polygons(self, multi_polygon_gdf):
        """Test creating instance with multiple polygons."""
        area = Shoreline_Extraction_Area(gdf=multi_polygon_gdf)

        assert len(area.gdf) == 2
        assert all(geom.geom_type == "Polygon" for geom in area.gdf.geometry)

    def test_init_with_invalid_geometry_raises_error(self, linestring_gdf):
        """Test that invalid geometry types raise InvalidGeometryType error."""
        with pytest.raises(InvalidGeometryType):
            Shoreline_Extraction_Area(gdf=linestring_gdf)

    def test_init_with_empty_gdf(self):
        """Test creating instance with empty GeoDataFrame."""
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        area = Shoreline_Extraction_Area(gdf=empty_gdf)

        assert area.gdf.empty
        assert area.gdf.crs.to_string() == "EPSG:4326"


class TestShorelineExtractionAreaStyleLayer:
    """Test style_layer method of Shoreline_Extraction_Area."""

    def test_style_layer_with_geojson_dict(self, sample_polygon_geojson):
        """Test style_layer with GeoJSON dictionary."""
        area = Shoreline_Extraction_Area()
        layer_name = "test_layer"

        result = area.style_layer(sample_polygon_geojson, layer_name)

        # Check return type is GeoJSON layer
        assert isinstance(result, GeoJSON)
        assert result.name == layer_name
        # Check custom purple styling is applied
        assert result.style["color"] == "#cb42f5"
        assert result.style["fill_color"] == "#cb42f5"
        assert result.style["opacity"] == 1
        assert result.style["fillOpacity"] == 0.1
        assert result.style["weight"] == 3

    def test_style_layer_with_gdf(self, valid_bbox_gdf):
        """Test style_layer with GeoDataFrame."""
        area = Shoreline_Extraction_Area(gdf=valid_bbox_gdf)
        layer_name = "gdf_layer"

        result = area.style_layer(area.gdf, layer_name)

        assert isinstance(result, GeoJSON)
        assert result.name == layer_name
        # Check styling properties are correctly applied
        assert result.style["color"] == "#cb42f5"
        assert result.style["opacity"] == 1

    def test_style_layer_inherits_from_parent(self, valid_bbox_gdf):
        """Test that style_layer calls parent class with correct parameters."""
        area = Shoreline_Extraction_Area(gdf=valid_bbox_gdf)
        layer_name = "test_inheritance"

        result = area.style_layer(area.gdf, layer_name)

        # Verify the layer was created (indicates parent method was called)
        assert isinstance(result, GeoJSON)
        assert result.name == layer_name
        # Verify custom style was applied (not default parent style)
        assert result.style["color"] == "#cb42f5"  # Not default "#555555"

    def test_style_layer_with_empty_geojson_raises_error(self):
        """Test that empty GeoJSON raises ValueError."""
        area = Shoreline_Extraction_Area()
        layer_name = "empty_layer"
        empty_geojson = {}

        with pytest.raises(ValueError, match="Empty .* geojson cannot be drawn"):
            area.style_layer(empty_geojson, layer_name)


class TestShorelineExtractionAreaProperties:
    """Test class properties and constants."""

    def test_layer_name_constant(self):
        """Test that LAYER_NAME constant is correctly set."""
        assert Shoreline_Extraction_Area.LAYER_NAME == "shoreline_extraction_area"

    def test_inherits_from_feature(self):
        """Test that class properly inherits from Feature."""
        area = Shoreline_Extraction_Area()

        # Check inheritance attributes
        assert hasattr(area, "DEFAULT_CRS")
        assert hasattr(area, "FILE_EXT")
        assert area.DEFAULT_CRS == "EPSG:4326"
        assert area.FILE_EXT == ".geojson"

    def test_gdf_columns_after_initialization(self, valid_bbox_gdf):
        """Test that only geometry column is kept after cleaning."""
        # Add extra column to test cleaning
        valid_bbox_gdf["extra_column"] = "test_value"

        area = Shoreline_Extraction_Area(gdf=valid_bbox_gdf)

        # Only geometry column should remain
        assert list(area.gdf.columns) == ["geometry"]
        assert "extra_column" not in area.gdf.columns


class TestShorelineExtractionAreaEdgeCases:
    """Test edge cases and error conditions."""

    def test_with_none_gdf_creates_empty_instance(self):
        """Test passing None as gdf creates empty instance."""
        area = Shoreline_Extraction_Area(gdf=None)

        assert area.gdf.empty
        assert area.gdf.crs.to_string() == "EPSG:4326"

    def test_with_mixed_geometry_types_filters_valid_only(self):
        """Test that mixed geometry types get filtered to only valid polygons."""
        # Create mixed geometry GDF
        polygon = Polygon(
            [
                (-122.5, 37.5),
                (-122.4, 37.5),
                (-122.4, 37.6),
                (-122.5, 37.6),
                (-122.5, 37.5),
            ]
        )
        line = LineString([(-122.5, 37.5), (-122.4, 37.6)])
        point = Point(-122.45, 37.55)

        mixed_gdf = gpd.GeoDataFrame(
            {"geometry": [polygon, line, point]}, crs="EPSG:4326"
        )

        # Should raise InvalidGeometryType due to invalid geometries
        with pytest.raises(InvalidGeometryType):
            Shoreline_Extraction_Area(gdf=mixed_gdf)

    def test_with_no_crs_gdf_gets_default_crs(self):
        """Test that GDF without CRS gets default CRS assigned."""
        polygon = Polygon(
            [
                (-122.5, 37.5),
                (-122.4, 37.5),
                (-122.4, 37.6),
                (-122.5, 37.6),
                (-122.5, 37.5),
            ]
        )
        no_crs_gdf = gpd.GeoDataFrame({"geometry": [polygon]})  # No CRS specified

        area = Shoreline_Extraction_Area(gdf=no_crs_gdf)

        assert area.gdf.crs.to_string() == "EPSG:4326"
        assert not area.gdf.empty
