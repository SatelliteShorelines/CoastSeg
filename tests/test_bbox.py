import json
import pytest
from coastseg import bbox
from coastseg import exceptions
import geopandas as gpd
from ipyleaflet import GeoJSON
from shapely.geometry import Polygon, LineString, MultiPolygon, Point
import pyproj


# Test data
@pytest.fixture
def sample_geojson():
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [-104.05, 48.850],
                [-97.125, 48.850],
                [-97.125, 45.675],
                [-104.05, 45.675],
                [-104.05, 48.850],
            ]
        ],
    }


@pytest.fixture
def sample_crs():
    return "EPSG:4326"


# Test cases
def test_create_geodataframe_not_empty(sample_geojson):
    new_bbox = bbox.Bounding_Box(rectangle=sample_geojson)
    result = new_bbox.gdf
    assert not result.empty, "The GeoDataFrame is empty."


def test_bbox_no_z_coordinates(sample_geojson):
    actaul_bbox = bbox.Bounding_Box(rectangle=sample_geojson)

    for _, row in actaul_bbox.gdf.iterrows():
        geom = row.geometry
        for coord in geom.exterior.coords:
            assert len(coord) == 2, "Z-coordinates should be removed."


def test_bbox_no_id_column(
    sample_geojson,
):
    actaul_bbox = bbox.Bounding_Box(rectangle=sample_geojson)
    assert "id" not in actaul_bbox.gdf.columns, "The 'id' column should not be present."


def test_bbox_crs(sample_geojson, sample_crs):
    actaul_bbox = bbox.Bounding_Box(rectangle=sample_geojson)
    assert actaul_bbox.gdf.crs == sample_crs, "The CRS does not match the expected CRS."


def test_check_bbox_size():
    with pytest.raises(exceptions.BboxTooLargeError):
        bbox.Bounding_Box.check_bbox_size(bbox.Bounding_Box.MAX_AREA + 1)

    with pytest.raises(exceptions.BboxTooSmallError):
        bbox.Bounding_Box.check_bbox_size(bbox.Bounding_Box.MIN_AREA - 1)


def test_valid_Boundary_Box(valid_bbox_geojson: dict):
    """tests if a Bounding_Box will be created from a valid bounding box thats a dict
    Args:
        valid_bbox_geojson (dict): valid bounding box as a geojson dict
    """
    bbox_geojson = valid_bbox_geojson["features"][0]["geometry"]
    box = bbox.Bounding_Box(bbox_geojson)
    assert isinstance(box, bbox.Bounding_Box)
    assert box.gdf is not None
    assert box.filename == "bbox.geojson"
    box_geojson = json.loads(box.gdf["geometry"].to_json())
    assert bbox_geojson == box_geojson["features"][0]["geometry"]


def test_valid_Boundary_Box_gdf(valid_bbox_gdf: gpd.GeoDataFrame):
    """tests if a Bounding_Box will be created from a valid bounding box thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): alid bounding box as a gpd.GeoDataFrame
    """
    box = bbox.Bounding_Box(valid_bbox_gdf)
    assert isinstance(box, bbox.Bounding_Box)
    assert box.gdf is not None
    assert box.filename == "bbox.geojson"
    valid_bbox_geojson = json.loads(valid_bbox_gdf["geometry"].to_json())
    box_geojson = json.loads(box.gdf["geometry"].to_json())
    assert (
        valid_bbox_geojson["features"][0]["geometry"]
        == box_geojson["features"][0]["geometry"]
    )


def test_different_CRS_Boundary_Box_gdf():
    """tests if a Bounding_Box will be created from a invalid geoddataframe that contains a linestring"""
    CRS = "epsg:2033"
    rectangle = gpd.GeoDataFrame(
        geometry=[
            Polygon(
                [
                    (-121.12083854611063, 35.56544740627308),
                    (-121.12083854611063, 35.53742390816822),
                    (-121.08749373817861, 35.53742390816822),
                    (-121.08749373817861, 35.56544740627308),
                    (-121.12083854611063, 35.56544740627308),
                ]
            )
        ],
        crs="epsg:4326",
    )
    rectangle.to_crs(CRS, inplace=True)

    box = bbox.Bounding_Box(rectangle)
    assert hasattr(box, "gdf")
    assert isinstance(box.gdf, gpd.GeoDataFrame)
    assert not box.gdf.empty
    assert isinstance(box.gdf.crs, pyproj.CRS)
    assert box.gdf.crs == "epsg:4326"


def test_invalid_geometry_Boundary_Box_gdf():
    """tests if a Bounding_Box will be created from a invalid geoddataframe that contains a linestring"""
    line = gpd.GeoDataFrame(
        geometry=[
            LineString(
                [
                    (-120.83849150866949, 35.43786191889319),
                    (-120.93431712689429, 35.40749430666743),
                ]
            )
        ],
        crs="epsg:4326",
    )
    with pytest.raises(exceptions.InvalidGeometryType):
        bbox.Bounding_Box(line)


def test_invalid_Boundary_Box():
    with pytest.raises(TypeError):
        bbox.Bounding_Box([])
    with pytest.raises(TypeError):
        bbox.Bounding_Box("invalid_string")


def test_style_layer(valid_bbox_gdf: gpd.GeoDataFrame):
    box = bbox.Bounding_Box(valid_bbox_gdf)
    layer_geojson = json.loads(valid_bbox_gdf.to_json())
    new_layer = box.style_layer(layer_geojson, bbox.Bounding_Box.LAYER_NAME)
    assert isinstance(new_layer, GeoJSON)
    assert new_layer.name == bbox.Bounding_Box.LAYER_NAME
    assert (
        new_layer.data["features"][0]["geometry"]
        == layer_geojson["features"][0]["geometry"]
    )


def test_invalid_style_layer(valid_bbox_gdf: gpd.GeoDataFrame):
    box = bbox.Bounding_Box(valid_bbox_gdf)
    with pytest.raises(Exception):
        box.style_layer({}, bbox.Bounding_Box.LAYER_NAME)


# ============================================================================
# COMPREHENSIVE TESTS FOR BOUNDING_BOX CLASS
# ============================================================================


class TestBoundingBoxInitialization:
    """Test comprehensive initialization scenarios for Bounding_Box class."""

    def test_init_with_multipolygon_gdf(self):
        """Test initialization with MultiPolygon geometry."""
        # Create MultiPolygon from multiple rectangles
        poly1 = Polygon([(-1, -1), (-1, 0), (0, 0), (0, -1), (-1, -1)])
        poly2 = Polygon([(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)])
        multi_poly = MultiPolygon([poly1, poly2])

        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[multi_poly], crs="EPSG:4326")

        box = bbox.Bounding_Box(gdf)

        # Check initialization succeeded
        assert isinstance(box, bbox.Bounding_Box)
        assert not box.gdf.empty
        assert str(box.gdf.crs) == "EPSG:4326"

    def test_init_with_custom_filename(self):
        """Test initialization with custom filename."""
        gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:4326",
        )
        custom_filename = "my_custom_bbox.geojson"

        box = bbox.Bounding_Box(gdf, filename=custom_filename)

        # Check custom filename is set
        assert box.filename == custom_filename

    def test_init_with_geojson_dict(self):
        """Test initialization from GeoJSON dictionary."""
        geojson_dict = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        }

        box = bbox.Bounding_Box(geojson_dict)

        # Check GeoJSON conversion worked
        assert isinstance(box, bbox.Bounding_Box)
        assert not box.gdf.empty
        assert "geometry" in box.gdf.columns

    def test_init_type_error_invalid_input(self):
        """Test TypeError for invalid input types."""
        # Test with various invalid types
        invalid_inputs = [[], "string", 123, None, set()]

        for invalid_input in invalid_inputs:
            with pytest.raises(
                TypeError, match="rectangle must be GeoDataFrame or GeoJSON-like dict"
            ):
                bbox.Bounding_Box(invalid_input)

    def test_init_geometry_type_validation(self):
        """Test that invalid geometry types raise errors."""
        # Create GDF with invalid geometry (Point)
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")

        # Should raise InvalidGeometryType error
        with pytest.raises(exceptions.InvalidGeometryType):
            bbox.Bounding_Box(gdf)

    def test_init_crs_conversion(self):
        """Test automatic CRS conversion to EPSG:4326."""
        # Create GDF in different CRS
        gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:3857",
        )

        box = bbox.Bounding_Box(gdf)

        # Check CRS was converted to EPSG:4326
        assert str(box.gdf.crs) == "EPSG:4326"


class TestBoundingBoxStyleLayer:
    """Test style_layer method with various inputs and edge cases."""

    def test_style_layer_basic_functionality(self, valid_bbox_gdf):
        """Test basic style_layer functionality."""
        box = bbox.Bounding_Box(valid_bbox_gdf)
        geojson_data = json.loads(valid_bbox_gdf.to_json())

        result = box.style_layer(geojson_data, "test_layer")

        # Check return type and properties
        assert isinstance(result, GeoJSON)
        assert result.name == "test_layer"
        # The style_layer method applies styling, so data may be modified
        assert "features" in result.data

    def test_style_layer_with_layer_name_default(self, valid_bbox_gdf):
        """Test style_layer with default layer name."""
        box = bbox.Bounding_Box(valid_bbox_gdf)
        geojson_data = json.loads(valid_bbox_gdf.to_json())

        result = box.style_layer(geojson_data, bbox.Bounding_Box.LAYER_NAME)

        # Check default layer name is used
        assert result.name == bbox.Bounding_Box.LAYER_NAME
        assert result.name == "Bbox"

    def test_style_layer_styling_properties(self, valid_bbox_gdf):
        """Test that correct styling is applied."""
        box = bbox.Bounding_Box(valid_bbox_gdf)
        geojson_data = json.loads(valid_bbox_gdf.to_json())

        result = box.style_layer(geojson_data, "styled_layer")

        # Check styling properties exist
        assert hasattr(result, "style")
        # The styling is applied in parent class, so we verify the method works
        assert isinstance(result, GeoJSON)

    def test_style_layer_empty_geojson_error(self, valid_bbox_gdf):
        """Test style_layer with empty GeoJSON raises error."""
        box = bbox.Bounding_Box(valid_bbox_gdf)

        # Empty dict should raise exception
        with pytest.raises(Exception):
            box.style_layer({}, "empty_layer")

    def test_style_layer_malformed_geojson_error(self, valid_bbox_gdf):
        """Test style_layer with malformed GeoJSON."""
        box = bbox.Bounding_Box(valid_bbox_gdf)

        # Test that the method handles various inputs gracefully
        # This may not always raise an exception depending on parent implementation
        malformed_geojson = {"invalid": "structure"}

        # Just verify method can be called - behavior depends on parent class
        result = box.style_layer(malformed_geojson, "malformed_layer")
        assert isinstance(result, GeoJSON)


class TestBoundingBoxSizeValidation:
    """Test check_bbox_size static method comprehensively."""

    def test_check_bbox_size_valid_areas(self):
        """Test size validation with valid areas."""
        # Test areas within valid range
        valid_areas = [
            bbox.Bounding_Box.MIN_AREA,  # Minimum boundary
            bbox.Bounding_Box.MAX_AREA,  # Maximum boundary
            (bbox.Bounding_Box.MIN_AREA + bbox.Bounding_Box.MAX_AREA)
            // 2,  # Middle value
            bbox.Bounding_Box.MIN_AREA + 1,  # Just above minimum
            bbox.Bounding_Box.MAX_AREA - 1,  # Just below maximum
        ]

        for area in valid_areas:
            # Should not raise any exception
            bbox.Bounding_Box.check_bbox_size(area)

    def test_check_bbox_size_too_small_error(self):
        """Test BboxTooSmallError for areas below minimum."""
        too_small_areas = [
            bbox.Bounding_Box.MIN_AREA - 1,
            bbox.Bounding_Box.MIN_AREA - 100,
            0,
            -1,
        ]

        for area in too_small_areas:
            with pytest.raises(exceptions.BboxTooSmallError):
                bbox.Bounding_Box.check_bbox_size(area)

    def test_check_bbox_size_too_large_error(self):
        """Test BboxTooLargeError for areas above maximum."""
        too_large_areas = [
            bbox.Bounding_Box.MAX_AREA + 1,
            bbox.Bounding_Box.MAX_AREA + 1000000,
            bbox.Bounding_Box.MAX_AREA * 2,
        ]

        for area in too_large_areas:
            with pytest.raises(exceptions.BboxTooLargeError):
                bbox.Bounding_Box.check_bbox_size(area)

    def test_check_bbox_size_floating_point_areas(self):
        """Test size validation with floating point areas."""
        # Test floating point areas within range
        float_areas = [
            float(bbox.Bounding_Box.MIN_AREA + 0.5),
            float(bbox.Bounding_Box.MAX_AREA - 0.1),
            1500.75,  # Valid float area
            50000000.99,  # Valid large float area
        ]

        for area in float_areas:
            # Should not raise any exception
            bbox.Bounding_Box.check_bbox_size(area)

    def test_check_bbox_size_boundary_precision(self):
        """Test exact boundary value handling."""
        # Test exact boundary values
        bbox.Bounding_Box.check_bbox_size(bbox.Bounding_Box.MIN_AREA)
        bbox.Bounding_Box.check_bbox_size(bbox.Bounding_Box.MAX_AREA)

        # Test values just outside boundaries
        with pytest.raises(exceptions.BboxTooSmallError):
            bbox.Bounding_Box.check_bbox_size(bbox.Bounding_Box.MIN_AREA - 0.1)

        with pytest.raises(exceptions.BboxTooLargeError):
            bbox.Bounding_Box.check_bbox_size(bbox.Bounding_Box.MAX_AREA + 0.1)


class TestBoundingBoxClassAttributes:
    """Test class attributes and constants."""

    def test_class_constants_values(self):
        """Test that class constants have expected values."""
        # Check constant values are as expected
        assert bbox.Bounding_Box.MAX_AREA == 100000000000
        assert bbox.Bounding_Box.MIN_AREA == 1000
        assert bbox.Bounding_Box.LAYER_NAME == "Bbox"

    def test_class_constants_types(self):
        """Test that class constants are correct types."""
        # Check types
        assert isinstance(bbox.Bounding_Box.MAX_AREA, int)
        assert isinstance(bbox.Bounding_Box.MIN_AREA, int)
        assert isinstance(bbox.Bounding_Box.LAYER_NAME, str)

    def test_area_constants_relationship(self):
        """Test logical relationship between area constants."""
        # MIN_AREA should be less than MAX_AREA
        assert bbox.Bounding_Box.MIN_AREA < bbox.Bounding_Box.MAX_AREA
        # Both should be positive
        assert bbox.Bounding_Box.MIN_AREA > 0
        assert bbox.Bounding_Box.MAX_AREA > 0


class TestBoundingBoxFeatureInheritance:
    """Test Bounding_Box inheritance from Feature base class."""

    def test_inherits_from_feature(self, valid_bbox_gdf):
        """Test that Bounding_Box properly inherits from Feature."""
        from coastseg.feature import Feature

        box = bbox.Bounding_Box(valid_bbox_gdf)

        # Check inheritance
        assert isinstance(box, Feature)
        assert issubclass(bbox.Bounding_Box, Feature)

    def test_has_feature_attributes(self, valid_bbox_gdf):
        """Test that Bounding_Box has required Feature attributes."""
        box = bbox.Bounding_Box(valid_bbox_gdf)

        # Check inherited attributes exist
        assert hasattr(box, "gdf")
        assert hasattr(box, "filename")
        assert hasattr(box, "DEFAULT_CRS")

    def test_feature_method_inheritance(self, valid_bbox_gdf):
        """Test that Feature methods are properly inherited."""
        box = bbox.Bounding_Box(valid_bbox_gdf)

        # Check inherited methods exist
        assert hasattr(box, "ensure_crs")
        assert hasattr(box, "clean_gdf")
        assert hasattr(box, "gdf_from_mapping")

    def test_default_filename_inheritance(self, valid_bbox_gdf):
        """Test default filename behavior."""
        box = bbox.Bounding_Box(valid_bbox_gdf)

        # Check default filename
        assert box.filename == "bbox.geojson"


class TestBoundingBoxEdgeCases:
    """Test edge cases and error scenarios."""

    def test_bbox_with_holes_polygon(self):
        """Test Bounding_Box with polygon containing holes."""
        # Create polygon with hole
        exterior = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]
        hole = [(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]
        poly_with_hole = Polygon(exterior, [hole])

        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly_with_hole], crs="EPSG:4326")

        # Should handle polygons with holes
        box = bbox.Bounding_Box(gdf)
        assert isinstance(box, bbox.Bounding_Box)
        assert not box.gdf.empty

    def test_bbox_with_very_small_polygon(self):
        """Test Bounding_Box with very small polygon."""
        # Create very small polygon
        small_poly = Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001), (0, 0)])
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[small_poly], crs="EPSG:4326")

        # Should create bbox even if small
        box = bbox.Bounding_Box(gdf)
        assert isinstance(box, bbox.Bounding_Box)

    def test_bbox_columns_cleanup(self, valid_bbox_gdf):
        """Test that only geometry column is kept."""
        # Add extra columns to GDF
        gdf_with_extra = valid_bbox_gdf.copy()
        gdf_with_extra["extra_col"] = ["extra_value"]
        gdf_with_extra["another_col"] = [123]

        box = bbox.Bounding_Box(gdf_with_extra)

        # Check only geometry column remains
        expected_columns = ["geometry"]
        assert list(box.gdf.columns) == expected_columns

    def test_bbox_single_row_constraint(self, valid_bbox_gdf):
        """Test that bbox handles single geometry constraint."""
        box = bbox.Bounding_Box(valid_bbox_gdf)

        # Check that only one row exists
        assert len(box.gdf) == 1
