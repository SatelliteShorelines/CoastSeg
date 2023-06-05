import json
import pytest
from coastseg import bbox
from coastseg import exceptions
import geopandas as gpd
from ipyleaflet import GeoJSON


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
                [-104.05, 48.850]
            ]
        ]
    }

@pytest.fixture
def sample_crs():
    return "EPSG:4326"

# Test cases
def test_create_geodataframe_not_empty(sample_geojson):
    new_bbox=bbox.Bounding_Box(rectangle =sample_geojson )
    result = new_bbox.gdf
    assert not result.empty, "The GeoDataFrame is empty."

def test_bbox_no_z_coordinates(sample_geojson):
    actaul_bbox = bbox.Bounding_Box(rectangle=sample_geojson)
    
    for _, row in actaul_bbox.gdf.iterrows():
        geom = row.geometry
        for coord in geom.exterior.coords:
            assert len(coord) == 2, "Z-coordinates should be removed."

def test_bbox_no_id_column(sample_geojson, ):
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


def test_invalid_Boundary_Box():
    with pytest.raises(Exception):
        bbox.Bounding_Box([])
    with pytest.raises(Exception):
        bbox.Bounding_Box()


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
