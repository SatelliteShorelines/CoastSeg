import json
import os
from types import SimpleNamespace
from unittest.mock import patch
from coastseg import shoreline
from coastseg import transects
from coastseg import roi
from coastseg import exceptions
from coastseg import bbox
from coastseg import file_utilities
from coastseg import coastseg_map
from coastseg import common
from leafmap import Map
import pytest
import geopandas as gpd
from ipyleaflet import GeoJSON
from shapely.geometry import Point


def test_imports():
    """Test that all the internal coastseg packages are imported correctly"""
    from coastseg import exceptions
    from coastseg import exception_handler
    from coastseg import extracted_shoreline
    from coastseg import factory
    from coastseg import map_UI
    from coastseg import models_UI


def test_init_coastseg_map_no_map():
    coastsegmap = coastseg_map.CoastSeg_Map(create_map=False)
    assert coastsegmap.map == None


def test_get_roi_ids_no_rois():
    """
    Test case to verify the behavior of get_roi_ids() method when there are no ROIs present.
    """
    coastsegmap = coastseg_map.CoastSeg_Map()
    assert coastsegmap.get_roi_ids() == []


def test_get_roi_ids_with_rois(valid_coastseg_map_with_settings, valid_rois_filepath):
    """
    Test the get_roi_ids() method of CoastSegMap class when ROIs are loaded onto the map.

    Args:
        valid_coastseg_map_with_settings (CoastSegMap): A valid instance of CoastSegMap class.
        valid_rois_filepath (str): The filepath of the valid ROIs file.
    """
    actual_coastsegmap = valid_coastseg_map_with_settings
    # test if rois will be correctly loaded onto map
    actual_coastsegmap.load_feature_on_map("rois", file=valid_rois_filepath)
    assert actual_coastsegmap.get_roi_ids() == ["17", "30", "35"]


def test_save_config_invalid_inputs(
    valid_coastseg_map,
    valid_coastseg_map_with_incomplete_settings,
    valid_coastseg_map_with_settings,
    coastseg_map_with_rois,
    valid_rois_filepath,
):
    with pytest.raises(Exception):
        valid_coastseg_map.save_config()

    # test if exception is raised when settings is missing ["dates", "sat_list", "landsat_collection"]
    # save config will not work without ROIs loaded onto map
    valid_coastseg_map_with_incomplete_settings.load_feature_on_map(
        "rois", file=valid_rois_filepath
    )

    with pytest.raises(Exception):
        valid_coastseg_map_with_incomplete_settings.save_config()

    # test if exception is raised when coastseg_map missing rois
    with pytest.raises(Exception):
        # save config will not work without ROIs loaded onto map
        valid_coastseg_map_with_settings.save_config()


def test_load_feature_on_map_fail_load_default_shorelines(box_no_shorelines_transects):
    """Fail to load default shorelines if the bbox doesn't contain any shorelines"""
    coastsegmap = coastseg_map.CoastSeg_Map(create_map=False)
    coastsegmap.bbox = bbox.Bounding_Box(box_no_shorelines_transects)
    # attempt to load default shorelines on the map ( it should fail)
    with pytest.raises(exceptions.Object_Not_Found):
        coastsegmap.load_feature_on_map("shorelines")


def test_load_feature_on_map_fail_load_default_transects(box_no_shorelines_transects):
    """Fail to load default transects if the bbox doesn't contain any transects"""
    coastsegmap = coastseg_map.CoastSeg_Map(create_map=False)
    coastsegmap.bbox = bbox.Bounding_Box(box_no_shorelines_transects)
    # attempt to load default transects on the map ( it should fail)
    with pytest.raises(exceptions.Object_Not_Found):
        coastsegmap.load_feature_on_map("transects")


def test_load_feature_on_map_map_off(valid_bbox_gdf):
    """Fail to load default transects if the bbox doesn't contain any transects"""
    coastsegmap = coastseg_map.CoastSeg_Map(create_map=False)
    coastsegmap.bbox = bbox.Bounding_Box(valid_bbox_gdf)
    # attempt to load default transects on the map
    coastsegmap.load_feature_on_map("transects")
    # attempt to load default shorelines on the map
    coastsegmap.load_feature_on_map("transects")


def test_load_feature_on_map_map_on(valid_bbox_gdf):
    """Fail to load default transects if the bbox doesn't contain any transects"""
    coastsegmap = coastseg_map.CoastSeg_Map(create_map=True)
    coastsegmap.bbox = bbox.Bounding_Box(valid_bbox_gdf)
    # attempt to load default transects on the map
    coastsegmap.load_feature_on_map("transects")
    # attempt to load default shorelines on the map
    coastsegmap.load_feature_on_map("transects")


def test_save_config(coastseg_map_with_selected_roi_layer, tmp_path):
    """tests if save configs will save both a config.json and
    config_gdf.geojson to the filepath directory when coastseg_map's rois have roi_settings.
    Args:
        coastseg_map_with_selected_roi_layer (Coastseg_Map): instance of CoastSeg_Map with settings loaded, rois loaded,
                                                        and ROI with id 17 selected on map
                                                        ROIs on map have ids:["17","30","35"]
    Selected ROIs have id:["17"]
        tmp_path (WindowsPath): temporary directory
    """
    actual_coastsegmap = coastseg_map_with_selected_roi_layer
    filepath = str(tmp_path)
    roi_id = "17"
    date_str = "01-31-22_12_19_45"

    # modify the settings
    settings = actual_coastsegmap.get_settings()
    dates = settings["dates"]
    landsat_collection = settings["landsat_collection"]
    sat_list = settings["sat_list"]

    # Add roi_settings to  actual_coastsegmap.rois
    selected_layer = actual_coastsegmap.map.find_layer(roi.ROI.SELECTED_LAYER_NAME)
    roi_settings = common.create_roi_settings(
        settings,
        selected_layer.data,
        filepath,
        date_str,
    )
    actual_coastsegmap.rois.set_roi_settings(roi_settings)
    assert actual_coastsegmap.rois.roi_settings != {}
    # use the roi_settings to save config
    actual_coastsegmap.save_config(filepath)
    assert actual_coastsegmap.rois.roi_settings != {}
    expected_config_json_path = tmp_path / "config.json"
    assert expected_config_json_path.exists()
    with open(expected_config_json_path, "r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    # test if roi id was saved as key and key fields exist
    assert roi_id in data
    assert "dates" in data[roi_id]
    assert dates == data[roi_id]["dates"]
    assert "sat_list" in data[roi_id]
    assert sat_list == data[roi_id]["sat_list"]
    assert "roi_id" in data[roi_id]
    assert roi_id == data[roi_id]["roi_id"]
    assert "polygon" in data[roi_id]
    assert "landsat_collection" in data[roi_id]
    assert landsat_collection in data[roi_id]["landsat_collection"]
    assert "sitename" in data[roi_id]
    assert date_str in data[roi_id]["sitename"]
    assert "filepath" in data[roi_id]
    assert filepath == data[roi_id]["filepath"]
    expected_config_geojson_path = tmp_path / "config_gdf.geojson"
    assert expected_config_geojson_path.exists()


@pytest.mark.parametrize("named_temp_dir", [("CoastSeg", None)], indirect=True)
def test_save_config_empty_roi_settings(
    coastseg_map_with_selected_roi_layer, named_temp_dir
):
    """test_save_config_empty_roi_settings tests if save configs will save both a config.json and
    config_gdf.geojson to the filepath directory when coastseg_map's rois do not have roi_settings.
    It should also create roi_settings for coastseg_map's rois

    Args:
        coastseg_map_with_selected_roi_layer (Coastseg_Map): instance of CoastSeg_Map with settings loaded, rois loaded,
                                                        and ROI with id 17 selected on map
                                                        ROIs on map have ids:["17","30","35"]
    Selected ROIs have id:["17"]
        tmp_path (WindowsPath): temporary directory
    """
    # The named_temp_dir fixture created a temporary directory named 'CoastSeg'
    tmp_CoastSeg_path = named_temp_dir
    actual_coastsegmap = coastseg_map_with_selected_roi_layer
    assert actual_coastsegmap.rois.roi_settings == {}
    if type(tmp_CoastSeg_path) != str:
        filepath = str(tmp_CoastSeg_path)
    else:
        filepath = tmp_CoastSeg_path
    roi_id = "17"
    actual_coastsegmap.save_config(filepath)
    # roi_settings was empty before. save_config should have created it
    assert actual_coastsegmap.rois.roi_settings != {}
    expected_config_json_path = os.path.join(tmp_CoastSeg_path, "config.json")
    assert os.path.exists(expected_config_json_path)
    with open(expected_config_json_path, "r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    # test if roi id was saved as key and key fields exist
    assert roi_id in data
    assert "dates" in data[roi_id]
    assert "sat_list" in data[roi_id]
    assert "roi_id" in data[roi_id]
    assert "polygon" in data[roi_id]
    assert "landsat_collection" in data[roi_id]
    assert "sitename" in data[roi_id]
    assert "filepath" in data[roi_id]
    expected_config_geojson_path = os.path.join(tmp_CoastSeg_path, "config_gdf.geojson")
    assert os.path.exists(expected_config_geojson_path)


def test_load_json_config_without_rois(valid_coastseg_map_with_settings, tmp_data_path):
    # test if exception is raised when coastseg_map has no ROIs
    actual_coastsegmap = valid_coastseg_map_with_settings
    with pytest.raises(Exception):
        actual_coastsegmap.load_json_config("")


def test_load_json_config_downloaded(
    valid_coastseg_map_with_settings,
    valid_rois_filepath,
    config_json,
):
    config_path, temp_dir = config_json
    # tests if load_json_config will load contents into rois.roi_settings
    # create instance of Coastseg_Map with settings and ROIs initially loaded
    actual_coastsegmap = valid_coastseg_map_with_settings
    actual_coastsegmap.load_feature_on_map("rois", file=valid_rois_filepath)

    # test if settings are correctly loaded when valid json config loaded with 'filepath' & 'sitename' keys is loaded
    json_data = actual_coastsegmap.load_json_config(config_path)
    actual_coastsegmap.rois.roi_settings = common.process_roi_settings(
        json_data, temp_dir
    )

    assert isinstance(actual_coastsegmap.rois.roi_settings, dict)
    actual_config = file_utilities.read_json_file(config_path)
    for roi_id in actual_config["roi_ids"]:
        assert roi_id in actual_coastsegmap.rois.roi_settings
        for key in actual_config[roi_id]:
            assert (
                actual_coastsegmap.rois.roi_settings[roi_id][key]
                == actual_config[roi_id][key]
            )
    for roi_id, item in actual_config.get("settings", {}).items():
        assert actual_coastsegmap.settings[roi_id] == item


def test_valid_shoreline_gdf(valid_shoreline_gdf: gpd.GeoDataFrame):
    """tests if a Shoreline will be created from a valid shoreline thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): a valid shoreline as a gpd.GeoDataFrame
    """
    expected_shoreline = shoreline.Shoreline(shoreline=valid_shoreline_gdf)
    assert isinstance(expected_shoreline, shoreline.Shoreline)
    assert expected_shoreline.gdf is not None
    assert expected_shoreline.filename == "shoreline.geojson"


def test_valid_transects_gdf(valid_transects_gdf: gpd.GeoDataFrame):
    """tests if a Transects will be created from a valid transects thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): valid transects as a gpd.GeoDataFrame
    """
    expected_transects = transects.Transects(transects=valid_transects_gdf)
    assert isinstance(expected_transects, transects.Transects)
    assert expected_transects.gdf is not None
    assert expected_transects.filename == "transects.geojson"


def test_transect_compatible_roi(transect_compatible_roi: gpd.GeoDataFrame):
    """tests if a ROI will be created from valid rois thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): alid rois as a gpd.GeoDataFrame
    """
    expected_roi = roi.ROI(rois_gdf=transect_compatible_roi)
    assert isinstance(expected_roi, roi.ROI)
    assert expected_roi.gdf is not None
    assert expected_roi.filename == "rois.geojson"


def test_transect_compatible_roi(transect_compatible_roi: gpd.GeoDataFrame):
    """tests if a ROI will be created from valid rois thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): alid rois as a gpd.GeoDataFrame
    """
    expected_roi = roi.ROI(rois_gdf=transect_compatible_roi)
    assert isinstance(expected_roi, roi.ROI)
    assert expected_roi.gdf is not None
    assert expected_roi.filename == "rois.geojson"


def test_coastseg_map():
    """tests a CoastSeg_Map object is created"""
    coastsegmap = coastseg_map.CoastSeg_Map()
    assert isinstance(coastsegmap, coastseg_map.CoastSeg_Map)
    assert isinstance(coastsegmap.map, Map)
    assert hasattr(coastsegmap, "draw_control")
    assert hasattr(coastsegmap, "settings")
    default_settings = {
        "landsat_collection": "C02",
        "dates": ["2017-12-01", "2018-01-01"],
        "sat_list": ["L8"],
        "cloud_thresh": 0.8,
        "percent_no_data": 0.8,
        "dist_clouds": 300,
        "output_epsg": 4326,
        "check_detection": False,
        "adjust_detection": False,
        "save_figure": True,
        "min_beach_area": 4500,
        "min_length_sl": 100,
        "cloud_mask_issue": False,
        "sand_color": "default",
        "pan_off": "False",
        "max_dist_ref": 25,
        "along_dist": 25,
        "min_points": 3,
        "max_std": 15,
        "max_range": 30,
        "min_chainage": -100,
        "multiple_inter": "auto",
        "prc_multiple": 0.1,
        "apply_cloud_mask": True,
        "image_size_filter": True,
    }
    for key in default_settings:
        assert key in coastsegmap.settings
        assert coastsegmap.settings[key] == default_settings[key]


def test_set_settings():
    """tests if a ROI will be created from valid rois thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): alid rois as a gpd.GeoDataFrame
    """
    coastsegmap = coastseg_map.CoastSeg_Map()
    pre_process_settings = {
        # general parameters:
        "dates": ["2018-12-01", "2019-03-01"],
        "sat_list": ["L9"],
        "cloud_thresh": 0.9,  # threshold on maximum cloud cover
        "dist_clouds": 400,  # ditance around clouds where shoreline can't be mapped
        "output_epsg": 3857,  # epsg code of spatial reference system desired for the output
        # quality control:
        "check_detection": True,  # if True, shows each shoreline detection to the user for validation
        "adjust_detection": False,  # if True, allows user to adjust the position of each shoreline by changing the threshold
        "save_figure": True,  # if True, saves a figure showing the mapped shoreline for each image
        # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
        "min_beach_area": 400,  # minimum area (in metres^2) for an object to be labelled as a beach
        "min_length_sl": 100,  # minimum length (in metres) of shoreline perimeter to be valid
        "cloud_mask_issue": True,  # switch this parameter to True if sand pixels are masked (in black) on many images
        "sand_color": "default",  # 'default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
        "pan_off": "False",  # if True, no pan-sharpening is performed on Landsat 7,8 and 9 imagery
        "max_dist_ref": 20,
        "landsat_collection": "C02",
    }
    coastsegmap.set_settings(**pre_process_settings)
    actual_settings = set(list(coastsegmap.get_settings().keys()))
    expected_settings = set(list(pre_process_settings.keys()))
    assert expected_settings.issubset(actual_settings)
    assert set(["dates", "landsat_collection", "sat_list"]).issubset(actual_settings)
    for key in pre_process_settings:
        assert coastsegmap.settings[key] == pre_process_settings[key]


def test_select_roi_layer(
    valid_coastseg_map_with_settings,
    valid_rois_filepath,
):
    """tests if a ROI will be added to selected layer when clicked
    Simulates an ROI being clicked on map by manually adding ROI id to selected_set
    and creating a new layer
    Args:
        valid_coastseg_map_with_settings (Coastseg_Map): valid instance of coastseg map with settings already loaded
        valid_rois_filepath (str): filepath to geojson file containing valid rois
                                    ROIs with ids:[17,30,35]
    """
    actual_coastsegmap = valid_coastseg_map_with_settings
    # test if rois will added to coastsegmap and added to ROI layer
    actual_coastsegmap.load_feature_on_map("rois", file=valid_rois_filepath)
    # test if roi layer was added to map
    existing_layer = actual_coastsegmap.map.find_layer(roi.ROI.LAYER_NAME)
    assert existing_layer is not None
    # simulate an ROI being clicked on map
    ROI_id = "17"
    actual_coastsegmap.selected_set.add(ROI_id)

    selected_layer = GeoJSON(
        data=actual_coastsegmap.convert_selected_set_to_geojson(
            actual_coastsegmap.selected_set, layer_name=roi.ROI.LAYER_NAME
        ),
        name=roi.ROI.SELECTED_LAYER_NAME,
        hover_style={"fillColor": "blue", "fillOpacity": 0.1, "color": "aqua"},
    )
    actual_coastsegmap.replace_layer_by_name(
        roi.ROI.SELECTED_LAYER_NAME,
        selected_layer,
        on_click=actual_coastsegmap.selected_onclick_handler,
        on_hover=actual_coastsegmap.update_roi_html,
    )
    # test if roi layer was added to map
    selected_layer = actual_coastsegmap.map.find_layer(roi.ROI.SELECTED_LAYER_NAME)
    assert selected_layer is not None
    existing_layer = actual_coastsegmap.map.find_layer("Selected ROIs")
    assert existing_layer is not None
    assert "17" in actual_coastsegmap.selected_set
    assert isinstance(selected_layer.data, dict)
    assert isinstance(selected_layer.data["features"], list)
    assert isinstance(selected_layer.data["features"][0], dict)
    assert len(selected_layer.data["features"]) == 1
    roi_json = actual_coastsegmap.rois.gdf[
        actual_coastsegmap.rois.gdf["id"] == ROI_id
    ].to_json()
    roi_geojson = json.loads(roi_json)
    assert isinstance(roi_geojson, dict)
    # test if geojson in selected layer matches geojson in coastsegmap.rois.gdf
    assert (
        roi_geojson["features"][0]["geometry"]
        == selected_layer.data["features"][0]["geometry"]
    )


def test_load_rois_on_map_with_file(
    valid_coastseg_map_with_settings, valid_rois_filepath, valid_rois_gdf
):
    """tests if a ROI will be created from geojson file and added to the map
    Args:
        valid_coastseg_map_with_settings (Coastseg_Map): valid instance of coastseg map with settings already loaded
        valid_rois_filepath (str): filepath to geojson file containing valid rois
    """
    actual_coastsegmap = valid_coastseg_map_with_settings
    # test if rois will be correctly loaded onto map
    actual_coastsegmap.load_feature_on_map("rois", file=valid_rois_filepath)
    assert actual_coastsegmap.rois is not None
    assert isinstance(actual_coastsegmap.rois, roi.ROI)
    # test if rois geodataframe was created correctly
    assert isinstance(actual_coastsegmap.rois.gdf, gpd.GeoDataFrame)
    assert actual_coastsegmap.rois.gdf.equals(valid_rois_gdf)
    # test if roi layer was added to map
    existing_layer = actual_coastsegmap.map.find_layer(roi.ROI.LAYER_NAME)
    assert existing_layer is not None


def test_load_feature_on_map_generate_rois(valid_bbox_gdf):
    coastsegmap = coastseg_map.CoastSeg_Map()
    # if no bounding box loaded on map this should raise an error
    with pytest.raises(exceptions.Object_Not_Found):
        coastsegmap.load_feature_on_map(
            "rois",
            lg_area=20,
            sm_area=0,
            units="km²",
        )
    # load bbox on map
    coastsegmap.load_feature_on_map("bbox", gdf=valid_bbox_gdf)

    # Mock shoreline creation to avoid file downloads
    mock_shoreline_gdf = gpd.GeoDataFrame(
        {
            "id": ["test_shoreline_1"],
            "geometry": [valid_bbox_gdf.geometry.iloc[0]],
        },
        crs="EPSG:4326",
    )

    with patch.object(
        shoreline.Shoreline,
        "get_intersecting_shoreline_files",
        return_value=["test_file.geojson"],
    ):
        with patch.object(
            shoreline.Shoreline, "create_geodataframe", return_value=mock_shoreline_gdf
        ):
            # now that bbox is loaded on map, this should work
            # this will automatically load a shoreline within the bbox
            coastsegmap.load_feature_on_map(
                "rois",
                lg_area=20,
                sm_area=0,
                units="km²",
            )
            assert coastsegmap.rois is not None


def test_load_feature_on_map_rois_without_shorelines(box_no_shorelines_transects):
    coastsegmap = coastseg_map.CoastSeg_Map()
    # load bbox on map where no default shorelines are available
    coastsegmap.load_feature_on_map("bbox", gdf=box_no_shorelines_transects)
    # attempt to load rois within the bbox where no shorelines are available
    coastsegmap.load_feature_on_map(
        "rois",
        lg_area=20,
        sm_area=0,
        units="km²",
    )
    # validate that the rois exists
    assert coastsegmap.rois is not None
    assert coastsegmap.bbox is not None
    assert coastsegmap.shoreline is None


def test_load_feature_on_map_rois_custom(box_no_shorelines_transects):
    # this box has no default shorelines available but it can still load because its a custom ROI
    coastsegmap = coastseg_map.CoastSeg_Map()
    coastsegmap.load_feature_on_map("rois", gdf=box_no_shorelines_transects)


def test_update_loadable_shorelines_mixed_date_formats(coastseg_map_with_rois):
    """Test mixed date parsing and fallback behavior in loadable shoreline labels."""
    coastsegmap = coastseg_map_with_rois
    selected_id = "17"

    extracted_shorelines_gdf = gpd.GeoDataFrame(
        {
            "date": [
                "2021-01-01",
                "2021-01-02 05:30:00",
                "not-a-date",
            ],
            "satname": ["L8", "S2", "L9"],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs="EPSG:4326",
    )
    # Use a SimpleNamespace to mimic the expected structure of the extracted shoreline object
    # This allows us to set the gdf attribute directly without needing to create a full Shoreline object
    extracted_shoreline_obj = SimpleNamespace(gdf=extracted_shorelines_gdf)
    coastsegmap.rois.extracted_shorelines[selected_id] = extracted_shoreline_obj

    returned_shoreline_obj = coastsegmap.update_loadable_shorelines(selected_id)

    assert returned_shoreline_obj is extracted_shoreline_obj
    assert coastsegmap.extract_shorelines_container.trash_list == []

    # validate the load list contains the original labels since all dates should be parsed successfully with fallback to original string when parsing fails
    load_list = set(coastsegmap.extract_shorelines_container.load_list)
    assert "L8_2021-01-01 00:00:00" in load_list
    assert "S2_2021-01-02 05:30:00" in load_list
    assert "L9_not-a-date" in load_list

    assert list(returned_shoreline_obj.gdf["date"]) == sorted(
        list(returned_shoreline_obj.gdf["date"])
    )


def test_extract_all_shorelines_refreshes_display_after_updating_roi_ids(
    coastseg_map_with_rois,
):
    """Ensure post-extraction flow refreshes shoreline display after ROI IDs are updated."""
    coastsegmap = coastseg_map_with_rois
    roi_id = "17"

    extracted_shorelines_gdf = gpd.GeoDataFrame(
        {
            "date": ["2021-01-01 00:00:00"],
            "satname": ["L8"],
            "geometry": [Point(0, 0)],
        },
        crs="EPSG:4326",
    )
    extracted_shoreline_obj = SimpleNamespace(gdf=extracted_shorelines_gdf)

    coastsegmap.shoreline = SimpleNamespace(
        gdf=gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")
    )
    coastsegmap.transects = SimpleNamespace(gdf=gpd.GeoDataFrame())
    coastsegmap.set_session_name("test_session")

    with patch.object(coastsegmap, "validate_extract_shoreline_inputs"):
        with patch.object(coastsegmap, "get_missing_directories", return_value={}):
            with patch.object(coastsegmap, "create_session", return_value="session"):
                with patch.object(
                    coastsegmap,
                    "extract_shoreline_for_roi",
                    return_value=extracted_shoreline_obj,
                ):
                    with patch.object(coastsegmap, "save_session"):
                        with patch.object(coastsegmap, "compute_transects"):
                            with patch.object(
                                coastsegmap, "update_extracted_shorelines_display"
                            ) as display_mock:
                                coastsegmap.extract_all_shorelines(roi_ids=[roi_id])

    # The inner per-ROI loop calls update_extracted_shorelines_display once.
    # The second refresh comes via the on_roi_selected widget callback chain at runtime
    # (not exercised here because no widget is linked in the test fixture).
    assert display_mock.call_count == 1
    assert display_mock.call_args_list[0].args[0] == roi_id


def test_roi_extracted_shoreline_lookup_normalizes_roi_id(coastseg_map_with_rois):
    """Ensure extracted shoreline lookup works for string and integer ROI IDs."""
    coastsegmap = coastseg_map_with_rois
    extracted_shoreline_obj = SimpleNamespace(gdf=gpd.GeoDataFrame())

    coastsegmap.rois.add_extracted_shoreline(extracted_shoreline_obj, 17)

    assert coastsegmap.rois.get_extracted_shoreline("17") is extracted_shoreline_obj
    assert coastsegmap.rois.get_extracted_shoreline(17) is extracted_shoreline_obj


def test_validate_extract_shoreline_inputs_allows_missing_transects(
    coastseg_map_with_rois,
):
    """Issue #249: transects are optional.

    validate_extract_shoreline_inputs must not raise when self.transects is None -
    only ROIs and shorelines are required. Before the fix the transects
    validate_feature() call raised Object_Not_Found on a None transects layer.
    """
    coastsegmap = coastseg_map_with_rois
    coastsegmap.shoreline = SimpleNamespace(
        gdf=gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")
    )
    coastsegmap.transects = None
    coastsegmap.rois.roi_settings = {"17": {}}
    coastsegmap.set_session_name("test_session")

    settings = {
        "dates": ["2020-01-01", "2020-12-31"],
        "sat_list": ["L8"],
        "landsat_collection": "C02",
    }

    with (
        patch.object(coastsegmap, "get_settings", return_value=settings),
        patch.object(coastsegmap, "get_roi_ids", return_value=["17"]),
        patch("coastseg.coastseg_map.exception_handler.check_empty_dict"),
        patch("coastseg.coastseg_map.exception_handler.check_if_subset"),
        patch("coastseg.coastseg_map.exception_handler.check_if_dirs_missing"),
        patch("coastseg.coastseg_map.common.get_missing_roi_dirs", return_value={}),
    ):
        # Should complete without raising Object_Not_Found for the missing transects.
        coastsegmap.validate_extract_shoreline_inputs(roi_ids=["17"])
