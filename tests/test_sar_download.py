"""Tests for getting Sentinel-1 imagery onto disk and describing it.

Three parts:

1. the download itself and the resume of a partly-downloaded site,
2. the ``sentinel_1_properties`` settings and the polarization folder layout,
3. reading ``band_order`` back out of a scene's metadata file.

Earth Engine is never contacted.
"""

import json
import os

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from coastsat import SDS_download

from coastseg import coastseg_map, common
from coastseg import extracted_shoreline as es


# ===========================================================================
# Download and resume
#
# Tests for the Sentinel-1 download and the resume of a partly-downloaded site.
#
# The download itself lives in coastsat; CoastSeg only builds the ``inputs`` block and calls
# ``SDS_download.retrieve_images``. These tests pin that boundary:
#
# 1. an S1 download asks for both polarizations, and an explicit choice is not overwritten,
# 2. a site missing a polarization reaches coastsat untouched -- CoastSeg does not intercept
#    or warn, because coastsat backfills it during this same call and reports that itself,
# 3. re-running such a site re-queues the incomplete scene and backfills VV in place --
#    same sitename, no duplicate.
#
# Earth Engine is never contacted: ``retrieve_images`` is replaced by a capturing stub.
# ===========================================================================


@pytest.fixture
def captured_downloads(monkeypatch):
    """Replace coastsat's downloader with a stub that records what it was handed."""
    calls = []
    monkeypatch.setattr(
        SDS_download,
        "retrieve_images",
        lambda inputs, **kwargs: calls.append(inputs),
    )
    return calls


@pytest.fixture
def one_roi_gdf():
    """A single ROI, the shape CoastSeg_Map.download_imagery expects."""
    return gpd.GeoDataFrame(
        {"id": ["roi1"]},
        geometry=[
            Polygon(
                [
                    (-122.50, 37.70),
                    (-122.40, 37.70),
                    (-122.40, 37.80),
                    (-122.50, 37.80),
                ]
            )
        ],
        crs="EPSG:4326",
    )


def download_one_roi(monkeypatch, roi_gdf, destination, **settings):
    """Run CoastSeg_Map.download_imagery for one S1 ROI without writing a config."""
    coastseg_object = coastseg_map.CoastSeg_Map(create_map=False)
    monkeypatch.setattr(coastseg_object, "save_config", lambda *a, **k: None)
    resolved = coastseg_object.set_settings(
        sat_list=["S1"],
        dates=["2023-01-01", "2023-06-01"],
        landsat_collection="C02",
        **settings,
    )
    coastseg_object.download_imagery(
        rois=roi_gdf,
        settings=resolved,
        selected_ids={"roi1"},
        file_path=str(destination),
    )


# ---------------------------------------------------------------------------
# Both polarizations are requested
# ---------------------------------------------------------------------------


def test_download_imagery_asks_coastsat_for_both_polarizations(
    monkeypatch, tmp_path, one_roi_gdf, captured_downloads
):
    """The SAR segmentation model needs VV, VH and their difference.

    A download that quietly asks for one polarization produces a site the model cannot
    read, and the failure only shows up much later at extraction time.
    """
    download_one_roi(monkeypatch, one_roi_gdf, tmp_path)

    assert len(captured_downloads) == 1
    inputs = captured_downloads[0]
    assert inputs["sat_list"] == ["S1"]
    assert inputs["sentinel_1_properties"]["transmitterReceiverPolarisation"] == [
        "VV",
        "VH",
    ]


def test_an_explicit_polarization_choice_is_not_overwritten(
    monkeypatch, tmp_path, one_roi_gdf, captured_downloads
):
    """A user asking for VV only must not have the VV+VH default put back."""
    download_one_roi(
        monkeypatch,
        one_roi_gdf,
        tmp_path,
        sentinel_1_properties={
            "transmitterReceiverPolarisation": ["VV"],
            "instrumentMode": "IW",
        },
    )

    inputs = captured_downloads[0]
    assert inputs["sentinel_1_properties"]["transmitterReceiverPolarisation"] == ["VV"]


# ---------------------------------------------------------------------------
# An incomplete site reaches coastsat intact
# ---------------------------------------------------------------------------


def test_a_vh_only_site_is_handed_straight_to_coastsat(
    sar_legacy_site_with_meta, captured_downloads
):
    """A site missing a polarization must not be intercepted on the way to the download.

    CoastSeg used to warn here first. It no longer does: coastsat re-queues the
    incomplete scenes and backfills them in place during this very call, and reports
    that itself in terms of scenes rather than polarization folders. Anything CoastSeg
    printed beforehand would have told the user to re-run the download they had already
    started.
    """
    inputs = sar_legacy_site_with_meta["inputs"]

    coastseg_map.CoastSeg_Map(create_map=False).retrieve_images_for_rois([inputs], {})

    assert len(captured_downloads) == 1
    assert captured_downloads[0] is inputs


def test_the_incomplete_site_is_left_on_disk_untouched(
    sar_legacy_site_with_meta, captured_downloads
):
    """Nothing on the way to the download may delete the bands already downloaded.

    The backfill overwrites scenes in place under the same sitename, so the VH already
    on disk has to still be there when coastsat takes over.
    """
    coastseg_map.CoastSeg_Map(create_map=False).retrieve_images_for_rois(
        [sar_legacy_site_with_meta["inputs"]], {}
    )

    assert os.path.isfile(sar_legacy_site_with_meta["vh"])
    assert os.path.isfile(sar_legacy_site_with_meta["meta"])


# ---------------------------------------------------------------------------
# Resuming a VH-only download
# ---------------------------------------------------------------------------


def test_a_vh_only_scene_is_re_queued_for_download(sar_legacy_site_with_meta):
    """coastsat must not report the scene as already downloaded just because the date is
    on disk -- that is what left legacy sites single-polarization forever."""
    inputs = sar_legacy_site_with_meta["inputs"]
    metadata = es.get_metadata(inputs)

    assert metadata["S1"]["band_order"] == [["VH"]]

    incomplete = SDS_download.get_incomplete_sar_dates(
        inputs, metadata["S1"], ["VV", "VH"]
    )

    assert len(incomplete) == 1


def test_backfilling_vv_clears_the_incomplete_scene(sar_legacy_site_with_meta):
    """Once VV lands next to VH the scene is complete and must not be fetched again."""
    inputs = sar_legacy_site_with_meta["inputs"]
    sar_legacy_site_with_meta["add_vv"]()
    metadata = es.get_metadata(inputs)

    incomplete = SDS_download.get_incomplete_sar_dates(
        inputs, metadata["S1"], ["VV", "VH"]
    )

    assert incomplete == set()


def test_the_resumed_site_keeps_its_sitename_and_gains_no_duplicate(
    sar_legacy_site_with_meta,
):
    """The backfill overwrites the scene in place: one file per polarization, no _dup
    suffix, and both polarizations present."""
    sar_legacy_site_with_meta["add_vv"]()

    s1_root = os.path.join(sar_legacy_site_with_meta["site"], "S1")
    for polarization in ("VV", "VH"):
        tifs = [
            name
            for name in os.listdir(os.path.join(s1_root, polarization))
            if name.endswith(".tif")
        ]
        assert len(tifs) == 1
        assert "_dup" not in tifs[0]
        assert tifs[0].startswith(sar_legacy_site_with_meta["scene_date"])


# ===========================================================================
# Download settings and on-disk layout
#
# Tests for the Sentinel-1 download settings and on-disk layout.
#
# The SAR model needs both VV and VH, so these pin that the default requests both, that
# the setting survives a config round-trip, and that the polarization folders are found.
# ===========================================================================


def make_selected_rois(*roi_ids):
    return {
        "features": [
            {
                "properties": {"id": roi_id},
                "geometry": {"coordinates": [[[0, 0], [0, 1], [1, 1], [0, 0]]]},
            }
            for roi_id in roi_ids
        ]
    }


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_sentinel_1_properties_request_both_polarizations():
    properties = common.get_default_sentinel_1_properties()
    assert properties["transmitterReceiverPolarisation"] == ["VV", "VH"]
    assert properties["instrumentMode"] == "IW"


def test_create_roi_settings_defaults_to_vv_and_vh():
    roi_settings = common.create_roi_settings(
        settings={"sat_list": ["S1"], "dates": ["2023-05-01", "2023-06-01"]},
        selected_rois=make_selected_rois("1"),
        filepath="/data",
        date_str="01-01-25__00_00_00",
    )
    assert roi_settings["1"]["sentinel_1_properties"][
        "transmitterReceiverPolarisation"
    ] == ["VV", "VH"]


def test_create_roi_settings_gives_each_roi_its_own_properties_dict():
    """A shared mutable dict would let an edit to one ROI silently change every ROI."""
    roi_settings = common.create_roi_settings(
        settings={"sat_list": ["S1"], "dates": ["2023-05-01", "2023-06-01"]},
        selected_rois=make_selected_rois("1", "2"),
        filepath="/data",
        date_str="01-01-25__00_00_00",
    )

    first = roi_settings["1"]["sentinel_1_properties"]
    second = roi_settings["2"]["sentinel_1_properties"]
    assert first is not second

    first["transmitterReceiverPolarisation"].append("HH")
    assert second["transmitterReceiverPolarisation"] == ["VV", "VH"]


def test_create_roi_settings_honours_an_explicit_override():
    roi_settings = common.create_roi_settings(
        settings={
            "sat_list": ["S1"],
            "dates": ["2023-05-01", "2023-06-01"],
            "sentinel_1_properties": {
                "transmitterReceiverPolarisation": ["VH"],
                "instrumentMode": "IW",
            },
        },
        selected_rois=make_selected_rois("1"),
        filepath="/data",
        date_str="01-01-25__00_00_00",
    )
    assert roi_settings["1"]["sentinel_1_properties"][
        "transmitterReceiverPolarisation"
    ] == ["VH"]


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


def test_extract_roi_settings_keeps_sentinel_1_properties():
    """Without this the setting is dropped whenever a session config is reloaded."""
    properties = {
        "transmitterReceiverPolarisation": ["VV", "VH"],
        "instrumentMode": "IW",
    }
    json_data = {
        "roi_ids": ["1"],
        "1": {
            "dates": ["2023-05-01", "2023-06-01"],
            "sitename": "ID_1_datetime01-01-25__00_00_00",
            "polygon": [[[0, 0]]],
            "roi_id": "1",
            "sat_list": ["S1"],
            "landsat_collection": "C02",
            "filepath": "/data",
            "sentinel_1_properties": properties,
        },
    }

    roi_settings = common.extract_roi_settings(json_data)
    assert roi_settings["1"]["sentinel_1_properties"] == properties


def test_load_settings_keeps_sentinel_1_properties(tmp_path):
    properties = {
        "transmitterReceiverPolarisation": ["VV", "VH"],
        "instrumentMode": "IW",
    }
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"sat_list": ["S1"], "sentinel_1_properties": properties}),
        encoding="utf-8",
    )

    settings = common.load_settings(str(config))
    assert settings["sentinel_1_properties"] == properties


def test_global_settings_upgrade_a_legacy_roi_without_the_key():
    """Legacy ROI settings have no sentinel_1_properties key at all.

    update_roi_settings skips keys that are absent, so without add_if_missing the ROI
    would stay on coastsat's VH-only fallback forever.
    """
    roi_settings = {"1": {"sat_list": ["S1"], "dates": ["2023-05-01", "2023-06-01"]}}
    global_settings = {
        "sat_list": ["S1"],
        "dates": ["2023-05-01", "2023-06-01"],
        "sentinel_1_properties": {
            "transmitterReceiverPolarisation": ["VV", "VH"],
            "instrumentMode": "IW",
        },
    }

    updated = common.update_roi_settings_with_global_settings(
        roi_settings, global_settings
    )
    assert updated["1"]["sentinel_1_properties"][
        "transmitterReceiverPolarisation"
    ] == ["VV", "VH"]


def test_global_settings_propagation_does_not_alias_the_global_dict():
    global_settings = {
        "sentinel_1_properties": {"transmitterReceiverPolarisation": ["VV", "VH"]}
    }
    roi_settings = {"1": {"sat_list": ["S1"]}}

    updated = common.update_roi_settings_with_global_settings(
        roi_settings, global_settings
    )
    updated["1"]["sentinel_1_properties"]["transmitterReceiverPolarisation"].append("HH")

    assert global_settings["sentinel_1_properties"][
        "transmitterReceiverPolarisation"
    ] == ["VV", "VH"]


# ---------------------------------------------------------------------------
# On-disk layout
# ---------------------------------------------------------------------------


def test_get_filepath_returns_every_polarization_in_canonical_order(tmp_path):
    site = "ID_test_datetime01-01-25__00_00_00"
    # created out of order on purpose: the result must not depend on listdir order
    os.makedirs(tmp_path / site / "S1" / "VH")
    os.makedirs(tmp_path / site / "S1" / "VV")

    paths = es.get_filepath(str(tmp_path), site, "S1")

    assert [os.path.basename(path) for path in paths] == ["VV", "VH"]


def test_get_filepath_on_a_legacy_vh_only_site(tmp_path):
    site = "ID_legacy_datetime01-01-20__00_00_00"
    os.makedirs(tmp_path / site / "S1" / "VH")

    paths = es.get_filepath(str(tmp_path), site, "S1")

    assert [os.path.basename(path) for path in paths] == ["VH"]


def test_get_filepath_when_nothing_is_downloaded_yet(tmp_path):
    """Keep the legacy single-element shape so callers fail the way they always did."""
    paths = es.get_filepath(str(tmp_path), "ID_missing", "S1")
    assert [os.path.basename(path) for path in paths] == ["VH"]


# ===========================================================================
# band_order in the scene metadata
#
# Tests for reading Sentinel-1 `band_order` out of a scene's metadata .txt file.
#
# coastsat writes one metadata file per S1 scene carrying the polarizations it downloaded.
# CoastSeg keeps its own copy of `get_metadata`, which used to drop that key entirely.
# coastsat reads it back when it cannot list the files on disk -- for example to tell a
# complete scene from one still missing a band -- so it has to survive the round trip and
# stay index-aligned with `filenames`.
# ===========================================================================


LEGACY_META = """\
filename\t2024-08-02-08-39-41_S1_site_VH.tif
epsg\t32756
im_width\t168
im_height\t436
orbitProperties_pass\tASCENDING
transmitterReceiverPolarisation\t['VV', 'VH']
saved_polarization\tVH
instrumentMode\tIW
"""

DUAL_POL_META = """\
filename\t2023-12-03-19-15-49_S1_site_VV.tif
epsg\t32756
im_width\t168
im_height\t436
band_order\t['VV', 'VH']
orbitProperties_pass\tDESCENDING
transmitterReceiverPolarisation\t['VV', 'VH']
instrumentMode\tIW
"""


def write_meta(directory, filename, contents):
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(contents)
    return path


@pytest.fixture
def s1_site(tmp_path):
    """A site whose meta folder the caller fills in."""
    meta_dir = tmp_path / "site" / "S1" / "meta"
    meta_dir.mkdir(parents=True)
    return tmp_path, meta_dir


def read_s1_metadata(root, dates=("2020-01-01", "2030-01-01")):
    return es.get_metadata(
        {
            "filepath": str(root),
            "sitename": "site",
            "sat_list": ["S1"],
            "dates": list(dates),
        }
    )


# ---------------------------------------------------------------------------
# read_metadata_file
# ---------------------------------------------------------------------------


def test_band_order_is_parsed_into_a_list(tmp_path):
    path = write_meta(str(tmp_path), "scene.txt", DUAL_POL_META)

    assert es.read_metadata_file(path)["band_order"] == ["VV", "VH"]


def test_a_legacy_file_has_no_band_order(tmp_path):
    path = write_meta(str(tmp_path), "scene.txt", LEGACY_META)
    meta_info = es.read_metadata_file(path)

    assert meta_info["band_order"] == []
    assert meta_info["saved_polarization"] == "VH"


def test_a_malformed_band_order_does_not_break_the_read(tmp_path, caplog):
    """band_order is optional; a bad value must not take down the whole file."""
    path = write_meta(
        str(tmp_path), "scene.txt", DUAL_POL_META.replace("['VV', 'VH']", "[VV, VH")
    )

    with caplog.at_level("WARNING"):
        meta_info = es.read_metadata_file(path)

    assert meta_info["band_order"] == []
    assert meta_info["epsg"] == 32756  # the rest of the file still parsed
    assert "band_order" in caplog.text


# ---------------------------------------------------------------------------
# get_band_order_from_meta
# ---------------------------------------------------------------------------


def test_band_order_wins_over_saved_polarization():
    resolved = es.get_band_order_from_meta(
        {"band_order": ["VV", "VH"], "saved_polarization": "VH"}
    )
    assert resolved == ["VV", "VH"]


def test_a_legacy_scene_falls_back_to_saved_polarization():
    assert es.get_band_order_from_meta({"saved_polarization": "VH"}) == ["VH"]


def test_neither_key_resolves_to_empty():
    assert es.get_band_order_from_meta({"filename": "x.tif"}) == []


def test_the_requested_polarizations_are_not_mistaken_for_the_saved_ones():
    """transmitterReceiverPolarisation is what was asked for, not what landed on disk --
    a legacy file carries ['VV','VH'] there while holding only VH."""
    resolved = es.get_band_order_from_meta(
        {"transmitterReceiverPolarisation": ["VV", "VH"], "saved_polarization": "VH"}
    )
    assert resolved == ["VH"]


# ---------------------------------------------------------------------------
# get_metadata
# ---------------------------------------------------------------------------


def test_get_metadata_records_band_order_for_s1(s1_site):
    root, meta_dir = s1_site
    write_meta(str(meta_dir), "2023-12-03-19-15-49_S1_site.txt", DUAL_POL_META)

    metadata = read_s1_metadata(root)

    assert metadata["S1"]["band_order"] == [["VV", "VH"]]


def test_band_order_stays_index_aligned_on_a_mixed_site(s1_site):
    """A legacy VH-only scene sitting next to a new dual-pol one. band_order[i] must
    stay safe to index alongside filenames[i]."""
    root, meta_dir = s1_site
    write_meta(str(meta_dir), "2023-12-03-19-15-49_S1_site.txt", DUAL_POL_META)
    write_meta(str(meta_dir), "2024-08-02-08-39-41_S1_site_VH.txt", LEGACY_META)

    s1 = read_s1_metadata(root)["S1"]

    assert len(s1["band_order"]) == len(s1["filenames"]) == 2
    assert s1["band_order"] == [["VV", "VH"], ["VH"]]


def test_optical_satellites_do_not_gain_a_band_order_key(tmp_path):
    meta_dir = tmp_path / "site" / "L8" / "meta"
    meta_dir.mkdir(parents=True)
    write_meta(
        str(meta_dir),
        "2023-12-03-19-15-49_L8_site.txt",
        "filename\t2023-12-03-19-15-49_L8_site_ms.tif\nepsg\t32756\n",
    )

    metadata = es.get_metadata(
        {
            "filepath": str(tmp_path),
            "sitename": "site",
            "sat_list": ["L8"],
            "dates": ["2020-01-01", "2030-01-01"],
        }
    )

    assert "band_order" not in metadata["L8"]
