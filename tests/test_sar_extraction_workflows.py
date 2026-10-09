"""Tests for extracting Sentinel-1 shorelines in the zoo and CoastSat workflows.

Three parts:

1. both workflows reach the right segmenter and extract from its output,
2. the SAR settings the CoastSat workflow hands to coastsat,
3. the warnings raised when SAR shorelines fall back to Otsu thresholding.

Nothing here loads the real ~130 MB ONNX model.
"""

import os

import numpy as np
import pytest
from coastsat import SDS_shoreline, SDS_tools

from coastseg import coastseg_map, common, sar_model, zoo_model
from coastseg import extracted_shoreline as es


# ===========================================================================
# Both workflows
#
# Tests that Sentinel-1 shorelines can be extracted in both workflows.
#
# The two workflows reach segmentation by different routes:
#
# * **zoo** -- ``Zoo_Model.run_model`` segments first and writes ``*_res.npz``, then
#   ``extract_shorelines_with_dask`` traces contours from the class labels. The SAR branch
#   runs onnxruntime in-process; the optical branch shells out to the TensorFlow
#   environment. Picking the wrong one is silent until it fails much later.
# * **coastsat** -- ``Extracted_Shoreline.extract_shorelines`` hands everything to
#   ``SDS_shoreline.extract_shorelines``, which segments S1 itself.
#
# Nothing here loads the 130 MB ONNX model, launches the TensorFlow subprocess, or reads a
# real scene: the two segmenters and coastsat's extractor are replaced by capturing stubs.
# ===========================================================================


# ---------------------------------------------------------------------------
# Zoo workflow: which segmenter runs
# ---------------------------------------------------------------------------


@pytest.fixture
def routed_segmenters(monkeypatch):
    """Stub both segmentation backends and record which one was called."""
    calls = {}

    def fake_sar_segmentation(**kwargs):
        calls["sar"] = kwargs
        return {"written_npz": 1, "total_images": 1, "failures": []}

    def fake_external_segmentation(**kwargs):
        calls["external"] = kwargs

    monkeypatch.setattr(sar_model, "run_sar_segmentation", fake_sar_segmentation)
    monkeypatch.setattr(
        zoo_model, "run_segmentation_in_external_env", fake_external_segmentation
    )
    return calls


def model_directory(tmp_path, name, weights):
    """A model directory identified only by the extension of the file inside it."""
    directory = tmp_path / name
    directory.mkdir()
    (directory / weights).write_bytes(b"")
    return str(directory)


def run_model(monkeypatch, directory, roi_directory="roi"):
    """Invoke Zoo_Model.run_model against a directory, skipping model resolution."""
    model = zoo_model.Zoo_Model()
    monkeypatch.setattr(model, "resolve_model_directory", lambda setting: directory)
    model.run_model(
        {}, "sample_directory", "session_directory", roi_directory=roi_directory
    )


def test_a_sar_model_directory_routes_to_the_onnx_driver(
    monkeypatch, tmp_path, routed_segmenters
):
    """A directory holding an .onnx is a SAR model and runs in-process."""
    directory = model_directory(tmp_path, "sar_model_dir", "SAR_3_band_model.onnx")

    run_model(monkeypatch, directory, roi_directory="roi_data")

    assert "external" not in routed_segmenters
    assert routed_segmenters["sar"]["roi_directory"] == "roi_data"
    assert routed_segmenters["sar"]["model_directory"] == directory


def test_an_optical_model_directory_routes_to_the_tensorflow_env(
    monkeypatch, tmp_path, routed_segmenters
):
    """The complement: no .onnx means the SegFormer path in the separate environment.

    Together with the test above this pins is_sar_model_directory as the whole
    discriminator -- there is no img_type flag consulted here.
    """
    directory = model_directory(tmp_path, "segformer_dir", "weights.h5")

    run_model(monkeypatch, directory)

    assert "sar" not in routed_segmenters
    assert routed_segmenters["external"]["model_path"] == directory


def test_the_sar_path_requires_the_roi_directory(
    monkeypatch, tmp_path, routed_segmenters
):
    """The SAR model reads the S1 GeoTIFFs, not the jpg previews the optical path uses,
    so a missing ROI directory has to fail loudly rather than segment nothing."""
    directory = model_directory(tmp_path, "sar_model_dir", "model.onnx")

    with pytest.raises(ValueError, match="ROI directory"):
        run_model(monkeypatch, directory, roi_directory="")

    assert routed_segmenters == {}


# ---------------------------------------------------------------------------
# Zoo workflow: reaching the SAR model without naming its directory
# ---------------------------------------------------------------------------


@pytest.fixture
def designated_sar_directory(monkeypatch, tmp_path):
    """Point CoastSeg's designated SAR directory somewhere empty and unwritten.

    Absent, not merely empty: the model is 130 MB and downloaded on first use, so a
    fresh clone has no such directory at all.
    """
    directory = tmp_path / "models" / sar_model.DEFAULT_SAR_MODEL_DIRNAME
    monkeypatch.setattr(
        zoo_model.sar_model, "get_sar_model_directory", lambda: str(directory)
    )
    monkeypatch.setattr(
        zoo_model.sar_model, "load_sar_model_spec", lambda directory: None
    )
    return directory


def test_selecting_sar_without_a_local_path_uses_the_designated_directory(
    monkeypatch, designated_sar_directory
):
    """``img_type='SAR'`` alone is enough to reach the model.

    'SAR' names an image type, not a Zoo model id, so falling through to the Zoo
    downloader looks ``model_type`` up on Zenodo and fails with a 404 -- an error that
    says nothing about SAR. The UI never sees this because it fills in
    'local_model_path' itself; a script that sets only 'img_type' does.
    """

    def fail(*args, **kwargs):  # pragma: no cover - only runs if the branch is wrong
        raise AssertionError("SAR must not be resolved through the Zoo downloader")

    model = zoo_model.Zoo_Model()
    monkeypatch.setattr(model, "ensure_model_downloaded", fail)

    resolved = model.resolve_model_directory(
        {"img_type": "SAR", "model_type": "SAR_segmentation_model"}
    )

    assert resolved == str(designated_sar_directory)


def test_the_sar_directory_resolves_before_the_model_is_downloaded(
    designated_sar_directory,
):
    """An absent SAR directory is the fresh-clone state, not a configuration error.

    ``is_sar_model_directory`` deliberately answers True here so the first run routes
    to the ONNX path and downloads the model. Rejecting the path for not existing yet
    would make that download unreachable.
    """
    assert not designated_sar_directory.exists()

    model = zoo_model.Zoo_Model()
    resolved = model.resolve_model_directory(
        {"img_type": "SAR", "local_model_path": str(designated_sar_directory)}
    )

    assert resolved == str(designated_sar_directory)


def test_a_missing_optical_model_path_is_still_an_error(tmp_path):
    """The tolerance above is scoped to SAR; a typo'd optical path must still raise."""
    missing = tmp_path / "not_a_real_model_directory"

    model = zoo_model.Zoo_Model()
    with pytest.raises(FileNotFoundError, match="local_model_path"):
        model.resolve_model_directory(
            {"img_type": "RGB", "local_model_path": str(missing)}
        )


def test_a_missing_non_sar_directory_is_an_error_even_when_sar_is_selected(tmp_path):
    """Selecting SAR does not excuse a path that is not a SAR directory at all."""
    missing = tmp_path / "some_other_model"

    model = zoo_model.Zoo_Model()
    with pytest.raises(FileNotFoundError, match="local_model_path"):
        model.resolve_model_directory(
            {"img_type": "SAR", "local_model_path": str(missing)}
        )


# ---------------------------------------------------------------------------
# Zoo workflow: where the class mapping comes from
# ---------------------------------------------------------------------------


@pytest.fixture
def sar_spec(monkeypatch):
    """Stub the ONNX spec so no 130 MB model is loaded."""

    class Spec:
        classes = {0: "land", 1: "water"}

    monkeypatch.setattr(sar_model, "load_sar_model_spec", lambda directory: Spec())
    return Spec


@pytest.mark.parametrize(
    "contents",
    [
        pytest.param(["SAR_3_band_model.onnx"], id="downloaded-onnx-only"),
        pytest.param(["my_retrained.onnx"], id="user-supplied-onnx"),
    ],
)
def test_a_sar_model_directory_takes_its_classes_from_the_onnx(
    monkeypatch, tmp_path, sar_spec, contents
):
    """A SAR model directory has no ``*modelcard.json``, and must not be given the
    optical default mapping because of it.

    The model is fetched as a bare ``.onnx`` into coastsat's cache, and a user supplying
    their own drops in one file, so neither case ever produces a model card. Falling back
    to the 4-class Zoo mapping sets ``water_class_indices`` to ``[0, 1]``, which on SAR's
    ``{0: land, 1: water}`` labels marks *every* pixel as water: no land/water boundary,
    no contour, and extraction raises ``No_Extracted_Shoreline`` while segmentation looks
    perfectly healthy.
    """
    directory = tmp_path / "SAR_segmentation_model"
    directory.mkdir()
    for name in contents:
        (directory / name).write_bytes(b"")
    assert not list(directory.glob("*modelcard.json"))

    model = zoo_model.Zoo_Model()
    info = model.load_model_info({"local_model_path": str(directory)})

    assert info.class_mapping == {0: "land", 1: "water"}
    assert info.water_class_indices == [1]


def test_an_optical_model_directory_still_reads_its_model_card(monkeypatch, tmp_path):
    """The SAR branch must not intercept a directory holding ordinary Zoo weights."""
    directory = tmp_path / "segformer_RGB_4class"
    directory.mkdir()
    (directory / "weights.h5").write_bytes(b"")

    def fail(directory):  # pragma: no cover - only runs if the branch is wrong
        raise AssertionError("the optical path must not load a SAR spec")

    monkeypatch.setattr(sar_model, "load_sar_model_spec", fail)

    model = zoo_model.Zoo_Model()
    info = model.load_model_info({"local_model_path": str(directory)})

    # No model card either, so this is the documented optical default.
    assert info.water_class_indices == [0, 1]


# ---------------------------------------------------------------------------
# Zoo workflow: extraction from a segmented session
# ---------------------------------------------------------------------------


def test_the_zoo_session_tags_s1_shorelines_as_sar_model(monkeypatch, tmp_path):
    """S1 goes through extraction and is recorded as segmented by the SAR model.

    ``segmentation_method`` is what later tells a threshold filter to leave these
    shorelines alone, and ``combine_satellite_data`` builds its column list from the keys
    present, so a column missing on one satellite desynchronizes the merged lists.
    """
    processed = []

    def fake_process_satellite(satname, settings, metadata, session_path, *args, **kwargs):
        processed.append(satname)
        return {
            satname: {
                "dates": ["2023-05-01-04-26-29"],
                "shorelines": [np.array([[0.0, 0.0], [1.0, 1.0]])],
                "filename": ["2023-05-01-04-26-29_S1_site_VV.tif"],
                "cloud_cover": [0.0],
                "geoaccuracy": [1.0],
                "idx": [0],
                "satname": [satname],
                "segmentation_method": [es.get_segmentation_method(satname)],
            }
        }

    monkeypatch.setattr(es, "process_satellite", fake_process_satellite)

    output = es.extract_shorelines_with_dask(
        session_path=str(tmp_path),
        metadata={"S1": {"filenames": ["a"], "dates": ["2023-05-01-04-26-29"]}},
        settings={"inputs": {"sitename": "site", "filepath": str(tmp_path)}},
        save_location=str(tmp_path),
    )

    assert processed == ["S1"]
    assert output["satname"] == ["S1"]
    assert output["segmentation_method"] == ["sar_model"]
    assert len(output["segmentation_method"]) == len(output["shorelines"])


# ---------------------------------------------------------------------------
# CoastSat workflow
# ---------------------------------------------------------------------------


def test_the_coastsat_workflow_hands_coastsat_an_s1_only_run(monkeypatch, tmp_path):
    """coastsat decides to segment with its SAR model from the run it is handed.

    ``sat_list`` and the metadata keys are what select the S1 branch inside
    ``SDS_shoreline.extract_shorelines``; an ROI that loses either of them silently falls
    through to the optical path.
    """
    captured = {}

    def fake_extract_shorelines(metadata, settings, **kwargs):
        captured["metadata"] = metadata
        captured["settings"] = settings
        return {}

    monkeypatch.setattr(es.SDS_shoreline, "extract_shorelines", fake_extract_shorelines)
    monkeypatch.setattr(
        es, "get_metadata", lambda *a, **k: {"S1": {"filenames": ["scene_VV.tif"]}}
    )
    monkeypatch.setattr(es.common, "filter_metadata_with_dates", lambda m, *a, **k: m)
    monkeypatch.setattr(es, "get_reference_shoreline", lambda *a, **k: {})
    monkeypatch.setattr(es, "get_reference_shoreline_parts", lambda *a, **k: [])
    monkeypatch.setattr(es.SDS_tools, "remove_duplicates", lambda output: output)
    monkeypatch.setattr(
        es.SDS_tools, "remove_inaccurate_georef", lambda output, _: output
    )
    monkeypatch.setattr(es, "get_data_folder", lambda *a, **k: str(tmp_path))

    es.Extracted_Shoreline().extract_shorelines(
        shoreline_gdf=None,
        roi_settings={
            "roi_id": "abc1",
            "sitename": "site",
            "filepath": str(tmp_path),
            "sat_list": ["S1"],
            "dates": ["2023-01-01", "2024-01-01"],
        },
        settings={"output_epsg": 4326, "min_beach_area": 100},
    )

    assert list(captured["metadata"]) == ["S1"]
    assert captured["settings"]["inputs"]["sat_list"] == ["S1"]


def test_a_sar_shoreline_is_exempt_from_the_mndwi_threshold_filter():
    """A SAR shoreline carries no MNDWI threshold, so it must not be filtered on one.

    ``MNDWI_threshold`` holds a different quantity per segmentation method: an MNDWI value
    for the optical contour path, but NaN when a model mapped the shoreline. Comparing NaN
    to an MNDWI range fails every time and would delete every SAR shoreline.
    """
    output = {
        "shorelines": [np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]])],
        "MNDWI_threshold": [float("nan"), 0.1],
        "segmentation_method": ["sar_model", "mndwi"],
    }

    applies = es.threshold_filter_applies(output)

    assert list(applies) == [False, True]


# ===========================================================================
# CoastSat workflow settings
#
# Tests for the CoastSat (classic) Sentinel-1 workflow.
#
# That workflow hands everything to ``SDS_shoreline.extract_shorelines``, which segments S1
# with coastsat's trained ONNX model. coastsat reads three settings out of the dict CoastSeg
# builds for it, so these tests pin two things:
#
# 1. the settings actually survive CoastSeg's allowlists and reach coastsat, and
# 2. ``sar_model_path`` is stored relative to the CoastSeg directory, never absolute, so a
#    config.json stays usable on another machine.
# ===========================================================================


SAR_KEYS = ("sar_segmentation", "sar_water_threshold", "sar_model_path")


@pytest.fixture
def based_at(monkeypatch, tmp_path):
    """Point core_utilities.get_base_dir at tmp_path for both path helpers."""
    monkeypatch.setattr(common.core_utilities, "get_base_dir", lambda *a, **k: tmp_path)
    return tmp_path


def shoreline_settings(settings):
    """Build the dict CoastSeg hands to SDS_shoreline.extract_shorelines."""
    return es.Extracted_Shoreline().create_shoreline_settings(
        settings,
        {"roi_id": "abc1", "sitename": "site", "filepath": "data"},
        reference_shoreline={},
    )


# ---------------------------------------------------------------------------
# The settings reach coastsat
# ---------------------------------------------------------------------------


def test_the_default_is_the_model_not_otsu():
    """The trained segmenter is the default path for Sentinel-1."""
    settings = coastseg_map.CoastSeg_Map(create_map=False).set_settings()

    assert settings["sar_segmentation"] == "model"
    assert settings["sar_water_threshold"] == 0.5
    assert settings["sar_model_path"] == ""


@pytest.mark.parametrize(
    "key, value",
    [
        ("sar_segmentation", "otsu"),
        ("sar_water_threshold", 0.8),
    ],
)
def test_create_shoreline_settings_passes_the_sar_keys_to_coastsat(key, value):
    """coastsat reads these straight out of the dict; the allowlist used to drop them."""
    settings = coastseg_map.CoastSeg_Map(create_map=False).set_settings(**{key: value})

    assert shoreline_settings(settings)[key] == value


def test_empty_sar_model_path_survives_rather_than_being_stripped(monkeypatch):
    """'' is meaningful: it tells coastsat to resolve and download its own model.

    It must reach coastsat rather than being dropped by the SHORELINE_KEYS allowlist.
    With no model in CoastSeg's models/ directory it stays '' -- see
    test_an_unset_path_points_coastsat_at_a_manually_downloaded_model for the case
    where one is there.
    """
    monkeypatch.setattr(sar_model, "find_local_sar_model", lambda: "")
    settings = coastseg_map.CoastSeg_Map(create_map=False).set_settings()

    assert shoreline_settings(settings)["sar_model_path"] == ""


def test_an_unset_path_points_coastsat_at_a_manually_downloaded_model(monkeypatch):
    """A user who downloads the model by hand into models/SAR_segmentation_model
    expects both workflows to use it. The zoo path reads that directory directly, but
    coastsat looks only in its own pooch cache -- so without this the same 130 MB would
    be downloaded a second time for the coastsat run."""
    manual = r"C:\CoastSeg\models\SAR_segmentation_model\SAR_3_band_model.onnx"
    monkeypatch.setattr(sar_model, "find_local_sar_model", lambda: manual)
    settings = coastseg_map.CoastSeg_Map(create_map=False).set_settings()

    assert shoreline_settings(settings)["sar_model_path"] == manual


def test_an_explicit_path_is_not_overridden_by_the_local_model(monkeypatch):
    """An explicit choice always wins; the models/ directory is only a fallback."""
    monkeypatch.setattr(
        sar_model, "find_local_sar_model", lambda: r"C:\CoastSeg\models\local.onnx"
    )
    settings = coastseg_map.CoastSeg_Map(create_map=False).set_settings(
        sar_model_path="models/retrained.onnx"
    )

    handed_to_coastsat = shoreline_settings(settings)["sar_model_path"]
    assert handed_to_coastsat.endswith(os.path.join("models", "retrained.onnx"))
    assert "local.onnx" not in handed_to_coastsat


def test_settings_absent_from_the_input_are_simply_absent():
    """A settings dict predating these keys must not gain them with junk values."""
    result = shoreline_settings({"output_epsg": 4326, "min_beach_area": 100})

    assert not any(key in result for key in SAR_KEYS)


def test_the_override_reaches_extract_shorelines(monkeypatch, tmp_path):
    """End to end through Extracted_Shoreline.extract_shorelines.

    A unit test on SHORELINE_KEYS alone would not catch a later refactor that rebuilds
    the settings dict somewhere else, so capture what coastsat is actually handed.
    """
    captured = {}

    def fake_extract_shorelines(metadata, settings, **kwargs):
        captured.update(settings)
        return {}

    monkeypatch.setattr(
        es.SDS_shoreline, "extract_shorelines", fake_extract_shorelines
    )
    monkeypatch.setattr(es, "get_metadata", lambda *a, **k: {"S1": {"filenames": ["x"]}})
    monkeypatch.setattr(es.common, "filter_metadata_with_dates", lambda m, *a, **k: m)
    monkeypatch.setattr(es, "get_reference_shoreline", lambda *a, **k: {})
    monkeypatch.setattr(es, "get_reference_shoreline_parts", lambda *a, **k: [])
    monkeypatch.setattr(es.SDS_tools, "remove_duplicates", lambda output: output)
    monkeypatch.setattr(es.SDS_tools, "remove_inaccurate_georef", lambda output, _: output)
    monkeypatch.setattr(es, "get_data_folder", lambda *a, **k: str(tmp_path))

    settings = coastseg_map.CoastSeg_Map(create_map=False).set_settings(
        sar_segmentation="otsu", sar_water_threshold=0.75
    )
    es.Extracted_Shoreline().extract_shorelines(
        shoreline_gdf=None,
        roi_settings={
            "roi_id": "abc1",
            "sitename": "site",
            "filepath": str(tmp_path),
            "sat_list": ["S1"],
            "dates": ["2023-01-01", "2024-01-01"],
        },
        settings=settings,
    )

    assert captured["sar_segmentation"] == "otsu"
    assert captured["sar_water_threshold"] == 0.75


def test_the_sar_keys_round_trip_through_load_settings(tmp_path):
    written = {
        "sar_segmentation": "otsu",
        "sar_water_threshold": 0.7,
        "sar_model_path": "models/retrained.onnx",
    }
    loaded = common.load_settings(new_settings=written)

    assert {key: loaded[key] for key in SAR_KEYS} == written


def test_the_sar_keys_round_trip_through_extract_roi_settings():
    config = {
        "roi_ids": ["abc1"],
        "abc1": {
            "sitename": "site",
            "sat_list": ["S1"],
            "sar_segmentation": "otsu",
            "sar_water_threshold": 0.7,
            "sar_model_path": "models/retrained.onnx",
        },
    }
    roi_settings = common.extract_roi_settings(config)

    assert roi_settings["abc1"]["sar_segmentation"] == "otsu"
    assert roi_settings["abc1"]["sar_water_threshold"] == 0.7
    assert roi_settings["abc1"]["sar_model_path"] == "models/retrained.onnx"


# ---------------------------------------------------------------------------
# sar_model_path is stored relative, resolved absolute
# ---------------------------------------------------------------------------


def test_empty_path_stays_empty_through_both_helpers(based_at):
    """coastsat falls back to its packaged model on a falsy value."""
    assert common.relativize_sar_model_path("") == ""
    assert common.resolve_sar_model_path("") == ""


def test_an_absolute_path_inside_the_tree_is_stored_relative(based_at):
    absolute = str(based_at / "models" / "retrained.onnx")

    stored = common.relativize_sar_model_path(absolute)

    assert stored == "models/retrained.onnx"  # POSIX, so a Windows config loads on Linux
    assert not os.path.isabs(stored)
    assert os.path.normcase(common.resolve_sar_model_path(stored)) == os.path.normcase(
        absolute
    )


def test_relativizing_is_idempotent(based_at):
    stored = common.relativize_sar_model_path(str(based_at / "models" / "m.onnx"))

    assert common.relativize_sar_model_path(stored) == stored


def test_a_relative_path_resolves_against_the_base_dir_not_the_cwd(
    based_at, monkeypatch, tmp_path
):
    """coastsat runs os.path.abspath on whatever it is given, which would resolve
    against wherever the notebook happens to be running."""
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    resolved = common.resolve_sar_model_path("models/retrained.onnx")

    assert os.path.normcase(resolved) == os.path.normcase(
        str(based_at / "models" / "retrained.onnx")
    )
    assert os.path.normcase(str(elsewhere)) not in os.path.normcase(resolved)


def test_a_path_outside_the_tree_is_kept_verbatim_and_warns(based_at, tmp_path, caplog):
    outside = tmp_path.parent / "outside_the_repo" / "m.onnx"

    with caplog.at_level("WARNING"):
        stored = common.relativize_sar_model_path(str(outside))

    assert stored == outside.as_posix()
    assert "cannot be stored relatively" in caplog.text


def test_set_settings_never_persists_an_absolute_path(based_at):
    absolute = str(based_at / "models" / "retrained.onnx")

    settings = coastseg_map.CoastSeg_Map(create_map=False).set_settings(
        sar_model_path=absolute
    )

    assert settings["sar_model_path"] == "models/retrained.onnx"


def test_coastsat_receives_the_absolute_path(based_at):
    """Stored relative for portability, handed to coastsat absolute so it can be opened."""
    absolute = based_at / "models" / "retrained.onnx"
    settings = coastseg_map.CoastSeg_Map(create_map=False).set_settings(
        sar_model_path=str(absolute)
    )

    handed_to_coastsat = shoreline_settings(settings)["sar_model_path"]

    assert os.path.isabs(handed_to_coastsat)
    assert os.path.normcase(handed_to_coastsat) == os.path.normcase(str(absolute))


# ===========================================================================
# Otsu fallback warnings
#
# Tests for the warnings CoastSeg raises when SAR shorelines are not model-segmented.
#
# coastsat never raises when the SAR segmentation model is unusable -- ``load_sar_segmenter``
# returns None and every scene is thresholded with the legacy Otsu method instead. The run
# still produces shorelines, so nothing looks wrong, and coastsat records the fallback only
# in its own log inside the session folder. These tests pin the two places CoastSeg surfaces
# it to the user:
#
# 1. a pre-flight check, before extraction, for the whole-run fallback, and
# 2. a post-run tally of ``output['segmentation_method']``, which also catches the per-scene
#    fallbacks the pre-flight check cannot predict.
#
# Nothing here loads the real ~130 MB ONNX model: the segmenter is stubbed.
# ===========================================================================


S1_METADATA = {"S1": {"dates": ["2023-05-06-02-07-48"]}}


@pytest.fixture
def segmenter(monkeypatch):
    """Control what coastsat's load_sar_segmenter returns without touching the model."""

    def set_result(result):
        monkeypatch.setattr(
            SDS_shoreline, "load_sar_segmenter", lambda *a, **k: result
        )

    return set_result


# ---------------------------------------------------------------------------
# Pre-flight check
# ---------------------------------------------------------------------------


def test_an_unusable_model_warns_that_the_whole_run_falls_back(
    caplog, capsys, segmenter
):
    """The silent degrade is the whole point: a user who is not told keeps the worse
    shorelines and never knows the model was skipped."""
    segmenter(None)

    with caplog.at_level("WARNING"):
        fell_back = es.warn_if_sar_falls_back_to_otsu({}, S1_METADATA)

    assert fell_back is True
    assert "falls back to the legacy Otsu threshold" in caplog.text
    # printed too -- a notebook user does not read the logger
    assert "WARNING" in capsys.readouterr().out


def test_a_loadable_model_says_nothing(caplog, segmenter):
    segmenter(("session", {"channel_order": ["VV", "VH", "VV-VH"]}))

    with caplog.at_level("WARNING"):
        assert es.warn_if_sar_falls_back_to_otsu({}, S1_METADATA) is False

    assert caplog.text == ""


def test_asking_for_otsu_is_not_a_warning(caplog, capsys, segmenter):
    """Otsu chosen deliberately is a valid configuration, not a degraded run. Warning
    about it would train users to ignore the warning that matters."""
    segmenter(None)

    with caplog.at_level("INFO"):
        assert (
            es.warn_if_sar_falls_back_to_otsu(
                {"sar_segmentation": "otsu"}, S1_METADATA
            )
            is False
        )

    assert "WARNING" not in capsys.readouterr().out
    assert "thresholded rather than segmented" in caplog.text


@pytest.mark.parametrize("metadata", [{}, {"S2": {"dates": ["2023-05-06"]}}])
def test_an_optical_only_run_never_loads_the_sar_model(caplog, monkeypatch, metadata):
    """No S1 scenes means the SAR model is irrelevant, and loading a ~130 MB model to
    discover that would cost every optical run real time."""

    def explode(*args, **kwargs):
        raise AssertionError("the SAR model must not be loaded for an optical run")

    monkeypatch.setattr(SDS_shoreline, "load_sar_segmenter", explode)

    with caplog.at_level("WARNING"):
        assert es.warn_if_sar_falls_back_to_otsu({}, metadata) is False

    assert caplog.text == ""


# ---------------------------------------------------------------------------
# Post-run tally
# ---------------------------------------------------------------------------


def test_a_per_scene_fallback_is_reported_after_the_run(caplog, capsys):
    """A scene coastsat could not segment drops to Otsu on its own, which the pre-flight
    check cannot predict."""
    output = {
        "segmentation_method": [
            SDS_tools.SEGMENTATION_SAR_MODEL,
            SDS_tools.SEGMENTATION_SAR_OTSU,
            SDS_tools.SEGMENTATION_SAR_MODEL,
        ]
    }

    with caplog.at_level("WARNING"):
        counts = es.report_sar_segmentation_methods(output)

    assert counts == {
        SDS_tools.SEGMENTATION_SAR_MODEL: 2,
        SDS_tools.SEGMENTATION_SAR_OTSU: 1,
    }
    assert "1 of 3" in caplog.text
    assert "WARNING" in capsys.readouterr().out


def test_a_fully_model_segmented_run_does_not_warn(caplog, capsys):
    output = {
        "segmentation_method": [SDS_tools.SEGMENTATION_SAR_MODEL] * 3
    }

    with caplog.at_level("INFO"):
        counts = es.report_sar_segmentation_methods(output)

    assert counts[SDS_tools.SEGMENTATION_SAR_OTSU] == 0
    assert "WARNING" not in capsys.readouterr().out
    assert "All 3 Sentinel-1 shorelines" in caplog.text


def test_an_optical_only_output_is_not_reported_on(caplog, capsys):
    """Optical shorelines carry 'mndwi' and must not be counted as SAR."""
    output = {"segmentation_method": [SDS_tools.SEGMENTATION_MNDWI] * 4}

    with caplog.at_level("INFO"):
        counts = es.report_sar_segmentation_methods(output)

    assert counts == {
        SDS_tools.SEGMENTATION_SAR_MODEL: 0,
        SDS_tools.SEGMENTATION_SAR_OTSU: 0,
    }
    assert capsys.readouterr().out == ""


def test_output_without_the_column_is_handled():
    """CoastSeg's own zoo extraction records no SAR methods; this must not raise."""
    assert es.report_sar_segmentation_methods({}) == {
        SDS_tools.SEGMENTATION_SAR_MODEL: 0,
        SDS_tools.SEGMENTATION_SAR_OTSU: 0,
    }
