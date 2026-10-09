"""Tests for the Sentinel-1 ONNX segmentation model (``coastseg.sar_model``).

Three parts, in the order a scene meets them:

1. the preprocessing / postprocessing contract around the model,
2. how the model file itself is found or downloaded,
3. the artifacts a segmented session leaves on disk for shoreline extraction.

Only the two ``requires_model`` tests load the real ~130 MB model, and only when it is
already in ``models/SAR_segmentation_model``; nothing here touches the network.
"""

import json
import os

import numpy as np
import pytest
from coastsat import SDS_sar_model

from coastseg import common, extracted_shoreline, sar_model
from coastseg.model_info import ModelInfo


# ===========================================================================
# Preprocessing contract
#
# Unit tests for the Sentinel-1 ONNX segmentation preprocessing contract.
#
# Getting any preprocessing step wrong does not crash -- it silently degrades or destroys
# accuracy -- so each step of the documented pipeline is pinned here.
# ===========================================================================


SAR_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "SAR_segmentation_model",
)

requires_model = pytest.mark.skipif(
    not os.path.isdir(SAR_MODEL_DIR) or not any(
        name.endswith(".onnx") for name in os.listdir(SAR_MODEL_DIR)
    ),
    reason="the SAR ONNX model is not present in models/SAR_segmentation_model",
)


# ---------------------------------------------------------------------------
# Model spec
# ---------------------------------------------------------------------------


@requires_model
def test_load_spec_reads_metadata_props():
    """Every constant comes from the model file, not from hardcoded values."""
    spec = sar_model.load_sar_model_spec(SAR_MODEL_DIR)

    assert spec.channel_order == ("VV", "VH", "VV-VH")
    assert np.allclose(spec.mean, [-12.59, -20.26, 10.5465])
    assert np.allclose(spec.std, [5.26, 5.91, 7.6855])
    assert spec.output_stride == 32
    assert spec.nodata == 255
    assert spec.classes == {0: "land", 1: "water"}
    assert spec.input_name == "image"
    assert spec.prob_output == "water_prob"
    assert spec.input_scale == "dB"
    assert spec.speckle_filter["type"] == "none"


# The exported model exposes both of these, so which one carries P(water) is a real
# choice, not a hypothetical one. Thresholding logits at 0.5 does not raise -- it just
# maps the wrong shoreline.
BOTH_OUTPUTS = ["logits", "water_prob"]


def test_prob_output_prefers_the_named_water_probability():
    assert (
        sar_model._resolve_prob_output(BOTH_OUTPUTS, BOTH_OUTPUTS, "m.onnx")
        == "water_prob"
    )


def test_prob_output_is_not_guessed_by_position():
    """'water_prob' wins wherever it sits, so export order cannot change the result."""
    reordered = ["water_prob", "logits"]

    assert (
        sar_model._resolve_prob_output(reordered, reordered, "m.onnx") == "water_prob"
    )


def test_ambiguous_outputs_raise_instead_of_picking_one():
    """Falling back to the last output would silently threshold logits at 0.5."""
    names = ["logits", "features"]

    with pytest.raises(sar_model.SarModelError, match="water_prob"):
        sar_model._resolve_prob_output(names, names, "retrained.onnx")


def test_a_single_output_is_used_but_says_so(caplog):
    """One output is unambiguous; the run continues, but not silently."""
    with caplog.at_level("WARNING"):
        chosen = sar_model._resolve_prob_output(["prob"], ["prob"], "m.onnx")

    assert chosen == "prob"
    assert any("water_prob" in record.message for record in caplog.records)


def test_metadata_naming_an_output_the_graph_lacks_raises():
    """Otherwise session.run fails later with an opaque onnxruntime error."""
    with pytest.raises(sar_model.SarModelError, match="graph exposes"):
        sar_model._resolve_prob_output(
            ["water_prob"], ["logits", "probability"], "m.onnx"
        )


def test_water_class_indices_is_derived_from_class_names(sar_model_spec_stub):
    # This model's water class is 1, the inverse of the SegFormer convention.
    assert sar_model_spec_stub.water_class_indices == [1]


def test_required_polarizations_expands_difference_channels(sar_model_spec_stub):
    """A VV-VH channel contributes both operands, not a literal 'VV-VH' band."""
    assert sar_model_spec_stub.required_polarizations == ("VV", "VH")


def test_is_sar_model_directory(tmp_path, sar_scene_tifs):
    assert not sar_model.is_sar_model_directory("")
    assert not sar_model.is_sar_model_directory(str(tmp_path))
    (tmp_path / "some_model.onnx").write_bytes(b"")
    assert sar_model.is_sar_model_directory(str(tmp_path))


def test_the_designated_directory_routes_as_sar_while_still_empty(tmp_path):
    """The model is downloaded on first use, so the designated directory is empty (or
    absent) until then. If routing depended on the file being there, the first SAR run
    on a fresh clone would be sent down the TensorFlow path instead."""
    designated = tmp_path / sar_model.DEFAULT_SAR_MODEL_DIRNAME

    assert sar_model.is_sar_model_directory(str(designated))  # does not exist yet
    designated.mkdir()
    assert sar_model.is_sar_model_directory(str(designated))  # exists but empty


def test_find_sar_onnx_file_missing_raises_when_download_is_off(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .onnx file"):
        sar_model.find_sar_onnx_file(str(tmp_path), download=False)


# ---------------------------------------------------------------------------
# Band stack (steps 1-7)
# ---------------------------------------------------------------------------


def test_difference_channel_uses_raw_db_before_any_fill(
    sar_scene_tifs, sar_model_spec_stub
):
    """VV-VH must be derived before invalid pixels are filled.

    Otherwise a fill value leaks into the subtraction and the third channel is wrong
    wherever the scene has nodata.
    """
    stack, valid, _, _ = sar_model.build_sar_stack(
        [sar_scene_tifs["vv"], sar_scene_tifs["vh"]], sar_model_spec_stub
    )
    assert stack.shape == (3, 50, 70)
    assert np.allclose(stack[2], stack[0] - stack[1], equal_nan=True)


def test_validity_mask_is_the_intersection_of_both_polarizations(
    sar_scene_tifs, sar_model_spec_stub
):
    _, valid, _, _ = sar_model.build_sar_stack(
        [sar_scene_tifs["vv"], sar_scene_tifs["vh"]], sar_model_spec_stub
    )
    height, width = sar_scene_tifs["shape"]
    nodata_rows = sar_scene_tifs["nodata_rows"]
    assert valid.shape == (height, width)
    # VV's last rows are nodata, so they must be invalid in the combined mask
    assert not valid[-nodata_rows:, :].any()
    assert valid[: height - nodata_rows, :].all()


def test_missing_polarization_raises(sar_site_vh_only, sar_model_spec_stub):
    with pytest.raises(sar_model.MissingPolarizationError, match="VV"):
        sar_model.build_sar_stack([sar_site_vh_only["vh"]], sar_model_spec_stub)


def test_grid_mismatch_reprojects_onto_the_reference_grid(
    sar_grid_mismatch_tifs, sar_model_spec_stub, caplog
):
    """A VH raster on a different grid is warped onto VV's grid, never the reverse."""
    with caplog.at_level("WARNING"):
        stack, valid, geotransform, _ = sar_model.build_sar_stack(
            [sar_grid_mismatch_tifs["vv"], sar_grid_mismatch_tifs["vh"]],
            sar_model_spec_stub,
        )

    assert stack.shape == (3,) + sar_grid_mismatch_tifs["shape"]
    assert valid.shape == sar_grid_mismatch_tifs["shape"]
    # georeferencing stays on the VV (reference) grid
    assert geotransform[1] == 10.0
    assert any("reprojecting" in record.message for record in caplog.records)


def test_unsupported_speckle_filter_raises(sar_scene_tifs, sar_model_spec_stub):
    """A model trained with a filter this module cannot apply must stop, not proceed."""
    import dataclasses

    spec = dataclasses.replace(
        sar_model_spec_stub, speckle_filter={"type": "lee_sigma"}
    )
    with pytest.raises(sar_model.UnsupportedSpeckleFilterError, match="lee_sigma"):
        sar_model.build_sar_stack(
            [sar_scene_tifs["vv"], sar_scene_tifs["vh"]], spec
        )


# ---------------------------------------------------------------------------
# Fill / normalize / pad (steps 8-10)
# ---------------------------------------------------------------------------


def test_invalid_pixels_normalize_to_exactly_zero(sar_model_spec_stub):
    """Filling with the per-channel mean puts invalid pixels at a neutral 0."""
    spec = sar_model_spec_stub
    stack = np.zeros((3, 32, 32), dtype=np.float32)
    stack[:, :, :] = 999.0  # an extreme outlier everywhere
    valid = np.ones((32, 32), dtype=bool)
    valid[0, 0] = False

    tensor, _ = sar_model.preprocess(stack, valid, spec)
    assert np.allclose(tensor[0, :, 0, 0], 0.0)


def test_normalization_is_per_channel(sar_model_spec_stub):
    spec = sar_model_spec_stub
    stack = np.stack(
        [
            np.full((32, 32), -10.0, dtype=np.float32),
            np.full((32, 32), -18.0, dtype=np.float32),
            np.full((32, 32), 8.0, dtype=np.float32),
        ]
    )
    valid = np.ones((32, 32), dtype=bool)

    tensor, _ = sar_model.preprocess(stack, valid, spec)
    expected = (np.array([-10.0, -18.0, 8.0], dtype=np.float32) - spec.mean) / spec.std
    assert np.allclose(tensor[0, :, 5, 5], expected)


def test_pads_bottom_right_to_a_multiple_of_the_output_stride(sar_model_spec_stub):
    """50x70 becomes 64x96, padded with zeros on the bottom and right only."""
    spec = sar_model_spec_stub
    stack = np.ones((3, 50, 70), dtype=np.float32)
    valid = np.ones((50, 70), dtype=bool)

    tensor, original_shape = sar_model.preprocess(stack, valid, spec)

    assert tensor.shape == (1, 3, 64, 96)
    assert original_shape == (50, 70)
    assert np.all(tensor[0, :, 50:, :] == 0.0)
    assert np.all(tensor[0, :, :, 70:] == 0.0)
    # nothing was padded on the top or left
    assert not np.all(tensor[0, :, 0, 0] == 0.0)


def test_already_aligned_shape_is_not_padded(sar_model_spec_stub):
    stack = np.ones((3, 64, 32), dtype=np.float32)
    valid = np.ones((64, 32), dtype=bool)
    tensor, _ = sar_model.preprocess(stack, valid, sar_model_spec_stub)
    assert tensor.shape == (1, 3, 64, 32)


def test_oversized_scene_raises_instead_of_exhausting_memory(sar_model_spec_stub, monkeypatch):
    monkeypatch.setattr(sar_model, "MAX_INPUT_PIXELS", 100)
    stack = np.ones((3, 64, 64), dtype=np.float32)
    valid = np.ones((64, 64), dtype=bool)
    with pytest.raises(sar_model.SarModelError, match="pixel limit|above the"):
        sar_model.preprocess(stack, valid, sar_model_spec_stub)


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------


def test_postprocess_crops_padding_and_never_resizes(sar_model_spec_stub):
    probability = np.zeros((1, 1, 64, 96), dtype=np.float32)
    valid = np.ones((50, 70), dtype=bool)

    label, cropped = sar_model.postprocess(probability, (50, 70), valid, sar_model_spec_stub)

    assert label.shape == (50, 70)
    assert cropped.shape == (50, 70)


def test_threshold_boundary_is_inclusive(sar_model_spec_stub):
    probability = np.array([[0.5, 0.4999, 0.9]], dtype=np.float32)
    valid = np.ones((1, 3), dtype=bool)

    label, _ = sar_model.postprocess(probability, (1, 3), valid, sar_model_spec_stub)

    assert label.tolist() == [[1, 0, 1]]


def test_nodata_is_stamped_onto_invalid_pixels(sar_model_spec_stub):
    probability = np.full((1, 1, 32, 32), 0.9, dtype=np.float32)
    valid = np.ones((32, 32), dtype=bool)
    valid[0, :] = False

    label, cropped = sar_model.postprocess(
        probability, (32, 32), valid, sar_model_spec_stub
    )

    assert np.all(label[0, :] == 255)
    assert np.all(label[1:, :] == 1)
    assert np.all(np.isnan(cropped[0, :]))
    assert not np.any(np.isnan(cropped[1:, :]))


def test_labels_follow_the_model_class_indices_not_a_fixed_water_value(
    sar_model_spec_stub,
):
    """A model free to declare {0: "water", 1: "land"} must not come out inverted.

    Extraction decodes ``grey_label`` by testing equality against
    ``model_info.water_class_indices``, which comes from these same class names. Writing
    a fixed 1 for water would silently swap land and sea on such a model -- a plausible
    looking shoreline on the wrong side of the beach.
    """
    import dataclasses

    probability = np.array([[0.9, 0.1]], dtype=np.float32)
    valid = np.ones((1, 2), dtype=bool)

    reversed_spec = dataclasses.replace(
        sar_model_spec_stub, classes={0: "water", 1: "land"}
    )
    label, _ = sar_model.postprocess(probability, (1, 2), valid, reversed_spec)

    assert reversed_spec.water_class_indices == [0]
    assert label.tolist() == [[0, 1]]  # water pixel carries the water class index


def test_labels_still_use_1_for_water_on_the_conventional_mapping(sar_model_spec_stub):
    """The shipped {0: land, 1: water} model keeps producing exactly what it did."""
    probability = np.array([[0.9, 0.1]], dtype=np.float32)
    valid = np.ones((1, 2), dtype=bool)

    label, _ = sar_model.postprocess(probability, (1, 2), valid, sar_model_spec_stub)

    assert sar_model_spec_stub.water_class_indices == [1]
    assert label.tolist() == [[1, 0]]


# ---------------------------------------------------------------------------
# Scene / site helpers
# ---------------------------------------------------------------------------


def test_output_stem_matches_the_tensorflow_zoo_convention():
    """The stem must stay {date}_RGB_S1 so npz lookups keep matching."""
    assert (
        sar_model._output_stem("2023-05-01-04-26-29_S1_site_VV.tif")
        == "2023-05-01-04-26-29_RGB_S1"
    )


def test_output_stem_keeps_duplicate_suffix():
    """Duplicate acquisitions must not overwrite each other's artifacts."""
    assert (
        sar_model._output_stem("2023-05-01-04-26-29_S1_site_VV_dup0.tif")
        == "2023-05-01-04-26-29_RGB_S1_dup0"
    )


def test_collect_sar_scenes_pairs_polarizations(sar_scene_tifs):
    scenes = sar_model.collect_sar_scenes(sar_scene_tifs["site"])
    assert len(scenes) == 1
    (paths,) = scenes.values()
    assert set(paths) == {"VV", "VH"}


def test_collect_sar_scenes_on_a_legacy_site(sar_site_vh_only):
    scenes = sar_model.collect_sar_scenes(sar_site_vh_only["site"])
    (paths,) = scenes.values()
    assert set(paths) == {"VH"}


def test_polarization_token_is_not_confused_by_a_sitename():
    """A sitename containing 'VH' must not be read as a polarization token."""
    assert sar_model.get_polarization_from_filename("2023_S1_myVHsite_VV.tif") == "VV"


# ---------------------------------------------------------------------------
# Nodata helper
# ---------------------------------------------------------------------------


def test_sar_nodata_mask_reports_the_empty_part_of_the_swath(sar_scene_tifs):
    mask = sar_model.sar_nodata_mask([sar_scene_tifs["vv"], sar_scene_tifs["vh"]])
    height, width = sar_scene_tifs["shape"]
    assert mask.shape == (height, width)
    assert int(mask.sum()) == sar_scene_tifs["nodata_rows"] * width


def test_sar_nodata_mask_returns_none_when_unreadable(tmp_path):
    assert sar_model.sar_nodata_mask([str(tmp_path / "missing.tif")]) is None


# ---------------------------------------------------------------------------
# Real inference
# ---------------------------------------------------------------------------


@requires_model
def test_real_model_separates_water_from_land(sar_scene_tifs):
    """Low-backscatter pixels come back as water and high-backscatter ones as land."""
    spec = sar_model.load_sar_model_spec(SAR_MODEL_DIR)
    prediction = sar_model.segment_scene(
        [sar_scene_tifs["vv"], sar_scene_tifs["vh"]], spec
    )

    height, width = sar_scene_tifs["shape"]
    valid_rows = height - sar_scene_tifs["nodata_rows"]

    assert prediction.label.shape == (height, width)
    assert set(np.unique(prediction.label).tolist()) <= {0, 1, 255}
    assert (prediction.label[:valid_rows, : width // 2] == 1).mean() > 0.9
    assert (prediction.label[:valid_rows, width // 2 :] == 1).mean() < 0.1


# ===========================================================================
# Obtaining the ONNX model
#
# Tests for how the zoo SAR workflow obtains its ONNX model.
#
# The model is not shipped with CoastSeg -- it is ~130 MB and ``models/**/*.onnx`` is
# gitignored, so a fresh clone has none. ``find_sar_onnx_file`` downloads it from Hugging
# Face on first use, deferring to ``coastsat.SDS_sar_model`` so both SAR workflows share
# one sha256-pinned pooch cache instead of keeping a copy each.
#
# Nothing here touches the network: coastsat's resolver and fetcher are both stubbed. A
# test that reached Hugging Face would download 130 MB per run.
# ===========================================================================


@pytest.fixture
def no_real_downloads(monkeypatch, tmp_path):
    """Make the default model look absent and the download explode.

    Applied to every test in this section so one that accidentally reaches the download
    path fails loudly instead of silently pulling 130 MB. Tests that mean to download
    override these.
    """
    monkeypatch.setattr(
        SDS_sar_model, "get_sar_model_path", lambda *a, **k: str(tmp_path / "absent.onnx")
    )

    def explode():
        raise AssertionError("the test tried to download the real SAR model")

    monkeypatch.setattr(SDS_sar_model, "fetch_default_sar_model", explode)


@pytest.fixture
def fake_download(no_real_downloads, monkeypatch, tmp_path):
    """Stand in for the Hugging Face fetch, recording that it was called."""
    calls = []
    cached = tmp_path / "cache" / "SAR_3_band_model.onnx"
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"onnx")

    def fetch():
        calls.append(True)
        return str(cached)

    monkeypatch.setattr(SDS_sar_model, "fetch_default_sar_model", fetch)
    return {"calls": calls, "path": str(cached)}


# ---------------------------------------------------------------------------
# A local copy always wins
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_real_downloads")
def test_a_local_model_is_used_without_downloading(tmp_path, fake_download):
    """A retrained or manually placed model must never be silently replaced by the
    default one, and an offline machine that already has a model must keep working."""
    local = tmp_path / "SAR_segmentation_model"
    local.mkdir()
    (local / "my_retrained.onnx").write_bytes(b"local")

    resolved = sar_model.find_sar_onnx_file(str(local))

    assert resolved == str((local / "my_retrained.onnx").resolve())
    assert fake_download["calls"] == []


@pytest.mark.usefixtures("no_real_downloads")
def test_an_already_cached_default_is_not_re_downloaded(monkeypatch, tmp_path):
    """The second run must not re-fetch: coastsat's resolver already points at the
    cached copy, whichever workflow put it there."""
    cached = tmp_path / "SAR_3_band_model.onnx"
    cached.write_bytes(b"onnx")
    monkeypatch.setattr(SDS_sar_model, "get_sar_model_path", lambda *a, **k: str(cached))

    # fetch_default_sar_model is still the exploding stub from no_real_downloads
    resolved = sar_model.find_sar_onnx_file(str(tmp_path / "empty_dir"))

    assert resolved == os.path.abspath(str(cached))


# ---------------------------------------------------------------------------
# Downloading when there is nothing on disk
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_real_downloads")
def test_an_empty_model_directory_downloads_the_default(tmp_path, fake_download):
    empty = tmp_path / sar_model.DEFAULT_SAR_MODEL_DIRNAME
    empty.mkdir()

    resolved = sar_model.find_sar_onnx_file(str(empty))

    assert resolved == fake_download["path"]
    assert fake_download["calls"] == [True]


@pytest.mark.usefixtures("no_real_downloads")
def test_a_missing_model_directory_downloads_the_default(tmp_path, fake_download):
    """A fresh clone has no models/SAR_segmentation_model directory at all."""
    resolved = sar_model.find_sar_onnx_file(
        str(tmp_path / "never_created" / sar_model.DEFAULT_SAR_MODEL_DIRNAME)
    )

    assert resolved == fake_download["path"]
    assert fake_download["calls"] == [True]


@pytest.mark.usefixtures("no_real_downloads")
def test_download_is_skippable(tmp_path):
    """download=False keeps the old fail-fast behaviour for callers that want it."""
    with pytest.raises(FileNotFoundError, match="No .onnx file"):
        sar_model.find_sar_onnx_file(str(tmp_path), download=False)


# ---------------------------------------------------------------------------
# Failure is actionable
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_real_downloads")
def test_a_failed_download_explains_how_to_do_it_by_hand(monkeypatch, tmp_path):
    """Offline, behind a proxy, or Hugging Face down: the user needs the URL and the
    destination, not just 'download failed'."""

    def fail():
        raise SDS_sar_model.SarModelUnavailable("no internet")

    monkeypatch.setattr(SDS_sar_model, "fetch_default_sar_model", fail)

    with pytest.raises(sar_model.SarModelError) as error:
        sar_model.find_sar_onnx_file(str(tmp_path / "empty"))

    message = str(error.value)
    assert "no internet" in message
    assert "huggingface.co" in message
    assert sar_model.DEFAULT_SAR_MODEL_DIRNAME in message


@pytest.mark.usefixtures("no_real_downloads")
def test_the_download_failure_is_a_coastseg_error(monkeypatch, tmp_path):
    """Callers catch coastseg.sar_model.SarModelError; coastsat's exception type must
    not leak out of this module."""

    def fail():
        raise SDS_sar_model.SarModelUnavailable("nope")

    monkeypatch.setattr(SDS_sar_model, "fetch_default_sar_model", fail)

    with pytest.raises(sar_model.SarModelError):
        sar_model.find_sar_onnx_file(str(tmp_path / "empty"))


# ===========================================================================
# Session artifacts
#
# Tests for the artifacts the SAR path writes into a session directory.
#
# Shoreline extraction is model-agnostic: it only knows how to read `_res.npz` files and
# `model_info.json`. These tests pin that contract so the SAR segmenter's output stays
# consumable without any changes downstream.
#
# Inference is stubbed, so none of this loads the real 130 MB model.
# ===========================================================================


# Column boundary the stub predicts water to the left of. Chosen to sit inside the
# unpadded 70-pixel-wide fixture so the assertions do not depend on the padded width.
WATER_COLUMNS = 20


class StubSession:
    """Stands in for an onnxruntime session: water left of WATER_COLUMNS, land right."""

    def __init__(self):
        self.calls = []

    def run(self, output_names, feed):
        (tensor,) = feed.values()
        self.calls.append(tensor.shape)
        _, _, height, width = tensor.shape
        probability = np.zeros((1, 1, height, width), dtype=np.float32)
        probability[:, :, :, :WATER_COLUMNS] = 0.99
        return [probability]


@pytest.fixture
def stubbed_sar(monkeypatch, sar_model_spec_stub):
    """Run the batch driver without coastsat or the real model."""
    session = StubSession()
    monkeypatch.setattr(
        sar_model, "load_sar_model_spec", lambda *a, **k: sar_model_spec_stub
    )
    monkeypatch.setattr(sar_model, "get_session", lambda *a, **k: session)
    return session


def run_segmentation(tmp_path, site, **kwargs):
    return sar_model.run_sar_segmentation(
        roi_directory=site,
        session_directory=str(tmp_path / "session"),
        model_directory=str(tmp_path / "model"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def test_writes_the_expected_artifact_set(tmp_path, sar_scene_tifs, stubbed_sar):
    summary = run_segmentation(tmp_path, sar_scene_tifs["site"])
    written = sorted(os.listdir(summary["session_directory"]))

    assert written == [
        "2023-05-01-04-26-29_RGB_S1_predseg.png",
        "2023-05-01-04-26-29_RGB_S1_res.npz",
        "model_info.json",
        "model_settings.json",
        "segmentation_summary.json",
    ]
    assert summary["total_images"] == 1
    assert summary["written_npz"] == 1
    assert summary["failures"] == []


def test_npz_keys_and_dtypes(tmp_path, sar_scene_tifs, stubbed_sar):
    summary = run_segmentation(tmp_path, sar_scene_tifs["site"])
    npz_path = os.path.join(
        summary["session_directory"], "2023-05-01-04-26-29_RGB_S1_res.npz"
    )
    data = np.load(npz_path, allow_pickle=True)

    assert data["grey_label"].dtype == np.uint8
    assert set(np.unique(data["grey_label"]).tolist()) <= {0, 1, 255}
    assert data["grey_label"].shape == sar_scene_tifs["shape"]
    assert int(data["n_data_bands"]) == 3
    assert data["classes_to_index"].item() == {0: "land", 1: "water"}


def test_nclasses_is_the_class_count_not_the_max_label(
    tmp_path, sar_scene_tifs, stubbed_sar
):
    """Deriving nclasses from label.max() + 1 would give 256 once nodata is present."""
    summary = run_segmentation(tmp_path, sar_scene_tifs["site"])
    data = np.load(
        os.path.join(summary["session_directory"], "2023-05-01-04-26-29_RGB_S1_res.npz"),
        allow_pickle=True,
    )

    assert int(data["nclasses"]) == 2
    assert 255 in np.unique(data["grey_label"])


def test_probability_is_stored_with_nan_at_nodata(tmp_path, sar_scene_tifs, stubbed_sar):
    summary = run_segmentation(tmp_path, sar_scene_tifs["site"])
    data = np.load(
        os.path.join(summary["session_directory"], "2023-05-01-04-26-29_RGB_S1_res.npz"),
        allow_pickle=True,
    )
    nodata_mask = data["nodata_mask"]

    assert data["av_softmax_scores"].dtype == np.float32
    assert np.all(np.isnan(data["av_softmax_scores"][nodata_mask]))
    assert not np.any(np.isnan(data["av_softmax_scores"][~nodata_mask]))


def test_predseg_png_is_written_for_the_segmentation_filter(
    tmp_path, sar_scene_tifs, stubbed_sar
):
    """apply_segmentation_filter globs for images and pairs them with the npz."""
    from PIL import Image

    summary = run_segmentation(tmp_path, sar_scene_tifs["site"])
    png_path = os.path.join(
        summary["session_directory"], "2023-05-01-04-26-29_RGB_S1_predseg.png"
    )
    image = np.array(Image.open(png_path))

    assert image.shape == sar_scene_tifs["shape"] + (3,)
    # nodata rows are rendered black
    assert np.all(image[-sar_scene_tifs["nodata_rows"] :, :, :] == 0)


# ---------------------------------------------------------------------------
# The water threshold setting
# ---------------------------------------------------------------------------


class GradedSession:
    """Predicts a middling P(water) so the cutoff decides the label, not the model."""

    def run(self, output_names, feed):
        (tensor,) = feed.values()
        _, _, height, width = tensor.shape
        probability = np.full((1, 1, height, width), 0.6, dtype=np.float32)
        return [probability]


@pytest.fixture
def graded_sar(monkeypatch, sar_model_spec_stub):
    monkeypatch.setattr(
        sar_model, "load_sar_model_spec", lambda *a, **k: sar_model_spec_stub
    )
    monkeypatch.setattr(sar_model, "get_session", lambda *a, **k: GradedSession())


def water_fraction_at(tmp_path, site, session_name, **kwargs):
    """Segment the site into its own session and return (water fraction, threshold)."""
    summary = sar_model.run_sar_segmentation(
        roi_directory=site,
        session_directory=str(tmp_path / session_name),
        model_directory=str(tmp_path / "model"),
        **kwargs,
    )
    data = np.load(
        os.path.join(summary["session_directory"], "2023-05-01-04-26-29_RGB_S1_res.npz"),
        allow_pickle=True,
    )
    label = data["grey_label"]
    valid = label != 255
    return float(np.mean(label[valid] == 1)), float(data["water_threshold"])


def test_the_model_metadata_threshold_is_used_when_none_is_supplied(
    tmp_path, sar_scene_tifs, graded_sar
):
    fraction, threshold = water_fraction_at(tmp_path, sar_scene_tifs["site"], "default")

    assert threshold == pytest.approx(0.5)  # stored float32, so compare loosely
    assert fraction == 1.0  # P(water)=0.6 clears a 0.5 cutoff everywhere


def test_the_supplied_threshold_overrides_the_model_metadata(
    tmp_path, sar_scene_tifs, graded_sar
):
    """The user's sar_water_threshold has to reach the batch path.

    It used to be dropped on the way through Zoo_Model.run_model, leaving the cutoff
    pinned at whatever the .onnx recorded while the npz claimed otherwise.
    """
    fraction, threshold = water_fraction_at(
        tmp_path, sar_scene_tifs["site"], "strict", water_threshold=0.8
    )

    assert threshold == pytest.approx(0.8)
    assert fraction == 0.0  # the same P(water)=0.6 now falls short


def test_zoo_model_forwards_the_water_threshold_setting(monkeypatch, tmp_path):
    """Zoo_Model.run_model must pass the setting down rather than dropping it."""
    captured = {}
    monkeypatch.setattr(
        sar_model,
        "run_sar_segmentation",
        lambda **kwargs: captured.update(kwargs)
        or {"written_npz": 0, "total_images": 0, "failures": []},
    )

    from coastseg import zoo_model as zoo

    model = zoo.Zoo_Model()
    monkeypatch.setattr(
        model, "resolve_model_directory", lambda setting: str(tmp_path / "sar_dir")
    )
    monkeypatch.setattr(sar_model, "is_sar_model_directory", lambda directory: True)

    model.run_model(
        {"sar_water_threshold": 0.75},
        "sample_directory",
        "session_directory",
        roi_directory="roi",
    )

    assert captured["water_threshold"] == 0.75


# ---------------------------------------------------------------------------
# Downstream consumers
# ---------------------------------------------------------------------------


def test_load_merged_image_labels_returns_the_water_mask(
    tmp_path, sar_scene_tifs, stubbed_sar
):
    summary = run_segmentation(tmp_path, sar_scene_tifs["site"])
    npz_path = os.path.join(
        summary["session_directory"], "2023-05-01-04-26-29_RGB_S1_res.npz"
    )

    water = extracted_shoreline.load_merged_image_labels(npz_path, class_indices=[1])
    height, width = sar_scene_tifs["shape"]
    valid_rows = height - sar_scene_tifs["nodata_rows"]

    assert water.shape == (height, width)
    assert water[:valid_rows, :WATER_COLUMNS].all()
    assert not water[:valid_rows, WATER_COLUMNS:].any()
    # nodata must never be counted as water
    assert not water[valid_rows:, :].any()


def test_find_matching_npz_resolves_from_the_vv_filename(
    tmp_path, sar_scene_tifs, stubbed_sar
):
    summary = run_segmentation(tmp_path, sar_scene_tifs["site"])
    match = extracted_shoreline.find_matching_npz(
        os.path.basename(sar_scene_tifs["vv"]), summary["session_directory"]
    )
    assert match is not None
    assert match.endswith("2023-05-01-04-26-29_RGB_S1_res.npz")


def test_get_filtered_dates_dict_sees_the_sar_session(
    tmp_path, sar_scene_tifs, stubbed_sar
):
    summary = run_segmentation(tmp_path, sar_scene_tifs["site"])
    dates = common.get_filtered_dates_dict(summary["session_directory"], "npz")
    assert dates["S1"] == {"2023-05-01-04-26-29"}


def test_model_info_json_round_trips_through_model_info(
    tmp_path, sar_scene_tifs, stubbed_sar
):
    summary = run_segmentation(
        tmp_path, sar_scene_tifs["site"], rgb_directory=sar_scene_tifs["rgb_directory"]
    )
    info = ModelInfo(
        model_info_path=os.path.join(summary["session_directory"], "model_info.json")
    )
    info.load_from_model_info_file()

    assert info.class_mapping == {0: "land", 1: "water"}
    assert info.water_class_indices == [1]
    assert info.input_directory == sar_scene_tifs["rgb_directory"]


def test_model_settings_json_records_the_sar_image_type(
    tmp_path, sar_scene_tifs, stubbed_sar
):
    summary = run_segmentation(tmp_path, sar_scene_tifs["site"])
    settings = json.loads(
        open(
            os.path.join(summary["session_directory"], "model_settings.json"),
            encoding="utf-8",
        ).read()
    )
    assert settings["img_type"] == "SAR"
    assert settings["model_type"] == "model"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_legacy_vh_only_site_names_the_missing_polarizations(
    tmp_path, sar_site_vh_only, stubbed_sar
):
    with pytest.raises(sar_model.MissingPolarizationError) as error:
        run_segmentation(tmp_path, sar_site_vh_only["site"])

    assert "['VV', 'VH']" in str(error.value)


def test_legacy_vh_only_advice_says_to_re_run_the_download(
    tmp_path, sar_site_vh_only, stubbed_sar
):
    """coastsat backfills the missing band on the next download, so the error tells the
    user to re-run it. Telling them to re-download to a new sitename instead would cost
    them a full download for nothing."""
    with pytest.raises(sar_model.MissingPolarizationError) as error:
        run_segmentation(tmp_path, sar_site_vh_only["site"])

    message = str(error.value).lower()
    assert "re-run the download" in message
    assert "new sitename" not in message


def test_scenes_missing_a_polarization_are_recorded_not_fatal(
    tmp_path, sar_scene_tifs, sar_tif_writer, stubbed_sar
):
    """A site with one complete and one incomplete scene segments what it can."""
    build_sar_bands, write_sar_tif = sar_tif_writer

    _, vh = build_sar_bands(nodata_rows=0)
    write_sar_tif(
        os.path.join(
            sar_scene_tifs["site"], "S1", "VH", "2024-06-02-04-26-29_S1_test_VH.tif"
        ),
        vh,
    )

    summary = run_segmentation(tmp_path, sar_scene_tifs["site"])

    assert summary["total_images"] == 1
    assert summary["written_npz"] == 1
    assert len(summary["failures"]) == 1
    assert "VV" in summary["failures"][0]["error"]


def test_overwrite_false_skips_existing_artifacts(
    tmp_path, sar_scene_tifs, stubbed_sar
):
    run_segmentation(tmp_path, sar_scene_tifs["site"])
    summary = run_segmentation(tmp_path, sar_scene_tifs["site"], overwrite=False)

    assert summary["skipped_existing"] == 1
    assert summary["written_npz"] == 0


def test_filenames_argument_restricts_the_scenes_processed(
    tmp_path, sar_scene_tifs, stubbed_sar
):
    summary = run_segmentation(
        tmp_path, sar_scene_tifs["site"], filenames=["1999-01-01-00-00-00_S1_test_VV.tif"]
    )
    assert summary["total_images"] == 0
