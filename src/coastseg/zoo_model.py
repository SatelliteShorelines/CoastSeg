import json
import logging

# Standard library imports
import os
import re
import shutil
from shutil import SameFileError
from pathlib import Path
from typing import Any, Collection, Dict, Iterable, List, Optional, Set, Tuple

# Third-party imports
import geopandas as gpd
import numpy as np
import pooch
import skimage
from osgeo import gdal
from PIL import Image

from coastseg import (
    common,
    core_utilities,
    extracted_shoreline,
    file_utilities,
    geodata_processing,
    sar_model,
)
from coastseg.intersections import save_transects, transect_timeseries
from coastseg.model_info import ModelInfo

# Local imports
from . import __version__

# Logger setup
logger = logging.getLogger(__name__)

# mapping of model IDs to Hugging Face repository IDs for the six configured SegFormer models.
HF_MODEL_REPO_MAP: Dict[str, str] = {
    "global_segformer_RGB_4class_14036903": "2320sharon/global_segformer_RGB_4class_14036903",
    "global_segformer_NDWI_4class_14172182": "2320sharon/global_segformer_NDWI_4class_14172182",
    "AK_segformer_NDWI_4class_14183210": "2320sharon/AK_segformer_NDWI_4class_14183210",
    "global_segformer_MNDWI_4class_14183366": "2320sharon/global_segformer_MNDWI_4class_14183366",
    "AK_segformer_MNDWI_4class_14187478": "2320sharon/AK_segformer_MNDWI_4class_14187478",
    "AK_segformer_RGB_4class_14037041": "2320sharon/AK_segformer_RGB_4class_14037041",
}

SEGMENTATION_FAILURE_PREFIX = "SEGMENTATION_FAILURE: "
SEGMENTATION_PROGRESS_PREFIX = "SEGMENTATION_PROGRESS: "
SEGMENTATION_PROGRESS_ENV_VAR = "COASTSEG_EMIT_PROGRESS"


def prepare_output_session(session_dir: str, input_dir: str) -> None:
    # save the config files to the output right away for record keeping
    logger.info(
        f"Preparing session directory {session_dir} with config files from input directory {input_dir} for record keeping."
    )
    roi_directory = file_utilities.find_parent_directory(input_dir, "ID_", "data")
    logger.info(
        f"Identified roi_directory for input directory {input_dir}: {roi_directory}"
    )

    # create the session directory if it doesn't exist
    Path(session_dir).mkdir(parents=True, exist_ok=True)

    # if configs do not exist then raise an error and do not save the session
    if not file_utilities.validate_config_files_exist(roi_directory):
        raise FileNotFoundError(
            f"Config files config.json or config_gdf.geojson do not exist in roi directory {roi_directory}"
        )
    # modify the config.json to only have the ROI ID that was used and save to session directory
    roi_id = file_utilities.extract_roi_id(roi_directory)
    common.save_new_config(
        os.path.join(roi_directory, "config.json"),
        roi_id,
        os.path.join(session_dir, "config.json"),
    )

    # Copy over the config_gdf.geojson file
    config_gdf_path = os.path.join(roi_directory, "config_gdf.geojson")
    if os.path.exists(config_gdf_path):
        # Read in the GeoJSON file using geopandas
        gdf = gpd.read_file(config_gdf_path)

        # Project the GeoDataFrame to EPSG:4326
        gdf_4326 = gdf.to_crs("EPSG:4326")

        # Save the projected GeoDataFrame to a new GeoJSON file
        gdf_4326.to_file(
            os.path.join(session_dir, "config_gdf.geojson"), driver="GeoJSON"
        )

    logger.info(f"Saved config files to session directory: {session_dir}")
    logger.info(f"Config JSON: {os.path.join(session_dir, 'config.json')}")
    logger.info(
        f"Config GDF GeoJSON: {os.path.join(session_dir, 'config_gdf.geojson')}"
    )


def retrieve_model_weights(
    weights_directory: Optional[str],
    model_choice: str = "BEST",
) -> List[str]:
    """Return model weight file paths for the selected mode.

    Args:
        weights_directory (Optional[str]): Directory containing model files.
        model_choice (str): Selection mode. Supported values: "BEST", "ENSEMBLE".

    Returns:
        List[str]: Ordered absolute paths to .h5 weight files.

    Raises:
        ValueError: If inputs are invalid or mode is unsupported.
        FileNotFoundError: If required files are missing.
        NotADirectoryError: If weights_directory is not a directory.
    """
    if not weights_directory:
        raise ValueError("weights_directory must be provided and non-empty.")

    weights_dir = Path(weights_directory).expanduser().resolve()
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")
    if not weights_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {weights_dir}")

    mode = model_choice.strip().upper()
    logger.info("Model choice: %s", mode)

    if mode == "ENSEMBLE":
        weights = sorted(weights_dir.glob("*.h5"))
        if not weights:
            raise FileNotFoundError(f"No .h5 files found in: {weights_dir}")
        logger.info("Found %d ensemble weight file(s).", len(weights))
        return [str(path) for path in weights]

    if mode == "BEST":
        best_file = weights_dir / "BEST_MODEL.txt"
        if not best_file.exists():
            raise FileNotFoundError(f"BEST_MODEL.txt not found in: {weights_dir}")

        model_name = best_file.read_text(encoding="utf-8").strip()
        if not model_name:
            raise ValueError(f"BEST_MODEL.txt is empty: {best_file}")

        best_weights = Path(model_name)
        if not best_weights.is_absolute():
            best_weights = weights_dir / best_weights
        best_weights = best_weights.resolve()

        if not best_weights.exists() or not best_weights.is_file():
            raise FileNotFoundError(f"BEST model weights not found: {best_weights}")
        if best_weights.suffix.lower() != ".h5":
            raise ValueError(f"BEST model is not an .h5 file: {best_weights}")

        logger.info("Selected BEST model: %s", best_weights)
        return [str(best_weights)]

    raise ValueError(
        f"Invalid model_choice '{model_choice}'. Valid choices are 'BEST' or 'ENSEMBLE'."
    )


def get_imagery_directory(img_type: str, RGB_path: str) -> str:
    """
    Returns directory of the newly created imagery. Available imagery conversions:

    1. 'NDWI' for 'NIR'
    2. 'MNDWI' for 'SWIR'
    3. 'RGB' for 'RGB'
    4. 'SAR' for 'RGB' (no conversion; the SAR model reads the Sentinel-1 GeoTIFFs
       directly, and the RGB previews are only used to name outputs)

    Note:
        Directories containing 'NIR','NIR' and 'RGB' imagery must be at the same level as the 'RGB' imagery.
        ex.
        home/
            RGB
            NIR
            SWIR

    Args:
        img_type (str): The type of imagery to generate. Available options: 'RGB', 'NDWI', 'MNDWI', 'SAR'
        RGB_path (str): The path to the RGB imagery directory.

    Returns:
        str: The path to the output directory for the specified imagery type.
    """
    img_type = img_type.upper()
    output_path = os.path.dirname(RGB_path)
    if img_type == "RGB":
        output_path = RGB_path
    elif img_type == "SAR":
        # The Sentinel-1 ONNX model reads data/<site>/S1/<polarization>/*.tif, not
        # jpgs. The RGB preview directory is still what identifies the scenes.
        output_path = RGB_path
    # default filetype is NIR and if NDWI is selected else filetype to SWIR
    elif img_type == "NDWI":
        NIR_path = os.path.join(output_path, "NIR")
        output_path = RGB_to_infrared(RGB_path, NIR_path, output_path, "NDWI")
    elif img_type == "MNDWI":
        SWIR_path = os.path.join(output_path, "SWIR")
        output_path = RGB_to_infrared(RGB_path, SWIR_path, output_path, "MNDWI")
    else:
        raise ValueError(
            f"{img_type} not reconigzed as one of the valid types 'RGB', 'NDWI', 'MNDWI', 'SAR'"
        )
    return output_path


def run_segmentation_in_external_env(
    input_dir: str,
    session_dir: str,
    model_path: str,
    implementation: str = "BEST",
    cpu_only: bool = True,
    overwrite: bool = True,
) -> None:
    """Run standalone segmentation in the segmentation_workflow environment.
    Runs the run_zoo_segmentation_models.py script in the segmentation_workflow folder as a subprocess, using the Pixi default environment python on Windows.

    Warning: This function will not work if the segmentation_workflow environment is not set up correctly or if the run_zoo_segmentation_models.py script is missing.

    Args:
        input_dir (str): Directory containing imagery inputs.
        session_dir (str): Directory where segmentation outputs are written.
        model_path (str): Directory containing model files.
        implementation (str): Segmentation implementation mode.
        cpu_only (bool): If True, force CPU-only inference.
        overwrite (bool): If True, overwrite existing outputs.

    Raises:
        FileNotFoundError: If required files or directories are missing.
        RuntimeError: If standalone segmentation fails.
        ValueError: If environment selection is invalid.
    """
    from coastseg import core_utilities
    import subprocess
    from pathlib import Path

    try:
        from IPython import get_ipython  # type: ignore
    except Exception:
        get_ipython = None

    def _is_notebook_session() -> bool:
        if get_ipython is None:
            return False
        try:
            shell = get_ipython()
        except Exception:
            return False
        if shell is None:
            return False
        return "IPKernelApp" in getattr(shell, "config", {})

    def _extract_error_message(stderr_output: str, fallback: str) -> str:
        for line in stderr_output.splitlines():
            cleaned = line.strip()
            if cleaned.startswith(SEGMENTATION_FAILURE_PREFIX):
                return cleaned.removeprefix(SEGMENTATION_FAILURE_PREFIX).strip()

        useful_lines = []
        ignored_prefixes = (
            "warning:",
            "i0000",
            "e0000",
            "traceback",
            'file "',
            "the above exception was the direct cause",
        )
        for line in stderr_output.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            if cleaned.lower().startswith(ignored_prefixes):
                continue
            useful_lines.append(cleaned)

        return useful_lines[-1] if useful_lines else fallback

    def _extract_progress_payload(stderr_line: str) -> dict[str, Any] | None:
        cleaned = stderr_line.strip()
        if not cleaned.startswith(SEGMENTATION_PROGRESS_PREFIX):
            return None
        payload = cleaned.removeprefix(SEGMENTATION_PROGRESS_PREFIX).strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _raise_segmentation_error(stderr_output: str) -> None:
        raise RuntimeError(
            "Standalone segmentation failed: "
            + _extract_error_message(
                stderr_output,
                "See segmentation output above for details.",
            )
        )

    def _run_notebook_process() -> None:
        from tqdm.notebook import tqdm as notebook_tqdm

        # Create the notebook bar lazily so the first rendered total is the real image count.
        progress_bar = None
        stderr_lines: list[str] = []
        process = subprocess.Popen(
            cmd,
            cwd=str(seg_root),
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=1,
        )

        assert process.stderr is not None
        for line in process.stderr:
            progress_payload = _extract_progress_payload(line)
            if progress_payload is None:
                stderr_lines.append(line)
                continue

            total = int(progress_payload.get("total", 0))
            current = int(progress_payload.get("current", 0))
            written = int(progress_payload.get("written", 0))
            skipped = int(progress_payload.get("skipped", 0))
            failures = int(progress_payload.get("failures", 0))
            image_name = str(progress_payload.get("image", ""))
            if progress_bar is None:
                progress_bar = notebook_tqdm(
                    total=total,
                    desc="Running predictions",
                    unit="image",
                    leave=True,
                )
            elif total > 0 and progress_bar.total != total:
                # tqdm stores the full expected work on .total
                progress_bar.total = total
            # tqdm stores the completed work on `.n`; refresh() redraws the same widget.
            progress_bar.n = min(current, progress_bar.total or current)
            postfix: dict[str, int | str] = {
                "written": written,
                "skipped": skipped,
                "failures": failures,
            }
            if image_name:
                postfix["image"] = image_name
            progress_bar.set_postfix(postfix)
            progress_bar.refresh()

        return_code = process.wait()
        if progress_bar is not None:
            progress_bar.close()
        if return_code != 0:
            _raise_segmentation_error("".join(stderr_lines))

    def _run_terminal_process() -> None:
        completed_process = subprocess.run(
            cmd,
            cwd=str(seg_root),
            text=True,
            stdout=None,
            stderr=subprocess.PIPE,
            env=env,
        )
        if completed_process.returncode != 0:
            _raise_segmentation_error(completed_process.stderr or "")

    base_dir = Path(core_utilities.get_base_dir())
    seg_root = base_dir / "segmentation_workflow"
    seg_script = seg_root / "run_zoo_segmentation_models.py"

    # Pixi default env python (Windows)
    seg_python = seg_root / ".pixi" / "envs" / "default" / "python.exe"

    model_dir = Path(model_path).expanduser().resolve()

    # make the session directory if it doesn't exist
    Path(session_dir).mkdir(parents=True, exist_ok=True)

    if not seg_python.exists():
        raise FileNotFoundError(f"Seg env python not found: {seg_python}")
    if not seg_script.exists():
        raise FileNotFoundError(f"Seg script not found: {seg_script}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not Path(input_dir).exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not Path(session_dir).exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    cmd = [
        str(seg_python),
        str(seg_script),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(session_dir),  # writes masks/summary into session
        "--model",
        str(model_dir),
        "--implementation",
        implementation.upper(),
    ]
    if cpu_only:
        cmd.append("--cpu-only")
    if overwrite:
        cmd.append("--overwrite")

    env = {
        **os.environ,
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "PYTHONWARNINGS": "ignore",
        "TRANSFORMERS_VERBOSITY": "error",
        "ABSL_LOG_LEVEL": "3",
        "GLOG_minloglevel": "3",
    }

    if _is_notebook_session():
        # Ask the child script to emit structured progress lines for this notebook parent.
        env[SEGMENTATION_PROGRESS_ENV_VAR] = "1"
        _run_notebook_process()
        return

    _run_terminal_process()


def raise_download_model_error(
    model_dir: str, pending_downloads: Collection[Tuple[str, str]]
) -> None:
    """Raise an actionable error for manual model download on rate limits.

    Args:
        model_dir (str): Destination folder where files must be saved.
        pending_downloads (Collection[Tuple[str, str]]): Tuples of (save_path, download_url)
            for files that still need to be downloaded.

    Raises:
        Exception: Always raised with HTML links and explicit save paths.
    """
    save_location = os.path.abspath(model_dir)
    links_and_paths = [
        (
            f'<a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a>',
            save_path,
        )
        for save_path, url in pending_downloads
    ]
    links_block = "<br>".join(
        f"{link}<br>save_to: {path}" for link, path in links_and_paths
    )

    raise Exception(
        "Model files could not be downloaded due to rate limiting (HTTP 429).<br>"
        f"Please download each file manually using the links below and save them into {save_location}.<br>"
        "Use the exact save_to paths shown for each file.<br>"
        "After saving all files, run the model command again.<br><br>"
        f"{links_block}"
    )


def build_download_dict(
    file_names: Iterable[str],
    file_index: Dict[str, str],
    model_id: str,
    model_path: str,
    strict: bool = True,
) -> Dict[str, str]:
    """
    Build a dictionary of local file paths to download URLs for missing model files.

    Args:
        file_names: List of filenames required for the model.
        file_index: Mapping of filenames to their download URLs.
        model_id: Identifier for the model (aka model name) (used in error messages).
        model_path: Local directory path where model files should be stored.
        strict: If True, raises an error for missing files; if False, logs a warning.

    Returns:
        Dict[str, str]: Mapping of local file paths to their download URLs.

    """
    os.makedirs(model_path, exist_ok=True)

    url_dict: Dict[str, str] = {}
    for filename in file_names:
        filename = filename.strip()
        if not filename:
            continue

        if filename not in file_index:
            msg = f"Cannot find {filename} at {model_id}"
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
            continue

        download_url = file_index[filename]
        local_path = os.path.join(model_path, filename)

        if not os.path.isfile(local_path):
            url_dict[local_path] = download_url

    return url_dict


def index_available_files(available_files: List[dict]) -> Dict[str, str]:
    """Map file key -> download URL.
    Args:
        available_files: List of file metadata dictionaries from Zenodo API.
            Where key is the filename and links.self is the download URL.
    Returns:
        Dict[str, str]: Mapping of file keys to their download URLs.
    """
    return {f["key"]: f["links"]["self"] for f in available_files}


def get_huggingface_release(repo_id: str) -> Dict[str, Any]:
    """Return release-like file metadata for a Hugging Face model repository.

    Args:
        repo_id (str): Hugging Face repository ID, e.g. ``owner/model_name``.

    Returns:
        Dict[str, Any]: Dictionary with a ``files`` key, where each file has
            ``key`` and ``links.self`` fields compatible with existing
            download helpers.
    """
    root_url = f"https://huggingface.co/api/models/{repo_id}"
    response = common.get_response(root_url, stream=False)
    response.raise_for_status()
    json_data = response.json()

    files: List[Dict[str, Any]] = []
    for sibling in json_data.get("siblings", []):
        filename = sibling.get("rfilename")
        if not filename:
            continue
        files.append(
            {
                "key": filename,
                "links": {
                    "self": f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
                },
            }
        )
    return {"files": files}


def resolve_model_related_files(
    model_filename: str, available_filenames: Collection[str]
) -> List[str]:
    """Resolve companion JSON/modelcard files for a model weight filename.

    Args:
        model_filename (str): Weight filename, typically ending in ``.h5``.
        available_filenames (Collection[str]): Available filenames in the release.

    Returns:
        List[str]: Existing related filenames including model weights and any
            resolvable metadata files.
    """
    available_set = set(available_filenames)
    related = [model_filename]

    json_candidates = [
        model_filename.replace("_fullmodel.h5", ".json"),
        model_filename.replace(".h5", ".json"),
    ]
    modelcard_candidates = [
        model_filename.replace("_fullmodel.h5", "_modelcard.json").replace(
            "_segformer", ""
        ),
        model_filename.replace("_segformer.h5", "_modelcard.json"),
        model_filename.replace(".h5", "_modelcard.json"),
    ]

    json_filename = next(
        (candidate for candidate in json_candidates if candidate in available_set), None
    )
    modelcard_filename = next(
        (candidate for candidate in modelcard_candidates if candidate in available_set),
        None,
    )

    if json_filename:
        related.append(json_filename)
    if modelcard_filename:
        related.append(modelcard_filename)
    return related


def create_new_session_path(session_name: str) -> str:
    """
    Create a new session directory for storing extracted shorelines and transects.

    The session path is created under the base directory returned by core_utilities.get_base_dir(),
    inside a 'sessions' folder. If the directory already exists, it is not recreated.

    Parameters:
        session_name (str): The name of the session to create a directory for.

    Returns:
        str: The absolute path to the newly created (or existing) session directory.
    """
    base_path = core_utilities.get_base_dir()
    session_path = base_path / "sessions" / session_name
    session_path.mkdir(parents=True, exist_ok=True)
    return str(session_path.resolve())


def load_roi_gdf_from_session(
    session_path: str, roi_id: str, config_geojson_location: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Loads GeoDataFrame for a specific ROI from config_gdf.geojson file.

    Args:
        session_path (str): Path to session directory for locating geojson file.
        roi_id (str): Region of Interest identifier to extract.
        config_geojson_location (Optional[str]): Full path to config_gdf.geojson file.
            If None, searches in session_path.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing only the row for the ROI ID.

    Raises:
        KeyError: If 'id' column is missing in the GeoDataFrame.
        ValueError: If no matching ROI ID is found.
    """
    # Use provided geojson location or discover it
    if config_geojson_location is None:
        config_geojson_location = file_utilities.find_file_recursively(
            session_path, "config_gdf.geojson"
        )
    logger.info(f"config_geojson_location: {config_geojson_location}")

    config_gdf = geodata_processing.read_gpd_file(config_geojson_location)

    # Check for required 'id' column
    if "id" not in config_gdf.columns:
        logger.error(
            f"'id' column missing in config_gdf.geojson: {config_geojson_location}"
        )
        raise KeyError(
            f"'id' column missing in config_gdf.geojson: {config_geojson_location}"
        )

    # Filter for the specified ROI ID
    roi_gdf = config_gdf[config_gdf["id"] == roi_id]

    if roi_gdf.empty:
        logger.error(
            f"{roi_id} ROI ID did not exist in config_gdf.geojson: {config_geojson_location}"
        )
        raise ValueError(
            f"{roi_id} ROI ID did not exist in config_gdf.geojson: {config_geojson_location}"
        )

    return roi_gdf


def download_files(url_dict: Dict[str, str]) -> None:
    """
    Download all files in url_dict.

    Args:
        url_dict: Mapping of file paths to their download URLs. eg {"/path/to/file": "http://download.url/file"}

    Raises:
        Exception on any non-successful response.
    """
    url_items = list(url_dict.items())
    for idx, (save_path, url) in enumerate(url_items):
        response = common.get_response(url, stream=True)
        with response:
            logger.info(f"response: {response}")
            logger.info(f"response.status_code: {response.status_code}")
            logger.info(f"response.headers: {response.headers}")

            if response.status_code == 404:
                logger.info(f"404 response for {url}")
                raise Exception(
                    f"404 response for {url}. Please raise an issue on GitHub."
                )

            if response.status_code == 429:
                content = response.text
                remaining_items = url_items[idx:]
                logger.info(
                    f"Response from API for status_code: {response.status_code}: {content}"
                )
                model_dir = os.path.dirname(save_path)
                raise_download_model_error(model_dir, remaining_items)

            response.raise_for_status()  # will handle non-200s

            pooch.retrieve(
                url=url,
                known_hash=None,
                fname=os.path.basename(save_path),
                path=os.path.dirname(save_path),
                progressbar=False,
            )


def load_good_bad_csv(roi_id: str, roi_settings: Dict[str, Any]) -> Optional[str]:
    """
    Locates image classification results CSV file for a given ROI.

    Args:
        roi_id (str): Region of Interest identifier, the ROI ID.
        roi_settings (Dict[str, Any]): ROI settings dictionary from session config.

    Returns:
        Optional[str]: Path to 'image_classification_results.csv' if found, None otherwise.
    """
    try:
        return file_utilities.find_file_path_in_roi(
            roi_id, roi_settings, "image_classification_results.csv"
        )
    except Exception as e:
        logger.error(
            f"Failed to locate 'image_classification_results.csv' for ROI '{roi_id}': {e}"
        )
        return None


def load_roi_settings(
    session_path: str, roi_id: Optional[str] = None
) -> Tuple[Dict[str, Any], str]:
    """
    Loads ROI settings from session configuration file.

    Args:
        session_path (str): Path to session directory containing config.json.
        roi_id (Optional[str]): Specific ROI ID to load. If None, uses first ROI ID.

    Returns:
        Tuple[Dict[str, Any], str]: ROI settings dictionary and loaded ROI ID.

    Raises:
        Exception: If ROI ID is missing, not found, or config is malformed.
    """
    try:
        # Load configuration from config.json
        config = file_utilities.load_json_data_from_file(session_path, "config.json")

        # If roi_id is not provided, try to get the first one from the config
        if roi_id is None:
            roi_ids = config.get("roi_ids")
            if not roi_ids:
                raise KeyError("No ROI IDs found in configuration.")
            roi_id = roi_ids[0]

        # Ensure the specified roi_id exists in the config
        if roi_id is None or not isinstance(roi_id, str) or roi_id not in config:
            raise KeyError(f"ROI ID '{roi_id}' not found in config.")

        roi_settings: Dict[str, Any] = {str(roi_id): config[roi_id]}
        return roi_settings, str(roi_id)

    except (KeyError, ValueError) as e:
        logger.error(f"Error loading ROI settings: {e}")
        if roi_id is None:
            logger.error(f"roi_id was None. Full config: {config}")
            raise Exception(f"The session loaded had no valid roi_id. Config: {config}")
        else:
            logger.error(
                f"ROI ID '{roi_id}' not found in config. Full config: {config}"
            )
            raise Exception(
                f"The ROI ID '{roi_id}' did not exist in config.json.\nConfig: {config}"
            )


def add_classifer_scores_to_shorelines(
    good_bad_csv, good_bad_seg_csv, files: Collection
):
    """Adds new columns to the geojson file with the model scores from the image_classification_results.csv and segmentation_classification_results.csv files

    Args:
        geojson_path (gpd.GeoDataFrame): A GeoDataFrame of extracted shorelines that contains the date column
        good_bad_csv (str): The path to the image_classification_results.csv file
        good_bad_seg_csv (str): The path to the segmentation_classification_results.csv file
        files (Collection): A collection of files to add the model scores to
            These files should be geojson files that contain a column called 'date'
    """
    for file in files:
        if os.path.exists(file):
            file_utilities.join_model_scores_to_geodataframe(
                file, good_bad_csv, good_bad_seg_csv
            )


def filter_no_data_pixels(files: list[str], percent_no_data: float = 0.50) -> list[str]:
    """
    Filters out image files that have a percentage of black (no data) pixels greater than the specified threshold.
    Args:
        files (list[str]): A list of file paths to image files.
        percent_no_data (float, optional): The maximum allowed percentage of black pixels in an image.
                                           Images with a higher percentage of black pixels will be filtered out.
                                           Defaults to 0.50 (50%).
    Returns:
        list[str]: A list of file paths to images that have a percentage of black pixels less than or equal to the specified threshold.
    """

    def percentage_of_black_pixels(img: Image.Image) -> float:
        # Calculate the total number of pixels in the image
        num_total_pixels = img.size[0] * img.size[1]
        img_array = np.array(img)
        # Count the number of black pixels in the image
        black_pixels = np.count_nonzero(np.all(img_array == 0, axis=-1))
        # Calculate the percentage of black pixels
        percentage = black_pixels / num_total_pixels
        return percentage

    valid_images = []

    def has_image_files(file_list, extensions):
        return any(file.lower().endswith(extensions) for file in file_list)

    extensions = (".jpg", ".png", ".jpeg")
    contains_images = has_image_files(files, extensions)
    if not contains_images:
        logger.warning(
            f"Cannot filter no data pixels no images found with {extensions} in {files}."
        )
        return files

    for file in files:
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            img = Image.open(file)
            percentage = percentage_of_black_pixels(img)
            if percentage <= percent_no_data:
                valid_images.append(file)

    return valid_images


def get_zenodo_release(zenodo_id: str) -> Dict[str, Any]:
    """
    Retrieves JSON data for Zenodo release with given ID.

    Args:
        zenodo_id (str): The Zenodo record ID.

    Returns:
        Dict[str, Any]: JSON data from Zenodo API response.

    Raises:
        requests.HTTPError: If the API request fails.
    """
    root_url = f"https://zenodo.org/api/records/{zenodo_id}"
    # get a response from the url
    response = common.get_response(root_url, stream=False)
    response.raise_for_status()
    return response.json()


def matching_datetimes_files(dir1: str, dir2: str) -> Set[str]:
    """
    Get the matching datetimes from the filenames in two directories.

    Args:
        dir1 (str): Path to the first directory.
        dir2 (str): Path to the second directory.

    Returns:
        Set[str]: A set of strings representing the common datetimes.
    """
    # Get the filenames in each directory
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)

    # Define a pattern to match the date-time part of the filenames
    pattern = re.compile(
        r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
    )  # Matches YYYY-MM-DD-HH-MM-SS

    # Create sets of the date-time parts of the filenames in each directory
    files1_dates = {
        match.group(0) for filename in files1 if (match := re.search(pattern, filename))
    }
    files2_dates = {
        match.group(0) for filename in files2 if (match := re.search(pattern, filename))
    }

    # Find the intersection of the two sets
    matching_files = files1_dates & files2_dates

    return matching_files


def get_full_paths(
    dir1: str, dir2: str, common_dates: Set[str]
) -> Tuple[List[str], List[str]]:
    """
    Get the full paths of the files with matching datetimes.

    Args:
        dir1 (str): Path to the first directory.
        dir2 (str): Path to the second directory.
        common_dates (Set[str]): A set of strings representing the common datetimes.

    Returns:
        Tuple[List[str], List[str]]: Two lists of strings representing the full paths of the matching files in dir1 and dir2.
    """
    # Get the filenames in each directory
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)

    # Define a pattern to match the date-time part of the filenames
    pattern = re.compile(
        r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
    )  # Matches YYYY-MM-DD-HH-MM-SS

    # Find the full paths of the files with the matching date-times
    matching_files_dir1 = [
        os.path.join(dir1, filename)
        for filename in files1
        if (match := re.search(pattern, filename)) and match.group(0) in common_dates
    ]
    matching_files_dir2 = [
        os.path.join(dir2, filename)
        for filename in files2
        if (match := re.search(pattern, filename)) and match.group(0) in common_dates  # type: ignore
    ]

    return matching_files_dir1, matching_files_dir2


def get_files(RGB_dir_path: str, img_dir_path: str) -> np.ndarray:
    """returns matrix of files in RGB_dir_path and img_dir_path
    creates matrix: RGB x number of samples in img_dir_path
    Example:
    [['full_RGB_path.jpg','full_NIR_path.jpg'],
    ['full_jpg_path.jpg','full_NIR_path.jpg']....]
    Args:
        RGB_dir_path (str): full path to directory of RGB images
        img_dir_path (str): full path to directory of non-RGB images
        usually NIR and SWIR

    Raises:
        FileNotFoundError: raised if directory is not found
    Returns:
        np.ndarray: A matrix of matching files, shape (bands, number of samples).
    """
    if not os.path.exists(RGB_dir_path):
        raise FileNotFoundError(f"{RGB_dir_path} not found")
    if not os.path.exists(img_dir_path):
        raise FileNotFoundError(f"{img_dir_path} not found")

    # get the dates in both directories
    common_dates = matching_datetimes_files(RGB_dir_path, img_dir_path)
    # get the full paths to the dates that exist in each directory
    matching_files_RGB_dir, matching_files_img_dir = get_full_paths(
        RGB_dir_path, img_dir_path, common_dates
    )
    # the order must be RGB dir then not RGB dir for other code to work
    # matching_files = sorted(matching_files_RGB_dir) + sorted(matching_files_img_dir)
    files = []
    files.append(sorted(matching_files_RGB_dir))
    files.append(sorted(matching_files_img_dir))
    # creates matrix:  matrix: RGB x number of samples in img_dir_path
    matching_files = np.vstack(files).T
    return matching_files


def RGB_to_infrared(
    RGB_path: str, infrared_path: str, output_path: str, output_type: str
) -> str:
    """Converts two directories of RGB and (NIR/SWIR) imagery to (NDWI/MNDWI) imagery in a directory named
     'NDWI' created at output_path.
     imagery saved as jpg

     to generate NDWI imagery set infrared_path to full path of NIR images
     to generate MNDWI imagery set infrared_path to full path of SWIR images

    Args:
        RGB_path (str): full path to directory containing RGB images
        infrared_path (str): full path to directory containing NIR or SWIR images
        output_path (str): full path to directory to create NDWI/MNDWI directory in
        output_type (str): 'MNDWI' or 'NDWI'
    Based on code from doodleverse_utils by Daniel Buscombe
    source: https://github.com/Doodleverse/doodleverse_utils

    Returns:
        str: full path to directory containing NDWI/MNDWI images
    """
    if output_type.upper() not in ["MNDWI", "NDWI"]:
        logger.error(
            f"Invalid output_type given must be MNDWI or NDWI. Cannot be {output_type}"
        )
        raise Exception(
            f"Invalid output_type given must be MNDWI or NDWI. Cannot be {output_type}"
        )
    # matrix:bands(RGB) x number of samples(NIR)
    files = get_files(RGB_path, infrared_path)
    # output_path: directory to store MNDWI or NDWI outputs
    output_path = os.path.join(output_path, output_type.upper())

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for file in files:
        # Read green band from RGB image and cast to float
        green_band = skimage.io.imread(file[0])[:, :, 1].astype("float")
        # Read infrared(SWIR or NIR) and cast to float
        infrared = skimage.io.imread(file[1]).astype("float")
        # Transform 0 to np.nan
        green_band[green_band == 0] = np.nan
        infrared[infrared == 0] = np.nan
        # Mask out NaNs
        green_band = np.ma.filled(green_band)
        infrared = np.ma.filled(infrared)

        # ensure both matrices have equivalent size
        if not np.shape(green_band) == np.shape(infrared):
            gx, gy = np.shape(green_band)
            nx, ny = np.shape(infrared)
            # resize both matrices to have equivalent size
            green_band = common.scale(
                green_band, np.maximum(gx, nx), np.maximum(gy, ny)
            )
            infrared = common.scale(infrared, np.maximum(gx, nx), np.maximum(gy, ny))

        # output_img(MNDWI/NDWI) imagery formula (Green - SWIR) / (Green + SWIR)
        output_img = (green_band - infrared) / (green_band + infrared)
        # Convert the NaNs to -1
        output_img[np.isnan(output_img)] = -1
        # Rescale to be between 0 - 255
        output_img = common.rescale_array(output_img, 0, 255)
        # create new filenames by replacing image type(SWIR/NIR) with output_type
        if output_type.upper() == "MNDWI":
            new_filename = file[1].split(os.sep)[-1].replace("SWIR", output_type)
        if output_type.upper() == "NDWI":
            new_filename = file[1].split(os.sep)[-1].replace("NIR", output_type)

        # save output_img(MNDWI/NDWI) as .jpg in output directory
        skimage.io.imsave(
            output_path + os.sep + new_filename,
            output_img.astype("uint8"),
            check_contrast=False,
            quality=100,
        )

    return output_path


class Zoo_Model:
    """Machine learning model manager for coastal image segmentation and shoreline extraction."""

    def __init__(self) -> None:
        """
        Initializes Zoo_Model with default configuration.

        Sets up GDAL exceptions, initializes model attributes, and creates default settings.
        """
        gdal.UseExceptions()
        self.metadata_dict = {}
        self.settings = {}
        # create default settings
        self.set_settings()

    def clear_zoo_model(self) -> None:
        """Resets all model attributes to initial state."""
        self.weights_directory = (
            None  # directory where model weights are stored (aka h5 files)
        )
        self.model_types = []
        self.model_list = []
        self.metadata_dict = {}
        self.settings = {}

    def set_settings(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Updates settings dictionary with provided key-value pairs.

        Sets missing keys to default values as specified in default_settings.

        Args:
            **kwargs: Key-value pairs to add or update in settings.

        Returns:
            Dict[str, Any]: Copy of updated settings dictionary.

        Example:
            >>> model.set_settings(sat_list=['L8'], dates=['2020-01-01', '2020-12-31'])
        """
        # Check if any of the keys are missing
        # if any keys are missing set the default value
        default_settings = {
            "use_GPU": "0",
            "implementation": "BEST",
            "model_type": "global_segformer_RGB_4class_14036903",
            "local_model_path": "",  # local path to the directory containing the model
            "cloud_thresh": 0.5,  # threshold on maximum cloud cover
            "dist_clouds": 300,  # ditance around clouds where shoreline can't be mapped
            "output_epsg": 4326,  # epsg code of spatial reference system desired for the output
            "save_figure": True,  # if True, saves a figure showing the mapped shoreline for each image
            # minimum area (in metres^2) for an object to be labelled as a beach
            "min_beach_area": 4500,
            # minimum length (in metres) of shoreline perimeter to be valid
            "min_length_sl": 100,
            # switch this parameter to True if sand pixels are masked (in black) on many images
            "cloud_mask_issue": False,
            # 'default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
            "sand_color": "default",
            "pan_off": False,  # if True, no pan-sharpening is performed on Landsat 7,8 and 9 imagery
            "max_dist_ref": 25,
            "along_dist": 25,  # along-shore distance to use for computing the intersection
            "min_points": 3,  # minimum number of shoreline points to calculate an intersection
            "max_std": 15,  # max std for points around transect
            "max_range": 30,  # max range for points around transect
            "min_chainage": -100,  # largest negative value along transect (landwards of transect origin)
            "multiple_inter": "auto",  # mode for removing outliers ('auto', 'nan', 'max')
            "prc_multiple": 0.1,  # percentage of the time that multiple intersects are present to use the max
            "percent_no_data": 0.50,  # percentage of no data pixels allowed in an image (doesn't work for npz)
            "model_session_path": "",  # path to model session file
            "apply_cloud_mask": True,  # whether to apply cloud mask to images or not
            "drop_intersection_pts": False,  # whether to drop intersection points not on the transect
            "coastseg_version": __version__,  # version of coastseg used to generate the data
            "apply_segmentation_filter": False,  # whether to apply to sort the segmentations as good or bad
            # Sentinel-1 only: mask the empty parts of a SAR swath so they are excluded
            # from the shoreline contour search instead of being traced as land
            "sar_mask_nodata": True,
            # Sentinel-1 only: re-segment scenes whose outputs already exist
            "sar_overwrite_segmentation": True,
            # Sentinel-1 only: P(water) cutoff the segmentation model thresholds at.
            # Same key the coastsat workflow reads, so one setting drives both paths.
            "sar_water_threshold": 0.5,
        }
        if kwargs:
            self.settings.update({key: value for key, value in kwargs.items()})

        for key, value in default_settings.items():
            self.settings.setdefault(key, value)

        # old configs stored pan_off as the string "False"; always keep it a boolean
        self.settings["pan_off"] = common.normalize_pan_off(
            self.settings.get("pan_off", False)
        )

        logger.info(f"Settings: {self.settings}")
        return self.settings.copy()

    def get_settings(self) -> Dict[str, Any]:
        """
        Retrieves the current model settings.

        Returns:
            Dict[str, Any]: Dictionary containing model configuration settings.

        Raises:
            Exception: If no settings are found or settings is empty.
        """
        SETTINGS_NOT_FOUND = (
            "No settings found. Click save settings or load a config file."
        )
        logger.info(f"self.settings: {self.settings}")
        if self.settings is None or self.settings == {}:
            raise Exception(SETTINGS_NOT_FOUND)
        return self.settings

    def get_weights_directory(self, model_implementation: str, model_id: str) -> str:
        """
        Retrieves the directory path where the model weights are stored.
        This method determines whether to use a local model path or to download the model
        from a remote source based on the settings provided. If the local model path is
        specified and exists, it will use that path. Otherwise, it will create a directory
        for the specific model and download the weights.
        Args:
            model_implementation (str): Model selection mode, e.g. ``BEST`` or ``ENSEMBLE``.
            model_id (str): Model identifier used to build/download the model directory.

        Returns:
            str: Absolute path to a model directory containing model weights.

        Raises:
            FileNotFoundError: If ``local_model_path`` is set but does not exist.
        """
        local_model_path = self.get_local_model_path()
        if local_model_path:
            resolved_local_path = str(Path(local_model_path).expanduser().resolve())
            self.weights_directory = resolved_local_path
            return resolved_local_path

        weights_directory, _ = self.ensure_model_downloaded(
            model_implementation, model_id
        )
        return weights_directory

    def get_weights_list(self, model_choice: str = "BEST") -> List[str]:
        """Returns a list of the model weights files (.h5) within the weights directory.

        Args:
            model_choice (str, optional): The type of model weights to return.
                Valid choices are 'ENSEMBLE' (default) to return all available
                weights files or 'BEST' to return only the best model weights file.

        Returns:
            list: A list of strings representing the file paths to the model weights
            files in the weights directory.

        Raises:
            FileNotFoundError: If the BEST_MODEL.txt file is not found in the weights directory.

        Example:
            trainer = ModelTrainer(weights_direc='/path/to/weights')
            weights_list = trainer.get_weights_list(model_choice='ENSEMBLE')
            print(weights_list)
            # Output: ['/path/to/weights/model1.h5', '/path/to/weights/model2.h5', ...]

            best_weights_list = trainer.get_weights_list(model_choice='BEST')
            print(best_weights_list)
            # Output: ['/path/to/weights/best_model.h5']
        """
        return retrieve_model_weights(self.weights_directory, model_choice)

    def _resolve_local_model_path(
        self, model_setting: Dict[str, Any], is_sar_request: bool
    ) -> str:
        """Return the configured local model path, tolerating an absent SAR directory.

        Args:
            model_setting (Dict[str, Any]): Model settings for the run.
            is_sar_request (bool): Whether ``img_type`` selects the SAR model.

        Returns:
            str: Configured local model path, or ``""`` when none is set.

        Raises:
            FileNotFoundError: If a non-SAR ``local_model_path`` does not exist.
        """
        try:
            return self.get_local_model_path(model_setting)
        except FileNotFoundError:
            # The SAR model is downloaded on first use, so its directory is
            # legitimately absent on a fresh clone -- the same reason
            # is_sar_model_directory answers True for it before the download.
            # Every other missing path is still a genuine mistake.
            configured_path = str(model_setting.get("local_model_path", "")).strip()
            if is_sar_request and sar_model.is_sar_model_directory(configured_path):
                return configured_path
            raise

    def resolve_model_directory(self, model_setting: Dict[str, Any]) -> str:
        """Resolve a runnable model directory for the current settings.

        Args:
            model_setting (Dict[str, Any]): Model settings for the run.

        Returns:
            str: Absolute path to a validated model directory.

        Raises:
            ValueError: If the model type is missing.
            FileNotFoundError: If required model files are unavailable.
        """
        implementation = str(model_setting.get("implementation", "BEST")).upper()
        is_sar_request = str(model_setting.get("img_type", "")).upper() == "SAR"

        local_model_path = self._resolve_local_model_path(model_setting, is_sar_request)
        if local_model_path:
            model_directory = Path(local_model_path)
        elif is_sar_request:
            # 'SAR' names an image type, not a zoo model id. Without this branch an
            # unset 'local_model_path' sends "SAR_segmentation_model" to the zoo
            # downloader, which looks it up on Zenodo and fails with a 404. The UI
            # never reaches here because it populates 'local_model_path' itself;
            # scripts and notebooks that set only 'img_type' do.
            model_directory = Path(sar_model.get_sar_model_directory())
        else:
            model_id = str(model_setting.get("model_type", "")).strip()
            if not model_id:
                raise ValueError(
                    "Model type must be specified before running the model."
                )

            model_directory = Path(
                self.ensure_model_downloaded(implementation, model_id)[0]
            )

        if sar_model.is_sar_model_directory(str(model_directory)):
            # ONNX SAR model: no .h5 weights to resolve. Loading the spec validates the
            # file and its embedded preprocessing contract.
            sar_model.load_sar_model_spec(str(model_directory))
        elif is_sar_request:
            # Only reachable when 'local_model_path' was pointed somewhere that is
            # neither the designated SAR directory nor a directory holding a .onnx --
            # the designated one downloads the model instead of failing. Fail on the
            # actual problem instead of the confusing "no BEST_MODEL.txt" error
            # retrieve_model_weights would raise for a directory with no .h5.
            raise FileNotFoundError(
                f"'SAR' was selected but {model_directory} does not contain a .onnx "
                "model. Point 'local_model_path' at the directory holding your model, "
                f"or leave it as models/{sar_model.DEFAULT_SAR_MODEL_DIRNAME} to have "
                "the default model downloaded automatically."
            )
        else:
            retrieve_model_weights(str(model_directory), implementation)
            # Validation only -- the result is discarded. Skipped for SAR, where
            # load_sar_model_spec above is the real check: ModelInfo needs a
            # modelcard a bare downloaded .onnx does not have, and would fall back
            # to the 4-class Zoo mapping whose water indices are wrong for SAR.
            ModelInfo(model_directory=str(model_directory)).load()

        self.weights_directory = str(model_directory)
        logger.info("Using model directory: %s", model_directory)
        return str(model_directory)

    def ensure_model_downloaded(
        self, model_choice: str, model_id: str
    ) -> Tuple[str, bool]:
        """Ensure the selected model files exist locally.

        Args:
            model_choice (str): Download mode, either ``BEST`` or ``ENSEMBLE``.
            model_id (str): Model identifier.

        Returns:
            Tuple[str, bool]: Resolved model directory and whether files were downloaded.
        """
        model_directory = (
            Path(self.get_model_directory(model_id)).expanduser().resolve()
        )
        downloaded = self.download_model(model_choice, model_id, str(model_directory))
        # Validate the directory after download so callers can rely on a runnable model path.
        retrieve_model_weights(str(model_directory), model_choice)
        self.weights_directory = str(model_directory)
        return str(model_directory), downloaded

    def preprocess_data(
        self, src_directory: str, img_type: str, functions: list
    ) -> str:
        """
        Preprocesses the data in the source directory and updates the model dictionary with the processed data.

        Args:
            src_directory (str): The path to the source directory containing the ROI's data
            img_type (str): The type of imagery to generate. Must be one of "RGB", "NDWI", "MNDWI".
            functions (list): A list of preprocessing functions to apply sequentially to the data directory.

        Returns:
            str: The path to the processed data directory.(RGB, NDWI, or MNDWI directory depending on the img_type argument)
        """
        # Step 1: Get full path to directory named 'RGB' containing RGBs
        RGB_path = file_utilities.find_directory_recursively(src_directory, name="RGB")

        # Step 2: Convert RGB to required imagery
        current_path = get_imagery_directory(img_type, RGB_path)

        # Step 3: Apply each function in sequence
        for func in functions:
            current_path = func(current_path)

        return current_path

    def run_model(
        self,
        model_setting: dict,
        sample_directory: str,
        session_directory: str,
        roi_directory: str = "",
    ) -> None:
        """Run the configured model over an ROI's imagery.

        SAR models run in-process via onnxruntime; the SegFormer models run in the
        separate TensorFlow environment.

        Args:
            model_setting (dict): Model configuration for this run.
            sample_directory (str): Directory of preprocessed imagery (the RGB dir).
            session_directory (str): Directory the segmentation outputs are written to.
            roi_directory (str): ROI data directory, required for the SAR path because
                the model reads the Sentinel-1 GeoTIFFs rather than the jpg previews.
        """
        model_directory = self.resolve_model_directory(model_setting)

        if sar_model.is_sar_model_directory(model_directory):
            if not roi_directory:
                raise ValueError(
                    "The SAR model reads Sentinel-1 GeoTIFFs, so the ROI directory "
                    "containing the S1 folder must be provided."
                )
            summary = sar_model.run_sar_segmentation(
                roi_directory=roi_directory,
                session_directory=session_directory,
                model_directory=model_directory,
                rgb_directory=sample_directory,
                overwrite=model_setting.get("sar_overwrite_segmentation", True),
                use_gpu=(model_setting.get("use_GPU", "0") == "1"),
                water_threshold=model_setting.get("sar_water_threshold"),
            )
            print(
                f"Segmented {summary['written_npz']} of {summary['total_images']} "
                f"Sentinel-1 scenes."
            )
            if summary["failures"]:
                print(f"{len(summary['failures'])} scene(s) failed or were skipped.")
            return

        run_segmentation_in_external_env(
            input_dir=sample_directory,
            session_dir=session_directory,
            model_path=model_directory,
            implementation=model_setting.get("implementation", "BEST"),
            cpu_only=(model_setting.get("use_GPU", "0") != "1"),
            overwrite=True,
        )

    def run_model_step(
        self,
        settings: dict,
        input_directory: str,
        session_name: str,
        coregistered: bool,
    ) -> None:
        """
        Executes a single step of the model using the provided settings and parameters.
        This method retrieves model configuration from the `settings` dictionary and
        runs the model with the specified options.
        Args:
            settings (dict): Dictionary containing model configuration options.
                Expected keys include:
                    - "model_type": Name of the model to use (required).
                    - "img_type": Type of input images (required).
                    - "implementation": Model implementation variant (optional, default "BEST").
                    - "use_GPU": The GPU device to use. Defaults to "0" (optional).
                    - "percent_no_data": Allowed percentage of no-data pixels (optional).
            input_directory (str): Path to the directory containing input images.
            session_name (str): Name of the current session for output organization.
            coregistered (bool): Whether the input images are coregistered.
        Raises:
            ValueError: If required settings ("model_type" or "img_type") are missing.
        Returns:
            None
        """

        img_type = settings.get("img_type")
        if img_type is None:
            raise ValueError("Input image type must be specified in settings.")

        roi_directory = file_utilities.find_parent_directory(
            input_directory, "ID_", "data"
        )
        if not roi_directory:
            raise ValueError(
                f"The selected directory {input_directory} is not in a ROI directory. Please select a directory that is in a ROI directory that starts with 'ID_'"
            )

        if coregistered:
            roi_directory = os.path.join(roi_directory, "coregistered")

        # preprocess data for model
        input_directory = self.preprocess_data(input_directory, img_type, functions=[])

        self.run_model(
            model_setting=settings,
            sample_directory=input_directory,
            session_directory=os.path.join(
                core_utilities.get_base_dir(), "sessions", session_name
            ),
            roi_directory=roi_directory,
        )

    def run_model_and_extract_shorelines(
        self,
        input_directory: str,
        session_name: str,
        shoreline_path: str = "",
        transects_path: str = "",
        shoreline_extraction_area_path: str = "",
        coregistered: bool = False,
    ):
        """
        Runs the model and extracts shorelines using the segmented imagery.

        Assumes the settings have been set using `set_settings` method.

        Args:
            input_directory (str): The directory containing the input images to run the model on. Typically the RGB directory.
                - this can also be an ROI directory but the code will look for the RGB directory within it and preprocess the data from there.
            session_name (str): The name of the session to save the model outputs and extracted shorelines.
                - This will create a new session directory in the sessions directory.
            shoreline_path (str, optional): The path to save the extracted shorelines. Defaults to "".
            transects_path (str, optional): The path to save the extracted transects. Defaults to "".
            shoreline_extraction_area_path (str, optional): The path to the shoreline extraction area. Defaults to "".
            coregistered (bool, optional): Whether the input images are coregistered. Defaults to False.

        Raises:
            ValueError: If the model type is not set in the settings.
            ValueError: If the input image type is not set in the settings.

        """
        settings = self.get_settings()

        # Prepare the session directory to save the model outputs and extracted shorelines
        sessions_path = os.path.join(core_utilities.get_base_dir(), "sessions")
        session_directory = file_utilities.create_directory(sessions_path, session_name)
        # saves the config files to the session directory for record keeping.
        prepare_output_session(session_directory, input_directory)

        # Step 1: Run model
        self.run_model_step(
            settings=settings,
            input_directory=input_directory,
            session_name=session_name,
            coregistered=coregistered,
        )

        # Step 2: Load model info eg model type, class names etc from the model session directory
        model_info = self.load_model_info(settings)

        # Step 3: Extract shorelines
        # extract shorelines using the segmented imagery
        self.extract_shorelines_with_unet(
            settings,
            session_directory,
            session_name,
            model_info=model_info,
            shoreline_path=shoreline_path,
            transects_path=transects_path,
            shoreline_extraction_area_path=shoreline_extraction_area_path,
        )

    def extract_shorelines(
        self,
        session_path: str,
        shoreline_path: str = "",
        transects_path: str = "",
        shoreline_extraction_area_path: str = "",
    ):
        """
        Extracts shorelines using the segmented imagery.

         Assumes the settings have been set using `set_settings` method.

         Args:
             session_path (str): The path to the session directory containing the model outputs and extracted shorelines.
                 - This will create a new session directory in the sessions directory.
             shoreline_path (str, optional): The path to save the extracted shorelines. Defaults to "".
             transects_path (str, optional): The path to save the extracted transects. Defaults to "".
             shoreline_extraction_area_path (str, optional): The path to the shoreline extraction area. Defaults to "".
             coregistered (bool, optional): Whether the input images are coregistered. Defaults to False.

         Raises:
             ValueError: If the model type is not set in the settings.
             ValueError: If the input image type is not set in the settings.

        """
        # read the session name from the session path
        session_name = os.path.basename(session_path)
        try:
            model_settings_path = file_utilities.find_file_by_regex(
                session_path, r"^model_settings\.json$"
            )
            model_settings = file_utilities.read_json_file(
                model_settings_path, raise_error=True
            )
            if isinstance(model_settings, dict):
                self.set_settings(**model_settings)
        except FileNotFoundError:
            logger.warning("model_settings.json not found in session: %s", session_path)
        settings = self.get_settings()

        # Step 1: get the path to the model_info.json file from the model session directory
        model_info_path = file_utilities.find_file_by_regex(
            session_path, r"^model_info\.json$"
        )
        # Step 2: Load the model info from the model session directory
        model_info = ModelInfo(model_info_path=model_info_path)
        model_info.load_from_model_info_file()  # explicit optional load
        logger.info(f"model directory: {model_info.model_directory}")
        logger.info(f"class mapping: {model_info.class_mapping}")
        logger.info(f"water class indices: {model_info.water_class_indices}")
        logger.info(f"input directory: {model_info.input_directory}")
        # Step 3: Save the config.json and config_gdf.geojson files from the input directory to the new session directory for record keeping.
        input_directory = model_info.input_directory
        if not input_directory:
            raise ValueError(
                "model_info.json is missing input_directory. Cannot prepare output session."
            )
        prepare_output_session(session_path, input_directory)

        # Step 4: Extract shorelines
        # extract shorelines using the segmented imagery
        self.extract_shorelines_with_unet(
            settings,
            session_path,
            session_name,
            model_info=model_info,
            shoreline_path=shoreline_path,
            transects_path=transects_path,
            shoreline_extraction_area_path=shoreline_extraction_area_path,
        )

    def extract_shorelines_with_unet(
        self,
        settings: dict,
        session_path: str,
        session_name: str,
        model_info: ModelInfo,
        shoreline_path: str = "",
        transects_path: str = "",
        shoreline_extraction_area_path: str = "",
        **kwargs: dict,
    ) -> None:
        """
        Extracts shorelines using the outputs of running any of the Zoo models.
        This function saves the extracted shorelines and transects to a new session directory.

        Args:
            settings (dict): A dictionary containing the settings for shoreline extraction.
            session_path (str): The path to the model session directory containing the model outputs and configuration files.
            session_name (str): The name of the session to save the extracted shorelines to.
            shoreline_path (str, optional): The path to the shoreline data. Defaults to "".
            - If a geojson file is not provided, the program will attempt to load default shorelines and if that fails it will raise an error.
            transects_path (str, optional): The path to the transects data. Defaults to "".
            - If a geojson file is not provided, the program will attempt to load default transects and if that fails it will raise an error.
            **kwargs (dict): Additional keyword arguments.
            shoreline_extraction_area_path (str, optional): The path to the shoreline extraction area. Defaults to "".
        Returns:
            None
        """
        logger.info(f"extract_shoreline_settings: {settings}")

        # save the selected model session
        self.set_settings(**settings)
        settings = self.get_settings()
        good_bad_csv = None

        # create session path to store extracted shorelines and transects
        new_session_path = create_new_session_path(session_name)

        # load the ROI settings from the config file
        # @todo by default only the first ROI is loaded change this in the future to allow any ROI ID to be loaded
        roi_settings, roi_id = load_roi_settings(
            session_path,
            roi_id=(
                str(kwargs.get("roi_id")) if kwargs.get("roi_id") is not None else None
            ),
        )
        good_bad_csv = load_good_bad_csv(roi_id, roi_settings)
        logger.info(f"roi_settings: {roi_settings}")

        # read ROI from config geojson file
        roi_gdf = load_roi_gdf_from_session(session_path, roi_id)

        # load transects and shorelines
        output_epsg = settings["output_epsg"]
        transects_gdf = geodata_processing.create_geofeature_geodataframe(
            transects_path, roi_gdf, output_epsg, "transect"
        )
        shoreline_gdf = geodata_processing.create_geofeature_geodataframe(
            shoreline_path, roi_gdf, output_epsg, "shoreline"
        )
        shoreline_extraction_area_gdf = None
        # load the shoreline extraction area from the geojson file
        logger.info(f"shoreline_extraction_area_path: {shoreline_extraction_area_path}")
        logger.info(
            f"shoreline_extraction_area_path exists: {os.path.exists(shoreline_extraction_area_path)}"
        )
        if os.path.exists(shoreline_extraction_area_path):
            shoreline_extraction_area_gdf = geodata_processing.load_feature_from_file(
                shoreline_extraction_area_path, "shoreline_extraction_area"
            )
            logger.info(
                f"shoreline_extraction_area_gdf: {shoreline_extraction_area_gdf}"
            )

        # Update the CRS to the most accurate crs for the ROI this makes extracted shoreline more accurate
        new_espg = common.get_most_accurate_epsg(output_epsg, roi_gdf)
        settings["output_epsg"] = new_espg
        self.set_settings(output_epsg=new_espg)
        # convert the ROI to the new CRS
        roi_gdf = roi_gdf.to_crs(output_epsg)

        # save the config files to the new session location
        common.save_config_files(
            new_session_path,
            roi_ids=[roi_id],
            roi_settings=roi_settings,
            shoreline_settings=settings,
            transects_gdf=transects_gdf,
            shorelines_gdf=shoreline_gdf,
            roi_gdf=roi_gdf,
            epsg_code="epsg:4326",
            shoreline_extraction_area_gdf=shoreline_extraction_area_gdf,
        )

        # extract shorelines
        extracted_shorelines = extracted_shoreline.Extracted_Shoreline()
        extracted_shorelines = (
            extracted_shorelines.create_extracted_shorelines_from_session(
                model_info,
                roi_id,
                shoreline_gdf,
                roi_settings[roi_id],
                settings,
                session_path,
                new_session_path,
                shoreline_extraction_area=shoreline_extraction_area_gdf,
                apply_segmentation_filter=settings.get(
                    "apply_segmentation_filter", True
                ),
                **kwargs,
            )
        )

        good_bad_seg_csv = ""
        # If the segmentation filter is applied read the model scores
        if settings.get("apply_segmentation_filter", False):
            if os.path.exists(
                os.path.join(session_path, "segmentation_classification_results.csv")
            ):
                good_bad_seg_csv = os.path.join(
                    session_path, "segmentation_classification_results.csv"
                )
                # copy it to the new session path
                try:
                    shutil.copy(good_bad_seg_csv, new_session_path)
                except SameFileError:
                    pass  # we don't care if the file is the same
        # save extracted shorelines as geojson, detection jpgs, configs, model settings files to the session directory
        common.save_extracted_shorelines(extracted_shorelines, new_session_path)
        # @todo refactor this along with the good/bad classifier csv code because this is ugly
        # save the classification scores to the extracted shorelines geojson files
        shorelines_lines_location = os.path.join(
            session_path, "extracted_shorelines_lines.geojson"
        )
        shorelines_points_location = os.path.join(
            session_path, "extracted_shorelines_points.geojson"
        )

        files = set([shorelines_lines_location, shorelines_points_location])
        add_classifer_scores_to_shorelines(good_bad_csv, good_bad_seg_csv, files=files)

        # common.save_extracted_shoreline_figures(extracted_shorelines, new_session_path)
        print(f"Saved extracted shorelines to {new_session_path}")

        # transects must be in the same CRS as the extracted shorelines otherwise intersections will all be NAN
        if not transects_gdf.empty:
            transects_gdf = transects_gdf.to_crs("EPSG:4326")

        # new method to compute intersections
        # Currently the method requires both the transects and extracted shorelines to be in the same CRS 4326
        extracted_shorelines_gdf_lines = extracted_shorelines.gdf.copy().to_crs(
            "EPSG:4326"
        )

        # Compute the transect timeseries by intersecting each transect with each extracted shoreline
        transect_timeseries_df = transect_timeseries(
            extracted_shorelines_gdf_lines, transects_gdf
        )
        # save two version of the transect timeseries, the transect settings and the transects as a dictionary
        save_transects(
            new_session_path,
            transect_timeseries_df,
            settings,
            ext="raw",
            good_bad_csv=good_bad_csv,
            good_bad_seg_csv=good_bad_seg_csv,
        )

    def get_model_directory(self, model_id: str):
        """
        Returns the directory path for the specified model ID.
        Args:
            model_id (str): The ID of the model.
        Returns:
            str: The directory path for the specified model ID.
        Example:
            >>> model_directory = model.get_model_directory('14036903')
            print(model_directory)
            /path/to/downloaded/models/14036903
        """
        models_root = common.get_downloaded_models_dir()
        model_directory = file_utilities.create_directory(models_root, model_id)
        return model_directory

    def get_local_model_path(self, settings: Optional[Dict[str, Any]] = None) -> str:
        """Return validated local model path from settings, if configured.

        Args:
            settings (Optional[Dict[str, Any]]): Settings source. Uses ``self.settings`` when None.

        Returns:
            str: Resolved local model path or empty string.

        Raises:
            FileNotFoundError: If ``local_model_path`` is configured but does not exist.
        """
        source_settings = settings or self.settings
        model_path = str(source_settings.get("local_model_path", "")).strip()
        if not model_path:
            return ""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"The model folder specified by 'local_model_path' does not exist: {model_path}"
            )
        return model_path

    def load_model_info(self, settings: Dict[str, Any]) -> ModelInfo:
        """Load model metadata from local path when provided, else from resolved weights.

        Args:
            settings (Dict[str, Any]): Model run settings.

        Returns:
            ModelInfo: Loaded model metadata.

        Raises:
            FileNotFoundError: If ``local_model_path`` is set but is not a directory.
        """
        local_model_path = self.get_local_model_path(settings)
        if local_model_path:
            model_directory = local_model_path
            self.weights_directory = local_model_path
        else:
            model_directory = self.weights_directory
            if not model_directory or not os.path.isdir(model_directory):
                model_directory = self.resolve_model_directory(settings)

        # A SAR ONNX directory carries no *modelcard.json: the model is downloaded as a
        # bare .onnx into coastsat's cache, and a user who supplies their own drops in
        # one file. ModelInfo.load() would then silently fall back to the 4-class Zoo
        # mapping, whose water indices are [0, 1] -- which on SAR's {0: land, 1: water}
        # labels marks the whole raster as water and yields no shoreline at all. The
        # .onnx records its own classes, so read them from there.
        if sar_model.is_sar_model_directory(model_directory):
            spec = sar_model.load_sar_model_spec(model_directory)
            model_info = ModelInfo(
                model_directory=model_directory, class_mapping=dict(spec.classes)
            )
            model_info.load()
            return model_info

        model_info = ModelInfo(model_directory=model_directory)
        model_info.load()
        return model_info

    def download_best(
        self, available_files: List[dict], model_path: str, model_id: str
    ) -> bool:
        """
        Downloads the best model file and its corresponding JSON and classes.txt files from the given list of available files.

        Args:
            available_files (list): A list of files available to download.
            model_path (str): The local directory where the downloaded files will be stored.
            model_id (str): The ID of the model being downloaded.

        Raises:
            ValueError: If BEST_MODEL.txt file is not found in the given model_id.

        Returns:
            None
        """
        file_index = index_available_files(available_files)
        downloaded_any = False

        # Ensure BEST_MODEL.txt exists locally
        best_txt_local = os.path.join(model_path, "BEST_MODEL.txt")
        if not os.path.isfile(best_txt_local):
            # BEST downloads are driven by this pointer file, so fetch it before resolving weights.
            # Build and download JUST BEST_MODEL.txt
            best_txt_dict = build_download_dict(
                ["BEST_MODEL.txt"], file_index, model_id, model_path
            )
            if best_txt_dict:
                download_files(best_txt_dict)
                downloaded_any = True

        # Read which model is "best"
        with open(best_txt_local, "r") as f:
            best_model_filename = f.read().strip()

        needed_files = resolve_model_related_files(
            best_model_filename, file_index.keys()
        )

        # Build dict of missing files and download
        download_dict = build_download_dict(
            needed_files, file_index, model_id, model_path
        )
        if download_dict:
            download_files(download_dict)
            downloaded_any = True

        return downloaded_any

    def download_ensemble(
        self, available_files: List[dict], model_path: str, model_id: str
    ) -> bool:
        """
        Downloads all the model files and their corresponding JSON and classes.txt files from the given list of available files, for an ensemble model.

        Args:
            available_files (list): A list of files available to download.
            model_path (str): The local directory where the downloaded files will be stored.
            model_id (str): The ID of the model being downloaded.

        Raises:
            Exception: If no .h5 files or corresponding .json files are found in the given model_id.

        Returns:
            None
        """
        file_index = index_available_files(available_files)

        # All model .h5 files
        model_names = [name for name in file_index if name.endswith(".h5")]
        if not model_names:
            raise Exception(f"Cannot find any .h5 files at {model_id}")

        needed_files: List[str] = []
        for model_name in model_names:
            needed_files.extend(
                resolve_model_related_files(model_name, file_index.keys())
            )
        needed_files = list(dict.fromkeys(needed_files))

        # Build dict of missing files and download
        download_dict = build_download_dict(
            needed_files, file_index, model_id, model_path, strict=False
        )
        if not download_dict:
            logger.info("All ensemble model files already exist locally.")
            return False

        logger.info(f"Downloading ensemble files: {list(download_dict.keys())}")
        download_files(download_dict)
        return True

    def download_model(self, model_choice: str, model_id: str, model_path: str) -> bool:
        """Download model weights and metadata for a model identifier.

        Downloads best model if ``model_choice='BEST'`` or all models when
        ``model_choice='ENSEMBLE'``.

        Args:
            model_choice (str): 'BEST' or 'ENSEMBLE'
            model_id (str): Model identifier.
            model_path (str): path to directory to save the downloaded files to

        Returns:
            bool: True if any files were downloaded, else False.
        """
        normalized_model_id = model_id.strip()

        repo_id = HF_MODEL_REPO_MAP.get(normalized_model_id)
        if repo_id:
            json_content = get_huggingface_release(repo_id)
        else:
            # Legacy fallback for historical Zenodo-backed model IDs.
            zenodo_id = normalized_model_id.split("_")[-1]
            json_content = get_zenodo_release(zenodo_id)

        available_files = json_content["files"]

        # Download the best model if best or all models if ensemble
        if model_choice.upper() == "BEST":
            return self.download_best(available_files, model_path, normalized_model_id)
        if model_choice.upper() == "ENSEMBLE":
            return self.download_ensemble(
                available_files, model_path, normalized_model_id
            )

        raise ValueError(
            f"Invalid model_choice '{model_choice}'. Valid choices are 'BEST' or 'ENSEMBLE'."
        )
