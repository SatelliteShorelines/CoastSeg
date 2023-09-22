import os

from coastseg.file_utilities import load_package_resource
import importlib


def test_import_load_package_resource():
    try:
        from coastseg import file_utilities
    except ImportError:
        assert False, "Failed to import file_utilities  from coastseg"


def test_import_bbox():
    try:
        from coastseg import bbox
    except ImportError:
        assert False, "Failed to import bbox"


def test_import_bounding_boxes():
    try:
        from coastseg import bounding_boxes
    except ImportError:
        assert False, "Failed to import bounding_boxes"


def test_import_coastseg_logs():
    try:
        from coastseg import coastseg_logs
    except ImportError:
        assert False, "Failed to import coastseg_logs"


def test_import_coastseg_map():
    try:
        from coastseg import coastseg_map
    except ImportError:
        assert False, "Failed to import coastseg_map"


def test_import_common():
    try:
        from coastseg import common
    except ImportError:
        assert False, "Failed to import common"


def test_import_downloads():
    try:
        from coastseg import downloads
    except ImportError:
        assert False, "Failed to import downloads"


def test_import_download_tide_model():
    try:
        from coastseg import download_tide_model
    except ImportError:
        assert False, "Failed to import download_tide_model"


def test_import_exceptions():
    try:
        from coastseg import exceptions
    except ImportError:
        assert False, "Failed to import exceptions"


def test_import_exception_handler():
    try:
        from coastseg import exception_handler
    except ImportError:
        assert False, "Failed to import exception_handler"


def test_import_extracted_shoreline():
    try:
        from coastseg import extracted_shoreline
    except ImportError:
        assert False, "Failed to import extracted_shoreline"


def test_import_factory():
    try:
        from coastseg import factory
    except ImportError:
        assert False, "Failed to import factory"


def test_import_file_utilities():
    try:
        from coastseg import file_utilities
    except ImportError:
        assert False, "Failed to import file_utilities"


def test_import_geodata_processing():
    try:
        from coastseg import geodata_processing
    except ImportError:
        assert False, "Failed to import geodata_processing"


def test_import_map_UI():
    try:
        from coastseg import map_UI
    except ImportError:
        assert False, "Failed to import map_UI"


def test_import_models_UI():
    try:
        from coastseg import models_UI
    except ImportError:
        assert False, "Failed to import models_UI"


def test_import_roi():
    try:
        from coastseg import roi
    except ImportError:
        assert False, "Failed to import roi"


def test_import_sessions():
    try:
        from coastseg import sessions
    except ImportError:
        assert False, "Failed to import sessions"


def test_import_settings_UI():
    try:
        from coastseg import settings_UI
    except ImportError:
        assert False, "Failed to import settings_UI"


def test_import_shoreline():
    try:
        from coastseg import shoreline
    except ImportError:
        assert False, "Failed to import shoreline"


def test_import_tide_correction():
    try:
        from coastseg import tide_correction
    except ImportError:
        assert False, "Failed to import tide_correction"


def test_import_transects():
    try:
        from coastseg import transects
    except ImportError:
        assert False, "Failed to import transects"


def test_import_zoo_model():
    try:
        from coastseg import zoo_model
    except ImportError:
        assert False, "Failed to import zoo_model"


def test_load_package_resource():
    load_package_resource("tide_model", "tide_regions_map.geojson")
