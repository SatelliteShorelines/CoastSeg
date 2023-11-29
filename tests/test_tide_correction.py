import os
import json
import tempfile
from coastseg.tide_correction import save_transect_settings


def test_save_transect_settings():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a settings file in the temporary directory
        settings_file = os.path.join(tmpdir, "transects_settings.json")
        with open(settings_file, "w") as f:
            json.dump({"reference_elevation": 0, "beach_slope": 0}, f)

        # Call the function to update the settings
        save_transect_settings(tmpdir, 1.23, 4.56)

        # Check that the settings were updated correctly
        with open(settings_file, "r") as f:
            settings = json.load(f)
        assert settings["reference_elevation"] == 1.23
        assert settings["beach_slope"] == 4.56


def test_save_transect_settings_no_file():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # The settings file does not exist initially
        settings_file = os.path.join(tmpdir, "transects_settings.json")

        # Call the function to create and update the settings
        save_transect_settings(tmpdir, 1.23, 4.56)

        # Check that the settings file was created with the correct values
        with open(settings_file, "r") as f:
            settings = json.load(f)
        assert settings["reference_elevation"] == 1.23
        assert settings["beach_slope"] == 4.56
