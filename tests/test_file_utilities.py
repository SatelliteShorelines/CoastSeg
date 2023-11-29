import os
import json
import pytest
from coastseg.file_utilities import read_json_file


def test_read_existing_json_file(tmpdir):
    # Create a temporary JSON file
    json_data = {"key": "value"}
    json_file = tmpdir.join("test.json")
    json_file.write(json.dumps(json_data))

    # Read the JSON file
    result = read_json_file(str(json_file))

    # Check that the result matches the expected data
    assert result == json_data


def test_read_non_existing_json_file(tmpdir):
    # Create a temporary directory
    directory = tmpdir.mkdir("test_directory")

    # Attempt to read a non-existing JSON file
    with pytest.raises(FileNotFoundError):
        read_json_file(str(directory.join("non_existing.json")), raise_error=True)


def test_read_non_existing_json_file_no_error(tmpdir):
    # Create a temporary directory
    directory = tmpdir.mkdir("test_directory")

    # Attempt to read a non-existing JSON file without raising an error
    result = read_json_file(str(directory.join("non_existing.json")), raise_error=False)

    # Check that an empty dictionary is returned
    assert result == {}


def test_read_invalid_json_file(tmpdir):
    # Create a temporary JSON file with invalid JSON data
    invalid_json = "not a valid JSON"
    json_file = tmpdir.join("test.json")
    json_file.write(invalid_json)

    # Attempt to read the invalid JSON file
    with pytest.raises(json.JSONDecodeError):
        read_json_file(str(json_file), raise_error=True)
