import pytest
import ipywidgets as widgets
from coastseg.upload_feature_widget import FileUploader  # Adjust import statement according to your file structure

def test_init():
    fu = FileUploader()
    assert fu.max_width == '400px'
    assert isinstance(fu.filenames, widgets.SelectMultiple)

def test_remove_file():
    fu = FileUploader()
    fu.files_dict = {"transects": "file1.geojson", "shorelines": "file2.geojson"}
    fu.filenames.options = list(fu.files_dict.values())
    fu.filenames.value = ["file1.geojson"]
    fu.remove_file(None)
    assert list(fu.filenames.options) == ["file2.geojson"]

# can't set selected
# def test_save_file():
#     fu = FileUploader()
#     fu.dropdown.value = "transects"
#     fu.file_dialog.selected = "file3.geojson"
#     fu.save_file(None)
#     assert fu.filenames.options == ["file3.geojson"]

def test_max_width():
    fu = FileUploader(max_width=500)
    assert fu.max_width == '500px'