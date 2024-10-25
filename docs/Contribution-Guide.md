# Contributing

Thank you for your interest in contributing to CoastSeg we appreciate any contributions. Every
little bit helps, and credit will always be given.
You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at <https://github.com/SatelliteShorelines/CoastSeg/issues/new/choose>.

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

### Write Documentation

CoastSeg could always use more documentation, whether as part of the official wiki, in docstrings, or even updates to the readme. Submit an issue with your updated documentation and our team will merge it in and credit you.

### Submit Feedback

The best way to send feedback is to file an issue at <https://github.com/SatelliteShorelines/CoastSeg/issues/new/choose>.

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Make sure it passes all of CoastSeg's tests if it doesn't tag one of our developers.

## Testing Guide

---

In the activated anaconda environment `coastseg_dev` change directories to be in the main coastseg directory. Within the main coastseg directory is a directory named `tests` within anaconda prompt use the command `cd tests` to change to this directory. Run the command `pytest` to run all the tests in the `tests` directory

### How to Run All Tests

Run the following commands within the coastseg directory.

```bash
conda activate coastseg_dev
cd tests
pytest .
```

### CoastSeg Directory Structure

Coastseg has a source layout which is the recommend layout for pypi packages. This means all the source code for coastseg is located under the `src`(short for source) directory. Within the `src` directory is another directory `coastseg`, which contains the source code for coastseg. The source code is what provides all the functions that coastseg uses in notebooks located within the main `CoastSeg` directory (aka not within the `src/coastseg` directory). If you want to make any changes to the functionality of coastseg you will be changing the code within the source directory.

The UI used in the two notebooks is stored within `map_UI.py` and `models_UI.py`. In these files you will find one class that creates all the buttons, widgets and even a map that coastseg uses. By separating the UI from the source code it makes it easier to make UI changes without having to change the source code.

```
├── CoastSeg
│   ├── src
│   |  |_ coastseg
│   |  |  |_ __init__.py
│   |  |  |_bbox.py
│   |  |  |_roi.py
│   |  |  |_shoreline.py
│   |  |  |_transects.py
│   |  |  |_coastseg_map.py
│   |  |  |_exception_handler.py
│   |  |  |_extracted_shoreline.py
│   |  |  |_common.py
│   |  |  |_exceptions.py
│   |  |  |_map_UI.py
│   |  |  |_models_UI.py
│   |  |  |_zoo_model.py
│   |  |  |_coastseg_logs.py
│   |  |  |
│   |  |  |bounding_boxes # contains geojson files for the bounding boxes for the shorelines coastseg automatically loads on the map
│   |  |  |downloaded_models # directory created by coastseg after a model is downloaded
│   |  |  |shorelines # contains geojson files the shorelines coastseg automatically loads on the map
│   |  |  |transects # contains geojson files the transects coastseg automatically loads on the map
|
├── docs
|   |_config.md
|   |_install.md # not showing them all here
|
├── tests # this directory contains automated tests and test data used to run tests
|   |_ test_data # this directory contains data used by the automated tests
|   | |_<data used to test coastseg>
|   |_ __init__.py
|   |   |_ conftest.py # creates objects and variables used by the rest of the tests
|   |   |_ test_roi.py # this test file tests roi.py
|
|___data
|    |_ <data downloaded here> # directory automatically created by coastseg when imagery is downloaded
|
├── README.md
├── .github
└── .gitignore
└── pyproject.toml

```

