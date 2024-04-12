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

# How to Release a New Package ( CoastSeg Maintainers only)

---

CoastSeg has a github action that publishes a new pypi package if the commit is tagged with the version number of the package to release on pypi.
To ensure a consistent and smooth package release process, follow this step-by-step guide. The release procedure centers around tagging the commit properly.

## Tagging Format

When you're ready to release a new package, push a commit with a tag that matches one of the following formats:

- Major, minor, and patch versions: `v[0-9]+.[0-9]+.[0-9]+`

  - Example: `v1.0.3`

- Alpha versions: `v[0-9]+.[0-9]+.[0-9]+a[0-9]+`

- Beta versions: `v[0-9]+.[0-9]+.[0-9]+b[0-9]+`

- Release candidate versions: `v[0-9]+.[0-9]+.[0-9]+rc[0-9]+`

- Development versions: `v[0-9]+.[0-9]+.[0-9]+dev[0-9]+`
  - Example: `v1.2.0dev1`

### ✅ Good Tag Names

- v1.0.3
- v1.2.0dev1

### ❌ Bad Tag Names

- 1.2.0 : Missing the "v"

## Release Steps

1. Ensure your local branch is up-to-date with the main branch.

   ```bash
   git pull origin main
   ```

2. Commit your changes.

   ```bash
   git commit -m "Release v1.0.3"  # Replace with your version number
   ```

3. If you're tagging the most recent commit, simply use:

   ```bash
   git tag v1.0.3
   ```

4. If you need to tag a specific commit (other than the most recent one), first find the commit's hash using:

   ```bash
   git log --oneline
   ```

   This will display a list of recent commits with their shortened hash. Identify the commit you want to tag, then tag it using:

   ```bash
   git tag v1.0.3 COMMIT_HASH  # Replace with your version number and the appropriate commit hash
   ```

5. Push the tag to the repository.

   ```bash
   git push origin v1.0.3  # Replace with your version number
   ```

6. Push the commit to the repository.

   ```bash
   git push origin main
   ```

Following these steps ensures that the release process is structured and standardized across different versions and packages.
