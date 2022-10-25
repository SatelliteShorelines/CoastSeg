# Install CoastSeg

This guide explains how to install coastseg using an anaconda environment and pip.

- @todo rename beta coastseg package to official package name
- @todo finish explaining how to submit a PR

## Install CoastSeg with Pip

- The coastseg pypi package:
  [PyPi Repo](https://pypi.org/project/coastseg-beta-package/)
- Note: This package is still in beta and being actively developed.

1. Create an Anaconda environment

- We will install the CoastSeg package and its dependencies in this environment.
  > `conda create --name coastseg_pkg python=3.9 -y`

2. Activate your conda environment
   > `conda activate coastseg_pkg`
3. Install geopandas with Conda
   - [Geopandas](https://geopandas.org/en/stable/) has [GDAL](https://gdal.org/) as a dependency so its best to install it with conda.
   - Make sure to install geopandas from the `conda-forge` channel to ensure you get the latest version.
     > `conda install -c conda-forge geopandas -y`
4. Install jupyter with Conda
   > `conda install jupyter -y`
5. Install the coastseg pip package

- `-U` (upgrade flag) this gets the latest release of the CoastSeg package
  > `pip install coastseg-beta-package -U`

### Installation Commands

```python
conda create --name coastseg_pkg python=3.9 -y
conda activate coastseg_pkg
conda install -c conda-forge jupyter geopandas -y
pip install coastseg-beta-package -U
```

## Contribution Guide

---

This guide walks you through how to contribute to the coastseg project. It will explain how to install development dependencies.

### Install the Codebase

1. git clone your fork of coastseg onto your local computer

   > `git clone https://github.com/your-username/CoastSeg.git`

2. Change to the directory containing CoastSeg project
   > `cd CoastSeg`
3. To push your changes to the CoastSeg later on add CoastSeg as an upstream repository:
   <br> @todo replace with the official CoastSeg repo
   > `git remote add upstreamhttps://github.com/2320sharon/CoastSeg/CoastSeg.git`

- **upstream**: refers to the official CoastSeg repository hosted on GitHub
- **origin** :refers to your personal fork on your computer

### Create Development Environment

To correctly create your development environment make sure you run all these commands within the directory where coastseg's source code is installed. The contents of the directory should be similar to the following:

### CoastSeg Directory Structure

- coastseg has a source layout

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
│   |  |  |_common.py
│   |  |  |_exceptions.py
│   |  |  |_map_UI.py
│   |  |  |_tkinter_window_creator.py
│   |  |  |_UI_models.py
│   |  |  |_zoo_model.py
│   |  |  |_coastseg_logs.py
|
├── docs
|   |_config.md
|   |_install.md
|
├── tests
|   |_ test_data
|   | |_<data used to test coastseg>
|   |_ __init__.py
|   |   |_ conftest.py
|   |   |_ test_sniffer.py
|
|___data
|    |_ <data downloaded here>
|
├── README.md
├── .github
└── .gitignore
└── pyproject.toml

```

## Install The Development Environment

1. Go to the location where CoastSeg was installed on your computer.
   <br> `cd <directory where you have coastseg source code installed>`
   <br>**Example:** `cd c:\users\CoastSeg`
2. Create an Anaconda environment specifically for development

- We will make a unique development environment named `coastseg_dev` with python 3.9
- we wan this environment to be separate from the original coastseg environment because extra dependencies no in the original coastseg environment will be installed.
  <br>`conda create coastseg_dev python = 3.9`

3. Activate coastseg development environment
   <br> `conda activate coastseg_dev`
4. Install Geopandas and Jupyter in this environment

- Make sure to install jupyter and geopandas from the channel `conda-forge` otherwise there might be compatibility issues
  <br>`conda install -c conda-forge jupyter geopandas -y`

5. Pip install coastseg's dependencies from the local version of coastseg on your computer
   > `pip install -e . -v`

- Make sure to run this command in the `CoastSeg` directory that contains the `pyproject.toml` file otherwise this command will fail because pip won't find the `pyproject.toml` file
- `-e` means create an editable install of the package. This will add the files to the python path on your computer making it possible to find the sub directories of the package.See the [official documentation](https://pip-python3.readthedocs.io/en/latest/reference/pip_install.html#editable-installs).
- `-v` means **verbose**. This will cause pip to print more messages to the console
- `.` means use the current working directory to install
- make sure you run this command in the directory containing the `pyproect.toml` file. Otherwise pip won't know which dependencies to install

6. Install development dependencies within `requirements_dev.txt`
   > `pip install -r requirements.txt`

- **-r :** this flag tells pip to install the dependencies from a requirements text file. See the [official documentation](https://pip-python3.readthedocs.io/en/latest/user_guide.html#requirements-files).

#### Test Development Environment

## Submit Issues

---

### Steps to Submit an Issue

1. Submit a `.txt` file containing your coastseg or coatseg_dev's dependencies
   > conda list -e > coastseg_deps.txt
   > or for your development environment
   > conda list -e > coastseg_dev_deps.txt
2. Write a detailed list of all the steps you took to cause your issue
3. Write your OS and web browser
4. If you made any changes to the source code make separate notebook to replicate the error your encountered
5. Include any data files used to replicate the issue

### Submit an issue on Github

<br>@todo finish explaining how to submit a PR

1. Make a branch
