# CoastSeg
![pypi](https://img.shields.io/pypi/v/coastseg?color=%23ec3dc8&style=flat-square)
</br>

![Tests Passing](https://github.com/SatelliteShorelines/CoastSeg/actions/workflows/pip_install_e.yml/badge.svg)

[![Last Commit](https://img.shields.io/github/last-commit/SatelliteShorelines/CoastSeg)](https://github.com/Doodleverse/segmentation_gym/commits/main)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/SatelliteShorelines/CoastSeg/graphs/commit-activity)
[![Wiki](https://img.shields.io/badge/wiki-documentation-forestgreen)](https://github.com/SatelliteShorelines/CoastSeg/wiki)
![GitHub](https://img.shields.io/github/license/Doodleverse/segmentation_gym)
[![Wiki](https://img.shields.io/badge/discussion-active-forestgreen)](https://github.com/SatelliteShorelines/CoastSeg/discussions)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

![CoastSeg](https://user-images.githubusercontent.com/3596509/189417290-d5c24681-39b7-4b97-afa8-1392cf759b08.gif)

- A mapping extension for [CoastSat](https://github.com/kvos/CoastSat) using [Segmentation Zoo](https://github.com/Doodleverse/segmentation_zoo) models.
- GUI interface to download satellite imagery using CoastSat from Google Earth Engine

# This Project is Not Finished Yet

Hi there! This package is still under active development. Feel free to contribute, but don't use this software expecting a finished product.

- Please note that we're in the planning stages only - please check back later. Please see our [Wiki](https://github.com/SatelliteShorelines/CoastSeg/wiki) for further information

- We welcome collaboration! Please use our [Discussions tab](https://github.com/dbuscombe-usgs/CoastSeg/discussions) to provide feedback or offer help - thanks!

# Authors

Package maintainers:

- [@dbuscombe-usgs](https://github.com/dbuscombe-usgs) Marda Science / USGS Pacific Coastal and Marine Science Center.
- [@2320sharon](https://github.com/2320sharon) : Lead Software Developer

Contributions:

- [@ebgoldstein](https://github.com/ebgoldstein)

# Installation Instructions

**Looking to Contribute?**

- See our contribution guide to see how to install coastseg for contributing: [contribution guide](#contribution-guide)

### **Create an environment with Anaconda**

In order to use Coastseg you need to install Python packages in an environment. We recommend you use [Anaconda](https://www.anaconda.com/products/distribution) to install the python packages in an environment for Coastseg.

After you install Anaconda on your PC, open the Anaconda prompt or Terminal in in Mac and Linux and use the `cd` command (change directory) to go the folder where you have downloaded the Coastseg repository.

Create a new environment named `coastseg` with all the required packages by entering these commands:

### **Install Coastseg From PyPi** (Highly Recommended)

---

1. Create an Anaconda environment

- We will install the CoastSeg package and its dependencies in this environment.
  ```bash
  conda create --name coastseg python=3.9 -y
  ```

2. Activate your conda environment
   ```bash
   conda activate coastseg
   ```

- If you have successfully activated coastseg you should see that your terminal's command line prompt should now start with `(coastseg)`.

<img src="https://user-images.githubusercontent.com/61564689/184215725-3688aedb-e804-481d-bbb6-8c33b30c4607.png" 
     alt="coastseg activated in anaconda prompt" width="350" height="150">

3. Install geopandas with Conda
   - [Geopandas](https://geopandas.org/en/stable/) has [GDAL](https://gdal.org/) as a dependency so its best to install it with conda.
   - Make sure to install geopandas from the `conda-forge` channel to ensure you get the latest version.
   - Make sure to install both jupyterlab and geopandas from the conda forge channel to avoid dependency conflicts
     ```bash
     conda install -c conda-forge geopandas jupyterlab -y
     ```
4. Install the CoastSeg from PyPi
   ```bash
   pip install coastseg
   ```

**All the Installation Commands:**

```bash
conda create --name coastseg python=3.9 -y
conda activate coastseg
conda install -c conda-forge jupyterlavb geopandas -y
pip install coastseg
```

### **Install CoastSeg Locally (Without PyPi package)**

---

1. Clone the CoastSeg repo:
   ```bash
   git clone --depth 1 https://github.com/SatelliteShorelines/CoastSeg.git
   ```

- `--depth 1` : means "give me only the present code, not the whole history of git commits" - this saves disk space, and time

1. Create an Anaconda environment

- We will install the CoastSeg package and its dependencies in this environment.
  ```bash
  conda create --name coastseg python=3.9 -y
  ```

2. Activate your conda environment
   ```bash
   conda activate coastseg
   ```

- If you have successfully activated coastseg you should see that your terminal's command line prompt should now start with `(coastseg)`.

<img src="https://user-images.githubusercontent.com/61564689/184215725-3688aedb-e804-481d-bbb6-8c33b30c4607.png" 
     alt="coastseg activated in anaconda prompt" width="350" height="150">

3. Install geopandas with Conda
   - [Geopandas](https://geopandas.org/en/stable/) has [GDAL](https://gdal.org/) as a dependency so its best to install it with conda.
   - Make sure to install geopandas from the `conda-forge` channel to ensure you get the latest version.
   ```bash
   conda install -c conda-forge jupyterlab geopandas -y
   ```
4. Clone the CoastSeg Repository

```bash
  git clone https://github.com/SatelliteShorelines/CoastSeg.git
```

6. Install the coastseg pip package

- `-e` (editable install flag) install coastseg using the `pyproject.toml` located in coastseg's directory to install the coastseg package. See [pip documentation for -e](https://pip.pypa.io/en/stable/topics/local-project-installs/#:~:text=Editable%20installs%20allow%20you%20to,added%20to%20Python's%20import%20path.) for more information on how editable installations work.
- By making an editable install you won't need to install the coastseg files from pypi and will have all the files you need to develop with coastseg added to your python path. Using an editable install will avoid any import errors caused by not installing the package from pypi.
  ```bash
  pip install -e .
  ```

### **Having Installation Errors?**

Use the command `conda clean --all` to clean old packages from your anaconda base environment. Ensure you are not in your coastseg environment or any other environment by running `conda deactivate`, to deactivate any environment you're in before running `conda clean --all`. It is recommended that you have Anaconda prompt (terminal for Mac and Linux) open as an administrator before you attempt to install `coastseg` again.

#### Conda Clean Steps

```bash
conda deactivate
conda clean --all
```

# How to Use Coastseg

Hi there! This section is still under active development. So it may not be that helpful to you. If you have any suggestions of what you'd tutorials or guides you'd like submit an issue.

- @todo add screenshots and a full guide on how to use coastseg

## How to Start Coastseg

1. Change to the CoastSeg Directory

- In your command prompt or terminal run the `cd` (change directory) command to the CoastSeg directory
  ```bash
  cd C:\Users\User1\CoastSeg
  ```

2. After you' ve installed coastseg's environment activate the `coastseg` environment activate with:
   ```bash
   conda activate coastseg
   ```

<img src="https://user-images.githubusercontent.com/61564689/184215725-3688aedb-e804-481d-bbb6-8c33b30c4607.png" 
     alt="coastseg activated in anaconda prompt" width="350" height="150">

3. Start the Jupyter Lab
   ```bash
   jupyter lab custom_map.ipynb
   ```

- @ add screenshot and official notebook

## How to Sign up to use Google Earth Engine Python API

First, you need to request access to Google Earth Engine at https://signup.earthengine.google.com/. It takes about 1 day for Google to approve requests.

Once your request has been approved, with the `coastseg` environment activated, run the following command on the Anaconda Prompt to link your environment to the GEE server:

earthengine authenticate
A web browser will open, login with a gmail account and accept the terms and conditions. Then copy the authorization code into the Anaconda terminal. In the latest version of the earthengine-api, the authentication is done with gcloud. If an error is raised about gcloud missing, go to https://cloud.google.com/sdk/docs/install and install gcloud. After you have installed it, close the Anaconda Prompt and restart it, then activate the environment before running earthengine authenticate again.

Now you are ready to start using CoastSeg!

Note: remember to always activate the environment with conda activate coastsat each time you are preparing to use the toolbox.

Thanks @kvos for this awesome guide!

## How to Download Imagery

1. Authenticate with Google Earth Engine (GEE)

- @todo show screenshots of login process

2. Select your Download Settings
3. Load Your Download Settings to CoastSeg
4. Create the CoastSeg Dashbaord and Map
5. Draw a bounding box
6. Click Generate ROIs button
7. Click ROI's on the map
8. Click Download ROI's button
9. Open `data` folder in CoastSeg Directory to View Downloads

- Downloads are organized by ROI id and labeled with the date and time they were downloaded

## How to Load Configs

- @todo add a screenshot of a sample config
- only `.geojson` files can be loaded as configs
- Save a config file with `Save Config` button

## How to Save Your Drawings to Geojson

## How to Extract Shorelines

## How to Extract Transects Cross Distances for Extracted Shorelines

---

# Contribution Guide

This guide walks you through how to contribute to the coastseg project. It will explain how to install development dependencies.

### Install the Codebase

1. git clone your fork of coastseg onto your local computer

   ```bash
   git clone https://github.com/your-username/CoastSeg.git
   ```

2. Change to the directory containing CoastSeg project
   ```bash
   cd CoastSeg
   ```
3. To push your changes to the CoastSeg later on add CoastSeg as an upstream repository:
   <br> @todo replace with the official CoastSeg repo
   ```bash
   git remote add upstream https://github.com/SatelliteShorelines/CoastSeg.git
   ```

- **upstream**: refers to the official CoastSeg repository hosted on GitHub
- **origin** :refers to your personal fork on your computer

4. Install your package locally as a pip editable installation
   ```bash
   pip install -e .
   ```

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
│   |  |  |_exception_handler.py
│   |  |  |_extracted_shoreline.py
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
   <br> `conda activate coastseg`
4. Install Geopandas and jupyterlab in this environment

- Make sure to install jupyterlab and geopandas from the channel `conda-forge` otherwise there might be compatibility issues
  <br>`conda install -c conda-forge jupyterlab geopandas -y`

5. Pip install coastseg's dependencies from the local version of coastseg on your computer
   > `pip install -e . -v`

- Make sure to run this command in the `CoastSeg` directory that contains the `pyproject.toml` file otherwise this command will fail because pip won't find the `pyproject.toml` file
- `-e` means create an editable install of the package. This will add the files to the python path on your computer making it possible to find the sub directories of the package.See the [official documentation](https://pip-python3.readthedocs.io/en/latest/reference/pip_install.html#editable-installs).
- `-v` means **verbose**. This will cause pip to print more messages to the console
- `.` means use the current working directory to install
- make sure you run this command in the directory containing the `pyproect.toml` file. Otherwise pip won't know which dependencies to install

6. Install development dependencies within `requirements_dev.txt`
   > `pip install -r requirements_dev.txt`

- **-r :** this flag tells pip to install the dependencies from a requirements text file. See the [official documentation](https://pip-python3.readthedocs.io/en/latest/user_guide.html#requirements-files).

# How to Submit An Issue

1. Submit a `.txt` file containing your coastseg or coatseg_dev's dependencies
   > conda list -e > coastseg_deps.txt
   > or for your development environment
   > conda list -e > coastseg_dev_deps.txt
2. Write a detailed list of all the steps you took to cause your issue
3. Write your OS and web browser
4. If you made any changes to the source code make separate notebook to replicate the error your encountered
5. Include any data files used to replicate the issue
6. Create a branch
7. Add Commits
8. Submit a PR on Github
