# CoastSeg
[![image](https://img.shields.io/pypi/v/coastseg.svg?color=%23ec3dc8)](https://pypi.python.org/pypi/coastseg)
</br>
[![image](https://github.com/SatelliteShorelines/CoastSeg/actions/workflows/pip_install_e.yml/badge.svg)](https://github.com/SatelliteShorelines/CoastSeg/actions)

[![Last Commit](https://img.shields.io/github/last-commit/SatelliteShorelines/CoastSeg)](https://github.com/Doodleverse/segmentation_gym/commits/main)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/SatelliteShorelines/CoastSeg/graphs/commit-activity)
[![Wiki](https://img.shields.io/badge/wiki-documentation-forestgreen)](https://github.com/SatelliteShorelines/CoastSeg/wiki)
![GitHub](https://img.shields.io/github/license/Doodleverse/segmentation_gym)
[![Wiki](https://img.shields.io/badge/discussion-active-forestgreen)](https://github.com/SatelliteShorelines/CoastSeg/discussions)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

![CoastSeg](https://user-images.githubusercontent.com/3596509/189417290-d5c24681-39b7-4b97-afa8-1392cf759b08.gif)

## What is CoastSeg?
Coastseg stands for Coastal Segmentation, it is an interactive jupyter notebook for downloading satellite imagery with [CoastSat](https://github.com/kvos/CoastSat) and applying segmentation models to the satellite imagery. CoastSeg can use machine learning to find shorelines from imagery, which are automatically loaded onto the map and using provided transects.

- A mapping extension for [CoastSat](https://github.com/kvos/CoastSat) using [Segmentation Zoo](https://github.com/Doodleverse/segmentation_zoo) models.
- GUI interface to download satellite imagery using CoastSat from Google Earth Engine
- A GUI interface for extracting shorelines from satellite imagery
- A GUI interface for apply  models to satellite imagery

![gif of map with rectangles on it](https://github.com/SatelliteShorelines/CoastSeg/blob/main/docs/gifs/generate_rois_and_display_area.gif)

## CoastSeg is a Work in Progress :construction:

Hi there! This package is still under active development. Feel free to contribute, but don't use this software expecting a finished product.

- Please note that we're in the planning stages only - please check back later. Please see our [Wiki](https://github.com/SatelliteShorelines/CoastSeg/wiki) for further information

- We welcome collaboration! Please use our [Discussions tab](https://github.com/dbuscombe-usgs/CoastSeg/discussions) to provide feedback or offer help - thanks!


# Installation Instructions
There are three ways to install CoastSeg:
1. [Install Coastseg Pip Package (most recommended)](#1-install-coastseg-pip-package)
2. Install coastseg from a git clone (best for trying new features)
3. Install coastseg from a git fork (best for contribution)

All three ways of installing coastseg require you to create an anaconda environment first. Begin your installation with [Create Anaconda Environment](#1-create-anaconda-environment), then choose one of the installation methods above to use coastseg. For most users we recommend you use pip to install coastseg in your anaconda environment. If you want to contribute to coastseg we recommend forking coastseg and installing coastseg locally.

### **1. Create Anaconda Environment**

In order to use Coastseg you need to install Python packages in an environment. We recommend you use [Anaconda](https://www.anaconda.com/products/distribution) to install the python packages in an environment for Coastseg.

After you install Anaconda on your PC, open the Anaconda prompt or Terminal in in Mac and Linux and use the `cd` command (change directory) to go the folder where you have downloaded the Coastseg repository.

Create a new environment named `coastseg` with all the required packages by entering these commands:

## **1. Install Coastseg Pip Package**
---

1. Create an Anaconda environment

- If you haven't created an anaconda environment already create an environment named `coastseg`. 
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

## **2. Install CoastSeg Locally (Without PyPi package)**

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

5. Install the coastseg pip package

- `-e` (editable install flag) install coastseg using the `pyproject.toml` located in coastseg's directory to install the coastseg package. See [pip documentation for -e](https://pip.pypa.io/en/stable/topics/local-project-installs/#:~:text=Editable%20installs%20allow%20you%20to,added%20to%20Python's%20import%20path.) for more information on how editable installations work.
- By making an editable install you won't need to install the coastseg files from pypi and will have all the files you need to develop with coastseg added to your python path. Using an editable install will avoid any import errors caused by not installing the package from pypi.
  ```bash
  pip install -e .
  ```
#  3. How to Fork CoastSeg 


1. git clone your fork of coastseg onto your local computer

   ```bash
   git clone https://github.com/your-username/CoastSeg.git
   ```

2. Change to the directory containing CoastSeg project
   ```bash
   cd CoastSeg
   ```
3. To push your changes to the CoastSeg later on add CoastSeg as an upstream repository:

   ```bash
   git remote add upstream https://github.com/SatelliteShorelines/CoastSeg.git
   ```

- **upstream**: refers to the official CoastSeg repository hosted on GitHub
- **origin** :refers to your personal fork on your computer

4. Install your package locally as a pip editable installation
   ```bash
   pip install -e .
   ```


## **Having Installation Errors?**

Use the command `conda clean --all` to clean old packages from your anaconda base environment. Ensure you are not in your coastseg environment or any other environment by running `conda deactivate`, to deactivate any environment you're in before running `conda clean --all`. It is recommended that you have Anaconda prompt (terminal for Mac and Linux) open as an administrator before you attempt to install `coastseg` again.

#### Conda Clean Steps

```bash
conda deactivate
conda clean --all
```

# How to Use CoastSeg


## How to Sign up to use Google Earth Engine Python API

First, you need to request access to Google Earth Engine at https://signup.earthengine.google.com/. It takes about 1 day for Google to approve requests.

Once your request has been approved, with the `coastseg` environment activated, run the following command on the Anaconda Prompt(or terminal) to link your environment to the GEE server:
``` bash
`earthengine authenticate
```

A web browser will open, login with a gmail account and accept the terms and conditions. Then copy the authorization code into the Anaconda terminal. In the latest version of the earthengine-api, the authentication is done with gcloud. If an error is raised about gcloud missing, go to https://cloud.google.com/sdk/docs/install and install gcloud. After you have installed it, close the Anaconda Prompt and restart it, then activate the environment before running earthengine authenticate again. 

Now you are ready to start using CoastSeg!

Check out our [wiki](https://github.com/SatelliteShorelines/CoastSeg/wiki) for comprehensive guides for how to use coastseg to download imagery and apply image segmentation models to the imagery you download. 


# Contribution Guide
1. [How to submit an issue](https://github.com/SatelliteShorelines/CoastSeg/wiki/How-to-Submit-An-Issue)
2. See our contribution guide to see how to install coastseg for contributing: [contribution guide](#contribution-guide)

# Authors

Package maintainers:

- [@dbuscombe-usgs](https://github.com/dbuscombe-usgs) Marda Science / USGS Pacific Coastal and Marine Science Center.
- [@2320sharon](https://github.com/2320sharon) : Lead Software Developer

Contributions:

- [@ebgoldstein](https://github.com/ebgoldstein)