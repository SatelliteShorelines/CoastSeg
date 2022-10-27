# CoastSeg

[![Last Commit](https://img.shields.io/github/last-commit/SatelliteShorelines/CoastSeg)](https://github.com/Doodleverse/segmentation_gym/commits/main)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/SatelliteShorelines/CoastSeg/graphs/commit-activity)
[![Wiki](https://img.shields.io/badge/wiki-documentation-forestgreen)](https://github.com/SatelliteShorelines/CoastSeg/wiki)
![GitHub](https://img.shields.io/github/license/Doodleverse/segmentation_gym)
[![Wiki](https://img.shields.io/badge/discussion-active-forestgreen)](https://github.com/SatelliteShorelines/CoastSeg/discussions)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

![CoastSeg](https://user-images.githubusercontent.com/3596509/189417290-d5c24681-39b7-4b97-afa8-1392cf759b08.gif)

A mapping extension for [CoastSat](https://github.com/kvos/CoastSat) using [Segmentation Zoo](https://github.com/Doodleverse/segmentation_zoo) models

## NOT A FINISHED PRODUCT

Please note that we're in the planning stages only - please check back later. Please see our [Wiki](https://github.com/SatelliteShorelines/CoastSeg/wiki) for further information

We welcome collaboration! Please use our [Discussions tab](https://github.com/dbuscombe-usgs/CoastSeg/discussions) to provide feedback or offer help - thanks!

## Authors

Package maintainers:

- [@dbuscombe-usgs](https://github.com/dbuscombe-usgs) Marda Science / USGS Pacific Coastal and Marine Science Center.
- [@2320sharon](https://github.com/2320sharon)

Contributions:

- [@ebgoldstein](https://github.com/ebgoldstein)

lease use our [Discussions tab](https://github.com/dbuscombe-usgs/CoastSeg/discussions) if you're interested in this project.

## Installation Instructions

### Create an environment with Anaconda

In order to use Coastseg you need to install Python packages in an environment. We recommend you use [Anaconda](https://www.anaconda.com/products/distribution) to install the python packages in an environment for Coastseg.

After you install Anaconda on your PC, open the Anaconda prompt or Terminal in in Mac and Linux and use the `cd` command (change directory) to go the folder where you have downloaded the Coastseg repository.

Create a new environment named `coastseg` with all the required packages by entering these commands:

### Install Coastseg

```
conda create -n coastseg python=3.8
conda activate coastseg

## coastsat dependencies
conda install -c conda-forge earthengine-api astropy utm -y
conda install gdal geopandas scikit-image notebook pyqt -y
conda install -c conda-forge “numpy>=1.16.5, <=1.23.0" -y

## additional coastseg dependencies
conda install ipython cartopy  tqdm  -y
conda install -c conda-forge simplekml leafmap pydensecrf h5py -y
pip install area doodleverse_utils
conda install -c conda-forge tensorflow-gpu
pip install coastsat_package
```

### Activate Coastseg Environment

All the required packages have now been installed in an environment called coastseg. Always make sure that the environment is activated with:

`conda activate coastseg`
To confirm that you have successfully activated coastseg, your terminal command line prompt should now start with (coastseg).

<img src="https://user-images.githubusercontent.com/61564689/184215725-3688aedb-e804-481d-bbb6-8c33b30c4607.png" 
     alt="coastseg activated in anaconda prompt" width="250" height="150">

### Installation Errors

Use the command `conda clean --all` to clean old packages from your anaconda base environment. Ensure you are not in your coastseg environment or any other environment by running `conda deactivate`, to deactivate any environment you're in before running `conda clean --all`. It is recommended that you have Anaconda prompt (terminal for Mac and Linux) open as an administrator before you attempt to install `coastseg` again.

#### Conda Clean Steps

```
conda deactivate
conda clean --all
```

## Installation

We advise creating a new conda environment to run the program.

1. Clone the repo:

```
git clone --depth 1 https://github.com/dbuscombe-usgs/CoastSeg.git
```

(`--depth 1` means "give me only the present code, not the whole history of git commits" - this saves disk space, and time)

2. Create a conda environment called `coastseg`

```
conda env create --file install/coastseg.yml
conda activate coastseg
```
