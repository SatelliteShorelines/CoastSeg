# 📦 CoastSeg

A mapping extension for [CoastSat](https://github.com/kvos/CoastSat) using [Segmentation Zoo](https://github.com/Doodleverse/segmentation_zoo) models

## :warning: NOT A FINISHED PRODUCT :warning:
Please note that we're in the planning stages only - please check back later, or use our [Discussions tab](https://github.com/dbuscombe-usgs/CoastSeg/discussions) to provide feedback or offer help - thanks!

## 🌟 Highlights

1) CoastSeg will interface with, and extend the functionality of, [CoastSat](https://github.com/kvos/CoastSat) and [CoastSat.Slope](https://github.com/kvos/CoastSat.slope) by 
    * facilitating deep learning based image segmentation on coastal satellite imagery, and 
    * providing a web mapping interface to prepare data downloads, view data, analyze segmentation outputs, and subsequent analyses
    
The hope is to provide a standardized workflow that does not involve external processes such as a GIS or file browser, so analysis can be carried out on a cloud computer, among other potential advantages.

2) CoastSeg will facilitate the application of [Segmentation Zoo](https://github.com/Doodleverse/segmentation_zoo) models for deep-learning-based image segmentation on geospatial imagery with a web mapping interface. Our hope is that the use of deep learning models would facilitate custom mapping of coastal features beyond the current scope of [CoastSat](https://github.com/kvos/CoastSat).

3) CoastSeg will eventually facilitate custom image segmentation and mapping, by helping implement models that you have trained using [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym) and [Doodler](https://github.com/Doodleverse/dash_doodler) models


## ✍️ Authors

Package maintainers:
* [@dbuscombe-usgs](https://github.com/dbuscombe-usgs) Marda Science / USGS Pacific Coastal and Marine Science Center.

Contributions:
* [@2320sharon](https://github.com/2320sharon)
* [@ebgoldstein](https://github.com/ebgoldstein)

We welcome collaboration! Please use our [Discussions tab](https://github.com/dbuscombe-usgs/CoastSeg/discussions) if you're interested in this project.

## 🧰 Installation Instructions 
### Create an environment with Anaconda

In order to use Coastseg you need to install Python packages in an environment. We recommend you use [Anaconda](https://www.anaconda.com/products/distribution) to install the python packages in an environment for Coastseg.

After you install Anaconda on your PC, open the Anaconda prompt or Terminal in in Mac and Linux and use the `cd` command (change directory) to go the folder where you have downloaded the Coastseg repository.

Create a new environment named `coastseg` with all the required packages by entering these commands:

### Install Coastseg

```
conda create -n coastseg python=3.10
conda activate coastseg
conda install -c conda-forge geopandas earthengine-api scikit-image matplotlib astropy notebook tqdm -y
conda install -c conda-forge leafmap pydensecrf -y
pip install pyqt5 area doodleverse_utils tensorflow
```

#### Notes on `pip install tensorflow`

Windows users must use `pip` to install `tensorflow` because the conda version of tensorflow for windows is out of date as of 8/11/2022. The windows version is stuck on v1.14 on [conda-forge](https://anaconda.org/conda-forge/tensorflow).

### Activate Coastseg Environment

All the required packages have now been installed in an environment called coastseg. Always make sure that the environment is activated with:

`conda activate coastseg`
To confirm that you have successfully activated coastseg, your terminal command line prompt should now start with (coastseg).


<img src="https://user-images.githubusercontent.com/61564689/184215725-3688aedb-e804-481d-bbb6-8c33b30c4607.png" 
     alt="coastseg activated in anaconda prompt" width="250" height="150">

### ⚠️Installation Errors ⚠️

Use the command `conda clean --all` to clean old packages from your anaconda base environment. Ensure you are not in your coastseg environment or any other environment by running `conda deactivate`, to deactivate any environment you're in before running `conda clean --all`. It is recommended that you have Anaconda prompt (terminal for Mac and Linux) open as an administrator before you attempt to install `coastseg` again.

#### Conda Clean Steps

```
conda deactivate
conda clean --all
```


## :red_car: Roadmap

The current [CoastSat](https://github.com/kvos/CoastSat) workflow might be summarized thus:

![CoastSatFlow(2)](https://user-images.githubusercontent.com/3596509/153914900-017b02a6-f634-45b4-b1fd-5c66c4d6a1c3.jpg)


In a nutshell, CoastSeg will provide an alternative workflow in a web browser that attempts to streamline the CoastSat workflow, as well as provide additional functionality

![](https://user-images.githubusercontent.com/3596509/153467309-1583e449-1930-462b-815b-2bd37ee68928.png)


## :earth_americas: Workflow

### :new_moon: Module 1: ROI creation and image retrieval. 

The first step is to define regions and use CoastSat functionality to obtain pre-processed imagery

1. User will define a bounding box (region of interest) graphically / interactively
2. CoastSeg will automatically create smaller ROIs with specified overlap for imagery download and image segmentation
3. CoastSeg will provide tools for image QC, to define a subset of images for subsequent analyses
4. CoastSat will download and preprocess imagery

While we work on developing this tool, here is a movie of the current automated ROI creation process, that uses an existing global shoreline database to determine the approximate shoreline. ROIs are generated with a user-specified size and amount of overlap. ROIs are then selected for image retrieval

![](https://user-images.githubusercontent.com/3596509/153467223-e6d8f055-255d-4978-a055-c66a136d0dd7.gif)

### :waxing_crescent_moon: Module 2: image quality control (QC)
Satellite imagery is subject to various sources of noise, and cloud masking is not always optimal. Therefore some manual weeding of data is often necessary. This module will therefore facilitate image QC. At first, it will be a semi-automated tool. In time, we hope it could be a fully automated tool, using Machine Learning to flag good (usable) and bad (unusable) imagery.

In the initial manual version, the program will load each image one-by-one and the user decides to keep or discard the image. This app should keep track of the images that are labeled good and bad, so we can train automated processes to do this QC for us. This should be only for the preprocessed jpeg imagery, not the tiff imagery. This could be useful for a many other applications outside of CoastSeg.

### :first_quarter_moon: Module 3: image segmentation using pre-trained models for shoreline detection
This module is just a graphical interface for applying existing deep learning models to imagery for the purposes of shoreline detection

1. User selects a pre-trained segmentation model from [Segmentation Zoo](https://github.com/Doodleverse/segmentation_zoo) using a dropdown menu
2. User calls [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym) functionality to merge data bands (if necessary) and create npz format inputs for the segmentation model
3. User applies model to all filtered imagery (see QC module above)
4. Optionally, user interactively refines segmentation by selecting different or more models to ensemble until they are satisfied with the result
5. Outputs could replace the segmentation outputs of coastsat, or they could be cleverly combined as an ensemble prediction
6. create cloud-optimized geotiffs (COGs) of the label images (model outputs)


### :waxing_gibbous_moon: Module 4: Satellite-derived Shorelines
This module is just a graphical interface for existing CoastSat functionality, with some additional options for loading pre-made transects or interactively re-defining transects

1. Load transects for the ROIs in a session, or allow the user to draw/redraw transects interactively through the browser
2. Use CoastSat functionality to extract shorelines from transects
3. Apply CoastSat tidal corretions using the FES global tide model: estimate tide for each transect and timestamped image and apply the horizontal correction to each waterline (defined as the location of the waters edge) to convert to a shoreline (where the beach intersects mean sea level, obtained by tidal correction)
4. Apply CoastSat.slope to compute slopes and refine shoreline estimates
5. CoastSeg might also be used to interactively remove bad shoreline estimates


### :full_moon: Module 5: custom image segmentation for purposes other than shoreline detection
1. web-based Doodler for geospatial imagery, currently called HoloDoodler, for label creation with custom classes
2. wrapper utilities to convert labels to geospatial formats
3. wrapper utilities for [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym) functionality to train and apply new models from scratch


## :milky_way: Possible future directions

* Parallelism and modern geospatial formats:
    * xarrays facilitate segmentation workflows in Dask

* A local ipyleaflet tile server for data 
    * [Local Tile Server for Geospatial Rasters](https://github.com/banesullivan/localtileserver?s=09#ipyleaflet-tile-layers)
    * all ipyleaflet and folium functionality for data exploration and retrieval



