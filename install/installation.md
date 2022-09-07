# Create an environment with Anaconda

In order to use Coastseg you need to install Python packages in an environment. We recommend you use [Anaconda](https://www.anaconda.com/products/distribution) to install the python packages in an environment for Coastseg.

After you install Anaconda on your PC, open the Anaconda prompt or Terminal in in Mac and Linux and use the `cd` command (change directory) to go the folder where you have downloaded the Coastseg repository.

Create a new environment named `coastseg` with all the required packages by entering these commands:

## Install Coastseg

```
conda create -n coastseg python=3.8
conda activate coastseg_test

## coastsat dependencies
conda install -c conda-forge earthengine-api astropy -y
conda install gdal geopandas scikit-image notebook pyqt -y
conda install -c conda-forge “numpy>=1.16.5, <=1.23.0" -y


## additional coastseg dependencies
conda install ipython cartopy  tqdm  -y    ## pip no!
conda install -c conda-forge simplekml leafmap pydensecrf h5py -y
pip install area doodleverse_utils
conda install -c conda-forge tensorflow-gpu
```

## Activate Coastseg Environment

All the required packages have now been installed in an environment called coastseg. Always make sure that the environment is activated with:

`conda activate coastseg`
To confirm that you have successfully activated coastseg, your terminal command line prompt should now start with (coastseg).

<img src="https://user-images.githubusercontent.com/61564689/184215725-3688aedb-e804-481d-bbb6-8c33b30c4607.png" 
     alt="coastseg activated in anaconda prompt" width="250" height="150">

## ⚠️Installation Errors ⚠️

Use the command `conda clean --all` to clean old packages from your anaconda base environment. Ensure you are not in your coastseg environment or any other environment by running `conda deactivate`, to deactivate any environment you're in before running `conda clean --all`. It is recommended that you have Anaconda prompt (terminal for Mac and Linux) open as an administrator before you attempt to install `coastseg` again.

### Conda Clean Steps

```
conda deactivate
conda clean --all
```
