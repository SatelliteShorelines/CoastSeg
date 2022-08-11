# Create an environment with Anaconda

In order to use Coastseg you need to install Python packages in an environment. We recommend you use [Anaconda](https://www.anaconda.com/products/distribution) to install the python packages in an environment for Coastseg.

After you install Anaconda on your PC, open the Anaconda prompt or Terminal in in Mac and Linux and use the `cd` command (change directory) to go the folder where you have downloaded the Coastseg repository.

Create a new environment named `coastseg` with all the required packages by entering these commands:

## Install Coastseg

```
conda create -n coastseg python=3.10
conda activate coastseg
conda install -c conda-forge earthengine-api  jupyter -y
conda install -c conda-forge matplotlib=3.5.2 scikit-image=0.19.3 geopandas=0.11.1  astropy=5.1 tqdm=4.64.0 leafmap=0.10.3 pydensecrf=1.0rc3 -y
pip install pyqt5==5.15.7 area==1.1.1 doodleverse_utils==0.0.3 tensorflow==2.9.1
```

### Notes on `pip install tensorflow`

Windows users must use `pip` to install `tensorflow` because the conda version of tensorflow for windows is out of date as of 8/11/2022. The windows version is stuck on v1.14 on [conda-forge](https://anaconda.org/conda-forge/tensorflow).

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
