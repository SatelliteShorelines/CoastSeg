## Installation Instructions

We recommend that you use Windows 10, Windows 11, or Ubuntu Linux. Mac users, please see [here](https://github.com/Doodleverse/CoastSeg/wiki/01.-How-to-Install-CoastSeg#mac-users)

In order to use Coastseg you need to install Python packages in an environment. We recommend you use [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) to install the python packages in an environment for Coastseg.

After you install miniconda/Anaconda on your PC, open the Anaconda prompt or Terminal in Mac and Linux and use the `cd` command (change directory) to go the folder where you have downloaded the Coastseg repository.

We highly recommend you install CoastSeg using `conda` following the instructions in [Install from conda-forge](#install-from-conda-forge).

## Install from conda-forge

1. Create an miniconda/Anaconda environment and Activate it

- This command creates an anaconda environment named `coastseg` and installs `python 3.10` in it.
  ```bash
  conda create --name coastseg python=3.10 -y
  conda activate coastseg
  ```

2. Install coastseg

   ```bash
   conda install -c conda-forge coastseg
   ```

3. (Optional) Install Optional Dependencies
   - Only install these dependencies if you plan to use CoastSeg's Zoo workflow notebook.
   ```bash
   pip install tensorflow
   pip install transformers
   ```

## Install from Pypi

1. Create an miniconda/Anaconda environment

- This command creates an anaconda environment named `coastseg` and installs `python 3.10` in it.
  ```bash
  conda create --name coastseg python=3.10 -y
  ```

2. Activate your conda environment

   ```bash
   conda activate coastseg
   ```

- If you have successfully activated coastseg you should see that your terminal's command line prompt should now start with `(coastseg)`.

<img src="https://user-images.githubusercontent.com/61564689/184215725-3688aedb-e804-481d-bbb6-8c33b30c4607.png" 
     alt="coastseg activated in anaconda prompt" width="350" height="150">

3. Install Conda Dependencies
   - CoastSeg requires `jupyterlab` and `geopandas` to function properly so they will be installed in the `coastseg` environment.
   - [Geopandas](https://geopandas.org/en/stable/) has [GDAL](https://gdal.org/) as a dependency so its best to install it with conda.
   - Make sure to install geopandas from the `conda-forge` channel to ensure you get the latest version.
   - Make sure to install both jupyterlab and geopandas from the conda forge channel to avoid dependency conflicts
     ```bash
     conda install -c conda-forge geopandas -y
     ```
4. Install the CoastSeg from PyPi
   ```bash
   pip install coastseg
   ```
5. Uninstall the h5py installed by pip and reinstall with conda-forge
   - `pip install jsonschema==4.19.0` is a temporary command you have to run until issue https://github.com/stac-utils/pystac/issues/1214 is resolved
   ```bash
   pip install jsonschema==4.19.0 --user
   pip uninstall h5py -y
   conda install -c conda-forge h5py -y
   ```

## **Having Installation Errors?**

Use the command `conda clean --all` to clean old packages from your anaconda base environment. Ensure you are not in your coastseg environment or any other environment by running `conda deactivate`, to deactivate any environment you're in before running `conda clean --all`. It is recommended that you have Anaconda prompt (terminal for Mac and Linux) open as an administrator before you attempt to install `coastseg` again.
