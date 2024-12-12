## Installation Instructions

We recommend that you use Windows 10, Windows 11, or Ubuntu Linux. Mac users, please see [Mac install guide](https://satelliteshorelines.github.io/CoastSeg/mac-install-guide/)

In order to use Coastseg you need to install Python packages in an environment. We recommend you use [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) to install the python packages in an environment for Coastseg.

After you install miniconda/Anaconda on your PC, open the Anaconda prompt or Terminal in Mac and Linux and use the `cd` command (change directory) to go the folder where you have downloaded the Coastseg repository.

We highly recommend you install CoastSeg using `conda` following the instructions in [Install from conda-forge](#install-from-conda-forge).

### Step 1: Install CoastSeg from Github

 1. Make sure you have git installed and if not please download it [here](https://git-scm.com/downloads)
    - This install `git` as well as `git bash` on your computer
 2. Open a terminal (or if you are on windows open `git bash`) and run the command
    ```bash
    git --version
    ```
    - It should return something like this if you have it git installed
    ```bash
    git --version
    git version 2.47.0.windows.1
    ```
 3. Open a terminal that has git installed, then use the `cd` command ( `c`hange `d`irectory) to switch the location you want to install CoastSeg
 - Then use the `git clone https://github.com/SatelliteShorelines/CoastSeg.git --depth 1` to install the code from github
 - Once the git clone command finishes use the `cd` command ( `c`hange `d`irectory) to switch the CoastSeg directory containing the code you installed.

 ```bash
  cd path/to/your/directory 
  git clone https://github.com/SatelliteShorelines/CoastSeg.git --depth 1
  cd CoastSeg
 ```


## Method #1: Install from conda-forge (Recommended)

**0. Install Anaconda**

To get started, you'll need to install Anaconda, which is a free and open-source distribution of Python and R that comes with essential packages and tools for scientific computing and data science. Here’s how you can install it:

1. Install Anaconda by following the instructions at [Anaconda install page](https://docs.anaconda.com/anaconda/install/)

2. Verify the installation
 - Once the installation is complete, open your terminal (or Anaconda Prompt on Windows).
 - Type the following command to check if Anaconda is installed correctly:

 ```bash
  conda --version

 ```

**1.Create an miniconda/Anaconda environment and Activate it**

- This command creates an anaconda environment named `coastseg` and installs `python 3.10` in it.

  ```bash
  conda create --name coastseg python=3.10 -y
  conda activate coastseg
  ```

  **2.Install coastseg**

  ```bash
  conda install -c conda-forge coastseg
  ```

  **3.(Optional) Install Optional Dependencies**

  - Only install these dependencies if you plan to use CoastSeg's Zoo workflow notebook.
  - **Warning** installing tensorflow will not work correctly on Mac see for more details [Mac install guide](https://satelliteshorelines.github.io/CoastSeg/mac-install-guide/)

```bash
pip install tensorflow==2.16.2
pip install transformers
pip install tf-keras==2.16
``

## Method #2: Install from Pypi

**1.Create an miniconda/Anaconda environment**

- This command creates an anaconda environment named `coastseg` and installs `python 3.10` in it.
  ```bash
  conda create --name coastseg python=3.10 -y
  ```

**2.Activate your conda environment**

```bash
conda activate coastseg
```

- If you have successfully activated coastseg you should see that your terminal's command line prompt should now start with `(coastseg)`.

<img src="https://user-images.githubusercontent.com/61564689/184215725-3688aedb-e804-481d-bbb6-8c33b30c4607.png" 
     alt="coastseg activated in anaconda prompt" width="350" height="150">


**3.Install the CoastSeg from PyPi**

```bash
pip install coastseg
```

**4.Install GDAL from conda-forge**

- CoastSeg requires `gdal` to function properly and requires it to be installed from `conda-forge` due to how it is configured in the conda environment.
- Make sure to install `gdal` from the `conda-forge` channel to ensure you get the latest version and to avoid dependency conflicts.

```bash
conda install -c conda-forge  gdal -y
```


**5.(Optional) Install Optional Dependencies for the Zoo Workflow**

- Only install these dependencies if you plan to use CoastSeg's Zoo workflow notebook.
- **Warning** installing tensorflow will not work correctly on Mac see for more details [Mac install guide](https://satelliteshorelines.github.io/CoastSeg/mac-install-guide/)

```bash
pip install tensorflow==2.16.2
pip install transformers
pip install tf-keras==2.16
```

* If you get any errors about numpy try running `pip install numpy<2`

## **Having Installation Errors?**

Use the command `conda clean --all` to clean old packages from your anaconda base environment. Ensure you are not in your coastseg environment or any other environment by running `conda deactivate`, to deactivate any environment you're in before running `conda clean --all`. It is recommended that you have Anaconda prompt (terminal for Mac and Linux) open as an administrator before you attempt to install `coastseg` again.
