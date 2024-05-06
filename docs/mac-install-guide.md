## Mac users

CoastSeg users who only want to use the `coastsat_classifier` notebook and not the zoo notebook can install coastseg using the conda-forge package as shown below. However any users interested in using the zoo workflow should read the section [Install Tensorflow for Zoo](#install-tensorflow-for-zoo)

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

## Install Tensorflow for Zoo

CoastSeg requires Tensorflow (TF) to run the models in the zoo workflow, which doesn't play nicely with Mac. We advise you to use either Linux or Windows, if you can. We cannot troubleshoot Mac installations, but we can offer the following advice:

- TF for mac has its own instructions: https://developer.apple.com/metal/tensorflow-plugin/
- New Mac silicon runs TF, (and has its own TF branch), but the old intel Mac chips might not work with parts of TF.
- We are not sure if TF is compatible with M2 macs
- Our continuous integration tests check only the 'latest' version of Mac OS.
- If you get a working installation on Mac, please let us know, and we can edit these pages to communicate better advice. Thanks in advance
