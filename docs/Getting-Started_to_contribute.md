## Getting Started Guide for Contributors

---

Ready to contribute? Here's how to set up _CoastSeg_ for local development.
This guide will walk you through the installation process, testing procedures, and best practices for contributing to CoastSeg.

1. Make a Fork

Click the `fork` button located at the top right of the coastseg repository. This will create a fork copy of the coastseg repository that you can edit on your GitHub account. [Learn How to Fork from GitHub Docs](https://docs.github.com/en/get-started/quickstart/fork-a-repo)

![image](https://user-images.githubusercontent.com/61564689/212405847-047511d1-961f-43f3-a5e4-7145795ea17f.png)

2. Clone your fork locally:

- git clone your fork of coastseg onto your local computer
  ```bash
  git clone https://github.com/your-username/CoastSeg.git
  ```

3. Create an Anaconda Environment for CoastSeg Development

- We will install the CoastSeg package and its development dependencies in this environment.

  ```bash
  conda create --name coastseg_dev python=3.10 -y
  ```

4. Activate the Conda Environment

   ```bash
   conda activate coastseg_dev
   ```

5. Change Directory to the CoastSeg

- Go to the location where CoastSeg was installed on your computer.
  <br> `cd <directory where you have coastseg source code installed>`
  <br>**Example:** `cd c:\users\CoastSeg`
  ```bash
  cd CoastSeg
  ```

6. Install CoastSeg locally as a pip editable installation
   ```bash
   pip install -e .
   ```

- This command reads the required dependencies from CoastSeg's `pyproject.toml` file and installs them within your anaconda environment.
- Make sure to run this command in the `CoastSeg` directory that contains the `pyproject.toml` file otherwise this command will fail because pip won't find the `pyproject.toml` file
- `-e` means create an editable install of the package. This will add the files to the python path on your computer making it possible to find the sub directories of the package.See the [official documentation](https://pip-python3.readthedocs.io/en/latest/reference/pip_install.html#editable-installs).
- `.` means use the current working directory to install

7. Install Geopandas and JupyterLab Locally

   ```bash
   conda install -c conda-forge jupyterlab geopandas -y
   ```

8. Install the Development Dependencies
   ```bash
   pip install build pytest black
   ```

- `black` is a python formater that you can run on the code
- `pytest` is used to automatically run tests on the code

9. Create a branch for local development:

   ```
   git checkout -b name-of-your-bugfix-or-feature
   ```

   Now you can make your changes locally.

10. When you're done making changes, use `pytest` to check that your changes pass the tests.

    ```
    conda activate coastseg_dev
    cd CoastSeg
    cd tests
    pytest .
    ```

11. Format the code using Black
    To make your code adhere to python style standards use the `black` code formatter to automatically format the code. You'll need to change directories to the `src` directory, then to the sub directory `coastseg` and run the `black` here. If you were to run `black` in the main coastseg directory it would not format the code because the code for coastseg is located in directory `coastseg>src>coastseg`.

```bash
conda activate coastseg_dev
cd src
cd coastseg
black .
```

12. Commit your changes and push your branch to GitHub:

    ```
    git add .
    ```

    ```
    git commit -m "Your detailed description of your changes."
    ```

    ```
    git push origin name-of-your-bugfix-or-feature
    ```

13. Submit a pull request through the GitHub website.
