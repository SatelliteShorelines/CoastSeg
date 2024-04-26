This guide is designed to help you easily update CoastSeg, whether you're updating to a specific version, the latest version, applying a patch, or updating code and notebooks from GitHub.

## Step 1: Activate the Environment

---

1. Open your terminal or Anaconda Prompt.
   ```bash
     conda activate coastseg
   ```

## Step 2: Update the CoastSeg Version Installed

---

### Option 1: Update via Conda

Use the following command to update CoastSeg. This command will fetch the latest version of CoastSeg available in the Conda channels that you have access to.

```bash
  conda update coastseg
```

- If you're looking for a specific version of CoastSeg, you can specify it directly by using

  ```bash
    conda install coastseg=<version>
  ```

- Replace <version> with the desired version number, such as 1.2.3

### Option 2: Update via Pip

**Update to the Latest Version**

1.**Install the latest version of CoastSeg from PyPi:**

- Use the command below to upgrade to the latest version, which includes all recent features and fixes:

```bash
  pip install coastseg --upgrade
```

- Don't worry if you see the warning message below. This is normal

```bash
  "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts."
```

2.**Install jsonschema**

- To ensure functionality in Jupyter notebooks, install the required jsonschema version:

```bash
 pip install jsonschema==4.19.0 --user
```

**Update to a Specific Version**

1.**Install a Specific Version of CoastSeg from PyPi:**

- If you need to install a particular version, use the command below and replace <version> with the desired version number (e.g., 1.1.26).

  ```bash
   pip install coastseg==<version>
  ```

- Don't worry if you see the warning message below. This is normal

```bash
  "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts."
```

2. **Install jsonschema**

   -This is necessary to run coastseg in a jupyter notebook.

```bash
 pip install jsonschema==4.19.0 --user
```

## Step 3: Update Code and Notebooks from GitHub</h2>

---

(Optional) Follow these steps if you want the latest notebooks or code updates from the CoastSeg GitHub repository.

### Step 1: Open CoastSeg in Anaconda

1.Open Anaconda Prompt

2.Activate the coastseg environment

```bash
  conda activate coastseg
```

3.Go to your coastseg location

```bash
cd <coastseg location>
```

### Step 2: Check for a Remote Connection to CoastSeg Repository

-Run the command below. In the output of this command you should see `origin  https://github.com/Doodleverse/CoastSeg.git (fetch)`

```
git remote -v
```

![git remote output](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/adbb9783-0f0e-4081-ad3f-cbfb00964a9d)

- If you don't see this output, then run the following command
  ```bash
   git remote add origin  https://github.com/Doodleverse/CoastSeg.git
   git pull origin main
  ```

### Step 3: Pull the Latest Changes

1.  Run the command below
    ```
     git pull origin main
    ```
2.  If you recieve an error message like the one shown below then proceed to 3, otherwise go to [Go to Step 4: Verify Update Success](#step-4-verify-update-success)

    ```
        Please commit your changes or stash them before you merge
        Aborting
    ```

    <img width="437" alt="git_pull_fail" src="https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/fd7ebceb-11f4-4c68-8aad-19f4d5f85030">

3.  Run the command below:

-**WARNING** This will clear out anything you have written to the `certifications.json` make sure to save that file to a new location then move it back when you're done upgrading

```
       git fetch origin
       git reset --hard origin/main
       git pull origin main
```

### Step 4: Verify Update Success

```
git status
```

- This command should return the following message
- ```
  On branch main
  Your branch is up to date with 'origin/main'.
  ```
