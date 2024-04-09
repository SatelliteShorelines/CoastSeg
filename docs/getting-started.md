Check out the rest of the [wiki](https://github.com/Doodleverse/CoastSeg/wiki) for more tutorials

## Prerequisites

1. Sign up to use Google Earth Engine Python API

-Request access to Google Earth Engine at https://signup.earthengine.google.com/

-It takes about 1 day for Google to approve requests.

## Installation & SetUp

1. Activate the coastseg conda environment
   ```bash
   conda activate coastseg
   ```

- If you have successfully activated coastseg you should see that your terminal's command line prompt should now start with `(coastseg)`.

<img src="https://user-images.githubusercontent.com/61564689/184215725-3688aedb-e804-481d-bbb6-8c33b30c4607.png" 
     alt="coastseg activated in anaconda prompt" width="350" height="150">

2. Download CoastSeg from GitHub

```
git clone coastseg --depth 1 https://github.com/Doodleverse/CoastSeg.git
```

## Extract Shorelines

1. Launch Jupyter Lab

- Run this command in the coastseg directory to launch the notebook `SDS_coastsat_classifier`
  ```bash
  conda activate coastseg
  jupyter lab SDS_coastsat_classifier.ipynb
  ```

2. Authenticate with Google Earth Engine
   ![auth_cell_cropped](https://github.com/Doodleverse/CoastSeg/assets/61564689/642c8353-bfab-4458-a248-a8efce01f1ee)
3. Draw an Bounding Box
4. Generate ROI (Region of Interest)
5. Load Transects
   ![load_rois_then_transects_on_map_demo](https://github.com/Doodleverse/CoastSeg/assets/61564689/d53154b0-7a63-470f-91ec-dabdf7d4a100)

6. Modify the Settings
   - Change the satellites to L8 and L9
   - Change the dates to 12/01/2023 - 03/01/2024
   - Change the size of the reference shoreline buffer
   - Click `Save Settings`
7. Name the Session
8. Extract Shorelines
   ![save_settings_download_extract](https://github.com/Doodleverse/CoastSeg/assets/61564689/3548a9ce-a190-4c95-b495-0ff75484fdb2)

## Apply Tidal Correction to Extracted Shorelines (Optional)

1. Download the tide model
   - Before tidal correction can be applied the tide model must be downloaded
   - Follow the tutorial: [How to Download Tide Model](https://github.com/Doodleverse/CoastSeg/wiki/09.-How-to-Download-Tide-Model)
2. Load the Session with Extracted Shorelines
   - Re-open the jupyter notebook
   - Under the Kernal menu Click 'restart and clear outputs of all cells'
   - Click Load Session and load the same we made before ''
3. Click Correct Tides

   - Click the ROI ID from the dropdown
   - Enter Beach Slope
   - Enter Beach Elevation relative to Mean Sea Level

![load_session_correct_tides_demo](https://github.com/Doodleverse/CoastSeg/assets/61564689/d7a34d13-7c01-4a30-98b3-706a63195aa7)
