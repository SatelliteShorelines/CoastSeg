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

**1.Launch Jupyter Lab**

- Run this command in the coastseg directory to launch the notebook `SDS_coastsat_classifier`

```bash
conda activate coastseg
jupyter lab SDS_coastsat_classifier.ipynb
```

**2.Authenticate with Google Earth Engine**

![auth_cell_cropped](https://github.com/Doodleverse/CoastSeg/assets/61564689/642c8353-bfab-4458-a248-a8efce01f1ee)
**3.Draw an Bounding Box**
**4.Generate ROI (Region of Interest)**
**5.Load Transects**
![load_rois_then_transects_on_map_demo](https://github.com/Doodleverse/CoastSeg/assets/61564689/d53154b0-7a63-470f-91ec-dabdf7d4a100)

- Make sure there are transects inside the ROI you have selected otherwise you won't be able to extract shorelines

**6.Modify the Settings**

- Change the satellites to L8 and L9

- Change the dates to 12/01/2023 - 03/01/2024

- Change the size of the reference shoreline buffer

- Click `Save Settings`

**7.Name the Session**

- Let's call this 'sample_session'

- This is the name of the folder that will be saved in `CoastSeg/sessions`

- It will contain a subdirectory for each ROI that shorelines will be extracted for

**8.Download the ROIs**

- Click the ROIs you want to download on the map ( they will turn blue when selected)

- If no transects or reference shorelines are available for the region you have uploaded follow the guide here

- NEVER rename your ROIs this is because CoastSeg keeps track of the filename in the 'config.json' and you won't be able to load it into CoastSeg again if you do. The ROI's ID is the filename so that information will be lost if you rename it.

**9.Extract Shorelines**
![save_settings_download_extract](https://github.com/Doodleverse/CoastSeg/assets/61564689/3548a9ce-a190-4c95-b495-0ff75484fdb2)

- Extracting shorelines works by finding the land water interface in the image and drawing a line along it

- A time series of shoreline position along each transect is generated as well

![shoreline_transect_intersection](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/e87b8d34-d9a4-4b1e-b3de-8e0be1c16ecd)

## Apply Tidal Correction to Extracted Shorelines (Optional)

**1.Download the tide model**

- Before tidal correction can be applied the tide model must be downloaded

- Follow the tutorial: [How to Download Tide Model](https://satelliteshorelines.github.io/CoastSeg/How-to-Download-Tide-Model/)

**2.Load the Session with Extracted Shorelines**

- Re-open the jupyter notebook

- Under the 'Kernel' menu Click 'restart and clear outputs of all cells'

![restart kernel and clear outputs of all cells](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/a7d09bcb-6c35-48b2-b28a-a6821881e503)

- Click 'Load Session' and load 'sample_session'

![select load session and tide correct](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/581f8b4a-062e-4326-9ae8-0145026fb9ad)

**3.Click Correct Tides**

- Click the ROI ID from the dropdown

       -- You should see some extracted shorelines on the map if you don't then the ROI ID won't appear in the dropdown

- Enter Beach Slope

- Enter Beach Elevation relative to Mean Sea Level

![select roi id for tide correction](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/9e212590-1f1e-4c51-b223-2e49a329a524)

![load_session_correct_tides_demo](https://github.com/Doodleverse/CoastSeg/assets/61564689/d7a34d13-7c01-4a30-98b3-706a63195aa7)
