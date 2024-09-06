## Prerequisites

**1.Sign up to use Google Earth Engine Python API**

1. Follow the guide to setup Google Earth Engine: [Sign up for Google Earth Engine for the First Time](https://satelliteshorelines.github.io/CoastSeg/google-earth-engine-setup/)

2. Follow the guide to enable Google earth Engine for your project : [Check if Google Earth Engine API is Enabled](https://satelliteshorelines.github.io/CoastSeg/google-earth-enable-api/)

**2. Install Git**

1. Install git on your computer following the guide [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

**3. Install CoastSeg in Anaconda**

 Follow one of the install guides under "Installation" to install CoastSeg.

  - For most users the [basic install guide](https://satelliteshorelines.github.io/CoastSeg/basic-install-guide/) will work 

  - If you are on a secure network then you may need to follow the [secure install guide](https://satelliteshorelines.github.io/CoastSeg/basic-install-guide/)


## Installation & SetUp

**1.Activate the coastseg conda environment**

- Open anaconda prompt.
- Make sure you have coastseg installed if not follow the installation guide : [basic install guide](https://satelliteshorelines.github.io/CoastSeg/basic-install-guide/)

```bash
conda activate coastseg
```

- If you have successfully activated coastseg you should see that your terminal's command line prompt should now start with `(coastseg)`.

<img src="https://user-images.githubusercontent.com/61564689/184215725-3688aedb-e804-481d-bbb6-8c33b30c4607.png" 
     alt="coastseg activated in anaconda prompt" width="350" height="150">

**2.Download CoastSeg from GitHub**

- Before you run this step make you have install git on your computer a guide is linked [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

- Open 'git bash' (use this in the search bar on windows) and open the app. 

- Here is an example below where I have installed CoastSeg in a folder called 'projects', but you can install it wherever you'd like just make sure to remember the location.

- After the download finishes close git bash.

```
mkdir projects
cd projects
git clone --depth 1 https://github.com/SatelliteShorelines/CoastSeg.git
```

![git bash](https://github.com/user-attachments/assets/10fdd7c1-ef23-4234-ba01-07489c3f7007)


## Extract Shorelines

**1.Launch Jupyter Lab**

- If you have never used jupyter notebook before please check out this guide: [How to Use Jupyter Notebook](https://satelliteshorelines.github.io/CoastSeg/jupyter-notebook-guide/)

- Open Anaconda prompt

- Enter `jupyter lab SDS_coastsat_classifier.ipynb` in anaconda prompt and your notebook should automatically open in the browser

- 3. PLEASE MAKE SURE YOU RUN THIS IN THE COASTSEG DIRECTORY WHERE THE NOTEBOOK `SDS_coastsat_classifier.ipynb` IS LOCATED


```bash
cd <location you installed Coastseg>
conda activate coastseg
jupyter lab SDS_coastsat_classifier.ipynb
```

![jupyter notebbok open classifier](https://github.com/user-attachments/assets/ed1dddb2-3a01-4572-b2c1-4cc0ff794972)


**2.Authenticate with Google Earth Engine**

- Run the cell located under 'Authenticate with Google Earth Engine (GEE)'

![run_cell](https://github.com/user-attachments/assets/6bb955a7-edb3-4445-9337-d20d4bb95976)


**3.Run the Cell 'Create Coastseg Map Dashboard'**

- Run the cell to create the map dashboard


![run map dashboard](https://github.com/user-attachments/assets/fe46dcf3-352b-458e-a261-f2adbd4663f5)

- After you run the cell you should see the UI appear

- Scroll down to the interactive map

![ran map dashboard](https://github.com/user-attachments/assets/ddd44c63-9285-42a6-a69c-c7410d09a29c)

**4.Draw an Bounding Box**

- Draw a bounding box along the coast in this box is where ROIs will be created

**Option 1: Draw a Bounding Box with the box tool**

![draw_bbox](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/6b97866d-c54a-4c67-8383-530208fc643c)

**Option 2: Draw a Bounding Box with the polygon tool**

- This is useful if you have back - barrier shorelines that you don't want to include

![draw_bbox_polygon_remove_back_barrier](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/a63f023a-f9a8-4e48-9aca-bfa00dc262ea)

**5.Generate ROI (Region of Interest)**

- ROIs can only be generated along a shoreline

- If no shoreline is found then an error message will appear telling you no ROIs can be created. If this happens create your own reference shoreline following the guide here [How to Create Reference Shoreline](https://satelliteshorelines.github.io/CoastSeg/How-to-Create-Reference-Shorelines-%26-Transects%26ROIs/)

![generate_roi](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/ade2123f-3ea6-4dc0-ac5b-15f5f758e220)

**6.Load Transects**

- Make sure there are transects inside the ROI you have selected otherwise you won't be able to extract shorelines

- If there isn't a reference shoreline or any transects available for your site check out the guide on how to upload your own [here](https://satelliteshorelines.github.io/CoastSeg/how-to-upload-features/)

![load_rois_then_transects_on_map_demo](https://github.com/Doodleverse/CoastSeg/assets/61564689/d53154b0-7a63-470f-91ec-dabdf7d4a100)

**7.Modify the Settings**

- Change the satellites to L8 and L9

- Change the dates to 12/01/2023 - 03/01/2024

- Change the size of the reference shoreline buffer

- Click `Save Settings`

![save_settings_getting_started_circle](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/c14c2e01-bb1f-43d2-b932-b0ccfb82a598)

**8.Name the Session**

- Name the session 'demo_session'. This will be the name of the folder saved in `CoastSeg/sessions`.

- The folder will contain a subdirectory for each ROI from which shorelines were extracted.

- Keeping sessions and downloaded data separate allows users to create multiple sessions from the same downloaded data, enabling experimentation with different settings.


![save_demo_session](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/4340c734-e20d-4149-89c2-11e73d9905d3)

**9.Preview the available Imagery**

- Preview the amount of available imagery for the selected ROI between the dates

- In this example ROI 'cwm3' has 18 images available from LandSat 8 and 16 images available from LandSat 9 for the date range

![case study 1 preview imagery](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/db42fee9-682b-4e15-8470-b97a166e42a8)

**10. Download the ROIs**

- Click the ROIs you want to download on the map (they will turn blue when selected).

- If no transects or reference shorelines are available for the region you have uploaded, follow the guide here.

- NEVER rename your ROIs. This is because CoastSeg keeps track of the filename in the 'config.json' and you won't be able to load it into CoastSeg again if you do. The ROI's ID is the filename, so that information will be lost if you rename it.

- The downloaded data can be found in `CoastSeg/data` under a folder with a name in the format `ID_<ROI ID>_datetime<currenttime>`. For example, you can see the JPGs for all the images you downloaded in `CoastSeg/data/ID_<ROI ID>_datetime06-17-24__11_12_40/jpg_files/preprocessed/RGB`. You can read more on how ROIs are structured in this [guide](https://satelliteshorelines.github.io/CoastSeg/roi/)

- This is example of the RGB imagery from a downloaded ROI. Here you can see the user sorted some of the imagery into a bad folder so it would not be used to extract shorelines. A full tutorial on how to filter bad imagery is available [here](https://satelliteshorelines.github.io/CoastSeg/How-to-Filter-Out-Bad-Imagery/).
![coastseg/data/roi/](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/6f357c2e-8cdf-403d-8224-f8ba48946c2c)


![case_study_1_download_roi](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/1a30f9c7-fc4d-4e34-a57b-055624ff8464)




**11.Extract Shorelines**
![save_settings_download_extract](https://github.com/Doodleverse/CoastSeg/assets/61564689/3548a9ce-a190-4c95-b495-0ff75484fdb2)

- Extracting shorelines involves loading the ROI data for each selected ROI from `CoastSeg/data/<ROI ID>` and processing the downloaded rasters to extract shorelines. The resulting files are saved in `CoastSeg/sessions/<YOUR SESSION NAME>/ROI_ID_NAME` for each ROI. Note that the downloaded data is NOT copied to the session; instead, the `config.json` file in each session keeps track of the location of the downloaded ROI in `CoastSeg/data`. You can read more about what is in each session in this [guide](https://satelliteshorelines.github.io/CoastSeg/what-is-in-a-session/).


- Extracting shorelines works by finding the land water interface in the image and drawing a line along this boundary.

- Additionally, a time series of shoreline positions along each transect is generated.

- You can find images that show the extracted shoreline on each image in `CoastSeg/sessions/<YOUR SESSION NAME>/ROI_ID_NAME/jpg_files/detection` as illustrated below:

![coastseg/sessions/session/roi/jpg_files/detection](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/f5be75fb-1c75-46c2-bbec-b9b1e716fea4)


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
