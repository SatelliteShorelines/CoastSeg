For this case study we will select an easy site located in Santa Cruz, California.
This site has a simple shoreline, not much cloud cover, and lots of imagery available making it a great test site

## Prerequisites

1. Before you begin make sure you have a GEE account if not follow the guide in [How to Authenticate with Google Earth Engine(GEE)](https://satelliteshorelines.github.io/CoastSeg/how-to-auth/)

2. Have installed CoastSeg by following the [Installation Guide](https://satelliteshorelines.github.io/CoastSeg/basic-install-guide/)

## SetUp

**1.Activate the coastseg conda environment**

```bash
conda activate coastseg
```

- If you have successfully activated coastseg you should see that your terminal's command line prompt should now start with `(coastseg)`.

<img src="https://user-images.githubusercontent.com/61564689/184215725-3688aedb-e804-481d-bbb6-8c33b30c4607.png" 
     alt="coastseg activated in anaconda prompt" width="350" height="150">

**2.Download CoastSeg from GitHub**

- Only do this step if you haven't already cloned CoastSeg to your computer

```
git clone --depth 1 https://github.com/SatelliteShorelines/CoastSeg.git
```

**3.Change to the CoastSeg Directory**

- Open a command prompt like anaconda prompt and go the location you installed CoastSeg

```bash
cd coastseg
```

## Launch the Notebook

### 1.Launch Jupyter Lab

- Run this command in the coastseg directory to launch the notebook `SDS_coastsat_classifier`

```bash
conda activate coastseg
jupyter lab SDS_coastsat_classifier.ipynb
```

### 2.Authenticate with Google Earth Engine

- Run the cell located under 'Authenticate with Google Earth Engine (GEE)'

![auth_cell_cropped](https://github.com/Doodleverse/CoastSeg/assets/61564689/642c8353-bfab-4458-a248-a8efce01f1ee)

### 3.Draw an Bounding Box

- Draw a bounding box along the coast in this box is where ROIs will be created

- ROIs can only be generated along a shoreline

- If no shoreline is found then an error message will appear telling you no ROIs can be created. If this happens create your own reference shoreline following the guide here [How to Create Reference Shoreline](https://satelliteshorelines.github.io/CoastSeg/How-to-Create-Reference-Shorelines-%26-Transects%26ROIs/)

![case_study_1_bbox](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/a3e482fa-8498-4ea9-82b6-c8f1edab45ea)

### 4.Load Available Shorelines

- Click load shoreline to load the one of CoastSeg's default reference shorelines within the bounding box region

![case_study_1_shorelines](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/7723b864-3e47-4a99-ae66-db1a7be9cfa6)

### 5.Load Transects

- Make sure there are transects inside the ROI you have selected otherwise you won't be able to extract shorelines

- If there isn't a reference shoreline or any transects available for your site check out the guide on how to upload your own [here](https://satelliteshorelines.github.io/CoastSeg/how-to-upload-features/)

![case_study_1_transects](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/76a658cc-d5f9-4fcb-a4a4-9b9a474f4d4c)

### 6.Modify the Settings

- Change the satellites to L8 and L9

- Change the dates to 12/01/2023 - 03/01/2024

- Change the max bad pixel percentage to 66.0%
  -- This is the maximum percentage of pixels that can be covered in either cloud or no data pixels if a downloaded image exceeds this limit it will be deleted

- Change the cloud threshold to 60%
  -- This is the maximum percentage of pixels that can be covered in cloud if a downloaded image exceeds this limit it will be deleted

- Change the size of the reference shoreline buffer to 327 meters

  -- In the image below you can see the reference shoreline buffer in purple for this location. The reference shoreline buffer is the region in which a shoreline can be extracted. If its too small then the shoreline might not be found and if its too big then clouds in the reference shoreline buffer might get misidentified as shoreline.

![case_study_1_extracted_shoreline_rel_sl_buffer](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/c16dc3a2-a211-4e9f-85c0-fc144e9d4f83)

- Click `Save Settings`

![case_study_1_save_settings](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/aef76b30-3677-417b-8d19-5313649cb877)

### 7.Name the Session

- Let's call this 'case_study_1'

- This is the name of the folder that will be saved in `CoastSeg/sessions`

![case study 1 session name circle](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/bd85acd1-0cdc-4eba-b3b6-75b0402bbb76)

### 8.Preview the available Imagery

- Preview the amount of available imagery for the selected ROI between the dates

- In this example ROI 'cwm3' has 18 images available from LandSat 8 and 16 images available from LandSat 9 for 12/01/2023 - 03/01/2024

![case study 1 preview imagery](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/db42fee9-682b-4e15-8470-b97a166e42a8)

### 9.Download the ROI

- Click the ROI you want to download on the map ( they will turn blue when selected)

- Because we set the cloud threshold to 60% and the percent of bad pixels to 66% you can see that several downloads were skipped because they exceeded the limits

- When the download finishes CoastSeg will print the location where the downloads were saved in this case its 'CoastSeg\data\ID_tto3_datetime04-22-24\_\_03_47_52'

![case_study_1_download_roi](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/1a30f9c7-fc4d-4e34-a57b-055624ff8464)

### 10.Sort the Downloaded Imagery

- Open Coastseg/data and open the folder containing the ROI ID, in my case thats 'tto3', so I opened 'CoastSeg\data\ID_tto3_datetime04-22-24\_\_03_47_52'

- You can see the ROI ID in the hover menu located to the top right of the map

![roi_id_hover](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/f7b3e931-b511-4cd7-acbf-5badf0f7d382)

- Sort any bad images into the 'bad folder'

![case_study_1_bad_sort](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/aaba59a9-9523-4419-a80c-9fa860491984)

### 11.Extract Shorelines

- Extracting shorelines works by finding the land water interface in the image and drawing a line along it

- A time series of shoreline position along each transect is generated as well

![shoreline_transect_intersection](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/e87b8d34-d9a4-4b1e-b3de-8e0be1c16ecd)

### 12. Examine Detection Images for Extracted Shorelines

- The detection images for the extracted shorelines is at 'CoastSeg\sessions\case_study_1\ID_cwm3_datetime04-22-24\_\_02_57_16\jpg_files\detection'

- In these images you can see how well the shoreline were extracted depending on cloud cover, the size of the reference shoreline buffer and the rest of the extract shoreline settings

- There are a few images with some bad shorelines. Lets remove those in step 13

![case_study_1_detection_images](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/5bd10163-77bc-4fb1-8669-394ddb8a5bf5)

### 13. Remove Outlier/Bad Extracted Shorelines

- Use the Load Extracted Shoreline feature to view all the extracted shorelines on the map

- Find any bad shorelines and click the trash icon to put that shoreline in the trash

- Once you've put all the bad shorelines in the trash click the empty trash button and this will delete all those shorelines from all the files in the session directory.

![case_study_1_remove_outlier_shorelines](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/9e0c1c60-de27-4f5c-a0ab-66eeb20e64ae)

### 14. Open the Extracted Shoreline Session Outputs in QGIS

**Config_gdf.geojson**

- This screenshot show the contents of the config_gdf.geojson file in QGIS, you can see the ROI, the transects and the reference shoreline on the map

![case_study_1_qgis_config_gdf](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/79ff037e-77e2-4fa8-a5d0-216f0e71da50)

**extracted_shorelines_points.geojson & extracted_shorelines_lines.geojson**

- This screenshot show the contents of the extracted_shorelines_points.geojson & extracted_shorelines_lines.geojson files in QGIS

- These files contain the 2D shoreline vectors extracted directly from the satellite imagery. These are NOT the shoreline positions along the transects. The shoreline position along the transect is located in the timeseries files eg. raw_transect_time_series_points.geojson,raw_transect_time_series.csv, raw_transect_time_series_vectors.geojson, raw_transect_time_series_merged.csv

![case_study_1_extracted_shoreline_pts_qgis](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/650fc2da-59fd-4ca6-98c2-b6c4ab080058)

**raw_transect_time_series_points.geojson & raw_transect_time_series_vectors.geojson**

- These files contain the shoreline positions along the transects. This is the geojson format of the 'raw_transect_time_series_merged.csv' and 'raw_transect_time_series.csv'

![case_study_1_raw_timeseries_qgis](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/43a84621-e593-43b9-a535-b9e6d1c64db1)

## Apply Tidal Correction to Extracted Shorelines (Optional)

### 1.Download the tide model

- Before tidal correction can be applied the tide model must be downloaded

- Follow the tutorial: [How to Download Tide Model](https://satelliteshorelines.github.io/CoastSeg/How-to-Download-Tide-Model/)

### 2.Load the Session with Extracted Shorelines

- Re-open the jupyter notebook

- Under the 'Kernel' menu Click 'restart and clear outputs of all cells'

![restart kernel and clear outputs of all cells](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/a7d09bcb-6c35-48b2-b28a-a6821881e503)

- Click 'Load Session' and load 'case_study_1'

![select load session and tide correct](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/581f8b4a-062e-4326-9ae8-0145026fb9ad)

### 3.Click Correct Tides

- Click the ROI ID from the dropdown

       -- You should see some extracted shorelines on the map if you don't then the ROI ID won't appear in the dropdown

- Enter Beach Slope

- Enter Beach Elevation relative to Mean Sea Level

![case_study_1_tide_correction](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/8091247b-ad1b-4233-b09f-81363dde1202)

![load_session_correct_tides_demo](https://github.com/Doodleverse/CoastSeg/assets/61564689/d7a34d13-7c01-4a30-98b3-706a63195aa7)

### 4.View the Tidally Corrected TimeSeries in QGIS

4 new files will be generated:

1.'tidally_corrected_transect_time_series.csv'

2.'tidally_corrected_transect_time_series_merged.csv'

3.'tidally_corrected_transect_time_series_points.geojson'

4.'tidally_corrected_transect_time_series_vectors.geojson'

- This screenshot show the difference between the tidally_corrected_transect_time_series_vectors and raw_transect_time_series_vectors as you can see applying tidal correction shifts the raw shoreline position along the transect to account for the tide position

![case_study_1_raw_and_tide_corrected_timeseries_qgis](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/a232c79e-eb6c-42c8-bb1c-d90b28dd5d98)
