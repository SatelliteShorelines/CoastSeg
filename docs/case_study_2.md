For this case study we will select an difficult site located outside of Unalakleet Alaska.
This site has a simple shoreline,lots of cloud cover, and not imagery available due to ice and clouds covering the region most of the year making it a difficult test site.

![unakleet_site](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/157636ba-c1bc-4628-8d9a-04bef1ca1246)

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

![case_study_2_bbox_for_ak](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/aa4ca94f-3049-4801-99bc-98fe34b7c671)

### 4.Create and Load a Reference Shoreline

- This site does not have a shoreline available so we will have to make our own. Follow the tutorial at [How to Create Reference Shoreline](https://satelliteshorelines.github.io/CoastSeg/How-to-Create-Reference-Shorelines-%26-Transects%26ROIs/) for a guide on how to create your own reference shoreline

- Here is a guide on how to upload your own [here](https://satelliteshorelines.github.io/CoastSeg/how-to-upload-features/)

- I used geojson.io to create my reference shoreline

### 5.Create and Load Transects

- This site does not have any transects available so we will have to make our own. Follow the tutorial at [How to Create Reference Shoreline](https://satelliteshorelines.github.io/CoastSeg/How-to-Create-Reference-Shorelines-%26-Transects%26ROIs/) for a guide on how to create your own transects

- Here is a guide on how to upload your own [here](https://satelliteshorelines.github.io/CoastSeg/how-to-upload-features/) transects

- I used geojson.io to create my transects.This technique is in the tutorial [How to Create Reference Shoreline](https://satelliteshorelines.github.io/CoastSeg/How-to-Create-Reference-Shorelines-%26-Transects%26ROIs/)

![case_study_2_shoreline_transects_upload](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/cd780e5a-39f7-42ba-90ac-d632390c18dc)

### 6.Modify the Settings

- Change the satellites to L8 and L9

- Change the dates to 04/01/2023 - 011/01/2024

- Change the months to not select November since our site will be covered in sea ice

- The reason we choose these dates is because its less likely the ocean will be covered in ice. Shoreline extraction does not work properly with sea ice present

- Change the max bad pixel percentage to 50.0%
  -- This is the maximum percentage of pixels that can be covered in either cloud or no data pixels if a downloaded image exceeds this limit it will be deleted

- Change the cloud threshold to 50%
  -- This is the maximum percentage of pixels that can be covered in cloud if a downloaded image exceeds this limit it will be deleted

- Change the size of the reference shoreline buffer to 290 meters

  -- In the image below you can see the reference shoreline buffer in purple for this location. The reference shoreline buffer is the region in which a shoreline can be extracted. If its too small then the shoreline might not be found and if its too big then clouds in the reference shoreline buffer might get misidentified as shoreline.

![case_study_1_extracted_shoreline_rel_sl_buffer](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/c16dc3a2-a211-4e9f-85c0-fc144e9d4f83)

- Turn off the Apply Cloud Mask

  -- The cloud mask being off does not impact whether images are filtered out during the download process it only controls if the clouds are blotted off the images.

  -- The reason for this is that Alaska is very cloudy and often light clouds will be masked out close to the shoreline causing gaps in the shoreline due to the cloud distance threshold. If we don't mask out the clouds then we get a few more usable images

  -- In this example you can see the clouds where masked out in this image but these clouds were onshore so they would not have interfered with the cloud masking. However, the cloud masking was turned on and the 'cloud distance' parameter which removes any shorelines within 'cloud distance' to a cloud causes there to be gaps in the shoreline. You could lower cloud distance to something like 10-20m to fix this or turn off cloud masking

![case_study_2_why_cloud_mask_off](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/3d42653c-54fb-4568-a59c-d5ed75e3d844)

- Click `Save Settings`

![case_study_2_save_settings](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/b502e062-e518-4d26-8fa7-fe99de489693)

### 7.Name the Session

- Let's call this 'case_study_2'

- This is the name of the folder that will be saved in `CoastSeg/sessions`

![case_study_2_session](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/be91a824-66ac-4232-8a68-99160eca6ba6)

### 8.Preview the available Imagery

- Preview the amount of available imagery for the selected ROI between the dates

- In this example ROI 'cwm3' has 18 images available from LandSat 8 and 16 images available from LandSat 9 for 12/01/2023 - 03/01/2024

![case study 1 preview imagery](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/db42fee9-682b-4e15-8470-b97a166e42a8)

### 9.Add a Shoreline Extraction Area

- Our study site has a small pond located along the shoreline that will get picked up in the reference shoreline buffer and the pond's water land interface will get misidentified as a shoreline

To get around this issue we have 2 options:

1.Shrink the reference shoreline buffer

-- This would probably be the best solution for this specific, but if you have a dynamic coastline this may not be the best solution for you

2.Add a shoreline extraction area

**Example of Bad Shoreline**

- In this image below you can see the pond's water land interface gets misidentified as part of the shoreline

![case_study_2_why_shoreline_extraction_area_needed_pond](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/8c41a33a-722d-433b-a5da-b6b0667b9078)

**How to Draw a Shoreline Extraction Area**

- Draw a shoreline extraction area that does not include the pond

![case_study_2_shoreline_extraction_area](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/6c239b43-fdf4-4236-8021-56125b18ee0e)

**Example of Good Shoreline from New Shoreline Extraction Area**

- Here is what the shoreline detection image will look like with this new shoreline extraction area

![case_study_2_good_shoreline_extraction_area](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/478b38bf-8fae-44b0-a1d8-23570357357f)

### 10.Download the ROI

- Click the ROI you want to download on the map ( they will turn blue when selected)

- Because we set the cloud threshold to 60% and the percent of bad pixels to 66% you can see that several downloads were skipped because they exceeded the limits

- When the download finishes CoastSeg will print the location where the downloads were saved in this case its 'pgj2''CoastSeg\data\ID_pgj2_datetime04-22-24\_\_09_38_49'

![case_study_2_gen_and_download_roi](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/e8755f13-e2ef-45e4-a22f-42a2d46b150a)

### 11.Sort the Downloaded Imagery

- Open Coastseg/data and open the folder containing the ROI ID, in my case thats 'pgj2', so I opened 'CoastSeg\data\ID_pgj2_datetime04-22-24\_\_09_38_49'

- You can see the ROI ID in the hover menu located to the top right of the map

- Read this quick [guide](https://satelliteshorelines.github.io/CoastSeg/How-to-Filter-Out-Bad-Imagery/) on how to filter bad imagery in CoastSeg.

**Sort out Images with Sea Ice**

- Sort any bad images into the 'bad folder'

- Sort out images with snow, sea ice, or cloud cover

| Images                                                                                                                                  | Sample Image                                                                                                                          |
| --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| ![case_study_2_sea_ice_bad_sort](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/1559cec9-755a-41a5-b6ba-a87c33992220)  | ![case_study_2_sea_ice_example](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/eaf9ede6-cbed-4760-9225-cf020fc58b4e) |
| Filter out images with sea ice and snow                                                                                                 |                                                                                                                                       |
| ![case_study_2_sample_cloud_imgs](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/692b8742-2966-474b-9b41-94f1ff25278f) | ![case_study_2_bad_cloud](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/de0aedd2-4b9e-4d43-a533-dd31fd77360c)       |
| Filter out images with clouds                                                                                                           |

**Why is Sea Ice Bad for Extracting Shorelines?**

- Sea Ice forms offshore and changes where the land water interface is meaning that the reference shoreline no longer covers the region where the shoreline can be extracted

![case_study_2_sea_ice_detection_bad_example](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/15272a4b-fb98-42e5-9c76-955de96a6c31)

**Why is Snow Bad for Extracting Shorelines?**

- Snow will get misclassified as either whitewater or water sometimes making shoreline extraction variable.

- In this example the snow was mis classified as whitewater

![case_study_2_snow_detection_shoreline_](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/0325919a-b830-482d-bc79-9b55abbc3289)

**Why are Clouds Bad for Extracting Shorelines?**

- Clouds get misclassified as shorelines as you can see in the example below

- Clouds getting misclassified as shorelines is the reason behind why we typically mask out clouds

| Bad Shoreline Detection Due to Clouds                                                                                                      |
| ------------------------------------------------------------------------------------------------------------------------------------------ |
| ![case_study_2_clouds_bad_detection](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/1bc345b4-4a68-450d-bc24-e125309b4b1b) |
| ![case_study_2_bad_clouds1](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/ae7b6283-6ec5-4931-a56d-a99935815327)          |
| ![case_study_2_bad_clouds2](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/cb64bc8a-e1ab-4cfd-95a1-933e5f2ba156)          |

### 12.Extract Shorelines

- Extracting shorelines works by finding the land water interface in the image and drawing a line along it

- A time series of shoreline position along each transect is generated as well

![shoreline_transect_intersection](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/e87b8d34-d9a4-4b1e-b3de-8e0be1c16ecd)

![case_study_2_extract_shorelines](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/36e15597-65fc-4932-be4b-52b0c773fc9d)

### 13. Examine Detection Images for Extracted Shorelines

- The detection images for the extracted shorelines is at 'CoastSeg\sessions\case_study_2\

- In these images you can see how well the shoreline were extracted depending on cloud cover, the size of the reference shoreline buffer and the rest of the extract shoreline settings

- There are a few images with some bad shorelines. Lets remove those in step 13

![case_study_2_shoreline_detection_folder](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/60047725-e759-4aa7-b2ba-e3100a107c9a)

### 14. Remove Outlier/Bad Extracted Shorelines

- Use the Load Extracted Shoreline feature to view all the extracted shorelines on the map

- Find any bad shorelines and click the trash icon to put that shoreline in the trash

- Once you've put all the bad shorelines in the trash click the empty trash button and this will delete all those shorelines from all the files in the session directory.

![case_study_2_remove_outllier_shorelines](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/00bd9398-1a35-42f0-8c0a-8d2dac09c5c6)

### 15. Adjust the Settings to Extract Better shorelines

- For difficult sites such as Alaska you'll need to create a few sessions with different combinations of settings in order to extract the best shorelines

- For this example I had to create 4 sessions to find the best combination of settings. For this reason we HIGHLY recommed you create a small session first (1-2 years worth of imagery) and optimize the settings first before running a large session.

- One trick we recommended is to download your entire dataset first, then adjust the date range to a year when you extract your shorelines. This will cause CoastSeg to only extract shorelines from the date range you set in the settings rather than the entire range of dates you downloaded.

### 16. Open the Extracted Shoreline Session Outputs in QGIS

**Config_gdf.geojson**

- This screenshot show the contents of the config_gdf.geojson file in QGIS, you can see the ROI, the transects and the reference shoreline on the map

![case_study_2_config_gdf](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/6141c888-4b27-464d-8a88-27a193ac9502)

**extracted_shorelines_points.geojson & extracted_shorelines_lines.geojson**

- This screenshot show the contents of the extracted_shorelines_points.geojson & extracted_shorelines_lines.geojson files in QGIS

- These files contain the 2D shoreline vectors extracted directly from the satellite imagery. These are NOT the shoreline positions along the transects. The shoreline position along the transect is located in the timeseries files eg. raw_transect_time_series_points.geojson,raw_transect_time_series.csv, raw_transect_time_series_vectors.geojson, raw_transect_time_series_merged.csv

**Zoomed Out 2D Extracted Shorelines as Points and Lines**

![case_study_2_extracted_shorelines_points](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/8030612c-ed83-43bd-92f8-f8bd8c0cf592)

**Zoomed In 2D Extracted Shorelines as Points and Lines**

![case_study_2_extracted_shorelines_points_zoomed_in](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/9d5d17d7-1a20-4282-a0cb-01fedb6febf5)

**raw_transect_time_series_points.geojson & raw_transect_time_series_vectors.geojson**

- These files contain the shoreline positions along the transects. This is the geojson format of the 'raw_transect_time_series_merged.csv' and 'raw_transect_time_series.csv'

![case_study_2_timeseries_extracted_shorelines_points](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/8f82e270-92c5-4607-b5d9-475226a04346)

## Apply Tidal Correction to Extracted Shorelines (Optional)

### 1.Download the tide model

- Before tidal correction can be applied the tide model must be downloaded

- Follow the tutorial: [How to Download Tide Model](https://satelliteshorelines.github.io/CoastSeg/How-to-Download-Tide-Model/)

### 2.Load the Session with Extracted Shorelines

- Re-open the jupyter notebook

- Under the 'Kernel' menu Click 'restart and clear outputs of all cells'

![restart kernel and clear outputs of all cells](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/a7d09bcb-6c35-48b2-b28a-a6821881e503)

- Click 'Load Session' and load 'case_study_2'

![select load session and tide correct](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/581f8b4a-062e-4326-9ae8-0145026fb9ad)

### 3.Click Correct Tides

- Click the ROI ID from the dropdown

       -- You should see some extracted shorelines on the map if you don't then the ROI ID won't appear in the dropdown

- Enter Beach Slope

- Enter Beach Elevation relative to Mean Sea Level

![case_study_1_tide_correction](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/8091247b-ad1b-4233-b09f-81363dde1202)

**Example**

![case_study_2_correct_tides](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/19276668-3285-43a8-bfb0-7ca1edb165cd)

### 4.View the Tidally Corrected TimeSeries in QGIS

4 new files will be generated:

1.'tidally_corrected_transect_time_series.csv'

2.'tidally_corrected_transect_time_series_merged.csv'

3.'tidally_corrected_transect_time_series_points.geojson'

4.'tidally_corrected_transect_time_series_vectors.geojson'

- This screenshot show the difference between the tidally_corrected_transect_time_series_vectors and raw_transect_time_series_vectors as you can see applying tidal correction shifts the raw shoreline position along the transect to account for the tide position

![case_study_2_raw_and_tide_corrected_timeseries_qgis](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/59df41c7-2692-4f81-bd3a-cd997b2f7c37)
