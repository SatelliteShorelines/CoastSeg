# Extract Shorelines From Downloaded Imagery

---

**Extracted Shorelines** are vectors created from satellite imagery to outline coastlines.

CoastSeg will generate the following files containing the 2D shorelines extracted from the imagery:

1.`extracted_shorelines_lines.geojson`: this contains the 2D extracted shorelines formatted as lines

2.`extracted_shorelines_points.geojson`: this contains the 2D extracted shorelines formatted as points

3.`extracted_shorelines_dict.json`: this contains all the extracted shorelines stored in a json format containing metadata like the satellite the shoreline was derived from and more.

![shoreline_transect_intersection](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/e87b8d34-d9a4-4b1e-b3de-8e0be1c16ecd)

## Why is there not an extracted shoreline for my ROI?

Sometimes shorelines will be not be able to be extracted from any of the imagery downloaded for the ROI on the map due to image quality issues such as excessive cloud cover or too many 'no data'(black) pixels in the downloaded imagery. The message `The ROI id does not have a shoreline to extract. ` will print when this happens. When this occurs no extract shoreline vectors will appear on the map.

Check the imagery you downloaded. Sometimes you'll need to turn off "Apply Cloud Mask" in the settings because the cloud mask is covering the shoreline. Other times your reference shoreline buffer is too small. Go check `extract_shorelines_report.txt` report located in your session to see what happened with more details.

In the guide below we will outline a few senarios you might encounter when extracting shorelines. Also go check out our [case studies](https://satelliteshorelines.github.io/CoastSeg/case_study_2/) to see full examples of how to use CoastSeg to extract shorelines even at difficult locations like Alaska.

## How to Extract Shorelines From Your Imagery

If its your first time extracting shorelines from imagery we recommend using a small dataset to learn how the setting for extracting shoreline work, then downloading the full dataset. The first time through use the default settings, then change one setting to see how it impacts your results.

## Step 1: Load a Reference Shoreline

- Guide [How to delete parts of shoreline](https://github.com/SatelliteShorelines/CoastSeg/wiki/04.-How-to-Use-the-Map#how-to-delete-shorelines-from-the-map)

**Option 1 : Load CoastSeg's Shorelines**

1. Draw a bounding box
2. Click load shoreline

- In this example, we have a back barrier shoreline we want to remove. Here is hack for getting rid of the back barrier shoreline you can also follow the tutorial at [How to delete parts of shoreline](https://github.com/SatelliteShorelines/CoastSeg/wiki/04.-How-to-Use-the-Map#how-to-delete-shorelines-from-the-map).

  1. Load the reference shoreline in the ROIs
  2. Delete the ROIs where you don't want the shoreline to be loaded.
  3. Delete the reference shoreline
  4. Click Load Shoreline Button to load the reference shoreline in the remaining ROIs

![ROI_removal_and_shoreline_loading](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/9437c93d-096a-4a01-b0fd-9a97d228e8bd)

**Option 2 : Load Your Own Shorelines**

1. Select 'Shoreline' from under the drop down **Load Feature from File**
2. Load shorelines from your geojson file
   - Check out the guide for how to do this here [How to Load Features from Geojson Files on Map](https://github.com/SatelliteShorelines/CoastSeg/wiki/04.-How-to-Use-the-Map#how-to-load-features-from-geojson-files-on-map)

![how_to_load_default_sl](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/c26b14a8-fef9-462c-8ed5-cad93f1f3d25)

## Step 2: Load Transects

CoastSeg extract shorelines from your imagery by checking if the water line intersects with a transect. If you don't have any transects in your Roi, then you won't be able to extract shorelines.

**Option 1 : Load CoastSeg Transects**

1. Draw a bounding box
2. Click load transects
   - Guide on how to do this here [How to Load Transects on the Map](https://github.com/Doodleverse/CoastSeg/wiki/3.-How-to-Use-the-Map#how-to-delete-shorelines-from-the-map:~:text=your%20feature%20again.-,How%20to%20Load%20Transects%20on%20the%20Map,-To%20load%20transects)

**Option 2 : Load Your Own Transects**

1. Select 'transects' from under the drop down **Load Feature from File**
2. Load transects a geojson file

## Step 3: Assess and Download Imagery

- **Check Your Imagery Quality:** Ensure the selected ROI has minimal cloud cover and a balanced mix of land and water. See [Examples of Bad Imagery](#Examples-of-Bad-Imagery).

- **Download Small Dataset:** Initially, download 3-5 years of imagery for a small area to test and refine your settings before proceeding with a larger dataset.

**Examples of Bad Imagery**

If most of your imagery looks like these examples you may need to change your ROI.

| Example Image                                                                                                              | Description                                                | Solution                                                             |
| -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------- |
| ![bad_img_too_small](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/949f11d7-64e5-4886-aee9-f6aec296ea39) | ROI is too small with excessive water coverage.            | Make your ROI larger to include more land.                           |
| ![masked_out_clouds](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/39529ab1-33d6-4ef8-a2b4-5d80ec92cdbf) | Cloud masking failed and mistakenly masked the shoreline.  | Try turning off cloud masking by setting 'Apply Cloud Mask' to False |
| ![cloudy_imgs](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/c0c00983-29a3-427d-bffe-6d4ee02f2f2a)       | Image is likely too cloudy to extract accurate shorelines. | Try using imagery with less clouds or try turning off cloud masking  |

## Step 4: Examine Shorelines Extracted from 3-5 years of imagery

- When you click "Extract Shorelines," a new directory with the session name you provided will be created at `CoastSeg/sessions`. This directory will contain a subdirectory for each ROI you selected. Within each subdirectory, you will find all the files generated by extracting shorelines, which are detailed in [Extracted Shoreline Session Contents](#extracted-shoreline-session-contents).

- You can find images that show the extracted shorelines on each image in `CoastSeg/sessions/<YOUR SESSION NAME>/ROI_ID_NAME/jpg_files/detection`. An example is shown below:

![coastseg/sessions/session/roi/jpg_files/detection](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/f5be75fb-1c75-46c2-bbec-b9b1e716fea4)

| Description                                                                              | Example Image                                                                                                             | Suggested Adjustment                                                        |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| This shoreline is almost perfect, but it needs adjustment to capture the thin shoreline. | ![beach_area](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/70799078-4983-4b07-84d2-e7186d4745d1)       | Lower the `minimum beach area` to include thinner shorelines.               |
| More shoreline needs to be captured close to cloud-covered areas.                        | ![cloud_dis](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/539ba73d-5430-44b8-8f07-9032862a0ea3)        | Increase the `Cloud Distance` to capture more of the shoreline near clouds. |
| The reference shoreline buffer (in purple) is too narrow                                 | ![shoreline_buffer](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/39312caf-ceaf-4a3e-85a2-843ac80c45a6) | Increase the buffer size to better capture area dynamics.                   |

## Step 5: Experiment with the Settings

- Manipulate the settings until you get shorelines you like
- If you find that the cloud masking is covering your shorelines try turning off cloud masking and downloading your imagery again.

![settings](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/cb578710-2382-4e01-bef2-9fc9df2f02b9)

## Step 6: Save the Settings

- Click Save Settings button each time you extract shorelines

## Step 7: Download the Full Time Series

- Now that you have the settings saved for how you want the shorelines extracted, download the full time series.

## Step 8: Filter out the Bad Imagery

- Once your data is downloaded open the directories you downloaded and follow the guide to filter out the bad imagery [How to Filter Out Bad Imagery](https://github.com/SatelliteShorelines/CoastSeg/wiki/06.-How-to-Filter-Out-Bad-Imagery)

## Step 9: Extract Shorelines for the Full Time Series

- Now that the data is downloaded and filtered, name a new session, and click extract shorelines to extract shorelines from all your downloaded imagery.

## Step 10: Remove Outlier Extracted Shorelines

Interactively view the extracted shorelines on the map and remove outliers using the "Load Extracted Shorelines" controls. You can view a tutorial on how to use it at [YouTube Tutorial on How to Remove Outlier Shorelines](https://www.youtube.com/watch?v=WlfC1bukXI0)

### YouTube Video

[![Alt text for your video](http://img.youtube.com/vi/WlfC1bukXI0/0.jpg)](http://www.youtube.com/watch?v=WlfC1bukXI0)


## Extracted Shoreline Session Contents

A session contains all the files created during the process of extracting shorelines from specific ROIs whose data is stored in `CoastSeg/data`. Each ROI's settings, rasters, imagery, and metadata, generated during the download process, are organized into folders within `CoastSeg/data`, ensuring that the downloaded data remains independent of the extracted shorelines. Each session references the specific ROI(s) in `CoastSeg/data` it was derived from in the `config.json` file. While the structure of a session may vary depending on user actions, such as whether tide correction was applied, it will always contain the files `config_gdf.geojson` and `config.json` because they are needed to reference the downloaded ROI(s)from which the shorelines were extracted.


## Location of Sessions

Sessions are saved to `CoastSeg/sessions`. A session contains the all the files related to extract shorelines for a particular set of ROIs in the downloaded data located in `CoastSeg/data`. Each session follows a similar format depending on what actions were taken by the user. For instance, if tide correction was not applied during the session then none of the files with the tidal correction applied will be present.

## Sample Session Structure

Here is a sample session that had 2 ROIs where shorelines were successfully extracted and tide correction was applied to both ROIs.

```
├── CoastSeg
|
|___session
|    |_ ID_4_datetime11-22-22__11_15_15
│   |   |_ jpg_files
│   |   |  |_ detection # this folder contains images of the extracted shorelines overlaid on the original scene
│   |   |  |  |_ 2023-12-09-18-40-08_L9.jpg   # this is the shoreline extracted from satellite L9 on 2023-12-09 at 18:40:08
│   |   |  |  |_ <rest of shoreline detection images>
│   |   |_config.json
│   |   |_config_gdf.json
│   |   |_extracted_shorelines_lines.geojson
│   |   |_extracted_shorelines_points.geojson
│   |   |_raw_transect_time_series.csv
│   |   |_raw_transect_time_series_merged.csv
│   |   |_raw_transect_time_series_points.geojson
│   |   |_raw_transect_time_series_vectors.geojson
│   |   |_tidally_corrected_transect_time_series.csv        # this file will only appear if tidal correction was applied
│   |   |_tidally_corrected_transect_time_series_merged.csv # this file will only appear if tidal correction was applied
│   |   |_tidally_corrected_transect_time_series_points.geojson # this file will only appear if tidal correction was applied
│   |   |_tidally_corrected_transect_time_series_vectors.geojson # this file will only appear if tidal correction was applied
│   |   |_shoreline_settings.json
│   |   |_transects_cross_distances.json
│   |   |_transects_settings.json
│   |   |
|   |_ ID_3_datetime11-22-22__11_15_15
│   |   |_ jpg_files
│   |   |_config.json
│   |   |_config_gdf.json
│   |   |_<rest of files>
```
