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

- Under the sessions directory within the session you created look under `jpg_files\detection`

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
