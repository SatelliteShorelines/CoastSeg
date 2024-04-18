# Extract Shorelines From Downloaded Imagery

---

**Extracted Shorelines** are vectors created from satellite imagery to outline coastlines. To ensure accuracy:

- Each `.tiff` file is checked for cloud cover and quality.

- Suitable images are processed using a segmentation model to classify sand, water, and surf.

- Shorelines are then extracted and represented as colored vectors.

**Common Issues:**

- **Complex Shorelines:** Adjust the `sand color` setting if shorelines appear erratic.

- **Missing Shorelines:** This may occur due to poor image quality or excessive cloud cover.

## Why did isn't there an Extracted Shoreline

Sometimes shorelines will be not be able to be extracted from any of the imagery downloaded for the ROI on the map due to image quality issues such as excessive cloud cover or too many 'no data'(black) pixels in the downloaded imagery. When this is the case the extracted shoreline will not appear on the map.

The message `The ROI id does not have a shoreline to extract. ` will print when this happens

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

![ROI_removal_and_shoreline_loading](https://github.com/Doodleverse/CoastSeg/assets/61564689/0928145a-e5f4-4d3b-bda4-315e17b14263)

**Option 2 : Load Your Own Shorelines**

1. Select 'Shoreline' from under the drop down **Load Feature from File**
2. Load shorelines from your geojson file
   - Check out the guide for how to do this here [How to Load Features from Geojson Files on Map](https://github.com/SatelliteShorelines/CoastSeg/wiki/04.-How-to-Use-the-Map#how-to-load-features-from-geojson-files-on-map)

![how_to_make_ref_shoreline](https://github.com/Doodleverse/CoastSeg/assets/61564689/3cd70302-9bc0-411c-87f1-b25d86ac2280)

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

| Example Image                                                                                                               | Description                                                | Solution                                                            |
| --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------- |
| ![1985-09-17-15-02-56_RGB_L5](https://github.com/Doodleverse/CoastSeg/assets/61564689/37d470e2-e2fe-4489-966f-a2a2384b928b) | ROI is too small with excessive water coverage.            | Make your ROI larger to include more land.                          |
| ![2020-03-09-15-33-35_RGB_L8](https://github.com/Doodleverse/CoastSeg/assets/61564689/cb66cedb-9e90-446a-a6d1-3ac20f1ef0df) | Cloud masking failed and mistakenly masked the shoreline.  | Try turning off cloud masking or set 'cloud_mask_issue' to True     |
| ![2020-07-16-15-51-52_RGB_S2](https://github.com/Doodleverse/CoastSeg/assets/61564689/78071e57-a6dd-4ea1-b9aa-9731b9ad3723) | Image is likely too cloudy to extract accurate shorelines. | Try using imagery with less clouds or try turning off cloud masking |

## Step 4: Examine Shorelines Extracted from 3-5 years of imagery

- Under the sessions directory within the session you created look under `jpg_files\detection`

| Description                                                                              | Example Image                                                                                                             | Suggested Adjustment                                                        |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| This shoreline is almost perfect, but it needs adjustment to capture the thin shoreline. | ![2020-02-22-15-33-41_L8](https://github.com/Doodleverse/CoastSeg/assets/61564689/fea8ab34-8e86-42ce-91d6-7ff9f3c165de)   | Lower the `minimum beach area` to include thinner shorelines.               |
| More shoreline needs to be captured close to cloud-covered areas.                        | ![2020-10-19-15-34-01_L8](https://github.com/Doodleverse/CoastSeg/assets/61564689/4e7a1db5-9a76-41a9-8519-ca8d45e0b76e)   | Increase the `Cloud Distance` to capture more of the shoreline near clouds. |
| The reference shoreline buffer (in purple) is too narrow                                 | ![ref_sl_too_small](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/d3b8a195-c038-4a47-93ce-f1659958d586) | Increase the buffer size to better capture area dynamics.                   |

## Step 5: Experiment with the Settings

- Manipulate the settings until you get shorelines you like
- If you find that the cloud masking is covering your shorelines try turning off cloud masking and downloading your imagery again.

![image](https://github.com/Doodleverse/CoastSeg/assets/61564689/5e92e8c3-9540-4b7e-8829-d7e822f0fd20)

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
