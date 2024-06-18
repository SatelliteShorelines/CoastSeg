# Filtering Imagery


To extract the best shorelines, it's essential to use only high-quality imagery, which means filtering out any poor-quality images. You should filter out bad imagery in `data/roi_id/jpg_files/preprocessed/RGB` by moving any undesirable images to a designated subdirectory named 'bad'.

Filtering the RGB images does not delete any TIFF files but helps to improve the efficiency of shoreline extraction and enhance the quality of the extracted shorelines. For more information on the shoreline extraction process, check out the [wiki page about the shoreline extraction process](https://github.com/Doodleverse/CoastSeg/wiki/6.-How-to-Extract-Shorelines).


## Step-by-Step Guide

### Before You Begin

Download data with the `SDS_coastsat_classifier.ipynb` and ensure it is saved to your `data` directory within CoastSeg.

### 1. Open the `data` Directory

1. Navigate to the `data` directory within CoastSeg.
2. Click on the ROI directory you want to sort

In this example, you can see that ROI 3 is located within the `CoastSeg/data` folder. The `jpg_files > preprocessed > RGB` directory contains all the RGB imagery for all the satellites.

```
├── CoastSeg
|
|___data
|    |_ ID_3_datetime11-22-22__11_15_15
│   |   |_ L8
│   |   |_ L9
│   |   |_ S2
│   |   |_ jpg_files
│   |   |  |_ preprocessed
|   │   |   |  |_ RGB
|   |   │   |  |  |_ 2018-12-06-19-04-16_RGB_S2.jpg
|   |   │   |  |  |_ 2018-12-06-19-04-16_RGB_L8.jpg
|   |   │   |  |  |_ 2018-12-06-19-04-16_RGB_L9.jpg
|   │   |   |  |_ NIR
|   |   │   |  |  |_ 2018-12-06-19-04-16_NIR_S2.jpg
|   |   │   |  |  |_ <rest of images...>
|   │   |   |  |_ SWIR
|   |   │   |  |  |_ 2018-12-06-19-04-16_SWIR_S2.jpg
|   |   │   |  |  |_ <rest of images...>
│   |   |   |_detection
|   │   |   |  |_<jpgs of detected shorelines>
│   |   |_config.json
│   |   |_config_gdf.json
│   |   |_extracted_shorelines.geojson
```


### 2. Navigate to the RGB Directory

1. Go to `jpg_files > preprocessed > RGB` within the ROI directory.
   - Example on Windows: `CoastSeg\data\ID_yvk1_datetime06-05-23__06_57_26\jpg_files\preprocessed\RGB`

### 3. Move Bad Imagery

1. Create a subdirectory named 'bad' within the `data/roi_id/jpg_files/preprocessed/RGB` directory if it does not already exist.
2. Identify the images you want to remove.
3. Move these images to the 'bad' subdirectory.

![coastseg_screenshot_bad_subdir](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/f0423605-b8bc-4c1d-8eb2-c79be81e9a91)

### 4. Test Shoreline Extraction

Load your ROIs in CoastSeg and run `extract shorelines` again. None of the images that were sorted into the 'bad' directory will have their shorelines extracted. For more details, refer to the guide [How to Extract Shorelines](https://github.com/Doodleverse/CoastSeg/wiki/6.-How-to-Extract-Shorelines).