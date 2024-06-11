# What is a Session?

A session is a saved state that CoastSeg was in. A session can be used to load in the ROIs that were downloaded previously or the previously extracted shorelines.
A session is typically stored as a directory containing a minimum of two files a `config.json` and a `config_gdf.geojson` file that lets you restore the state coastseg was in.

### Where are Sessions Located?

Sessions are saved to `CoastSeg/sessions`. A session contains the all the files related to extract shorelines for a particular set of ROIs in the downloaded data located in `CoastSeg/data`. Each session follows a similar format depending on what actions were taken by the user. For instance, if tide correction was not applied during the session then none of the files with the tidal correction applied will be present.

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

# How to Save a New Session

1. Load ROIs on the map
2. Extract Shorelines
3. Write the session name in the textbox then click enter
   - If the session already exists and you click extract shorelines the data will be overwritten.

![image](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/7cf7541e-c516-4c9b-8f73-02a769ae8aa8)

# How to Load Downloaded Data

1. Click Load Session
2. Navigate from the `sessions` directory to the `data` directory.
3. Click on an ROI directory
4. Click the select button
5. On the map navigate to where the ROIs were loaded.

To load config files for ROIs that have been downloaded before go to the `data` directory within `coastseg` then find the directory of the ROIs you want to upload. Inside that directory, for example the `ID_3_datetime10-20-22__07_09_07` directory, there should be `geojson` file named `config_gdf.geojson`. You should notice the `ID_<ROI ID NUMBER>` in the directory name in this case this is ROI id 3. This will load the ROI in that directory and all the other selected ROIs,shorelines, transects, and the bounding box that were on the map when it was saved.

⭐ This works even for data downloaded on someone else's computer just copy and paste the ROI directory from their computer to your data directory

![load_data_session](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/6275a370-282d-48a4-a340-5b13cf4d885f)

# How to Load Extracted Shorelines from a Session

1. Click load session button
2. Select a directory from the sessions directory
   ![load_extracted_shorelines_session](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/4d3ba936-3d86-48e1-9d2d-ea67a355c71d)

## Config.json

Config.json files can be found in session directories as well as within ROI directories within the data directory.

### Sample Config.json in the data directory

The config files are organized into 2 sections:

1. Settings used to download each ROI organized by each ROI's id
2. Settings that were saved when the ROI was downloaded under the 'settings' section

a. Settings used to download each ROI organized by each ROI's id

- One setting to pay attention to is the `output epsg` this is the CRS the ROI and all its features are re-projected features into and these re-projected features are the ones used to calculate the shoreline transect intersections.

- The following settings are associated with ROI 0:
  - ROI 0 has imagery for dates "2018-01-01" - "2019-01-01" and it was saved to the `ID_0_datetime03-22-23__07_29_15` directory at
    `C:\\1_CoastSeg\\1_official_CoastSeg_repo\\CoastSeg\\data`.

```
{
    "0": {
        "dates": [
            "2018-01-01",
            "2019-01-01"
        ],
        "sitename": "ID_0_datetime03-22-23__07_29_15",
        "polygon": [
            [
                [
                    -75.75713505156588,
                    36.189897585260795
                ],
                [
                    -75.7514757042764,
                    36.17420897122136
                ],
                [
                    -75.74012332310643,
                    36.176873810012815
                ],
                [
                    -75.74578267039591,
                    36.191909415885604
                ],
                [
                    -75.75713505156588,
                    36.189897585260795
                ]
            ]
        ],
        "roi_id": "0",
        "sat_list": [
            "L5",
            "L7",
            "L8"
        ],
        "landsat_collection": "C02",
        "filepath": "C:\\1_CoastSeg\\1_official_CoastSeg_repo\\CoastSeg\\data"
    },
    "roi_ids": [
        "0"
    ],
    "settings": {
        "landsat_collection": "C02",
        "dates": [
            "2018-01-01",
            "2019-03-01"
        ],
        "sat_list": [
            "L5",
            "L7",
            "L8"
        ],
        "cloud_thresh": 0.68,
        "dist_clouds": 100,
        "output_epsg": 32618,
        "check_detection": false,
        "adjust_detection": false,
        "save_figure": true,
        "min_beach_area": 2100,
        "min_length_sl": 200,
        "cloud_mask_issue": false,
        "sand_color": "default",
        "pan_off": "False",
        "max_dist_ref": 396,
        "along_dist": 25,
        "min_points": 3,
        "max_std": 15.0,
        "max_range": 30.0,
        "min_chainage": -100.0,
        "multiple_inter": "auto",
        "prc_multiple": 0.1
    }
}
```
