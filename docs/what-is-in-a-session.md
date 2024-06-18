# Understanding Sessions in CoastSeg

A **session** in CoastSeg is a self-contained folder that holds all the data and configurations related to the extraction of shorelines for a specific set of Regions of Interest (ROIs). Each session is uniquely identified by its name and stores the extracted shorelines, ROIs, transects, reference shorelines, settings,  and other related files used to extract the shorelines in a structured format. This organization allows users to easily manage and revisit their analyses.

## What is a Session?

A session contains all the files created during the process of extracting shorelines from specific ROIs whose data is stored in `CoastSeg/data`. Each ROI's settings, rasters, imagery, and metadata, generated during the download process, are organized into folders within `CoastSeg/data`, ensuring that the downloaded data remains independent of the extracted shorelines. Each session references the specific ROI(s) in `CoastSeg/data` it was derived from in the `config.json` file. While the structure of a session may vary depending on user actions, such as whether tide correction was applied, it will always contain the files `config_gdf.geojson` and `config.json` because they are needed to reference the downloaded ROI(s)from which the shorelines were extracted.

## Overview

- **[Location of Sessions](#location-of-sessions)**
- **[Sample Session Structure](#sample-session-structure)**
- **[Session Files](#session-files)**
  - [Configuration Files](#configuration-files)
  - [Extracted Shorelines](#extracted-shorelines)
  - [Raw Transect Time Series](#raw-transect-time-series)
  - [Tidally Corrected Transect Time Series](#tidally-corrected-transect-time-series)
  - [Shoreline and Transect Settings](#shoreline-and-transect-settings)
  - [Config Files](#config-files)


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

## Session Files

Each session contains the following files that contain information related to the shorelines extracted from each ROI. Below is an explanation of each file:

## Configuration Files

- **config.json**: Contains the settings that were used to download the ROI, the location of the downloaded ROI, and the settings that were used to extract shorelines.
- **config_gdf.json**: Contains all the features that were on the map when shorelines were extracted this includes:
  - Selected ROIs
  - Reference Shoreline
  - Transects
  - Bounding Box (if present)
  - Shoreline Extraction Area (if present)

## Extracted Shorelines

The raw shorelines that are extracted from each image are all stored together in geojson files in two formats lines and points. These shorelines are intersected with the transects to generate the timeseries of shoreline change along the transects.

- **extracted_shorelines_lines.geojson**: A GeoJSON file containing the extracted shorelines represented as lines.
- **extracted_shorelines_points.geojson**: A GeoJSON file containing the extracted shorelines represented as points.
- **extracted_shorelines_dict.json**: this contains all the extracted shorelines stored in a json format containing metadata like the satellite the shoreline was derived from and more.


## Raw Transect Time Series

- **raw_transect_time_series.csv**: Contains raw transect time series data in CSV format in the projected CRS (listed as `output_epsg` in the `config.json` settings)

     -- This file contains the intersection between the shoreline and all the transects for each date. This intersection is the distance along the transect the shoreline was captured at.

- **raw_transect_time_series_merged.csv**: Merged raw transect time series data in CSV format in the projected CRS (listed as `output_epsg` in the `config.json` settings)

     -- This file contains the intersection between the shoreline and all the transects for each date along with the x and y point in crs epsg 4326 of where the shoreline intersected the transect, and the x and y coordinate in crs epsg 4326 of the end point of the transect.

  ![raw_timeseries_merged.csv](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/39f6f725-481b-48f8-8be4-4829939107d8)

- **raw_transect_time_series_points.geojson**: Raw transect time series data represented as points in GeoJSON format in crs epsg 4326.
- **raw_transect_time_series_vectors.geojson**: Raw transect time series data represented as vectors in GeoJSON format in crs epsg 4326.

## Tidally Corrected Transect Time Series

These files will only be present if tidal correction was applied during the analysis:

- **tidally_corrected_transect_time_series.csv**: Tidally corrected transect time series data in CSV format in the projected CRS (listed as `output_epsg` in the `config.json` settings)
  - This file contains the intersection between the shoreline and all the transects for each date. This intersection is the distance along the transect the shoreline was captured at.
- **tidally_corrected_transect_time_series_merged.csv**: Merged tidally corrected transect time series data in the projected CRS (listed as `output_epsg` in the `config.json` settings)
  - This file contains the intersection between the shoreline and all the transects for each date along with the x and y point in crs epsg 4326 of where the shoreline intersected the transect, the x and y coordinate in crs epsg 4326 of the end point of the transect, as well as the tide that was used for the tide correction.

![tidally_corrected_timeseries_merged.csv](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/a8e03dcb-0681-4f98-b122-6894652e4f61) - This file contains the intersection between the shoreline and all the transects for each date along with the x and y point in crs epsg 4326 of where the shoreline intersected the transect, and the x and y coordinate in crs epsg 4326 of the end point of the transect.

- **tidally_corrected_transect_time_series_points.geojson**: Tidally corrected transect time series data represented as points in GeoJSON format.
- **tidally_corrected_transect_time_series_vectors.geojson**: Tidally corrected transect time series data represented as vectors in GeoJSON format.

## Shoreline and Transect Settings

- **shoreline_settings.json**: Contains settings specific to the shoreline extraction process.
- **transects_cross_distances.json**: Contains cross-distance data for transects.
- **transects_settings.json**: Contains settings specific to the transect analysis.

### Config Files

Config files are used to save the current state of the map and all the data downloaded during a session using CoastSeg. The config file is actually composed of two separate files a `config_gdf.geojson` file and `config.json` file.
Under the `data` directory where all the downloaded ROIs are stored. Within each ROI's directory you will find a `config_gdf.geojson` file and a `config.json` file.

- `config_gdf.geojson` file that contains contains a geodataframe ( crs `espg 4326`) with all the ROIs, shorelines, transects, and the bounding box that were loaded on the map at the time the ROI was downloaded.
    
    -- It does NOT contain the extracted shorelines. Those are in separate geojson files.

- `config.json` file that contains the settings used to download each ROI as well as the settings that were loaded into CoastSeg when either the shorelines were extracted or the download occured.

### Config_gdf.geojson

**Example: Loading the config_gdf.geojson into QGIS**

- You can see the Bounding Box, ROI, reference shorelines and transects for this session

![config_gdf_qgis](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/0fab574b-3714-46d2-bf78-c3bc9cb0622f)

### Config.json

Config.json files can be found in session directories as well as within ROI directories within the data directory.

**Sample Config.json in the data directory**

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
