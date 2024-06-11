
### Where are sessions located?

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

- **extracted_shorelines_lines.geojson**: A GeoJSON file containing the extracted shorelines represented as lines.
- **extracted_shorelines_points.geojson**: A GeoJSON file containing the extracted shorelines represented as points.

## Raw Transect Time Series

- **raw_transect_time_series.csv**: Contains raw transect time series data in CSV format in the projected CRS (listed as `output_epsg` in the `config.json` settings)
  - This file contains the intersection between the shoreline and all the transects for each date. This intersection is the distance along the transect the shoreline was captured at.
- **raw_transect_time_series_merged.csv**: Merged raw transect time series data in CSV format in the projected CRS (listed as `output_epsg` in the `config.json` settings)

  - This file contains the intersection between the shoreline and all the transects for each date along with the x and y point in crs epsg 4326 of where the shoreline intersected the transect, and the x and y coordinate in crs epsg 4326 of the end point of the transect.

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
Under the `data` directory where all the downloaded ROIs are stored. Within each ROI's directory you will find a `config.geojson` file and a `config.json` file.

- `config.geojson` file that contains contains a geodataframe ( crs `espg 4326`) with all the ROIs, shorelines, transects, and the bounding box that were loaded on the map at the time the ROI was downloaded.

**Example: Loading the config.geojson into QGIS**

- You can see the Bounding Box, ROI, reference shorelines and transects for this session

![config_gdf_qgis](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/0fab574b-3714-46d2-bf78-c3bc9cb0622f)

- `config.json` file contains the settings that were used to download the imagery from Google Earth Engine(GEE).
