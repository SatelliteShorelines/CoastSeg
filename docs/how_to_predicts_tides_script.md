# ⚠️ Before you begin ⚠️

Make sure you have downloaded the FES2014 tide model following the guide 'How to Download the Tide Model'

## How to Predict Tides at a Single Point

1.**Choose a point**

To use the new `tide_predictions` script you're first going to need to go https://geojson.io/#map=2/0/20 and place a point in the sea around parranporth. Save this point to geojson file. This is the location where the tide model will predict the tides.

![image](https://github.com/SatelliteShorelines/coastsat_package/assets/61564689/c5d38b4e-182d-4be1-83f9-25ca584629ff)

- This parameter is used after the `-C `
- Example: `-C "C:\development\doodleverse\coastseg\CoastSeg\sessions\dnz3_extract_shorelines_10_yr\ID_dnz3_datetime12-22-23__09_10_44\config_gdf.geojson"`

  2.**Get the location of the transect_time_series.csv file**

- This parameter is used after the `-T `
- Example: `-T "C:\development\doodleverse\coastseg\CoastSeg\sessions\dnz3_extract_shorelines_10_yr\ID_dnz3_datetime12-22-23__09_10_44\transect_time_series.csv"`

  3.**Get the location of the tide_model**

- This parameter is used after the `-M`
- Example: `-M "C:\development\doodleverse\coastseg\CoastSeg\tide_model"`

  4.**Get the location of the tide model regions geojson file**

- This file contains the regions the tide model was split into so tide predictions could run quicker: https://github.com/SatelliteShorelines/CoastSeg/blob/main/scripts/tide_regions_map.geojson
- The script needs to know where this file is so it knows which region of the tide model the point belongs to and uses that tide model to predict the tides
- Example:
  `-R "C:\development\doodleverse\coastseg\CoastSeg\scripts\tide_regions_map.geojson"`

  5.**Assemble the command**

- Assemble all commands into a single line. I recommended using notepad to combine all the commands into a single line

```bash
python predict_tides.py -C "C:\Users\sf230\Downloads\parranporth.geojson" -T "C:\development\doodleverse\coastsat_package\coastsat_package\data\NARRA\transect_time_series.csv"  -M "C:\development\doodleverse\coastseg\CoastSeg\tide_model" -R "C:\development\doodleverse\coastseg\CoastSeg\scripts\tide_regions_map.geojson"
```

### Parameters:

1.**-C (or -c) [GeoJSON FILE PATH ]**

- Description: Path to the geojson file containing the point
- Example: `-C "path_to_config_file"`
- `GeoJSON FILE PATH `: Path to the geojson file.

  2.**-T (or -t) [RAW_TIMESERIES_FILE_PATH]**

- Description: Path to the raw timeseries file.
- Example: `-T "path_to_timeseries_file"`
- `RAW_TIMESERIES_FILE_PATH`: Path to a csv file containing the time series created by extracting shorelines with coastseg. This timeseries
  represents the intersection of each transect with the shoreline extracted for a particular date and time. It is not tidally corrected.

### Optional Configuration Options

If you didn't install the tide model in the default location you will need to modify the following variables

**-P (or -p) [TIDE_PREDICTIONS_FILE_NAME]**

- Description: File name for saving a csv file containing the tide predictions for each date time in the timeseries provided.
- By Default this file is named "tidal_predictions.csv"
- Example: `-P "tidal_predictions.csv"`

**-R (or -r) [MODEL_REGIONS_GEOJSON_PATH]**

- Description: Path to the model regions GeoJSON file.
- By default the program looks for `tide_regions_map.geojson` in the `scripts` directory
- Example: `-R "c:\coastseg\scripts\tide_regions_map.geojson"`
- `MODEL_REGIONS_GEOJSON_PATH`: Path to the location of the geojson file containing the regions used to create the clipped tide model in the previous steps. This file is typically located in the scripts directory within coastseg. "c:\coastseg\scripts\tide_regions_map.geojson"`

**-M (or -m) [FES_2014_MODEL_PATH]**

- Description: Path to the FES 2014 tide model directory.
- Example: `-M "c:\coastseg\tide_model"`
- `FES_2014_MODEL_PATH`: Path to the FES 2014 tide model, by default attempts to load from `coastseg\tide_model` if you installed the tide_model from in a different location then CoastSeg/tide_model then modify this variable to have the full location to the directory containing the clipped 2014 fes tide model.
