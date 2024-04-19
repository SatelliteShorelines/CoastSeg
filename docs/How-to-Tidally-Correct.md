![shoreline_transect_intersection](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/e87b8d34-d9a4-4b1e-b3de-8e0be1c16ecd)

⚠️<b>Make sure you have downloaded the FES2014 tide model following the guide 'How to Download the Tide Model'<b>⚠️

## Option 1: Tide Correction Button Method (Recommended)

Ever since coastseg 1.1.0 released you can now use the `correct tides` button in coastseg to automatically correct the timeseries. This button will automatically find which files need to be tidally corrected and apply the tidal correction for you.

⚠️ <b>Important</b> ⚠️

If the tide model was NOT downloaded to `CoastSeg/tide_model` then the tide correction button will NOT work. The tide correction button will try to load the tide model from `<CoastSeg location>/tide_model` and an error will occur. Instead follow the instructions in at [Option 2: Tide Correction Script](#option-2-tide-correction-script)

### Steps

**WARNING**
You must have downloaded the tide model to the `tide_model` folder within coastseg for this button to work correctly.

1.**Load a Session**

- Load a session containing extracted shorelines.
- The following files will be saved to each ROI in the session (any existing files will be overwritten)

  -- `predicted_tides.csv`: contains the tide prediction from the tide model for each transect end point for each date a shoreline was detected on it.

  -- `tidally_corrected_transect_time_series_points.geojson` : contains the tidally_corrected shoreline transect intersections points from `raw_transect_time_series_merged.csv`.

  -- `tidally_corrected_raw_transect_time_series_vectors.geojson` : contains the tidally_corrected shoreline transect intersections from `raw_transect_time_series_merged.csv`. as vectors by connecting points that occurred on the same date together.

  2.**Enter the beach slope**

- This is the beach slope in meters for the ROI you have selected.

  3.**Enter the beach elevation**

- This is the beach elevation in meters relative to MSL(Mean Sea Level) for the ROI you have selected.

  3.**Select the ROIs' ids to tidally correct**

  4.**Click compute tides and wait a few minutes.**

If you encounter any issues or have any questions please submit an issue at [CoastSeg Issues](https://github.com/Doodleverse/CoastSeg/issues/new/choose)

![image](https://github.com/Doodleverse/CoastSeg/assets/61564689/c72ee626-86fd-46ae-98d0-0921b0082d8b)

## Option 2: Tide Correction Script

⚠️ If the tide model was NOT downloaded to `CoastSeg/tide_model`, then you must use the tide correction script.

### Example 1: Apply Tide Correction to a session and provide tide model location

1.**Choose a session**

- In this example I will be using session `dnz3_extract_shorelines_10_yr` and using ROI `ID_dnz3_datetime12-22-23__09_10_44`
- Example: `C:\development\doodleverse\coastseg\CoastSeg\sessions\dnz3_extract_shorelines_10_yr\ID_dnz3_datetime12-22-23__09_10_44`

  2.**Get the location of the config_gdf.geojson file**

- This parameter is used after the `-C `
- Example: `-C "C:\development\doodleverse\coastseg\CoastSeg\sessions\dnz3_extract_shorelines_10_yr\ID_dnz3_datetime12-22-23__09_10_44\config_gdf.geojson"`

  3.**Get the location of the transect_time_series.csv file**

- This parameter is used after the `-T `
- Example: `-T "C:\development\doodleverse\coastseg\CoastSeg\sessions\dnz3_extract_shorelines_10_yr\ID_dnz3_datetime12-22-23__09_10_44\transect_time_series.csv"`

  4.**Get the location of the tide_model**

- This parameter is used after the `-M`
- Example: `-M "C:\development\doodleverse\coastseg\CoastSeg\tide_model"`

  5.Select a Beach Elevation

- This is the elevation of the beach in meters relative to MSL(Mean Sea Level) for the ROI you have selected
- Example: `-E 3.2`

  6.Select a Beach Slope

- This is the slope of the beach in meters for the ROI you have selected
- Example: `-S 2.1`

  7.**Assemble the command**

- Assemble all commands into a single line. I recommended using notepad to combine all the commands into a single line

```
python apply_tidal_correction.py -E 3.2 -S 2.1 -C
"C:\development\doodleverse\coastseg\CoastSeg\sessions\dnz3_extract_shorelines_10_yr\ID_dnz3_datetime12-22-23__09_10_44\config_gdf.geojson" -T
"C:\development\doodleverse\coastseg\CoastSeg\sessions\dnz3_extract_shorelines_10_yr\ID_dnz3_datetime12-22-23__09_10_44\transect_time_series.csv"
-M "C:\development\doodleverse\coastseg\CoastSeg\tide_model"
```

8.**Run the command**

- It takes a few minutes to predict the tides

![image](https://github.com/Doodleverse/CoastSeg/assets/61564689/13582ba7-63c6-46d4-93c1-aa68db23626c)

### Parameters:

1.**-C (or -c) [CONFIG_FILE_PATH]**

- Description: Path to the configuration file.
- Example: `-C "path_to_config_file"`
- `CONFIG_FILE_PATH`: Path to the configuration file. This is the `config_gdf.geojson` in the session directory generated when shorelines were extracted

  2.**-T (or -t) [RAW_TIMESERIES_FILE_PATH]**

- Description: Path to the raw timeseries file.
- Example: `-T "path_to_timeseries_file"`
- `RAW_TIMESERIES_FILE_PATH`: Path to a csv file containing the time series created by extracting shorelines with coastseg. This timeseries
  represents the intersection of each transect with the shoreline extracted for a particular date and time. It is not tidally corrected.

  3.**-E (or -e) [REFERENCE_ELEVATION]**

- Description: Set the reference elevation value. This is a float number.
- Example: `-E 3`

  4.**-S (or -s) [BEACH_SLOPE]**

- Description: Set the beach slope value. This is a float number.
- Example: `-S 2`

### Optional Configuration Options

If you didn't install the tide model in the default location you will need to modify the following variables

5.**-P (or -p) [TIDE_PREDICTIONS_FILE_NAME]**

- Description: File name for saving a csv file containing the tide predictions for each date time in the timeseries provided.
- By Default this file is named "tidal_predictions.csv"
- Example: `-P "tidal_predictions.csv"`

  6.**-O (or -o) [TIDALLY_CORRECTED_FILE_NAME]**

- Description: File name for saving the tidally corrected time series csv file.
- By Default this file is named "tidally_corrected_time_series.csv"
- Example: `-O "tidally_corrected_time_series.csv"`

  7.**-R (or -r) [MODEL_REGIONS_GEOJSON_PATH]**

- Description: Path to the model regions GeoJSON file.
- By default the program looks for `tide_regions_map.geojson` in the `scripts` directory
- Example: `-R "c:\coastseg\scripts\tide_regions_map.geojson"`
- `MODEL_REGIONS_GEOJSON_PATH`: Path to the location of the geojson file containing the regions used to create the clipped tide model in the previous steps. This file is typically located in the scripts directory within coastseg. "c:\coastseg\scripts\tide_regions_map.geojson"`

  8.**-M (or -m) [FES_2014_MODEL_PATH]**

- Description: Path to the FES 2014 tide model directory.
- Example: `-M "c:\coastseg\tide_model"`
- `FES_2014_MODEL_PATH`: Path to the FES 2014 tide model, by default attempts to load from `coastseg\tide_model` if you installed the tide_model from
  step 3 in a different location modify this variable to have the full location to the directory containing the clipped 2014 fes tide model.

### Running the script:

Use the parameters as described while executing the script. Here's an example usage:
**Example 1: Only Required Parameter**

```bash
cd scripts
python apply_tidal_correction.py -C "path_to_config" -T "path_to_timeseries" -E 3 -S 2
```

```
python apply_tidal_correction.py -E 3 -S 2 -C "C:\development\doodleverse\coastseg\CoastSeg\sessions\fire_island\ID_ham1_datetime08-03-23__10_58_34\config_gdf.geojson" -T "C:\development\doodleverse\coastseg\CoastSeg\sessions\fire_island\ID_ham1_datetime08-03-23__10_58_34\transect_time_series.csv"
```

**Example 2: All Available Parameters**

```bash
cd scripts
python apply_tidal_correction.py -C "path_to_config" -T "path_to_timeseries" -E 3 -S 2 -P "predictions.csv" -O "corrected.csv" -R "regions.geojson" -M "model_directory"
```

```bash
cd scripts
python apply_tidal_correction.py -E 3 -S 2 -C "C:\development\doodleverse\coastseg\CoastSeg\sessions\fire_island\ID_ham1_datetime08-03-23__10_58_34\config_gdf.geojson" -T "C:\development\doodleverse\coastseg\CoastSeg\sessions\fire_island\ID_ham1_datetime08-03-23__10_58_34\transect_time_series.csv" -P "tidal_predictions.csv" -O "tidally_corrected_time_series.csv" -R "C:\development\doodleverse\coastseg\CoastSeg\scripts\tide_regions_map.geojson" -M "C:\development\doodleverse\coastseg\CoastSeg\tide_model"
```

## Results

The results will be located in the scripts directory where the `apply_tidal_correction.py` script is located.

- `tidal_predictions.csv`: Contains the tide predictions for each datetime in the timeseries csv file given by `RAW_TIMESERIES_FILE_PATH`.
- `"tidally_corrected_time_series.csv"`: Contains the tidally corrected time series data based on the timeseries csv file given by `RAW_TIMESERIES_FILE_PATH`.

## Credits

Thank you [DEA-Coastlines](https://github.com/GeoscienceAustrali/dea-coastlines/wiki/Setting-up-tidal-models-for-DEA-Coastlines) for making a guide on how to use pyTMD and [pyTMD](https://pytmd.readthedocs.io/en/latest/api_reference/aviso_fes_tides.html) for making a easy to use script to download the AVISO FES 2014 Model.
The `model_tides` in this code has been modified and the original function was originally written by Robbi Bishop-Taylor for the `dea-tools` package https://github.com/GeoscienceAustralia/dea-notebooks/blob/develop/Tools/dea_tools/coastal.py#L466-L473
