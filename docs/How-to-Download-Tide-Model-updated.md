# Overview

This guide will walk you through signing up for the necessary account and then choosing between two methods to download and clip the tide model for use in CoastSeg.

# Part 1: Download the Tide Model

### Step 1: Register
- **Sign Up:** Begin by registering on the AVISO platform. Visit the [AVISO Registration Page ](https://www.aviso.altimetry.fr/en/data/data-access/registration-form.html) to create your account.

### Step 2: Select the Tide Model

- **Select the FES Model** During the registration process, you'll be presented with various data models to choose from. Ensure you select the **FES (Finite Element Solution - Oceanic Tides Heights)** model.


### Step 3: Account Verification
- **Confirmation Email:** After completing the registration, AVISO will send you a confirmation email. Check your inbox (and spam folder if necessary) for this email.
- **Verify Your Account**: Click on the link provided in the email to verify your account. This step is crucial to activate your account and gain access to the data models.

### Step 4: Accessing Your Dashboard
- **Login:** Once your account is verified, [log into your AVISO account.](https://www.aviso.altimetry.fr/en/my-aviso-plus.html)
- **Navigate to Products:** On your account dashboard, you'll find a left side menu. Click on the **My products** option.
![image](https://github.com/Doodleverse/CoastSeg/assets/61564689/bf5382f0-6bc0-4867-893d-c8f84a3d3760)

### Step 5: Confirm Your Subscription
- **Check Your Subscriptions:** In the "Your current subscriptions" section, ensure that the **FES (Finite Element Solution - Oceanic Tides Heights)** model is listed. This confirms that you've successfully subscribed to the desired model.

![FES_products](https://github.com/user-attachments/assets/87d23089-1f46-43c7-bb96-d1850dc7a9c4)

# Part 2: Download and Clip the Tide Model

You have two options to download and clip the tide model: use the Jupyter Notebook (Recommended) or using a Script (Alternative). 

### Space and Time Requirements

- **Storage Space:** Ensure you have at least 14GB of free space available for fes2014.
   - If you want both fes2014 and fes2022 you will need at least 35 GB of free space.
- **Time Commitment:** The download and setup process for the fes2014 model takes approximately 1-2 hours. This is a one-time requirement.

## Option 1: Using the Download_tide_model.ipynb Notebook (Recommended)

The easiest way to download and clip the tide model is to use this notebook. Run all the code in the notebook and you're done!

1. **Prepare:** Ensure you have you are in an activated coastseg environment and know your AVISO email and password.
2. **Open Notebook:** Launch `Download_tide_model.ipynb` in Jupyter Notebook.

```
cd CoastSeg
jupyter lab Download_tide_model.ipynb
```

3.**Run Code:** Execute all cells in the notebook. The code will guide you through the download and clipping process automatically.

4.**Check Results:** Confirm that the model files are correctly downloaded and clipped in the specified directory.

- **Example Output from Downloading Tide Model Step in Notebook**
  ![download_tide_model_notebook](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/eecae8d2-cb5e-4e50-a587-3260ff9469b7)

- **Example Output from Clipping Tide Model Step in Notebook**
  ![clip_tide_model](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/abba33ae-1c63-4c48-be14-4e51e8224870)

  5.**Downloaded Tide Model Contents**

![tide model contents](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/2a8f2425-993f-4184-90cf-01b6ff65af4a)

# Option 2: Using the Script (Alternative)

## Setup

1. Create Directory: Navigate to your CoastSeg installation directory and create a new folder named tide_model.
2. Script Location: Change directory to the scripts folder within CoastSeg.

```
cd <location of your coastseg directory>
mkdir tide_model
cd scripts
```

![activate_coastseg_tide-Model_download](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/9706d481-5538-456d-8c79-b80d0c27b055)

## Download & Clip

1.**Run Download Script:** Execute the `aviso_fes_tides.py` script with the necessary parameters, including your AVISO credentials and target directory for the tide model.

- ⚠️ Enter your AVISO password when prompted. Note that the password won't display as you type. After you have finished typing, press Enter to proceed.
- Replace `C:\Users\Sample\coastseg` with the actual location of your CoastSeg directory.
- ![AVISO Password Prompt](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/4e858552-6869-4511-8a87-5caa0f12dd38)

  Example command:

```
 python aviso_fes_tides.py --user your_email@example.com -D C:\Users\Sample\coastseg\tide_model --load --log --tide FES2014 -G
```

2.**Clip Model to Regions:** After downloading, run the `clip_and_write_new_nc_files_cmd_line.py` script with the paths to your tide model directory and the regions geojson file.

To make the tide predictions compute faster we will clip the model it to various regions across the planet. After we've clipped the model to each region, when we want to predict the tide at a particular latitude and longitude we first figure out which region the point lies within, then we load the tide model we clipped to that region.

- `-R or --regions_file` the full path to the regions geojson file. You should find this file `tide_regions_map.geojson` in the `scripts`
- `-T or --tide_model` the full path to the directory containing the tide model downloaded
- Replace `C:\Users\Sample\coastseg` with the actual location of your CoastSeg directory.

```
python clip_and_write_new_nc_files_cmd_line.py -T C:\Users\Sample\coastseg\tide_model -R C:\Users\Sample\coastseg\scripts\tide_regions_map.geojson
```

3.**Validate:** Ensure each region directory under tide_model contains clipped tide model files.

- Each region will have the same format as region0 the full list of files is omitted for brevity.

```
├── CoastSeg
|
|___scripts
|    |_ aviso_fes_tides.py
|    |_ clip_and_write_new_nc_files_cmd_line.py
|    |_ tide_regions_map.geojson
|
|___tide_model
|    |_ fes2014
|    |     |_load_tide
|    |     |       |__2n2.nc.gz
|    |     |       |__eps2.nc.gz
|    |     |       |__ ....
|    |     |_ocean_tide
|    |     |       |__2n2.nc.gz
|    |     |       |__eps2.nc.gz
|    |     |       |__ ....
|    |
|    |_ region0
|    |     |_fes2014
|    |     |       |__load_tide
|    |     |       |       |__2n2.nc
|    |     |       |       |__eps2.nc
|    |     |       |       |__....
|    |     |       |__ocean_tide
|    |     |       |       |__2n2.nc
|    |     |       |       |__eps2.nc
|    |     |       |       |__....
|    |_ region1
|    |_ region2
|    |_ region3
|    |_ region4
|    |_ region5
|    |_ region6
|    |_ region7
|    |_ region8
|    |_ region9
|    |_ region10

```

4.**Downloaded Tide Model Contents**

![tide model contents](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/2a8f2425-993f-4184-90cf-01b6ff65af4a)

## Running into Problems?

---

1. Your account may not be verified yet, which means you can't download the model
2. Occasionally the AVISO server goes down. Wait a few hours and try again and see if its working.

### Troubleshooting

If you are running coastseg on a secure network you may need to make the following modifications to your `.condarc` file.

1. If you get an error message similar to `CondaValueError: You have chosen a non-default solver backend(libmamba) but it was not recongized`
   - Solution: comment out the line in the .condarc file `solve:libmamba`
2. You may also need to modify the `ssl_verify` to either True or False depending on your network security.
   - Always switch `ssl_verify` to True if you ever set it it to False for debugging purposes.

# How to Perform Tide Correction

⚠️ You must have run the script to download the FES2014 and/or FES2022 tide model before attempting to correct tides
This script provides utilities for modeling tides and correcting time series data based on tide predictions.

## ⚠️ Important: Read Before Using Tide Correction Button

If the tide model was NOT downloaded to `CoastSeg/tide_model` then the tide correction button will NOT work. The tide correction button will try to load the tide model from `<CoastSeg location>/tide_model` and an error will occur. Instead follow the instructions in at [Option 2: Tide Correction Script](#option-2-tide-correction-script)

If you downloaded the tide model to a different location follow the instructions at [Option 2: Tide Correction Script](#option-2-tide-correction-script)
to use the script `apply_tidal_correction.py`

## Option 1: Tide Correction Button Method

Ever since coastseg 1.1.0 released you can now use the `correct tides` button in coastseg to automatically correct the timeseries. This button will automatically find which files need to be tidally corrected and apply the tidal correction for you.

## ⚠️ Important

If the tide model was NOT downloaded to `CoastSeg/tide_model` then the tide correction button will NOT work. The tide correction button will try to load the tide model from `<CoastSeg location>/tide_model` and an error will occur. Instead follow the instructions in at [Option 2: Tide Correction Script](#option-2-tide-correction-script)

### Steps

**WARNING**
You must have downloaded the tide model to the `tide_model` folder within coastseg for this button to work correctly.

1.**Enter the beach slope**

- This is the beach slope in meters for the ROI you have selected.

  2.**Enter the beach elevation**

- This is the beach elevation in meters for the ROI you have selected.

  3.**Select the ROIs' ids to tidally correct**

  4.**Click compute tides and wait a few minutes.**

If you encounter any issues or have any questions please submit an issue at [CoastSeg Issues](https://github.com/Doodleverse/CoastSeg/issues/new/choose)

![predict-tide_script](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/4ce506c2-403e-44e7-9ecd-6a4a4898b870)

## Option 2: Tide Correction Script

⚠️ If the tide model was NOT downloaded to `CoastSeg/tide_model`, then you must use the tide correction script.

### Example 1: Apply Tide Correction to a session and provide tide model location

1. **Choose a session**
   - In this example I will be using session `dnz3_extract_shorelines_10_yr` and using ROI `ID_dnz3_datetime12-22-23__09_10_44`
   - Example: `C:\development\doodleverse\coastseg\CoastSeg\sessions\dnz3_extract_shorelines_10_yr\ID_dnz3_datetime12-22-23__09_10_44`
2. **Get the location of the config_gdf.geojson file**
   - This parameter is used after the `-C `
   - Example: `-C "C:\development\doodleverse\coastseg\CoastSeg\sessions\dnz3_extract_shorelines_10_yr\ID_dnz3_datetime12-22-23__09_10_44\config_gdf.geojson"`
3. **Get the location of the transect_time_series.csv file**
   - This parameter is used after the `-T `
   - Example: `-T "C:\development\doodleverse\coastseg\CoastSeg\sessions\dnz3_extract_shorelines_10_yr\ID_dnz3_datetime12-22-23__09_10_44\transect_time_series.csv"`
4. **Get the location of the tide_model**
   - This parameter is used after the `-M`
   - Example: `-M "C:\development\doodleverse\coastseg\CoastSeg\tide_model"`
5. Select a Beach Elevation
   - This is the elevation of the beach in meters for the ROI you have selected
   - Example: `-E 3.2`
6. Select a Beach Slope
   - This is the slope of the beach in meters for the ROI you have selected
   - Example: `-S 2.1`
7. **Assemble the command**
   - Assemble all commands into a single line. I recommended using notepad to combine all the commands into a single line
   ```
   python apply_tidal_correction.py -E 3.2 -S 2.1 -C
   "C:\development\doodleverse\coastseg\CoastSeg\sessions\dnz3_extract_shorelines_10_yr\ID_dnz3_datetime12-22-23__09_10_44\config_gdf.geojson" -T
   "C:\development\doodleverse\coastseg\CoastSeg\sessions\dnz3_extract_shorelines_10_yr\ID_dnz3_datetime12-22-23__09_10_44\transect_time_series.csv"
   -M "C:\development\doodleverse\coastseg\CoastSeg\tide_model"
   ```
8. **Run the command**
   - It takes a few minutes to predict the tides

![image](https://github.com/Doodleverse/CoastSeg/assets/61564689/13582ba7-63c6-46d4-93c1-aa68db23626c)

### Parameters:

1. **-C (or -c) [CONFIG_FILE_PATH]**

   - Description: Path to the configuration file.
   - Example: `-C "path_to_config_file"`
   - `CONFIG_FILE_PATH`: Path to the configuration file. This is the `config_gdf.geojson` in the session directory generated when shorelines were extracted

2. **-T (or -t) [RAW_TIMESERIES_FILE_PATH]**

   - Description: Path to the raw timeseries file.
   - Example: `-T "path_to_timeseries_file"`
   - `RAW_TIMESERIES_FILE_PATH`: Path to a csv file containing the time series created by extracting shorelines with coastseg. This timeseries
     represents the intersection of each transect with the shoreline extracted for a particular date and time. It is not tidally corrected.

3. **-E (or -e) [REFERENCE_ELEVATION]**

   - Description: Set the reference elevation value. This is a float number.
   - Example: `-E 3`

4. **-S (or -s) [BEACH_SLOPE]**
   - Description: Set the beach slope value. This is a float number.
   - Example: `-S 2`

### Optional Configuration Options

If you didn't install the tide model in the default location you will need to modify the following variables

5. **-P (or -p) [TIDE_PREDICTIONS_FILE_NAME]**

   - Description: File name for saving a csv file containing the tide predictions for each date time in the timeseries provided.
   - By Default this file is named "tidal_predictions.csv"
   - Example: `-P "tidal_predictions.csv"`

6. **-O (or -o) [TIDALLY_CORRECTED_FILE_NAME]**

   - Description: File name for saving the tidally corrected time series csv file.
   - By Default this file is named "tidally_corrected_time_series.csv"
   - Example: `-O "tidally_corrected_time_series.csv"`

7. **-R (or -r) [MODEL_REGIONS_GEOJSON_PATH]**

   - Description: Path to the model regions GeoJSON file.
   - By default the program looks for `tide_regions_map.geojson` in the `scripts` directory
   - Example: `-R "c:\coastseg\scripts\tide_regions_map.geojson"`
   - `MODEL_REGIONS_GEOJSON_PATH`: Path to the location of the geojson file containing the regions used to create the clipped tide model in the previous steps. This file is typically located in the scripts directory within coastseg. "c:\coastseg\scripts\tide_regions_map.geojson"`

8. **-M (or -m) [FES_2014_MODEL_PATH]**
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

Thank you [DEA-Coastlines](https://github.com/GeoscienceAustrali/dea-coastlines/wiki/Setting-up-tidal-models-for-DEA-Coastlines) for making a guide on how to use pyTMD and [pyTMD](https://pytmd.readthedocs.io/en/latest/api_reference/aviso_fes_tides.html) for making a easy to use script to download the AVISO FES 2014 and FES 2022 Model.
The `model_tides` in this code has been modified and the original function was originally written by Robbi Bishop-Taylor for the `dea-tools` package https://github.com/GeoscienceAustralia/dea-notebooks/blob/develop/Tools/dea_tools/coastal.py#L466-L473 . For more informaion on the FES 2014 model please visit https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/global-tide-fes/description-fes2014.html and https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/global-tide-fes/release-fes22.html for the FES 2022 model. 
