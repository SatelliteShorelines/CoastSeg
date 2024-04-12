# How to Run a Script Using Anaconda Prompt

Running Python scripts using the Anaconda Prompt can be an efficient way to execute your scripts, especially when using libraries that are available in your Anaconda environment. Here's a step-by-step guide to help you run Python scripts using the Anaconda Prompt:

## Prerequisites

- Make sure you have Anaconda or Miniconda installed on your machine.

## List of Available Scripts:

1. transects_swap_points.py
   - A script that reads a `config_gdf.geojson` and swaps the origin & end point for each transect.
   - The new transects (default name : `reversed_transects.geojson` ) are saved to the scripts directory
2. shorten_transects.py
   - A script that reads a geojson file containing transects and can shorten or length each transect depending on the parameters used
   - The new transects (default name : `shortened_transects.geojson` ) are saved to the scripts directory
3. get_transects_points.py
   - A script that reads a `config_gdf.geojson` and creates two geojson files `origin_points.geojson` containing the origin point & `end_points.geojson` for each transect.
   - Both of these geojson files are saved to the scripts directory
4. georeference_jpgs.py
- A script that takes a list of ROI directories and creates georeferenced jpegs for the "RGB", "NIR", "SWIR", "MNDWI", "NDWI" subdirectories
- [See a guide on how to use it](#guide-for-script-georeference_jpgspy)
5. preview_images_for_dates.py
- This script prints the available satellite imagery for ROIs given a list of date ranges. It also saves the results to a file named `results.json`
- [See a guide on how to use it](#guide-for-script-preview_images_for_datespy)

## Steps

### 1. Open Anaconda Prompt

- Navigate to the Start menu or application directory and open the Anaconda Prompt.

### 2. Navigate to the Scripts Directory

```bash
cd path_to_your_directory\coastseg
cd scripts
```

Replace path_to_your_directory with the location where the coastseg directory is located on your machine.

### 3. Running the Script

```
python script_name.py [options]
```

Example:
To run the script named transects_swap_points.py, you can execute:

```
python transects_swap_points.py -i 'C:\path_to_directory\CoastSeg\sessions\ID_rmh16_datetime08-22-23__12_49_23\config_gdf.geojson'
```

### 4. Getting Help

For most scripts, you can get a description of the available options and how to use them by using the -h or --help flag:

```
python script_name.py -h
```

Or:

```
python script_name.py --help
```

### 5. Viewing Script Documentation

Most well-maintained scripts include a header comment or a docstring at the beginning of the file that provides a brief overview of the script's purpose, usage, and available options. You can open the script in any text editor or IDE like Notepad++, VSCode, or similar to view this documentation.


# Guide for Script preview_images_for_dates.py

## How to Use

1. Make sure you performed [steps 1](#1-open-anaconda-prompt) and [steps 2](#2-navigate-to-the-scripts-directory)
2. Run the script from the command line in an activated coastseg environment by providing the locations of the ROI geojson file and the list of dates as arguments:

   ```python
   python preview_images_for_dates.py "path/to/your/regions.geojson" "2002-01-01,2023-01-15" "1984-02-01,1995-02-15" "2023-03-01,2023-03-15"
   ```
### Example
```
python preview_images_for_dates.py "C:\development\doodleverse\coastseg\CoastSeg\rois.geojson" "2002-01-01,2023-01-15" "1984-02-01,1995-02-15" "2023-03-01,2023-03-15"
```
# Guide for Script merge_sessions.py (under construction 🚧 )
Use the `merge_sessions.py` script to combine multiple session folder into a single merged session folder. The script reads the information from the session folders you provide, then it merges the geojson files, json files, and csv files respectively together into a single file that is saved to the new merged session folder. 


 This script combines the following data together:
1. Extracted shorelines in geojson files
2. time series csv files
3. config_gdf.geojson files
4. shoreline_dict.json
5. cross_distance.json

⚠️ **Warning** ⚠️

Merged Sessions should NEVER be loaded into CoastSeg.

## Uses Cases
<table>
<tr>
<td>

### Use Case 1: Merge multiple ROIs at different locations together
- Example: You have multiple ROIs that cover an entire coastline that you want to merge into a single session. When the extracted shorelines are merged together the resulting session contains long continuous shorelines stored in a single file.

<img src="https://github.com/Doodleverse/CoastSeg/assets/61564689/c816db24-86e2-4a30-994e-23e0cab670b8.png" width="385" height="400"/>

</td>
<td>

### Use Case 2: Merge multiple sessions for a single ROI together
- Merging multiple date ranges for a single ROI into a single session.
   - For example: if you have 2 sessions with extracted shorelines and session 1 contains shorelines for dates  2022-2023 and session 2 contains shorelines for dates 2021 - 2022. You can merge them to have a single session with shorelines for dates 2021 - 2023.

<img src="https://github.com/Doodleverse/CoastSeg/assets/61564689/ed04ddf8-ee3a-4d24-a0e5-fe010c51cf54.png" width="385" height="400"/>

</td>
</tr>
</table>

### Warning
If more than 1 session contains <u>different shoreline geometries in the SAME location</u> that were captured by the <u>same satellite at the exact same datetime</u>, then these shorelines will be combined into a single shoreline resulting in double or triple shorelines.
   - For example, if the red and green shoreline were captured by the same satellite at the exact same datetime, then they would be merged into a single shoreline with this shape. <u>This means that a single transect would intersect the same shoreline at multiple points. </u>
     - If you are merging any ROIs that are in the same location and have some of the EXACT dates & satellites in common make sure to remove any duplicates using the edit extracted shorelines feature.


<p align="center">
  <img src="https://github.com/Doodleverse/CoastSeg/assets/61564689/db040c09-9586-41ce-8920-e4e7ef55d514.png" width="350" height="400"/>
  <img src="https://github.com/Doodleverse/CoastSeg/assets/61564689/b2e213c4-5907-4e2b-8f5e-becdaaa29aeb" width="480" height="440"/>
</p>


## How to Use the merge_sessions.py Script 

### Required Parameters

Run the code below to view the full list of parameters available for the `merge_sessions.py` script.

```bash
python merge_sessions.py --help
```

| Parameter | Usage                       | Description                                                                                                                                                                                                                          |
|-----------|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-i`      | `-i "<location1>" "<location2>"` | Provide each session location enclosed in double quotes and separated by a space . <br> Example:  `-i "C:\CoastSeg\sessions\sample_session1\ID_ewr3_datetime12-20-23__03_25_23" "C:\CoastSeg\sessions\session_2\ID_ewr1_datetime12-20-23__03_25_23"`                                                                                                                                                         |
| `--crs`   | `--crs "EPSG:32610"`        | Read the CRS that should be used to convert the data to from `output_epsg` located within the `config.json` file from any of your sessions. <br> > output_epsg: epsg code defining the spatial reference system of the shoreline coordinates. It has to be a cartesian coordinate system (i.e., projected) and not a geographical coordinate system (in latitude and longitude angles). See [spatialreference.org](http://spatialreference.org/) to find the EPSG number corresponding to your local coordinate system. If you do not use a local projection, your results may not be accurate. |
| `-n`      | `-n "example_name"` | The merged session directory name will be `"example_name"`.                                                                                                                                                                  |


![image](https://github.com/Doodleverse/CoastSeg/assets/61564689/c12f7445-7457-4892-b084-709d56fe9217)



## Example 1 : Merging 2 ROIs from different locations into a single session
----
 

#### Step 1 : Get the session locations


| Session Number | Session Name        | Location Path                                                                   |
|----------------|---------------------|---------------------------------------------------------------------------------|
| 1              | "sample_session1"   | `"C:\CoastSeg\sessions\sample_session1\ID_ewr3_datetime12-20-23__03_25_23"`     |
| 2              | "session_2"         | `"C:\CoastSeg\sessions\session_2\ID_ewr1_datetime12-20-23__03_25_23"`           |

⭐ Notice that the ROI directory within the session being used is `ID_gac6_datetime10-30-23__01_44_50` not just the session name `test_case4_overlapping`.


#### Step 2 : Get the CRS

 1. Open any session folder
 2. Open the `config.json`
    - you can use notepad, VSCode, etc.
 3. Get the CRS code from the setting  `output_epsg`
 4. Enter the CRS code parameter `--crs "epsg:<YOUR CRS CODE>"`
    - Example `--crs "EPSG:32610"`

#### Step 3 : Name the merged session
  - Choose a name for the new merged session directory
  - `-n "merged_session_example"`

#### Step 4: Run the script
   - separate each session location with a space (use double quotes on Windows):
```bash
      python merge_sessions.py -i "C:\CoastSeg\sessions\sample_session1\ID_ewr3_datetime12-20-23__03_25_23" 
      "C:\CoastSeg\sessions\session_2\ID_ewr1_datetime12-20-23__03_25_23"  --crs "EPSG:32610" -n "merged_session_example" 
```

<br>
<br>
<br>

## Example 2: Customizing Parameters
   In this example we merged the same sessions but we use the other parameters.
   -   `-s "C:\CoastSeg\merged_session\example1"` : This tell the script to save the merged session at `C:\CoastSeg\merged_session\example1`
   -   `-pm 0.3` : This tell the script to set the advanced setting `prc_multiple` to 0.3
   -   `-mp 2` : This tell the script to set the advanced setting `min points` to 2

```bash
python merge_sessions.py -i "C:\CoastSeg\sessions\sample_session1\ID_ewr3_datetime12-20-23__03_25_23" "C:\CoastSeg\sessions\session_2\ID_ewr1_datetime12-20-23__03_25_23"  --crs "EPSG:32610" -s "C:\CoastSeg\merged_session\example1" -n "merged_session1" -pm 0.3 -mp 2
```



# Guide for Script georeference_jpgs.py

## Overview

This script georeferences JPEG images using the georeferencing information available in corresponding TIFF images. Georeferencing provides spatial location information to images, allowing them to be placed at a specific location on the Earth's surface.

## How to Use

1. Make sure you performed [steps 1](#1-open-anaconda-prompt) and [steps 2](#2-navigate-to-the-scripts-directory)
2. Run the script from the command line in an activated coastseg environment by providing the locations of the ROI directories as arguments:

   ```python
   python georeference_jpegs.py "C:\development\doodleverse\CoastSeg\data\ID_quj9_datetime09-28-23__05_12_40" "C:\development\doodleverse\CoastSeg\data\ID_tyg8_datetime09-29-23__05_18_40"
   ```

## Output

For each specified ROI directory, the script will:

1. Detect JPEG images in the predefined subdirectories "RGB", "NIR", "SWIR", "MNDWI", "NDWI".
2. Find the corresponding TIFFs with georeferencing info.
3. Apply the georeferencing data from the TIFFs to the JPEGs.
4. Save the georeferenced JPEGs in a subdirectory named georeferenced within each subdirectory.

   - For example, if your RGB directory contains JPEGs and its path is:
     `path_to_roi_dir1/jpg_files/preprocessed/RGB/`
   - The georeferenced JPEGs will be saved in: `path_to_roi_dir1/jpg_files/preprocessed/RGB/georeferenced/`

### What are Georeferenced JPEGs?

Georeferenced JPEGs are standard JPEG images with associated spatial data. This spatial data allows the JPEG to be mapped to a specific location on Earth. The georeferencing information usually includes details about the image's projection, coordinates, and resolution. This is crucial for various applications in geographic information systems (GIS), remote sensing, and cartography, enabling the combination of the image with other spatial datasets in a meaningful way.


