# How to Run a Script Using Anaconda Prompt

Running Python scripts using the Anaconda Prompt can be an efficient way to execute your scripts, especially when using libraries that are available in your Anaconda environment. Here's a step-by-step guide to help you run Python scripts using the Anaconda Prompt:

## Prerequisites

- Make sure you have Anaconda or Miniconda installed on your machine.

## List of Available Scripts:

1.transects_swap_points.py

   - A script that reads a `config_gdf.geojson` and swaps the origin & end point for each transect.
   - The new transects (default name : `reversed_transects.geojson` ) are saved to the scripts directory

2.shorten_transects.py

   - A script that reads a geojson file containing transects and can shorten or length each transect depending on the parameters used
   - The new transects (default name : `shortened_transects.geojson` ) are saved to the scripts directory

3.get_transects_points.py

   - A script that reads a `config_gdf.geojson` and creates two geojson files `origin_points.geojson` containing the origin point & `end_points.geojson` for each transect.
   - Both of these geojson files are saved to the scripts directory

4.georeference_jpgs.py

- A script that takes a list of ROI directories and creates georeferenced jpegs for the "RGB", "NIR", "SWIR", "MNDWI", "NDWI" subdirectories
- [See a guide on how to use it](#guide-for-script-georeference_jpgspy)

5.preview_images_for_dates.py

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


