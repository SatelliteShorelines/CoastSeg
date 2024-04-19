## Available Scripts:

1.transects_swap_points.py

- A script that reads a `config_gdf.geojson` and swaps the origin & end point for each transect.
- The new transects (default name : `reversed_transects.geojson` ) are saved to the scripts directory

  2.shorten_transects.py

- A script that reads a geojson file containing transects and can shorten or length each transect depending on the parameters used
- The new transects (default name : `shortened_transects.geojson` ) are saved to the scripts directory

## Prerequisites for running the script

**1. Open Anaconda Prompt**

- Navigate to the Start menu or application directory and open the Anaconda Prompt.

**2. Navigate to the Scripts Directory**

- Replace path_to_your_directory with the location where the coastseg directory is located on your machine.

```bash
cd path_to_your_directory\coastseg
cd scripts
```

## Run the Script transects_swap_points.py

- This will swap each transect's origin and end point

```
python script_name.py [options]
```

**1. Example: Use config_gdf.geojson**
To run the script named transects_swap_points.py, you can execute:

```
python transects_swap_points.py -i 'C:\path_to_directory\CoastSeg\sessions\ID_rmh16_datetime08-22-23__12_49_23\config_gdf.geojson'
```

**2. Example: Use transects.geojson**

```
python transects_swap_points.py -i 'C:\path_to_directory\CoastSeg\transects.geojson'
```

## Run the Script shorten_transects.py

**Example 1: Shorten the transects by 500m**

- `-s` shorten the length of the transect by 500 meters by moving the origin seaward

```
python shorten_transects.py  -i "C:\development\doodleverse\coastseg\CoastSeg\reversed_transects.geojson" -s 500

```

**Example 2: Shorten the transects by 500m from the origin and lengthen by 100m from the seaward point**

- `-s` shorten the length of the transect by 500 meters by moving the origin towards the end point

- `-l` lengthen the length of the transect by 100 meters by moving the end point more seaward

```
python shorten_transects.py -i "C:\development\doodleverse\coastseg\CoastSeg\reversed_transects.geojson" -s  500 -l 100

```

**Example 3: Shorten the transects by 500m from the origin and lengthen by 100m from the seaward point and save to shortened_transects2.geojson**

- `-s` shorten the length of the transect by 500 meters by moving the origin towards the end point

- `-l` lengthen the length of the transect by 100 meters by moving the end point more seaward

- `-o` save the new transects to file "shortened_transects2.geojson"

```
python shorten_transects.py -i "C:\development\doodleverse\coastseg\CoastSeg\reversed_transects.geojson" -s  500 -l 100 -o "shortened_transects2.geojson"

```

## Getting Help

---

For most scripts, you can get a description of the available options and how to use them by using the -h or --help flag:

```
python script_name.py -h
```

Or:

```
python script_name.py --help
```

You can also open the script in any text editor or IDE like Notepad++, VSCode, or similar to view the documentation.
