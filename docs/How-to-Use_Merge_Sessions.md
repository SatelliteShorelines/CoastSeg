## Prerequisites for running the script¶

1. Open Anaconda Prompt

- Navigate to the Start menu or application directory and open the Anaconda Prompt.
2. Navigate to the Scripts Directory

- Replace path_to_your_directory with the location where the coastseg directory is located on your machine.

```bash
cd path_to_your_directory\coastseg
cd scripts 
```

## Run the Script merge_sessions.py

- This will merge two CoastSeg sessions into one combined session
- This new session will be saved to the scripts folder in CoastSeg

## Needed Variables

`-i` locations of the ROI sessions to be merged
```
ex: -i "C:\CoastSeg\sessions\session_2022\ID_ewr1_datetime12-20-23__03_25_23" "C:\CoastSeg\sessions\session_2022\ID_ewr3_datetime12-20-23__03_25_23"
```

`-c` coordinate reference system (CRS) for the merged session

```
ex: -c "EPSG:32610"
```

`-n` name for the merged session folder that will be created at save_location (default location in scripts folder)

```
ex: -n "merged_session_2022"
```

## Example

```
python merge_sessions.py -i "E:\anaconda3\CoastSeg-main\sessions\Merge1_test\ID_dgh4_datetime07-25-25__09_23_18" "E:\anaconda3\CoastSeg-main\sessions\Merge2_test\ID_dgh5_datetime07-25-25__09_26_49" -c "EPSG:32610" -n "merged_session_test"

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

