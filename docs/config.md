# Config Files

Config files are used to save the current state of the map and all the data downloaded during a session using CoastSeg. The config file is actually composed of two separate files a `.geojson` file and `.json` file.
Within each ROI's directory you will find a `config.geojson` file containing all the ROIs, shorelines, transects, and the bounding box that were loaded on the map at the time the ROI was downloaded or when the `Save Config` button was pressed.

The `.geojson` file contains a geodataframe with crs `espg 4326` that contains all the shorelines, transects,selected ROI, and the bounding box(bbox) that were present on the map when the download button was clicked. The `.json` file contains the settings (eg `preprocess settings`,`postprocess settings`, and `shoreline extraction settings`) and user specified inputs (eg `dates`, `landsat_collection`,`satellite list`) that were used to download the imagery from Google Earth Engine(GEE). Each set of config files are saved in the each ROI's directory.

## Saving Config Files

---

Config files are automatically saved in the `data` directory within each ROI directory when `Download Imagery` is clicked.
Config files can be saved at any time when the `Save Config ` button is clicked.

**ROIs Downloaded**
If the ROIs selected on the map have been downloaded before and exist in the `data` directory, then the config files will be stored in within each ROI directory.

**ROIs Not Downloaded**
</br>If the ROIs on the map have not been downloaded before then the config files will be saved in the coastseg directory.

### Steps to Save Config Files

1. Click the ROIs on the map you want to save
2. Click the `Save Config Files` button.

# Loading Config Files

## Load Config Files For Downloaded ROIs

To load config files for ROIs that have been downloaded before go to the `data` directory within `coastseg` then find the directory of the ROIs you want to upload. Inside that directory, for example the
`ID_3_datetime10-20-22__07_09_07` directory, there should be `geojson` file named `config_gdf_id_3.geojson`. You should notice the `ID_<ROI ID NUMBER>` in the directory name and the geojson file match, in this case this is ROI id 3. Upload the `config_gdf_id_<roi id number>.geojson`.
This will load the ROI in that directory and all the other selected ROIs,shorelines, transects, and the bounding box that were on the map when it was saved.

![Alt text](https://github.com/SatelliteShorelines/CoastSeg/blob/main/docs/gifs/load_config_demo.gif)

### Steps to Load Config File

1. Click `Load Config`
2. Open `data` directory
3. Open ROI directory ex. **ID_3_datetime10-20-22\_\_07_09_07**
4. Select `config_gdf_id_<roi id number>.geojson` ex. **config_gdf_id_3.geojson**

## Load Config Files For ROIs that Have Not Been Downloaded

To load config files for ROIs that have not been downloaded before first click the `Load Config` button, open the coastseg directory and select `config_gdf.geojson`. This will loaded the selected ROIs, shorelines, transects, and bounding box that was saved with the `Save Config` button.

**Note:** You will not be able to extract shorelines or compute transects until you download the data for these ROIs.

#### Steps

1. Click `Load Config`
2. Open `coastseg` directory
3. Select `config_gdf.geojson`
