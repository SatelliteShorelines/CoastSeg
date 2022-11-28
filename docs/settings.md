# Settings

Use the settings drop down to modify the settings used by CoastSeg to download ROIs, extract shorelines and compute transect and shoreline intersections.

![Settings Demo](https://github.com/SatelliteShorelines/CoastSeg/blob/main/docs/gifs/settings_demo.gif)

## Settings Explained

---

`sat_list`: satellite missions to consider (e.g., sat_list = ['L5', 'L7', 'L8', 'L9', 'S2'] for Landsat 5, 7, 8, 9 and Sentinel-2 collections)

`dates`: dates over which the images will be retrieved (e.g., dates = ['2017-12-01', '2018-01-01'])

`cloud_thresh`: threshold on maximum cloud cover that is acceptable on the images (value between 0 and 1 - this may require some initial experimentation).

`dist_clouds`: buffer around cloud pixels where shoreline is not mapped (in metres)

`output_epsg`: epsg code defining the spatial reference system of the shoreline coordinates. It has to be a cartesian coordinate system (i.e. projected). See http://spatialreference.org/ to find the EPSG number corresponding to your local coordinate system.

- If you use a CRS such as `4326` or `4327` geographical coordinate system which uses latitude and longitude shorelines cannot be extracted from imagery so CoastSeg will automatically convert the CRS to the most accurate CRS based on where the bounding box is drawn on the map.
- All outputted geojson files will be in CRS `4326` as this is the CRS used to load features onto CoastSeg's map.

`save_figure`: if set to True a figure of each mapped shoreline is saved under /filepath/sitename/jpg_files/detection, even if the two previous parameters are set to False. Note that this may slow down the process.

`min_beach_area`: minimum allowable object area (in metres^2) for the class 'sand'. During the image classification, some features (for example, building roofs) may be incorrectly labelled as sand. To correct this, all the objects classified as sand containing less than a certain number of connected pixels are removed from the sand class. The default value is 4500 m^2, which corresponds to 20 connected pixels of 15 m^2. If you are looking at a very small beach (<20 connected pixels on the images), try decreasing the value of this parameter.

`min_length_sl`: minimum length (in metres) of shoreline perimeter to be valid. This can be used to discard small features that are detected but do not correspond to the actual shoreline. The default value is 500 m. If the shoreline that you are trying to map is shorter than 500 m, decrease the value of this parameter.

`cloud_mask_issue`: the cloud mask algorithm applied to Landsat images by USGS, namely CFMASK, does have difficulties sometimes with very bright features such as beaches or white-water in the ocean. This may result in pixels corresponding to a beach being identified as clouds and appear as masked pixels on your images. If this issue seems to be present in a large proportion of images from your local beach, you can switch this parameter to True and CoastSat will remove from the cloud mask the pixels that form very thin linear features, as often these are beaches and not clouds. Only activate this parameter if you observe this very specific cloud mask issue, otherwise leave to the default value of False.

`sand_color`: this parameter can take 3 values: default, latest, dark or bright. Only change this parameter if you are seing that with the default the sand pixels are not being classified as sand (in orange). If your beach has dark sand (grey/black sand beaches), you can set this parameter to dark and the classifier will be able to pick up the dark sand. On the other hand, if your beach has white sand and the default classifier is not picking it up, switch this parameter to bright. The latest classifier contains all the training data and can pick up sand in most environments (but not as accurately). At this stage the different classifiers are only available for Landsat images (soon for Sentinel-2 as well).

`pan_off`: by default Landsat 7, 8 and 9 images are pan-sharpened using the panchromatic band and a PCA algorithm. If for any reason you prefer not to pan-sharpen the Landsat images, switch it off by setting pan_off to True

`along_dist`: defines the along-shore distance around the transect over which shoreline points are selected to compute the intersection. The default value is 25 m, which means that the intersection is computed as the median of the points located within 25 m of the transect (50 m alongshore-median). This helps to smooth out localized water levels in the swash zone.

## Settings Controlled Automatically by CoastSeg
---

`sitename`: name of the site (this is the name of the subfolder where the images and other accompanying files will be stored)

`filepath`: filepath to the directory where the data will be stored

`polygon`: the coordinates of the region of interest (longitude/latitude pairs in WGS84)

## Credits
Thank you to the amazing developer @kvos of CoastSat for defining these parameters.
