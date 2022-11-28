# What is an ROI?

An **ROI** is a **region of interest** it defines a rectangular area where satellite imagery will be downloaded from GEE (Google Earth Engine). GEE limits the area of land you are able to download to accommodate for this limitation ROIs of the sizes allowed by GEE are generated along the coastline within a bounding box drawn by the user. If the data for an ROI is downloaded it will have its own directory with the ROI ID in the directory name, for example `ID_3_datetime11-22-22__11_15_15` is the directory for ROI 3.

## ROI Directory Organization

Each ROI can have multiple sets of satellite imagery associated with it. For instance, a single ROI may have Sentinel 2, Landsat 8 and 9 imagery downloaded. Each type of satellite imagery will have its own directory within the ROI's directory. All the imagery and metadata associated with the Landsat 8 imagery for an ROI will be located in a subdirectory named `L8`. For the Sentinel 2 imagery this subdirectory would be named `S2`.

The jpgs for all satellite imagery downloaded for a given ROI are in subdirectory `jpg_files`. `jpg_files` contains the RGB, NIR, and SWIR subdirectories which contain the jpg images for all the selected satellites. This means all the imagery from L8, L9, and S2 would be located at each subdirectory with the satellite name in the image name. For example the RGB image from S2 would be in subdirectories `jpg_files\RGB` with a filename similar to: `2018-12-06-19-04-16_RGB_S2.jpg`.

### Sample ROI Directory Structure

```
├── CoastSeg
|
|___data
|    |_ ID_3_datetime11-22-22__11_15_15
│   |   |_ L8
│   |   |_ L9
│   |   |_ S2
│   |   |_ jpg_files
│   |   |  |_ preprocessed
|   │   |   |  |_ RGB
|   |   │   |  |  |_ 2018-12-06-19-04-16_RGB_S2.jpg
|   |   │   |  |  |_ 2018-12-06-19-04-16_RGB_L8.jpg
|   |   │   |  |  |_ 2018-12-06-19-04-16_RGB_L9.jpg
|   │   |   |  |_ NIR
|   |   │   |  |  |_ 2018-12-06-19-04-16_NIR_S2.jpg
|   |   │   |  |  |_ <rest of images...>
|   │   |   |  |_ SWIR
|   |   │   |  |  |_ 2018-12-06-19-04-16_SWIR_S2.jpg
|   |   │   |  |  |_ <rest of images...>
│   |   |   |_detection
|   │   |   |  |_<jpgs of detected shorelines>
│   |   |_config.json
│   |   |_config_gdf.json
│   |   |_extracted_shorelines.geojson
```

# How to Generate ROI on the Map

Use the sliders `Small ROI Length` and `Large ROI Length`
to control the side length of the ROI squares you want to generate. Click the button `Generate ROIs` to create a grid of overlapping ROIs on the map. You can then click each ROI on the map to indicated that satellite imagery for this ROI should be downloaded when the button `Download Imagery` is clicked.

![Alt text](https://github.com/SatelliteShorelines/CoastSeg/blob/main/docs/gifs/generate_rois_and_display_area.gif)

## How to Change the Size of the ROI

The size of the side lengths of the ROI can be controlled using the two sliders `Small ROI Length` and `Large ROI Length`. These sliders control the side lengths of the ROI squares. You can control the amount of overlap between ROIs by changing the side lengths of each ROI or have 0 overlap by setting `Small ROI Length` to 0.

## How to View the Area of an ROI

Hover over any ROI on the map and look under the `ROI Data` section on right side of the map to view its area and the ROI ID.

## How to View the ID of an ROI

Hover over any ROI on the map and look under the `ROI Data` section on right side of the map to view its ROI ID.
