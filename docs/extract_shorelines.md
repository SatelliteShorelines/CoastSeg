# Extract Shorelines From Downloaded Imagery

---

Extracted Shorelines are shoreline vectors in the imagery you downloaded from GEE. The extracted shorelines are created by analyzing each `.tiff` file for each set of satellite imagery downloaded to check if a shoreline classifier can be applied to it. If the `.tiff` file meets the quality standards by being under the maximum threshold for cloud cover and other metrics the image is analyzed by a special image segmentation model which classifies the sand, water and surf in the image. Using the labeled images from the image segmentation model a shoreline vector will be detected in the image and saved to a dictionary entry for the ROI it belonged to. The extracted shoreline will appear as colored vectors along the coastline.

## Why does the Extract Shoreline Look Weird?

The classifier to extract shorelines fails for complex shorelines sometimes. If the extracted shorelines zig and zags across the coastline this is likely the result of the classifier failing to classify an image. You may want to try changing the `sand color` parameter in the settings dropdown depending on if the sand on the coast is dark or light.

## Why did isn't there an Extracted Shoreline

1. Sometimes shorelines will be not be able to be extracted from any of the imagery downloaded for the ROI on the map due to image quality issues such as excessive cloud cover or too many 'no data'(black) pixels in the downloaded imagery. When this is the case the extracted shoreline will not appear on the map.
   The message `The ROI id does not have a shoreline to extract. ` will print when this happens

## How to Extract Shorelines From Your Imagery

See the extracted shorelines for the ROIs selected on the map by first selecting your ROIs, downloading their data from GEE and then clicking extract shorelines.

1. Click the ROIs to extract shorelines from on the map
2. Download the ROIs data from GEE or already have it downloaded.
3. Click the button `extract shorelines`
4. You should see extracted shorelines appear within some of your ROIs.
5. If this works you can compute the time-series of shoreline change along shore-normal transects by clicking the `Compute Transects` Button.

![Alt text](https://github.com/SatelliteShorelines/CoastSeg/blob/main/docs/gifs/extract_shorelines_and_transects.gif)
