# Compute Transect Cross Distance for Extracted Shorelines

## IMPORTANT

You cannot compute time-series of shoreline change along shore-normal transects unless at least one shoreline was successfully extracted from the download imagery.

---

# How Compute Transect Cross Distance for Extracted Shorelines

1. Click `Extract Shorelines` Button

- if you see extracted shorelines within at least one ROI on the map it means you can compute time-series of shoreline change along shore-normal transects

2.  Click `Compute Transects` Button

- This will save a `transects_cross_distances<ROIID>.json` which contains the intersection between the transect and the shoreline on each date for each transect.

3.  Click `Save Transects CSV` Button

- This will save a `transect_time_series.csv` which contains the intersection between the transect and the shoreline on each date for each transect.
- A message will print stating the location where the csv was saved
- The csv will appear the ROI directory for each ROI that had extracted shorelines and valid intersections between the transect and the extracted shorelines
  ![Alt text](https://github.com/SatelliteShorelines/CoastSeg/blob/main/docs/gifs/extract_shorelines_and_transects.gif)
