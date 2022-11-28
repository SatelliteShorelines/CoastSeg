## How to Deselect Map Layers

---

**Not able to hover over a feature on the map?** The problem is its under another layer on the map. Most likely the feature you want to hover over is under the ROI and selected ROI layers. Use the layers menu checkbox to deselect the ROI and selected ROI layers from the map and hover over your feature again.

![Alt text](https://github.com/SatelliteShorelines/CoastSeg/blob/main/docs/gifs/change_layer_demo.gif)

## How to Load Transects on the Map
To load transects on the map first draw a bounding box along the coastline. Then click the button `Load Transects` to load transects within the bounding box.
**WARNING**: Not all coastlines have transects available and if no transects exist within the bounding box a pop up will indicate that no transects were found. If this happens try drawing a bounding box somewhere else.

![Alt text](https://github.com/SatelliteShorelines/CoastSeg/blob/main/docs/gifs/load_shoreline_transect_demo.gif)

## How to Load Shorelines on the Map
To load shorelines on the map first draw a bounding box along the coastline. Then click the button `Load Shoreline` to load shorelines within the bounding box.
**WARNING**: Not all coastlines have shoreline vectors available and if no shorelines exist within the bounding box a pop up will indicate that no shorelines were found. If this happens try drawing a bounding box somewhere else.

![Alt text](https://github.com/SatelliteShorelines/CoastSeg/blob/main/docs/gifs/load_shoreline_transect_demo.gif)

## How to Load Features from Geojson Files on Map
To load shorelines, transects or bounding boxes onto the map from a geojson file use the `Load <feature name> file` button. It will open a file dialog window where you can select the geojson file you want to load on the map. If the geojson file is too large or has too many large features the map may become very slow. It is highly recommended to only load small features onto the map.

![Alt text](https://github.com/SatelliteShorelines/CoastSeg/blob/main/docs/gifs/load_geojson_files_demo.gif)
