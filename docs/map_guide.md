## How to Deselect Map Layers

---

**Not able to hover over a feature on the map?** The problem is its under another layer on the map. Most likely the feature you want to hover over is under the ROI and selected ROI layers. Use the layers menu checkbox to deselect the ROI and selected ROI layers from the map and hover over your feature again.

![change_layer_demo](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/9c8d65d5-85c8-457b-993c-6dc7122f332f)

## How to Load Transects on the Map

---

To load transects on the map first draw a bounding box along the coastline. Then click the button `Load Transects` to load transects within the bounding box.
**WARNING**: Not all coastlines have transects available and if no transects exist within the bounding box a pop up will indicate that no transects were found. If this happens try drawing a bounding box somewhere else.

![load_shoreline_transect_demo](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/7b47bc38-6e10-4f3c-93d9-117990cb1fec)

## How to Load Shorelines on the Map

---

To load shorelines on the map first draw a bounding box along the coastline. Then click the button `Load Shoreline` to load shorelines within the bounding box.
**WARNING**: Not all coastlines have shoreline vectors available and if no shorelines exist within the bounding box a pop up will indicate that no shorelines were found. If this happens try drawing a bounding box somewhere else.

![load_shoreline_transect_demo](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/7b47bc38-6e10-4f3c-93d9-117990cb1fec)

## How to Load Features from Geojson Files

---

**1.Click the dropdown from under 'Load Feature from File'**

- Select the feature you want on the map.

**2.Click the load button from under 'Load Feature from File'**

- This opens a file dialog window where you can select the geojson file you want to load on the map.

![load feature from file](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/3e314de6-a742-46e9-b390-36d66e7aaba9)

**3.Select a geoJSON file containing the feature**

- If the geojson file is too large or has too many large features the map may become very slow. It is highly recommended to only load small features onto the map.

![load_geojson_files_demo](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/4fedfb32-b36b-4203-a9af-228f927ca0a8)
