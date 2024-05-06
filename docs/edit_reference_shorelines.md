## How to Load Shorelines on the Map

To load shorelines on the map first draw a bounding box along the coastline. Then click the button `Load Shoreline` to load shorelines within the bounding box.
**WARNING**: Not all coastlines have shoreline vectors available and if no shorelines exist within the bounding box a pop up will indicate that no shorelines were found. If this happens try drawing a bounding box somewhere else.

![how_to_load_shorelines](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/f74674ea-6b76-42de-b9aa-79ef8cf9c860)

## How to Edit the Reference Shorelines on the Map

You need to have only one set of reference shorelines, otherwise when coastseg attempts to map your shorelines together it looks like this:
<img src="https://github.com/Doodleverse/CoastSeg/assets/61564689/e701e594-40b7-457f-a97e-4cd0b85a2a7e" 
     alt="bad shorelines" width="500" height="550">

1. Click all the shorelines you want to delete
2. Under **Remove Feature from Map** select 'selected shorelines' from the drop-down
3. Click the 'Remove Selected' button
   ![how_to_delete_parts_shoreline](https://github.com/Doodleverse/CoastSeg/assets/61564689/6097ff6c-69e2-4f9f-8637-bec5d21bc988)

## How to Load Features from Geojson Files on Map

To load shorelines, transects or bounding boxes onto the map from a geojson file use the `Load <feature name> file` button. It will open a file dialog window where you can select the geojson file you want to load on the map. If the geojson file is too large or has too many large features the map may become very slow. It is highly recommended to only load small features onto the map.

![how_to_edit_reference_shoreline_map](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/23871592-5ea6-4572-bf03-35a4bcae2cfa)
