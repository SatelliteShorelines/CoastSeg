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

![how_to_edit_reference_shoreline_map](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/23871592-5ea6-4572-bf03-35a4bcae2cfa)

## How to Load Shorelines from Geojson Files on the Map

See this [guide](https://satelliteshorelines.github.io/CoastSeg/How-to-Create-Reference-Shorelines-%26-Transects%26ROIs/) on how to create and load your own reference shorelines onto the map :
