# Table of Contents

## Reference Shoreline

In order to extract shorelines with CoastSeg you will need a reference shoreline. If you have interior and exterior shorelines you will need to select only one reference shoreline and adjust the shoreline buffer so that the other shoreline doesn't get picked up.

In this example we will be creating a reference Shoreline for Fire Island in New York.

## Option 1 : QGIS

1.  Load base imagery from google satellite
2.  Create a new shapefile
3.  Digitize the shoreline from the imagery
4.  Save out to geojson format
    <img src="https://github.com/Doodleverse/CoastSeg/assets/61564689/51450c7b-6003-46bb-a3c8-3590dc09891e" alt="QGIS screenshot" width="850" height="480">

## Option 2: geojson.io

1. Use the Line tool in https://geojson.io/ to create a reference shoreline
2. Download the geojson file.

<img src="https://github.com/Doodleverse/CoastSeg/assets/61564689/155918d4-3ec4-4f62-9f5d-2014c67edb6a" alt="geojson.io screenshot" width="850" height="480">

### Load the Reference Shoreline

1. Move the geojson file into the coastseg folder
2. Click the "load shoreline file" button
3. Scroll down to the bottom and click the reference shoreline file
4. click select to load the reference shoreline on the map

![load_shoreline_file_demo](https://github.com/Doodleverse/CoastSeg/assets/61564689/3de0b48d-3b3c-4e45-a980-931a73a47298)

## Transects

### Option 1: Create Transects in QGIS

- The guide below demonstrates how to create transects that you can load into Coastseg. It was created by Catherine Janda & Sharon Fitzpatrick
  [Create your own transects in QGIS.docx](https://github.com/Doodleverse/CoastSeg/files/13925675/Create.your.own.transects.in.QGIS.docx)

### Option 2: Create Transects in geojson.io

1.Always put the origin (starting point) of the transect on land and the end point on the sea.

![geojsonio_make_transects](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/fae3919e-c181-4585-8d2a-9ca6dfeb3fc4)

2.Save the transects to a geoJSON file

![save geojson transects](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/9a9ccfae-96cd-49e9-b2fa-a53c48debfdb)
