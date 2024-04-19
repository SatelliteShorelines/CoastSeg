# Table of Contents

## How to Create Reference Shoreline

In order to extract shorelines with CoastSeg you will need a reference shoreline. If you have interior and exterior shorelines you will need to select only one reference shoreline and adjust the shoreline buffer so that the other shoreline doesn't get picked up.

In this example we will be creating a reference Shoreline for Fire Island in New York.

## Option 1 : How to Create Reference Shorelines in QGIS

1.  Load base imagery from google satellite
2.  Create a new shapefile
3.  Digitize the shoreline from the imagery
4.  Save the shorelines in crs 'epsg:4326'

  <img src="https://github.com/Doodleverse/CoastSeg/assets/61564689/51450c7b-6003-46bb-a3c8-3590dc09891e" alt="QGIS screenshot" width="850" height="480">

## Option 2: How to Create Reference Shorelines in geojson.io

1. Use the Line tool in https://geojson.io/ to create a reference shoreline

2. Make sure to draw the shoreline with many shoreline points

- This matters because coastseg dilates each point the shoreline consists of to create the reference shoreline buffer

- If you use a very small reference shoreline buffer (<100m) and not enough points your reference shoreline will look like the image below.

![bad_ref_sl](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/8067e4b8-b288-4127-863d-3e14c21afdd1)

2. Download the geojson file.

<img src="https://github.com/Doodleverse/CoastSeg/assets/61564689/155918d4-3ec4-4f62-9f5d-2014c67edb6a" alt="geojson.io screenshot" width="850" height="480">

### How to Load the Reference Shoreline into CoastSeg

1. Move the geojson file into the coastseg folder
2. Click the "load shoreline file" button
3. Scroll down to the bottom and click the reference shoreline file
4. click select to load the reference shoreline on the map

![load_shoreline_file_demo](https://github.com/Doodleverse/CoastSeg/assets/61564689/3de0b48d-3b3c-4e45-a980-931a73a47298)

## How to Create Transects

### Option 1: Create Transects in QGIS

- The guide below demonstrates how to create transects that you can load into Coastseg. It was created by Catherine Janda & Sharon Fitzpatrick
  [Create your own transects in QGIS.docx](https://github.com/Doodleverse/CoastSeg/files/13925675/Create.your.own.transects.in.QGIS.docx)

### Option 2: Create Transects in geojson.io

**1.Always put the origin (starting point) of the transect on land and the end point on the sea.**

![geojsonio_make_transects](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/fae3919e-c181-4585-8d2a-9ca6dfeb3fc4)

**2.Modify the geoJSON file to give each transect a unique "id" (Optional)**

- If you don't give each transect an ID one will be automatically assigned by CoastSeg

![how to add an id geojson io](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/b5be22e7-722c-4037-aac6-209a4eb692d7)
</br>

**2.Save the transects to a geoJSON file**

![save geojson transects](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/9a9ccfae-96cd-49e9-b2fa-a53c48debfdb)
