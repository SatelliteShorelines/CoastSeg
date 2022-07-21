import os.path
import json
import glob
# New imports
import pandas as pd
import geopandas as gpd
from CoastSeg import bbox


transect_folder=os.path.abspath(os.getcwd())+os.sep+"Coastseg"+os.sep+"transects"
transects=glob.glob(transect_folder+os.sep+"*.geojson")
transects


# Loads the geojson from a transects file
with open(transects[1]) as f:
    data = json.load(f)
data

transect_layer_names=[]
for transect in transects:
    transects_layer_name=os.path.basename(transect)
    transect_layer_names.append(transects_layer_name)

transects_total_bounds=[]
for transect_file in transects:
    print(transect_file)
    transects_layer_name=os.path.splitext(os.path.basename(transect_file))[0]
    print(transects_layer_name)
    transects_gpd=bbox.read_gpd_file(transect_file)
    print(transects_gpd.total_bounds)
    print(type(transects_gpd.total_bounds))
    bounds=transects_gpd.total_bounds
    print(bounds,"\n")
#     print(bounds.tolist())
    transects_total_bounds.append(bounds.tolist())
#     transects_total_bounds.append(transects_gpd.total_bounds)

dictionary = [list(a) for a in zip(transect_layer_names, transects_total_bounds)]
print(dictionary) 

df = pd.DataFrame(dictionary, columns=['filename', 'bbox'])
print (df)

df.to_csv("transects_bounding_boxes2.csv")

transects_gpd=bbox.read_gpd_file(transects[1])
transects_gpd

# Add a geodataframe to leafmap
coastseg_map.m.add_gdf(transects_in_bbox, layer_name="Cable line")

# Alt Code for load
intersecting_transect_files = get_transect_filenames(gpd_bbox)
for transect_file in intersecting_transect_files:
    print(transect_file)
    data,transects_layer_name=load_transect(transect_file)
    transects_in_bbox=clip_transect(data, gpd_bbox)
    if transects_in_bbox.empty:
        print("Skipping ",transects_layer_name)
    else:
        transect_layer=style_transect(transects_in_bbox)
        geojson_layer = GeoJSON(data=transect_layer, name=transects_layer_name, hover_style={"fillColor": "red"})
        coastseg_map.m.add_layer(geojson_layer)
#         coastseg_map.m.add_geojson(
#         transect_layer, layer_name=transects_layer_name)