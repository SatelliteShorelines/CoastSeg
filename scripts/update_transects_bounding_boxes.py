import os
import glob
import pandas as pd
import geopandas as gpd
import shapely

# Written by:  Sharon Fitzpatrick
# This script will update the bounding box csv file for the transects
# It will be used to determine if the ROI intersects with any of the transects for that region
# All the transects to be included must be in CoastSeg/src/coastseg/transects
# The bounding box csv file will be saved in CoastSeg/src/coastseg/bounding_boxes


# full path to transects directory which contains all the transect geojson files
transects_folder=os.path.join(os.getcwd(),"src","coastseg","transects")
print(f"The transects folder is located at {os.path.abspath(transects_folder)}")

# get all the transects in the transects folder
transects=glob.glob(transects_folder+os.sep+"*.geojson")
# create a list of the transect layer names from the filenames of the transects
transect_layer_names=[os.path.basename(transect) for transect in transects]
print("Please be patient, this may take a few minutes...")
print(f"The bounding box csv file will be created for the following transects: {transect_layer_names}")
# get the bounding box for each transect this will be used to determine if the any of transects for that region could intersect with ROI
transects_total_bounds=[gpd.read_file(transect_file).total_bounds for transect_file in transects]

df = pd.DataFrame(transects_total_bounds,columns=['minx', 'miny', 'maxx', 'maxy'])
df['filename'] = transect_layer_names

# lets overwrite the old transects_total_bounds with the new dataframe
csv_file = "transects_bounding_boxes.csv"
bounding_box_path = os.path.join(os.getcwd(),"src","coastseg","bounding_boxes")
csv_path = transect_folder=os.path.join(bounding_box_path,csv_file)
df.to_csv(csv_path,index=False)

transects_df=pd.read_csv(csv_path)

print(f'The new bounding box csv file has been created at {os.path.abspath(csv_path)}')





