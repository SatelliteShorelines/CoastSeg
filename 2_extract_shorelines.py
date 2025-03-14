import os
from coastseg import coastseg_logs
from coastseg import coastseg_map
from coastseg import core_utilities

base_dir = core_utilities.get_base_dir()

# Run this script only after running 1_download_imagery.py 

# create a new coastseg_map object
coastsegmap=coastseg_map.CoastSeg_Map(create_map=False)

# Enter the name of the session you want to load here
session_name = 'sample_session1'
# session_name = 'paper_dfg2_extract_shorelines_buffer_62'
session_path = os.path.join(os.path.abspath(base_dir),'sessions', session_name)
print(f"Loading session from {session_path}")
# r"C:\development\doodleverse\coastseg\CoastSeg\sessions\paper_dfg2_extract_shorelines_buffer_62"
coastsegmap.load_fresh_session(session_path)

# name the session where you want to save the new extracted shorelines to. If you don't rename the session, it will overwrite the existing session
session_name = 'sample_session2'
coastsegmap.set_session_name(session_name)

# Modify the settings for the shoreline extraction here
# example: change the settings for the shoreline extraction to only extract shorelines with a minimum beach area of 500 m^2
new_settings = { 
    'min_length_sl': 500,       # minimum length (m) of shoreline perimeter to be valid
    'max_dist_ref':300,         # maximum distance (m) from reference shoreline to search for valid shorelines. This detrmines the width of the buffer around the reference shoreline  
    'cloud_thresh': 0.5,        # threshold on maximum cloud cover (0-1). If the cloud cover is above this threshold, no shorelines will be extracted from that image
    'dist_clouds': 300,         # distance(m) around clouds where shoreline will not be mapped
    'min_beach_area': 500,      # minimum area (m^2) for an object to be labelled as a beach
    'sand_color': 'default',    # 'default', 'latest', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
    "apply_cloud_mask": True,   # apply cloud mask to the imagery. If False, the cloud mask will not be applied.
}

coastsegmap.set_settings(**new_settings)

# loading the shorelines and transects are optional if the session already contains them
# The shorelines and transects loaded here will be used to extract the shorelines for the selected ROI
# load the default shoreline provided by CoastSeg
coastsegmap.load_feature_on_map("shoreline", zoom_to_bounds=True)

# load the default transects provided by CoastSeg
coastsegmap.load_feature_on_map("transect", zoom_to_bounds=True)


# get the ROI from the loaded session
roi= coastsegmap.rois
# get the select all the ROI IDs from the file and store them in a list
roi_ids =  list(roi.gdf.id)

print(f"Extracting shorelines for ROI with ID {roi_ids}")

# extract the shorelines for the selected ROI and save them to the /sessions/session_name folder
# coastsegmap.extract_all_shorelines(roi_ids = roi_ids)


# Tide Correction (optional)
# WARNING: Before running this snippet, you must download the tide model to the CoastSeg/tide_model folder.
# WE RECOMMEND USING FES2022.
# 
# Tutorial on How to Download the Tide Model:
# https://github.com/Doodleverse/CoastSeg/wiki/09.-How-to-Download-and-clip-Tide-Model
#
# The Tide Model must be downloaded to CoastSeg/tide_model.
# Two Tide Models are available: 'FES2014' or 'FES2022'.
#
# Parameters:
beach_slope = 0.02 # Slope of the beach (m/m)
reference_elevation = 0 # Reference elevation (m, relative to user-specified vertical datum)
tides_file = '' #(Optional) Enter the full path to the CSV file containing the tide data if you don't want to use the tide model. See accepted formats : https://satelliteshorelines.github.io/CoastSeg/tide-file-format/
slopes_file ='' #(Optional) Enter the full path to the CSV file containing the beach slopes if you don't want to use a constant slope. See accepted formats: https://satelliteshorelines.github.io/CoastSeg/slope-file-format/
if slopes_file:
    beach_slope = slopes_file

# Uncomment the line below to run the tide correction
coastsegmap.compute_tidal_corrections(roi_ids, beach_slope, reference_elevation,model='FES2022',tides_file=tides_file)
