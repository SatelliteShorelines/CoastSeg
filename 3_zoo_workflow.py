import os
from coastseg import coastseg_logs
from coastseg import zoo_model
from coastseg.tide_correction import compute_tidal_corrections
from coastseg import file_utilities

# The Zoo Model is a machine learning model that can be used to extract shorelines from satellite imagery.
# This script will only run a single ROI at a time. If you want to run multiple ROIs, you will need to run this script multiple times.

# The model can be run using the following settings:
model_setting = {
            "sample_direc": None, # directory of jpgs  ex. C:/Users/username/CoastSeg/data/ID_lla12_datetime11-07-23__08_14_11/jpg_files/preprocessed/RGB/",
            "use_GPU": "0",  # 0 or 1 0 means no GPU
            "implementation": "BEST",  # BEST or ENSEMBLE 
            "model_type": "segformer_RGB_4class_8190958", # model name ex. segformer_RGB_4class_8190958
            "otsu": False, # Otsu Thresholding
            "tta": False,  # Test Time Augmentation
        }
# Available models can run input "RGB" # or "MNDWI" or "NDWI"
img_type = "RGB"
# percentage of no data allowed in the image eg. 0.75 means 75% of the image can be no data
percent_no_data = 0.75

# 1. Set the User configuration Settings
# ---------------------------
# a. ENTER THE NAME OF THE SESSION TO SAVE THE MODEL PREDICTIONS TO
model_session_name = "sample_session_demo1"
# b. ENTER THE DIRECTORY WHERE THE INPUT IMAGES ARE STORED
# -  Example of the directory where the input images are stored ( this should be the /data folder in the CoastSeg directory)
sample_directory = "C:\development\doodleverse\coastseg\CoastSeg\data\ID_wra5_datetime03-04-24__03_43_01\jpg_files\preprocessed\RGB"


# 2. Run the zoo model
# -----------------
# Create an instance of the zoo model to run the model predictions
zoo_model_instance = zoo_model.Zoo_Model()
# Set the model settings to read the input images from the sample directory
model_setting["sample_direc"] = sample_directory

# run the zoo model with the settings
zoo_model_instance.run_model(
    img_type,
    model_setting["implementation"],
    model_session_name,
    model_setting["sample_direc"],
    model_name=model_setting["model_type"],
    use_GPU="0",
    use_otsu=model_setting["otsu"],
    use_tta=model_setting["tta"],
    percent_no_data=percent_no_data,
)

# 2. extract shorelines from running zoo model
# ------------------------------------------
# ENTER THE SESSION NAME TO STORE THE EXTRACTED SHORELINES SESSION
session_name = "sample_session_demo1_extracted_shorelines"
# OPTIONAL: If you have a transects and shoreline file, you can extract shorelines from the zoo model outputs
transects_path = "" # path to the transects geojson file
shoreline_path = "" # path to the shoreline geojson file

shoreline_settings = { 
    'min_length_sl': 500,       # minimum length (m) of shoreline perimeter to be valid
    'max_dist_ref':300,         # maximum distance (m) from reference shoreline to search for valid shorelines. This detrmines the width of the buffer around the reference shoreline  
    'cloud_thresh': 0.5,        # threshold on maximum cloud cover (0-1). If the cloud cover is above this threshold, no shorelines will be extracted from that image
    'dist_clouds': 300,         # distance(m) around clouds where shoreline will not be mapped
    'min_beach_area': 500,      # minimum area (m^2) for an object to be labelled as a beach
    'sand_color': 'default',    # 'default', 'latest', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
    "apply_cloud_mask": True,   # apply cloud mask to the imagery. If False, the cloud mask will not be applied.
}

# get session directory location
model_session_directory = os.path.join(os.getcwd(), "sessions", model_session_name)

# load in shoreline settings, session directory with model outputs, and a new session name to store extracted shorelines
zoo_model_instance.extract_shorelines_with_unet(
    shoreline_settings,
    model_session_directory,
    session_name,
    shoreline_path,
    transects_path,
)


# 3. OPTIONAL: Run Tide Correction
# ------------------------------------------
# Tide Correction (optional)
# Before running this snippet, you must download the tide model to the CoastSeg/tide_model folder
# Tutorial: https://github.com/Doodleverse/CoastSeg/wiki/09.-How-to-Download-and-clip-Tide-Model
#  You will need to uncomment the line below to run the tide correction

beach_slope = 0.02 # Slope of the beach (m)
reference_elevation = 0 # Elevation of the beach Mean Sea Level (M.S.L) (m)

# UNCOMMENT THESE LINES TO RUN THE TIDE CORRECTION
# roi_id = file_utilities.get_ROI_ID_from_session(session_name) # read ROI ID from the config.json file found in the extracted shoreline session directory
# compute_tidal_corrections(
#     session_name, [roi_id], beach_slope, reference_elevation
# )