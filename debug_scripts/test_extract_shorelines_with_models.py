import os
from typing import Optional
import argparse

# internal python imports
from coastseg import zoo_model
from transformers import TFSegformerForSemanticSegmentation

# how to run this script
# Replace parameter after -P with path to ROI's RGB directory
# python test_models.py -P <your path here>"
# Example
# python test_models.py -P C:\development\doodleverse\coastseg\CoastSeg\data\ID_12_datetime06-05-23__04_16_45\jpg_files\preprocessed\RGB"

# Create the parser
parser = argparse.ArgumentParser(description="Run models on provided input directory.")
parser.add_argument(
    "-P",
    "--path",
    type=str,
    help="Path to an ROI's RGB directory from the data directory",
)
args = parser.parse_args()

# get input directory from the args
input_directory = args.path

# uncomment the following lines for local testing only DUCK
CoastSeg_location = r"C:\development\doodleverse\coastseg\CoastSeg" # path to CoastSeg
roi_name = "ID_ppy1_datetime07-19-24__10_59_31" # name of the ROI directory in CoastSeg
input_directory = os.path.join(CoastSeg_location, "data", roi_name, "jpg_files", "preprocessed", "RGB") # this is the path to the RGB directory of the ROI



def create_settings(new_settings: Optional[dict] = None) -> dict:
    settings ={
        'min_length_sl': 100,       # minimum length (m) of shoreline perimeter to be valid
        'max_dist_ref':600,         # maximum distance (m) from reference shoreline to search for valid shorelines. This detrmines the width of the buffer around the reference shoreline  
        'cloud_thresh': 0.5,        # threshold on maximum cloud cover (0-1). If the cloud cover is above this threshold, no shorelines will be extracted from that image
        'dist_clouds': 100,         # distance(m) around clouds where shoreline will not be mapped
        'min_beach_area': 100,      # minimum area (m^2) for an object to be labelled as a beach
        'sand_color': 'default',    # 'default', 'latest', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
        "apply_cloud_mask": True,   # apply cloud mask to the imagery. If False, the cloud mask will not be applied.
    }
    settings.update(new_settings or {})
    return settings

def create_model_settings(input_directory:str,img_type:str,implementation: str="BEST"):
    """
    Creates a settings dictionary for model configuration.
    Args:
        input_directory (str): The directory containing the input images.
        img_type (str): The type of images (e.g., 'RGB', 'NIR').
        implementation (str, optional): The implementation type, either 'BEST' or 'ENSEMBLE'. Defaults to 'BEST'.
    Returns:
        dict: A dictionary containing the model settings.
    Raises:
        AssertionError: If the input directory does not exist.
    """
    model_setting = {
                "sample_direc": None, # directory of jpgs  ex. C:/Users/username/CoastSeg/data/ID_lla12_datetime11-07-23__08_14_11/jpg_files/preprocessed/RGB/",
                "use_GPU": "0",  # 0 or 1 0 means no GPU
                "implementation": "BEST",  # BEST or ENSEMBLE 
                "model_type":"global_segformer_RGB_4class_14036903", # model name ex. global_segformer_RGB_4class_14036903
                "otsu": False, # Otsu Thresholding
                "tta": False,  # Test Time Augmentation
            }
    # assrt the input directory exists
    assert os.path.exists(input_directory), f"Input directory {input_directory} does not exist"
    model_setting["sample_direc"] = input_directory
    model_setting["img_type"] = img_type
    return model_setting

def test_model(img_types:list, available_models_dict:dict, input_directory:str):
    """
    Tests models on different types of images.
    Parameters:
    img_types (list): A list of image types to be tested.
    available_models_dict (dict): A dictionary where keys are image types and values are lists of models available for those image types.
    input_directory (str): The directory where input images are stored. Should be the RGB directory for the ROI containing the images 
    Raises:
    Exception: If session_name is not generated correctly.
    Prints:
    Various debug information including the session name, implementation type, selected image type, selected model, and sample directory.
    """
    for img_type_index in range(len(img_types)):
        print(f"Running models for image input type {img_type_index}")
        selected_img_type = img_types[img_type_index]
        model_list = available_models_dict.get(selected_img_type)
        if not model_list:
            print(f"No models available for {selected_img_type}")
            continue

        for model_selected in model_list:
            print(f"Running model {model_selected}")
            implementation = "BEST"  # "ENSEMBLE" or "BEST"
            session_name = (
                model_selected
                + "_"
                + selected_img_type
                + "_"
                + implementation
                + "_"
                + "session"
            )
            if not session_name:
                raise Exception("Something went wrong...")

            print(f"session_name: {session_name}")
            print(f"implementation: {implementation}")
            print(f"selected_img_type: {selected_img_type}")
            print(f"model_selected: {model_selected}")
            print(f"sample_directory: {input_directory}")

            zoo_model_instance = zoo_model.Zoo_Model()
            # create and set settings for the model
            model_setting = create_model_settings(input_directory, selected_img_type, implementation)
            extract_shoreline_settings = create_settings()
            model_setting.update(extract_shoreline_settings)
            zoo_model_instance.set_settings(**model_setting)

            # run the model and extract shorelines 
            zoo_model_instance.run_model_and_extract_shorelines(
                        model_setting["sample_direc"],
                        session_name=session_name,
                        shoreline_path="",
                        transects_path="",
                        shoreline_extraction_area_path = "",
                    )

available_models_dict = {
    "RGB": [
        "global_segformer_RGB_4class_14036903", # global segformer model
        "AK_segformer_RGB_4class_14037041", # AK segformer model
    ],
    "MNDWI": [
        "global_segformer_MNDWI_4class_14183366", # global segformer model
        "AK_segformer_MNDWI_4class_14187478", # AK segformer model
    ],
    "NDWI": [
      "global_segformer_NDWI_4class_14172182", # global segformer model
      "AK_segformer_NDWI_4class_14183210", # AK segformer model
    ],
}

# img_types = ["RGB"]
img_types = ["RGB", "MNDWI", "NDWI"]

test_model(img_types, available_models_dict, input_directory)