import os
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
parser = argparse.ArgumentParser(description="Run models on provided inpput directory.")
parser.add_argument(
    "-P",
    "--path",
    type=str,
    help="Path to an ROI's RGB directory from the data directory",
)
args = parser.parse_args()

# get input directory from the args
input_directory = args.path
# for local testing only
input_directory = r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_12_datetime06-05-23__04_16_45\jpg_files\preprocessed\RGB"

# move them into their parent directories
parent_directory_names = [
    "segformer_RGB_4class_8190958",
    "sat_RGB_4class_6950472",
    "segformer_MNDWI_4class_8213443",
    "sat_MNDWI_4class_7352850",
    "segformer_NDWI_4class_8213427",
    "sat_NDWI_4class_7352859",
]

available_models_dict = {
    "RGB": [
        "segformer_RGB_4class_8190958",
        "sat_RGB_4class_6950472",
    ],
    "MNDWI": [
        "segformer_MNDWI_4class_8213443",
        "sat_MNDWI_4class_7352850",
    ],
    "NDWI": [
        "segformer_NDWI_4class_8213427",
        "sat_NDWI_4class_7352859",
    ],
}


img_types = ["RGB", "MNDWI", "NDWI"]

for img_type_index in range(len(img_types)):
    print(f"Running models for image input type {img_type_index}")
    selected_img_type = img_types[img_type_index]
    model_list = available_models_dict.get(selected_img_type)
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

        # load the basic zoo_model settings
        model_dict = {
            "sample_direc": input_directory,
            "use_GPU": "0",
            "implementation": implementation,
            "model_type": model_selected,
            "otsu": False,
            "tta": False,
        }

        zoo_model_instance = zoo_model.Zoo_Model()

        zoo_model_instance.run_model(
            img_types[img_type_index],
            model_dict["implementation"],
            session_name,
            model_dict["sample_direc"],
            model_name=model_dict["model_type"],
            use_GPU="0",
            use_otsu=model_dict["otsu"],
            use_tta=model_dict["tta"],
            percent_no_data=50.0,
        )
