import os
import zipfile
import re
import glob

# internal python imports
from coastseg import zoo_model

# def unzip_and_rename(directory):
#     for item in os.listdir(directory):  # loop through items in dir
#         item_path = os.path.join(directory, item)
#         if item.endswith('.zip'):  # check for ".zip" extension
#             zip_ref = zipfile.ZipFile(item_path)  # create zipfile object
#             zip_ref.extractall(directory)  # extract file to dir
#             zip_ref.close()  # close file
#             os.remove(item_path)  # delete zipped file

#     for item in os.listdir(directory):
#         item_path = os.path.join(directory, item)
#         if os.path.isdir(item_path):
#             new_folder_name = re.sub('_model.*', '', item)
#             new_folder_path = os.path.join(directory, new_folder_name)
#             os.rename(item_path, new_folder_path)

# # unzip and rename the downloaded files
# uncomment these lines when unzipping your first fir the first time
# dir_path = put your directory here
# unzip_and_rename(dir_path)

# move them into their parent directories
parent_directory_names=['sat_5band_2class_7448390', 'sat_5band_4class_7344606', 'sat_MNDWI_2class_7557080', 'sat_MNDWI_4class_7352850', 'sat_NDWI_2class_7557072', 'sat_NDWI_4class_7352859', 'sat_RGB_2class_7865364', 'sat_RGB_4class_6950472']
# I did this manually

available_models_dict = {
    "RGB": [
        "sat_RGB_2class_7865364",
        "sat_RGB_4class_6950472",
    ],
    "RGB+MNDWI+NDWI": [
        "sat_5band_4class_7344606",
        "sat_5band_2class_7448390",
    ],
    "MNDWI": [
        "sat_MNDWI_4class_7352850",
        "sat_MNDWI_2class_7557080",
    ],
    "NDWI": [
        "sat_NDWI_4class_7352859",
        "sat_NDWI_2class_7557072",
    ]
}

img_types = ['RGB',"RGB+MNDWI+NDWI","MNDWI","NDWI"]
# for img_type_index  in range(len(img_types)):
# for img_type_index  in [2,3]:
for img_type_index  in range(len(img_types)):
    print(img_type_index)
    for model_index in range(2):
        print(model_index)
        implementation=  "BEST" # "ENSEMBLE" or "BEST"
        selected_img_type = img_types[img_type_index]
        # selected_img_type = "RGB"
        model_selected = available_models_dict.get(selected_img_type)[model_index]
        # model_selected = 'sat_RGB_2class_7865364'
        session_name = model_selected + '_'+selected_img_type+'_'+implementation+'_'+'session'
        if not session_name:
            raise Exception('Something went wrong...')

        sample_directory= r'C:\1_USGS\1_CoastSeg\1_official_CoastSeg_repo\CoastSeg\data\ID_0_datetime04-25-23__02_19_04\jpg_files\preprocessed\RGB'

        print(f'session_name: {session_name}')
        print(f'implementation: {implementation}')
        print(f'selected_img_type: {selected_img_type}')
        print(f'model_selected: {model_selected}')
        print(f'sample_directory: {sample_directory}')

        # load the basic zoo_model settings
        model_dict = {
            "sample_direc": sample_directory,
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