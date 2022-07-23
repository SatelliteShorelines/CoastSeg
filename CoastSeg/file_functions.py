import os
import glob
import shutil
from datetime import datetime


def get_jpgs_from_data(file_ext:str='RGB') -> str:
    """Returns the folder where all jpgs were copied from the data folder in CoastSeg.
    This is where the model will save the computed segmentations."""
    # Data folder location
    src_path = os.getcwd() + os.sep + "data"
    if os.path.exists(src_path):
        rename_jpgs(src_path)
        # Create a new folder to hold all the data
        location = os.getcwd()
        name = "segmentation_data"
        new_folder = mk_new_dir(name, location)
        if file_ext == 'RGB':
            glob_str = src_path + str(os.sep + "**" + os.sep) * 3 +'RGB'+ os.sep + "*.jpg"
            copy_files_to_dst(src_path, new_folder, glob_str)
        elif file_ext == 'MNDWI':
            # Copy the RGB images to the destination
            glob_str = src_path + str(os.sep + "**" + os.sep) * 3 +'RGB'+ os.sep + "*RGB*.jpg"
            RGB_path=new_folder+os.sep+'RGB'
            if not os.path.exists(RGB_path):
                os.mkdir(RGB_path)
            copy_files_to_dst(src_path, RGB_path, glob_str)
            # Copy the NIR images to the destination
            glob_str = src_path + str(os.sep + "**" + os.sep) * 3 +'NIR'+ os.sep + "*NIR*.jpg"
            NIR_path=new_folder+os.sep+'NIR'
            if not os.path.exists(NIR_path):
                os.mkdir(NIR_path)
            copy_files_to_dst(src_path, NIR_path, glob_str)
        return new_folder
    else:
        print("ERROR: Cannot find the data directory in CoastSeg")
        raise Exception("ERROR: Cannot find the data directory in CoastSeg")


def rename_jpgs(src_path: str) -> None:
    """ Renames all the jpgs in the data directory in CoastSeg
    Args:
        src_path (str): full path to the data directory in CoastSeg
    """
    files_renamed = False
    for folder in os.listdir(src_path):
        folder_path = src_path + os.sep + folder
        # Split the folder name at the first _
        folder_id = folder.split("_")[0]
        folder_path = folder_path + os.sep + "jpg_files" + os.sep + "preprocessed"
        jpgs = glob.glob1(folder_path + os.sep, "*jpg")
        # Append folder id to basename of jpg if not already there
        for jpg in jpgs:
            if folder_id not in jpg:
                # print(jpg)
                files_renamed = True
                base, ext = os.path.splitext(jpg)
                new_name = folder_path + os.sep + base + "_" + folder_id + ext
                old_name = folder_path + os.sep + jpg
                os.rename(old_name, new_name)
        if files_renamed:
            print(f"Renamed files in {src_path} ")


def generate_datestring() -> str:
    """"Returns a datetime string in the following format %Y-%m-%d__%H_hr_%M_min.
    EX: "ID02022-01-31__13_hr_19_min"
    """
    date = datetime.now()
    return date.strftime('%Y-%m-%d__%H_hr_%M_min%S')


def mk_new_dir(name: str, location: str):
    """Create new folder with  datetime stamp at location
    Args:
        name (str): name of folder to create
        location (str): location to create folder
    """
    if os.path.exists(location):
        new_folder = location + os.sep + name + "_" + generate_datestring()
        os.mkdir(new_folder)
        return new_folder
    else:
        raise Exception("Location provided does not exist.")


def copy_files_to_dst(src_path: str, dst_path: str, glob_str: str) -> None:
    """Copies all files from src_path to dest_path
    Args:
        src_path (str): full path to the data directory in CoastSeg
        dst_path (str): full path to the images directory in Sniffer
    """
    if not os.path.exists(dst_path):
        print(f"dst_path: {dst_path} doesn't exist.")
    elif not os.path.exists(src_path):
        print(f"src_path: {src_path} doesn't exist.")
    else:
        for file in glob.glob(glob_str):
            shutil.copy(file, dst_path)
        print(f"\nCopied files that matched {glob_str}  \nto {dst_path}")
