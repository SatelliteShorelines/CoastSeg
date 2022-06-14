import os
import glob
import shutil
from datetime import datetime

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


def generate_datestring():
    """"
    Returns a string in the following format %Y-%m-%d__%H_hr_%M_min.
    EX: "ID02022-01-31__13_hr_19_min"
    """
    date = datetime.now()
    print(date)
    return date.strftime('%Y-%m-%d__%H_hr_%M_min')


def mk_new_dir(name:str,location:str):
    """make a new folder with a datatime stamp at the location
    Args:
        name (str): name of folder to create
        location (str): location to create folder
    """
    if os.path.exists(location):
        new_folder=location+os.sep+name+"_"+generate_datestring()
        os.mkdir(new_folder)
        return new_folder
    else:
        raise Exception("Location provided does not exist.")
    
def copy_files_to_dst(src_path: str, dst_path: str, glob_str: str) -> None:
    """copy_files_to_dst copies all the files from src_path to dest_path
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
        print(f"Copied files that matched {glob_str}  to {dst_path}")
