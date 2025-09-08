import os
from osgeo import gdal
import argparse

# ---------------------------------------------------------------------
# Georeferencing JPEGs Script
# ---------------------------------------------------------------------
# Author: Sharon Fitzpatrick
# Date : 10/4/2023
#
# Description:
# This script georeferences JPEG images using spatial information from corresponding TIFF images.
# Georeferencing adds spatial location metadata to the JPEGs, enabling them to be mapped onto specific
# locations on Earth's surface.
#
# How to use:
# 1. Ensure all required dependencies are installed, including `os` and `gdal`
# 2. Run this script from the command line and provide the ROI directories as arguments:
#    - Make sure to enclose each path in double quotes
#    - Example: python georeference_jpegs.py "C:\development\doodleverse\CoastSeg\data\ID_quj9_datetime09-28-23__05_12_40"
#    - Example python georeference_jpegs.py "path_to_roi_dir1" "path_to_roi_dir2" ... "path_to_roi_dirN"
#        - (Replace `path_to_roi_dirX` with your actual directory paths.)
#
# Output:
# For each specified ROI directory, the script:
# - Identifies JPEGs in predefined subdirectories: "RGB", "NIR", "SWIR", "MNDWI", "NDWI"
# - For each jpeg it finds the matching TIFFs with the georeferencing info.
# - Applies the georeferencing metadata from the TIFFs to the JPEGs.
# - Saves the georeferenced JPEGs in a `georeferenced` subfolder within each subdirectory.
#   For example, if the `RGB` directory's path is `path_to_roi_dir1/jpg_files/preprocessed/RGB/`,
#   the georeferenced JPEGs will be saved in:
#   `path_to_roi_dir1/jpg_files/preprocessed/RGB/georeferenced/`
#
# Georeferenced JPEGs:
# These are standard JPEG images but with additional files (.wld, .xml) containing the spatial data. The spatial data allows the JPEG to
# be correctly positioned on the Earth's surface in GIS software. The georeferencing information includes
# projection details, coordinate data, and image resolution.
# ---------------------------------------------------------------------


# helper function
def get_satellite_name(filename: str):
    """Returns the satellite name in the jpg name. Does not work tiffs"""
    try:
        return filename.split("_")[2].split(".")[0]
    except IndexError:
        # logger.error(f"Unable to extract satellite name from filename: {filename}")
        return None


def georeference_jpeg_using_tif(jpeg_path: str, tif_path: str, output_path: str):
    """
    Georeference a JPEG image using the georeferencing information from a TIFF image.

    The function reads the georeferencing info from the TIFF and applies it to the JPEG.
    The georeferenced JPEG is then saved to the specified output path.

    Parameters:
    - jpeg_path (str): The path to the input JPEG image.
    - tif_path (str): The path to the input TIFF image with georeferencing info.
    - output_path (str): The path where the georeferenced JPEG will be saved.

    Returns:
    None
    """
    # Open the TIFF dataset to get the georeferencing information
    tif_dataset = gdal.Open(tif_path)
    geotransform = tif_dataset.GetGeoTransform()
    projection = tif_dataset.GetProjection()

    # Open the JPEG dataset
    jpeg_dataset = gdal.Open(jpeg_path)

    # Create an in-memory copy of the JPEG with the georeferencing info from the TIFF
    driver = gdal.GetDriverByName("MEM")
    mem_dataset = driver.CreateCopy("", jpeg_dataset, 0)
    mem_dataset.SetGeoTransform(geotransform)
    mem_dataset.SetProjection(projection)

    # Define output format and options for translation
    options = gdal.TranslateOptions(format="JPEG", creationOptions=["WORLDFILE=YES"])
    gdal.Translate(output_path, mem_dataset, options=options)

    # Close datasets
    tif_dataset = None
    jpeg_dataset = None
    mem_dataset = None


def find_matching_tif(jpg_filename: str, tif_directory: str):
    """
    Find the corresponding TIFF file for a given JPEG filename by matching date components.

    The function splits the JPEG filename to extract the date and searches for a TIFF
    with a matching date in its name.

    Parameters:
    - jpg_filename (str): The name of the JPEG file (not the full path).
    - tif_directory (str): The directory where TIFF files are stored.

    Returns:
    str or None: The name of the matching TIFF file, or None if no match is found.
    """
    # Step 1: Split by "_"
    components = jpg_filename.split("_")
    # Extract the date component
    date_component = components[0]

    # Step 2: Find the matching TIFF by date in the tif_directory
    for tif_file in os.listdir(tif_directory):
        if tif_file.endswith(".tif") and date_component in tif_file:
            return tif_file
    return None  # return None if no matching TIFF is found


def get_jpeg_filenames(jpeg_directory):
    """Return a list of JPEG filenames from the given directory."""
    return [f for f in os.listdir(jpeg_directory) if f.endswith((".jpg", ".jpeg"))]


def group_jpegs_by_satellite(jpeg_filenames: list[str]):
    """
    Group JPEG filenames by satellite name.

    Parameters:
    - jpeg_filenames (list of str): List of JPEG filenames to be grouped.

    Returns:
    - dict: A dictionary where the keys are satellite names (e.g., 'L5', 'L7', etc.)
            and the values are lists of filenames associated with each satellite.
    """
    sat_dict = {
        "L5": [],
        "L7": [],
        "L8": [],
        "L9": [],
        "S2": [],
    }
    for filename in jpeg_filenames:
        satname = get_satellite_name(filename)
        if satname in sat_dict:
            sat_dict[satname].append(filename)
    return sat_dict


def georeference_jpegs_for_satellite(
    satname, jpg_filenames, jpeg_directory, roi_dir, output_directory
):
    """
    Process each satellite group and georeference the JPEGs based on their
    corresponding TIFFs in the specified directory.

    Parameters:
    - satname (str): The name of the satellite (e.g., 'L5', 'L7', etc.).
    - jpg_filenames (list of str): List of JPEG filenames associated with the given satellite.
    - jpeg_directory (str): Path to the directory containing the JPEG files.
    - roi_dir (str): Root directory path for TIFF and JPEG data.

    Returns:
    - None: The function operates in-place, creating georeferenced JPEGs in the
            specified output directory.
    """
    tif_directory = os.path.join(roi_dir, satname, "ms")
    os.makedirs(output_directory, exist_ok=True)

    for jpg_filename in jpg_filenames:
        matching_tif = find_matching_tif(jpg_filename, tif_directory)
        if matching_tif:
            # Construct full paths
            jpeg_path = os.path.join(jpeg_directory, jpg_filename).strip()
            tif_path = os.path.join(tif_directory, matching_tif).strip()
            if os.path.exists(tif_path):
                output_path = os.path.join(output_directory, jpg_filename)
                georeference_jpeg_using_tif(jpeg_path, tif_path, output_path)
                print(f"Created jpg at {output_path}")


def georeference_directory(img_dir_path: str, roi_dir: str) -> None:
    """
    Georeference all JPEG images in the specified directory based on corresponding TIFFs.

    Parameters:
    - img_dir_path (str): Path to the image directory containing JPEG files to be georeferenced.
    - roi_dir (str): Root directory containing the subdirectories for the TIFF and JPEG datasets.

    Returns:
    - None: The function works in-place, producing georeferenced JPEGs in a 'georeferenced' subdirectory.

    Note:
    The TIFF files are expected to be present in a structure like: `roi_dir/satellite_name/ms`.
    """
    # Check if the directory exists
    if not os.path.exists(img_dir_path):
        return

    jpeg_filenames = get_jpeg_filenames(img_dir_path)

    # Get all the jpegs associated with each satellite e.g. all the jpgs for L8
    satellite_groups = group_jpegs_by_satellite(jpeg_filenames)
    output_directory = os.path.join(img_dir_path, "georeferenced")

    for satname, jpg_filenames in satellite_groups.items():
        georeference_jpegs_for_satellite(
            satname, jpg_filenames, img_dir_path, roi_dir, output_directory
        )


def georeference_images_in_directory(roi_directory: str):
    """
    Iterate through specified subdirectories within the ROI directory and georeference all JPEGs based on their corresponding TIFFs.

    Parameters:
    - roi_directory (str): Root directory containing the TIFF and JPEG datasets organized by image types like RGB, NIR, etc.

    Returns:
    - None: The function processes images in-place, producing georeferenced JPEGs in 'georeferenced' subdirectories.

    Note:
    The function iterates over predefined image types (e.g., RGB, NIR) and applies georeferencing to each type's directory.
    """
    jpeg_directory = os.path.join(roi_directory, "jpg_files", "preprocessed")
    possible_directories = ["RGB", "NIR", "SWIR", "MNDWI", "NDWI"]

    for img_dir in possible_directories:
        subdirectory = os.path.join(jpeg_directory, img_dir)
        georeference_directory(subdirectory, roi_directory)


def main():
    # REPLACE ROI DIR with the directory you want to georeference jpegs for
    # roi_dir = r"C:\CoastSeg\data\ID_pmb7_datetime10-03-23__02_12_09"
    # roi_dirs = [
    #     r"C:\CoastSeg\data\ID_quj9_datetime09-28-23__05_12_40",
    #     r"C:\CoastSeg\data\ID_kpd7_datetime08-28-23__01_47_44",
    # ]

    parser = argparse.ArgumentParser(
        description="Georeference JPEGs using corresponding TIFFs for the given ROI directories."
    )
    parser.add_argument(
        "roi_dirs",
        nargs="+",
        help='List of ROI directories to process. If using windows enclose path in doouble quotes. Ex. python georeference_jpgs.py "C:\development\doodleverse\CoastSeg\data\ID_quj9_datetime09-28-23__05_12_40" ',
    )
    args = parser.parse_args()

    for roi_dir in args.roi_dirs:
        if not os.path.exists(roi_dir):
            print(f"Skipping this directory because it does not exist. {roi_dir}")
            continue
        georeference_images_in_directory(roi_dir)


if __name__ == "__main__":
    main()
