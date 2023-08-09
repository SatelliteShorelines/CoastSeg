import argparse
import xarray as xr
import json
import numpy as np
from glob import glob
import os
import gzip
import shutil

# Hello!
# to run this file from the command line
# replace user with your avios email
# replace argument after --tide_model with the location of your downloaded tide_model
# replace argument after --regions_file with the location to the regions file you want to clip the tide_model to
# python clip_and_write_new_nc_files_cmd_line.py --tide_model C:\1_USGS\1_CoastSeg\1_official_CoastSeg_repo\CoastSeg\tide_models --regions_file C:\1_USGS\1_CoastSeg\1_official_CoastSeg_repo\CoastSeg\scripts\tide_regions_map.geojson


def clip_specific_region(files, geometries, region_number=10):
    """
    Clips netCDF files to a specific region and writes the clipped data to new netCDF files.

    This function iterates over a list of file paths, where each file is expected to be a netCDF file.
    For each file, it opens the file as an xarray Dataset and clips the Dataset to a specific region specified
    by the 'region_number' argument. The region is retrieved from the global 'geometries' variable and is defined
    as a longitude-latitude bounding box.

    After clipping the Dataset to the region, the function writes the clipped data to a new netCDF file.
    The new file has the same name as the original file but with '_clipped_regionN' appended to the name,
    where N is the 'region_number'.

    Args:
        files (list): A list of file paths to netCDF files to be clipped.
        region_number (int, optional): The index of the region in 'geometries' to which to clip the data.
            Defaults to 10.

    Prints:
        Progress messages indicating which variable and region the function is currently processing.

    Returns:
        None
    """
    for counter, k in enumerate(files):
        print(f" working on variable {counter}")
        ds_disk = xr.open_dataset(k, engine="netcdf4")
        ## cycle through each region
        region = geometries[region_number]
        print(f" working on region {region_number}")
        lon = np.array(region["coordinates"][0])[:, 0]
        lat = np.array(region["coordinates"][0])[:, 1]
        if min(lon) < 0:
            # re-order along longitude to go from -180 to 180
            ds_disk = ds_disk.assign_coords(
                {"lon": (((ds_disk.lon + 180) % 360) - 180)}
            )
            ds_disk = ds_disk.reindex({"lon": np.sort(ds_disk.lon)})
            lon = np.array(region["coordinates"][0])[:, 0]
        # Find existing coords between min&max
        lats = ds_disk.lat[
            np.logical_and(ds_disk.lat >= min(lat), ds_disk.lat <= max(lat))
        ].values

        # If there was nothing between, just plan to grab closest
        if len(lats) == 0:
            lats = np.unique(ds_disk.lat.sel(lat=np.array(lat), method="nearest"))
        lons = ds_disk.lon[
            np.logical_and(ds_disk.lon >= min(lon), ds_disk.lon <= max(lon))
        ].values

        if len(lons) == 0:
            lons = np.unique(ds_disk.lon.sel(lon=np.array(lon), method="nearest"))

        # crop and keep attrs
        output = ds_disk.sel(lat=lats, lon=lons)
        output.attrs = ds_disk.attrs
        for var in output.data_vars:
            output[var].attrs = ds_disk[var].attrs
        output.to_netcdf(
            path=k.replace(".nc", f"_clipped_region{region_number}.nc"),
            mode="w",
            format=None,
            group=None,
            engine=None,
            encoding=None,
            unlimited_dims=None,
            compute=True,
            invalid_netcdf=False,
        )


def clip_and_write_new_nc_files(files, geometries, dest_dir):
    """
    Clips netCDF files to specified regions and writes the clipped data to new netCDF files.

    This function iterates over a list of file paths, where each file is expected to be a netCDF file.
    For each file, it opens the file as an xarray Dataset and clips the Dataset to each region specified in the
    global 'geometries' variable. Each region is defined as a longitude-latitude bounding box.

    After clipping the Dataset to a region, the function writes the clipped data to a new netCDF file.
    The new file has the same name as the original file but with '_clipped_regionN' appended to the name,
    where N is the index of the region.

    Args:
        files (list): A list of file paths to netCDF files to be clipped.

    Prints:
        Progress messages indicating which variable and region the function is currently processing.

    Returns:
        None
    """
    for counter, current_file in enumerate(files):
        print(f" Working on variable {counter}")
        ds_disk = xr.open_dataset(current_file, engine="netcdf4")

        ## cycle through each region
        for region_number, region in enumerate(geometries):
            print(f"working on region {region_number}")

            lon = np.array(region["coordinates"][0])[:, 0]
            lat = np.array(region["coordinates"][0])[:, 1]

            if min(lon) < 0:
                # re-order along longitude to go from -180 to 180
                ds_disk = ds_disk.assign_coords(
                    {"lon": (((ds_disk.lon + 180) % 360) - 180)}
                )
                ds_disk = ds_disk.reindex({"lon": np.sort(ds_disk.lon)})
                lon = np.array(region["coordinates"][0])[:, 0]

            # Find existing coords between min&max
            lats = ds_disk.lat[
                np.logical_and(ds_disk.lat >= min(lat), ds_disk.lat <= max(lat))
            ].values

            # If there was nothing between, just plan to grab closest
            if len(lats) == 0:
                lats = np.unique(ds_disk.lat.sel(lat=np.array(lat), method="nearest"))

            lons = ds_disk.lon[
                np.logical_and(ds_disk.lon >= min(lon), ds_disk.lon <= max(lon))
            ].values

            if len(lons) == 0:
                lons = np.unique(ds_disk.lon.sel(lon=np.array(lon), method="nearest"))

            # crop and keep attrs
            output = ds_disk.sel(lat=lats, lon=lons)
            output.attrs = ds_disk.attrs
            for var in output.data_vars:
                output[var].attrs = ds_disk[var].attrs

            region_name = f"region{region_number}"
            region_dir = os.path.join(dest_dir, region_name)
            fes2014_dir = os.path.join(region_dir, "fes2014")
            directory_name = os.path.basename(os.path.dirname(current_file))
            tide_dir = os.path.join(fes2014_dir, directory_name)

            os.makedirs(tide_dir, exist_ok=True)
            filename = os.path.basename(current_file)

            destination_path = os.path.join(tide_dir, filename)

            if not os.path.exists(destination_path):
                output.to_netcdf(
                    path=destination_path,
                    mode="w",
                    format=None,
                    group=None,
                    engine=None,
                    encoding=None,
                    unlimited_dims=None,
                    compute=True,
                    invalid_netcdf=False,
                )
                print(f"Saved file to: {destination_path}")
            else:
                print(f"File already exists: {destination_path}")

            # output.to_netcdf(path=destination_path, mode='w', format=None, group=None, engine=None, encoding=None, unlimited_dims=None, compute=True, invalid_netcdf=False)
            # print(f"Saved file to: {destination_path}")


def get_geometries_from_file(file_path):
    """
    Retrieves the geometries from a GeoJSON file.

    Args:
        file_path (str): The file path of the GeoJSON file.

    Returns:
        list: A list of geometries extracted from the file.

    """
    with open(file_path) as f:
        gj = json.load(f)
    features = gj["features"]
    geometries = []
    for f in features:
        geometries.append(f["geometry"])
    return geometries


def unzip_gzip_files(directory):
    """
    Unzips all gzip (.gz) files in a given directory and removes the original .gz files.

    This function iterates over all files in the specified directory. For each file that ends with '.gz',
    it unzips the file by copying the contents of the .gz file to a new file that has the same name but without the '.gz' extension.
    After a .gz file is unzipped, the original .gz file is removed from the directory.

    Args:
        directory (str): The path of the directory in which to find .gz files to unzip.

    Prints:
        A message for each .gz file that is unzipped and removed, indicating the path of the file.

    Returns:
        None
    """
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # If the file is a .gz file
        if filename.endswith(".gz"):
            # Construct full file path
            file_path = os.path.join(directory, filename)
            # Construct output file path (same as input but without .gz)
            output_path = file_path[:-3]

            # Open the .gz file and the output file
            with gzip.open(file_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    # Copy the contents of the .gz file to the output file
                    shutil.copyfileobj(f_in, f_out)

            print(f"Unzipped file {file_path}")

            # Remove the .gz file
            os.remove(file_path)
            print(f"Removed file {file_path}")


def create_region_directories(base_dir, region_names):
    """
    Creates subdirectories for each region directory.

    Args:
        base_dir (str): The base directory path.
        region_names (list): A list of region names.

    Returns:
        None

    """
    for region_name in region_names:
        region_dir = os.path.join(base_dir, region_name)
        fes2014_dir = os.path.join(region_dir, "fes2014")
        load_tide_dir = os.path.join(fes2014_dir, "load_tide")
        ocean_tide_dir = os.path.join(fes2014_dir, "ocean_tide")

        os.makedirs(region_dir, exist_ok=True)  # Create region directory
        os.makedirs(fes2014_dir, exist_ok=True)  # Create 'fes2014' directory
        os.makedirs(load_tide_dir, exist_ok=True)  # Create 'load_tide' directory
        os.makedirs(ocean_tide_dir, exist_ok=True)  # Create 'ocean_tide' directory


# Create the parser
parser = argparse.ArgumentParser(description="Clip and write new netCDF files.")

# Add the arguments
parser.add_argument(
    "--tide_model", "-T", default="", type=str, help="Path to tide model directory"
)

parser.add_argument(
    "--regions_file", "-R", default="", type=str, help="Path to the regions file"
)

# Parse the arguments
args = parser.parse_args()

# Assign the arguments to variables
tide_model_directory = args.tide_model
regions_file = args.regions_file

# # Please enter the full location to the regions file
# regions_file = r'C:\1_USGS\1_CoastSeg\1_official_CoastSeg_repo\CoastSeg\scripts\tide_regions_map.geojson'
# # Please enter the full location to the directory containing the tide models you downloaded
# tide_model_directory = r'C:\1_USGS\1_CoastSeg\1_official_CoastSeg_repo\CoastSeg\tide_models'

# create paths to tide models
fes2014_model_directory = os.path.join(tide_model_directory, "fes2014")
load_tide_dir = os.path.join(fes2014_model_directory, "load_tide")
ocean_tide_dir = os.path.join(fes2014_model_directory, "ocean_tide")
# load geometeries from regions file
geometries = get_geometries_from_file(regions_file)

print(f"{len(geometries)} tide regions")
# unzip the compressed files in each directory
unzip_gzip_files(load_tide_dir)
unzip_gzip_files(ocean_tide_dir)

# create a list of all the nc files in both the load and ocean tide directories
load_tide_nc_files = [
    f for f in glob(os.path.join(load_tide_dir, "*.nc")) if "clipped" not in f
]
ocean_tide_nc_files = [
    f for f in glob(os.path.join(ocean_tide_dir, "*.nc")) if "clipped" not in f
]

# create region directory structure and move files
region_names = [f"region{i}" for i in range(0, len(geometries))]
create_region_directories(tide_model_directory, region_names)

# create the new clipped files in the region directory
clip_and_write_new_nc_files(ocean_tide_nc_files, geometries, tide_model_directory)
clip_and_write_new_nc_files(load_tide_nc_files, geometries, tide_model_directory)
