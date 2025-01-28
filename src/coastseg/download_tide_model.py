"""
This script was originally written by Tyler Sutterley and modified by Sharon Fitzpatrick for Coastseg.
aviso_fes_tides.py
Written by Tyler Sutterley (11/2022)
Downloads the FES (Finite Element Solution) global tide model from AVISO
Decompresses the model tar files into the constituent files and auxiliary files
    https://www.aviso.altimetry.fr/data/products/auxiliary-products/
        global-tide-fes.html
    https://www.aviso.altimetry.fr/en/data/data-access.html

CALLING SEQUENCE:
    python aviso_fes_tides.py --user <username> --tide FES2014
    where <username> is your AVISO data dissemination server username

COMMAND LINE OPTIONS:
    --help: list the command line options
    --directory X: working data directory
    --user X: username for AVISO FTP servers (email)
    -N X, --netrc X: path to .netrc file for authentication
    --tide X: FES tide model to download
        FES1999
        FES2004
        FES2012
        FES2014
    --load: download load tide model outputs (fes2014)
    --currents: download tide model current outputs (fes2012 and fes2014)
    --log: output log of files downloaded
    -M X, --mode X: Local permissions mode of the files downloaded

PYTHON DEPENDENCIES:
    future: Compatibility layer between Python 2 and Python 3
        https://python-future.org/

PROGRAM DEPENDENCIES:
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Update 1/7/2025: added option to download FES2022 by Sharon Fitzpatrick Batiste
    Updated 11/2022: added encoding for writing ascii files
        use f-strings for formatting verbose or ascii output
    Updated 04/2022: use argparse descriptions within documentation
    Updated 10/2021: using python logging for handling verbose output
    Updated 07/2021: can use prefix files to define command line arguments
    Updated 05/2021: use try/except for retrieving netrc credentials
    Updated 04/2021: set a default netrc file and check access
    Updated 10/2020: using argparse to set command line parameters
    Updated 07/2020: add gzip option to compress output ascii and netCDF4 files
    Updated 06/2020: added netrc option for alternative authentication
    Updated 05/2019: new authenticated ftp host (changed 2018-05-31)
    Written 09/2017
"""
# Python built-in modules
from __future__ import print_function
import calendar
import io
import logging
import os
import tarfile
import time
import traceback

# Standard libraries
import ftplib
import gzip
import json
import posixpath
import shutil
from glob import glob

# Third-party libraries
import numpy as np
import xarray as xr
from tqdm.auto import tqdm
import lzma
import pathlib
import re

# Local application/library specific imports
import pyTMD.utilities
from coastseg.file_utilities import progress_bar_context, load_package_resource
from coastseg import core_utilities


# FILE SIZES for files in these files
LOAD_TIDE_FILES = {
    "2n2.nc.gz": 65738083,
    "eps2.nc.gz": 59648992,
    "j1.nc.gz": 63674897,
    "k1.nc.gz": 92483370,
    "k2.nc.gz": 79058819,
    "l2.nc.gz": 66194514,
    "la2.nc.gz": 60549789,
    "m2.nc.gz": 100588509,
    "m3.nc.gz": 58990134,
    "m4.nc.gz": 62106516,
    "m6.nc.gz": 61730021,
    "m8.nc.gz": 57998586,
    "mf.nc.gz": 61632949,
    "mks2.nc.gz": 59124980,
    "mm.nc.gz": 58505186,
    "mn4.nc.gz": 61385688,
    "ms4.nc.gz": 61621572,
    "msf.nc.gz": 57210912,
    "msqm.nc.gz": 54526221,
    "mtm.nc.gz": 55814427,
    "mu2.nc.gz": 67397939,
    "n2.nc.gz": 84849067,
    "n4.nc.gz": 58485930,
    "nu2.nc.gz": 68199478,
    "o1.nc.gz": 89167839,
    "p1.nc.gz": 78789560,
    "q1.nc.gz": 70454645,
    "r2.nc.gz": 59159817,
    "s1.nc.gz": 61028520,
    "s2.nc.gz": 92191832,
    "s4.nc.gz": 60449978,
    "sa.nc.gz": 55141801,
    "ssa.nc.gz": 54751841,
    "t2.nc.gz": 65009788,
}

OCEAN_TIDE_FILES = {
    "2n2.nc.gz": 79421243,
    "eps2.nc.gz": 79466239,
    "j1.nc.gz": 79181150,
    "k1.nc.gz": 79165326,
    "k2.nc.gz": 79135202,
    "l2.nc.gz": 79173251,
    "la2.nc.gz": 79224898,
    "m2.nc.gz": 79281037,
    "m3.nc.gz": 80194478,
    "m4.nc.gz": 80192222,
    "m6.nc.gz": 80193774,
    "m8.nc.gz": 80194731,
    "mf.nc.gz": 72780320,
    "mks2.nc.gz": 80164550,
    "mm.nc.gz": 69732455,
    "mn4.nc.gz": 79925609,
    "ms4.nc.gz": 80055330,
    "msf.nc.gz": 77420794,
    "msqm.nc.gz": 75451767,
    "mtm.nc.gz": 73869165,
    "mu2.nc.gz": 79540681,
    "n2.nc.gz": 79375675,
    "n4.nc.gz": 80149977,
    "nu2.nc.gz": 79410976,
    "o1.nc.gz": 79253088,
    "p1.nc.gz": 79109813,
    "q1.nc.gz": 79168065,
    "r2.nc.gz": 79300001,
    "s1.nc.gz": 79115934,
    "s2.nc.gz": 79245970,
    "s4.nc.gz": 80037548,
    "sa.nc.gz": 78641422,
    "ssa.nc.gz": 66958840,
    "t2.nc.gz": 79210226,
}

# PURPOSE: compare the modification time of two files
def newer(t1: int, t2: int) -> bool:
    return (pyTMD.utilities.even(t1) <= pyTMD.utilities.even(t2))

def normalize_longitude(ds):
    """Normalize dataset longitudes from 0-360 to -180 to 180."""
    ds = ds.assign_coords(
                        {"lon": (((ds.lon + 180) % 360) - 180)}
                    )
    if ds.lon[-1] == 0:
        ds = ds.isel(lon=slice(None, -1))
    return ds.reindex({"lon": np.sort(ds.lon)})

def get_coordinates(ds, region):
    """Get coordinates within the given region, or the nearest if not found."""
    # get the lats and lons from the region
    lon, lat = np.array(region["coordinates"][0])[:, 0], np.array(region["coordinates"][0])[:, 1]
    # normalize the longitude to be between -180 to 180 because of the geojson format the region is in
    if min(lon) < 0:
        ds = normalize_longitude(ds)
    
    # Find existing coords between min&max
    # Get the lats between the min and max latitude of the region
    lats = ds.lat[np.logical_and(ds.lat >= min(lat), ds.lat <= max(lat))].values
    # if no lats are found, get the nearest lat to the region but only get the unique values
    if len(lats) == 0:
        lats = np.unique(ds.lat.sel(lat=lat, method="nearest"))
    
    lons = ds.lon[np.logical_and(ds.lon >= min(lon), ds.lon <= max(lon))].values
    if len(lons) == 0:
        lons = np.unique(ds.lon.sel(lon=lon, method="nearest"))

    return lats, lons,ds

def write_clipped_data(ds, dest_dir, filename, region_number, model_name, directory_name):
    """
    Saves the content of ds to a new netCDF file at the follow location:
    dest_dir/region{region_number}/{model_name}/{directory_name}/{filename}

    Parameters:
    ds (xarray.Dataset): The dataset to be written to a netCDF file.
    dest_dir (str): The destination directory where the netCDF file will be saved.
    filename (str): The name of the file to be saved.
    region_number (int): The region number used to create a subdirectory.
    model_name (str): The model name used to create a subdirectory.
    directory_name (str): The directory name used to create a subdirectory.
    Returns:
    None
    """
    """Write clipped dataset to a new netCDF file."""
    region_name = f"region{region_number}"
    tide_dir = os.path.join(dest_dir, region_name, model_name, directory_name)
    os.makedirs(tide_dir, exist_ok=True)
    
    destination_path = os.path.join(tide_dir, os.path.basename(filename))
    if not os.path.exists(destination_path):
        ds.to_netcdf(
            path=destination_path,
            mode="w"
        )

def clip_and_write_new_nc_files(files, geometries, dest_dir, progress_bar_name="", use_progress_bar=True, model_name="fes2014",subdirectory_name="ocean_tide"):
    """
        Clips netCDF files to specified regions and writes the clipped data to new netCDF files.

        Saves the netCDF files in the following format
        dest_dir/region{region_number}/{model_name}/{directory_name}/{filename}

        Example:
            model_name = "fes2014"
            subdirectory_name = "ocean_tide"

            dest_dir/region0/fes2014/ocean_tide/filename.nc
            dest_dir/region/fes2014/ocean_tide/filename.nc
            dest_dir/region1/fes2014/load_tide/filename.nc

        Parameters:
            files (list of str): List of file paths to the netCDF files to be clipped.
            geometries (list of dict): List of geometries defining the regions to clip to.
            dest_dir (str): Destination directory where the new clipped netCDF files will be saved.
            progress_bar_name (str, optional): Name to display on the progress bar. Default is an empty string.
            use_progress_bar (bool, optional): Whether to display a progress bar. Default is True.
            model_name (str, optional): Name of the model used for naming the output files. Default is "fes2014".
            directory_name (str, optional): Name of the subdirectory to save the output files. Default is "ocean_tide".

        Returns:
        None

    """
    progress_description = f"Clipping {progress_bar_name} files"

    # Iterate through each file with a progress bar if enabled
    file_iterator = tqdm(files, desc=progress_description, disable=not use_progress_bar)
    for current_file in file_iterator:
        ds_disk = xr.open_dataset(current_file, engine="netcdf4")
        file_iterator.set_description(f"Processing {os.path.basename(current_file)}")

        # Iterate through each region
        for region_number, region in enumerate(geometries):
            lats, lons,ds = get_coordinates(ds_disk.copy(deep=True), region)
            output = ds.sel(lat=lats, lon=lons)
            output.attrs = ds.attrs
            for var in output.data_vars:
                output[var].attrs = ds[var].attrs

            write_clipped_data(output, dest_dir, current_file, region_number, model_name,subdirectory_name)


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


def unzip_gzip_files(directory, use_progress_bar: bool = True):
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
    gzipped_files = [f for f in os.listdir(directory) if f.endswith(".gz")]
    if not gzipped_files:
        return
    with progress_bar_context(
        use_progress_bar,
        total=len(gzipped_files),
        description=f"Unzipping {len(gzipped_files)} {os.path.basename(directory)}",
    ) as update:
        for filename in gzipped_files:
            update(f"Unzipped {filename}")
            # Construct full file path
            file_path = os.path.join(directory, filename)
            # Construct output file path (same as input but without .gz)
            output_path = file_path[:-3]
            # Open the .gz file and the output file
            with gzip.open(file_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    # Copy the contents of the .gz file to the output file
                    shutil.copyfileobj(f_in, f_out)
            # Remove the .gz file
            os.remove(file_path)


def create_region_directories(base_dir, region_names,model_name='fes2014'):
    """
    Creates a series of directories for each region and creates subdirectories for each tide model in each region.

    base_dir
    |__ region1
    |   |__ model_name
    |       |__ load_tide
    |       |__ ocean_tide
    |__ region2
    |   |__ model_name
    |       |__ load_tide
    |       |__ ocean_tide


    Args:
        base_dir (str): The base directory path.
        region_names (list): A list of region names.
        model_name (str): The name of the model. Defaults to 'fes2014'.


    Returns:
        None

    """
    for region_name in region_names:
        region_dir = os.path.join(base_dir, region_name)
        model_dir = os.path.join(region_dir, model_name)
        load_tide_dir = os.path.join(model_dir, "load_tide")
        ocean_tide_dir = os.path.join(model_dir, "ocean_tide")

        os.makedirs(region_dir, exist_ok=True)  # Create region directory
        os.makedirs (model_dir, exist_ok=True)  # Create 'fes2014' directory
        os.makedirs(load_tide_dir, exist_ok=True)  # Create 'load_tide' directory
        os.makedirs(ocean_tide_dir, exist_ok=True)  # Create 'ocean_tide' directory


def check_files(directory_path: str, files_dict: dict) -> list:
    """
    Checks if each file from the given dictionary exists in the directory and compares its size with the corresponding
    value in the dictionary. Returns a list of names of files that either don't exist or have a different size.

    Args:
        directory_path (str): The path to the directory.
        files_dict (dict): A dictionary containing the filenames as keys and sizes as values.

    Returns:
        list: A list of names of files that either don't exist or have a different size.

    """
    missing_files = []
    for file, size in files_dict.items():
        file_path = os.path.join(directory_path, file)
        if not os.path.exists(file_path) or os.path.getsize(file_path) != size:
            missing_files.append(file)
    return missing_files


# PURPOSE: pull file from a remote ftp server and decompress if tar file
def ftp_download(logger, ftp, remote_path, local_dir,
        LZMA=True,
        TARMODE=False,
        FLATTEN=False,
        GZIP=False,
        CHUNK=8192,
        MODE=0o775
    ):
    # remote and local directory for data product
    remote_file = posixpath.join('auxiliary','tide_model',*remote_path)
    # if compressing the output file
    opener = gzip.open if GZIP else open
    # Printing files transferred
    remote_ftp_url = posixpath.join('ftp://', ftp.host, remote_file)
    logger.info(f'{remote_ftp_url} -->')
    if TARMODE:
        # copy remote file contents to bytesIO object
        fileobj = io.BytesIO()
        ftp.retrbinary(f'RETR {remote_file}', fileobj.write, blocksize=CHUNK)
        fileobj.seek(0)
        # open the tar file
        tar = tarfile.open(name=remote_path[-1], fileobj=fileobj, mode=TARMODE)
        # read tar file and extract all files
        member_files = [m for m in tar.getmembers() if tarfile.TarInfo.isfile(m)]
        for m in member_files:
            member = posixpath.basename(m.name) if FLATTEN else m.name
            base, sfx = posixpath.splitext(m.name)
            # extract file contents to new file
            output = f'{member}.gz' if sfx in ('.asc','.nc') and GZIP else member
            local_file = local_dir.joinpath(*posixpath.split(output))
            # check if the local file exists
            if local_file.exists():
                # check the modification time of the local file
                # if remote file is newer: overwrite the local file
                continue
            # print the file being transferred
            logger.info(f'\t{str(local_file)}')
            # recursively create output directory if non-existent
            local_file.parent.mkdir(mode=MODE, parents=True, exist_ok=True)
            # extract file to local directory
            with tar.extractfile(m) as fi,opener(local_file, 'wb') as fo:
                shutil.copyfileobj(fi, fo)
            # get last modified date of remote file within tar file
            # keep remote modification time of file and local access time
            os.utime(local_file, (local_file.stat().st_atime, m.mtime))
            local_file.chmod(mode=MODE)
    elif LZMA:
        # get last modified date of remote file and convert into unix time
        mdtm = ftp.sendcmd(f'MDTM {remote_file}')
        mtime = calendar.timegm(time.strptime(mdtm[4:],"%Y%m%d%H%M%S"))
        # output file name for compressed and uncompressed cases
        stem = posixpath.basename(posixpath.splitext(remote_file)[0])
        base, sfx = posixpath.splitext(stem)
        # extract file contents to new file
        output = f'{stem}.gz' if sfx in ('.asc','.nc') and GZIP else stem
        local_file = local_dir.joinpath(output)
        # check if the local file exists
        if local_file.exists() and newer(mtime,local_file.stat().st_mtime):
            # check the modification time of the local file
            # if remote file is newer: overwrite the local file
            return
        # print the file being transferred
        logger.info(f'\t{str(local_file)}')
        # recursively create output directory if non-existent
        local_file.parent.mkdir(mode=MODE, parents=True, exist_ok=True)
        # copy remote file contents to bytesIO object
        fileobj = io.BytesIO()
        ftp.retrbinary(f'RETR {remote_file}', fileobj.write, blocksize=CHUNK)
        fileobj.seek(0)
        # decompress lzma file and extract contents to local directory
        with lzma.open(fileobj) as fi,opener(local_file, 'wb') as fo:
            shutil.copyfileobj(fi, fo)
        # get last modified date of remote file within tar file
        # keep remote modification time of file and local access time
        os.utime(local_file, (local_file.stat().st_atime, mtime))
        local_file.chmod(mode=MODE)
    else:
        # copy readme and uncompressed files directly
        stem = posixpath.basename(remote_file)
        base, sfx = posixpath.splitext(stem)
        # output file name for compressed and uncompressed cases
        output = f'{stem}.gz' if sfx in ('.asc','.nc') and GZIP else stem
        local_file = local_dir.joinpath(output)
        # get last modified date of remote file and convert into unix time
        mdtm = ftp.sendcmd(f'MDTM {remote_file}')
        mtime = calendar.timegm(time.strptime(mdtm[4:],"%Y%m%d%H%M%S"))
        # check if the local file exists
        if local_file.exists() and newer(mtime, local_file.stat().st_mtime):
            # check the modification time of the local file
            # if remote file is newer: overwrite the local file
            return
        # print the file being transferred
        logger.info(f'\t{str(local_file)}\n')
        # recursively create output directory if non-existent
        local_file.parent.mkdir(mode=MODE, parents=True, exist_ok=True)
        # copy remote file contents to local file
        with opener(local_file, 'wb') as f:
            ftp.retrbinary(f'RETR {remote_file}', f.write, blocksize=CHUNK)
        # keep remote modification time of file and local access time
        pathlib.os.utime(local_file, (local_file.stat().st_atime, mtime))
        local_file.chmod(mode=MODE)

# PURPOSE: download local AVISO FES files with ftp server
# by downloading tar files and extracting contents
def aviso_fes_tar(MODEL, f, logger,
        DIRECTORY: str | pathlib.Path | None = None,
        LOAD: bool = False,
        CURRENTS: bool = False,
        GZIP: bool = False,
        MODE: int = 0o775,
        use_progress_bar: bool = True,
        LOG=True,
        LOGFILE = "AVISO_FES_tides.log"
    ):
    # check if local directory exists and recursively create if not
    localpath = os.path.join(DIRECTORY, MODEL.lower())
    os.makedirs(localpath, MODE) if not os.path.exists(localpath) else None

    # path to remote directory for FES
    FES = {}
    # mode for reading tar files
    TAR = {}
    # flatten file structure
    FLATTEN = {}

    # 1999 model
    FES['FES1999']=[]
    FES['FES1999'].append(['fes1999_fes2004','readme_fes1999.html'])
    FES['FES1999'].append(['fes1999_fes2004','fes1999.tar.gz'])
    TAR['FES1999'] = [None,'r:gz']
    FLATTEN['FES1999'] = [None,True]
    # 2004 model
    FES['FES2004']=[]
    FES['FES2004'].append(['fes1999_fes2004','readme_fes2004.html'])
    FES['FES2004'].append(['fes1999_fes2004','fes2004.tar.gz'])
    TAR['FES2004'] = [None,'r:gz']
    FLATTEN['FES2004'] = [None,True]
    # 2012 model
    FES['FES2012']=[]
    FES['FES2012'].append(['fes2012_heights','readme_fes2012_heights_v1.1'])
    FES['FES2012'].append(['fes2012_heights','fes2012_heights_v1.1.tar.lzma'])
    TAR['FES2012'] = []
    TAR['FES2012'].extend([None,'r:xz'])
    FLATTEN['FES2012'] = []
    FLATTEN['FES2012'].extend([None,True])
    if CURRENTS:
        subdir = 'fes2012_currents'
        FES['FES2012'].append([subdir,'readme_fes2012_currents_v1.1'])
        FES['FES2012'].append([subdir,'fes2012_currents_v1.1_block1.tar.lzma'])
        FES['FES2012'].append([subdir,'fes2012_currents_v1.1_block2.tar.lzma'])
        FES['FES2012'].append([subdir,'fes2012_currents_v1.1_block3.tar.lzma'])
        FES['FES2012'].append([subdir,'fes2012_currents_v1.1_block4.tar.lzma'])
        TAR['FES2012'].extend([None,'r:xz','r:xz','r:xz','r:xz'])
        FLATTEN['FES2012'].extend([None,False,False,False,False])
    # 2014 model
    FES['FES2014']=[]
    FES['FES2014'].append(['fes2014_elevations_and_load',
        'readme_fes2014_elevation_and_load_v1.2.txt'])
    FES['FES2014'].append(['fes2014_elevations_and_load',
        'fes2014b_elevations','ocean_tide.tar.xz'])
    TAR['FES2014'] = []
    TAR['FES2014'].extend([None,'r'])
    FLATTEN['FES2014'] = []
    FLATTEN['FES2014'].extend([None,False])
    if LOAD:
        FES['FES2014'].append(['fes2014_elevations_and_load',
            'fes2014a_loadtide','load_tide.tar.xz'])
        TAR['FES2014'].extend(['r'])
        FLATTEN['FES2014'].extend([False])
    if CURRENTS:
        subdir = 'fes2014a_currents'
        FES['FES2014'].append([subdir,'readme_fes2014_currents_v1.2.txt'])
        FES['FES2014'].append([subdir,'eastward_velocity.tar.xz'])
        FES['FES2014'].append([subdir,'northward_velocity.tar.xz'])
        TAR['FES2014'].extend(['r'])
        FLATTEN['FES2014'].extend([False])

    num_files_to_process = len(FES[MODEL])
    with progress_bar_context(
        use_progress_bar,
        total=num_files_to_process,
        description=f"Downloading files from AVISO FTP",
    ) as update:
        for remotepath, tarmode, flatten in zip(FES[MODEL], TAR[MODEL], FLATTEN[MODEL]):
            # download file from ftp and decompress tar files
            update(f"Downloading files...{localpath}\n{remotepath}", 0)
            ftp_download_file(
                logger, f, remotepath, localpath, tarmode, flatten, GZIP, MODE
            )
            update(f"Downloading files...{localpath}\n{remotepath}", 1)

        # close the ftp connection
        f.quit()
        # close log file and set permissions level to MODE
        if LOG:
            os.chmod(os.path.join(DIRECTORY, LOGFILE), MODE)

# PURPOSE: download local AVISO FES files with ftp server
# by downloading individual files
def aviso_fes_list(MODEL, f, logger,
        DIRECTORY: str | pathlib.Path | None = None,
        LOAD: bool = False,
        CURRENTS: bool = False,
        EXTRAPOLATED: bool = False,
        GZIP: bool = False,
        MODE: oct = 0o775
    ):
    # validate local directory
    DIRECTORY = pathlib.Path(DIRECTORY).expanduser().absolute()

    # path to remote directory for FES
    FES = {}
    # 2022 model
    FES['FES2022'] = []
    FES['FES2022'].append(['fes2022b','ocean_tide_20241025'])
    if LOAD:
        FES['FES2022'].append(['fes2022b','load_tide'])
    if EXTRAPOLATED:
        FES['FES2022'].append(['fes2022b','ocean_tide_extrapolated'])

    # for each model file type
    for subdir in FES[MODEL]:
        local_dir = DIRECTORY.joinpath(*subdir)
        file_list = ftp_list(f, subdir, basename=True, sort=True)
        for fi in tqdm(file_list, desc=f"Downloading {MODEL} files"):
            remote_path = [*subdir, fi]
            LZMA = fi.endswith('.xz')
            ftp_download(logger, f, remote_path, local_dir,
                LZMA=LZMA,
                GZIP=GZIP,
                CHUNK=32768,
                MODE=MODE
            )

# PURPOSE: List a directory on a ftp host
def ftp_list(ftp, remote_path, basename=False, pattern=None, sort=False):
    # list remote path
    output = ftp.nlst(posixpath.join('auxiliary','tide_model',*remote_path))
    # reduce to basenames
    if basename:
        output = [posixpath.basename(i) for i in output]
    # reduce using regular expression pattern
    if pattern:
        i = [i for i,f in enumerate(output) if re.search(pattern,f)]
        # reduce list of listed items
        output = [output[indice] for indice in i]
    # sort the list
    if sort:
        i = [i for i,j in sorted(enumerate(output), key=lambda i: i[1])]
        # sort list of listed items
        output = [output[indice] for indice in i]
    # return the list of items
    return output


# PURPOSE: download local AVISO FES files with ftp server
def aviso_fes_tides(
    MODEL,
    DIRECTORY=None,
    USER="",
    PASSWORD="",
    LOAD=False,
    CURRENTS=False,
    GZIP=False,
    LOG=True,
    MODE=None,
    use_progress_bar: bool = True,
    EXTRAPOLATED: bool = False,
):
    
    # connect and login to AVISO ftp server
    f = ftplib.FTP("ftp-access.aviso.altimetry.fr", timeout=1000)
    f.login(USER, PASSWORD)
    # check if local directory exists and recursively create if not
    # localpath = os.path.join(DIRECTORY, MODEL.lower())
    # os.makedirs(localpath, MODE) if not os.path.exists(localpath) else None
    # create log file with list of downloaded files (or print to terminal)
    if LOG:
        # format: AVISO_FES_tides_2002-04-01.log
        today = time.strftime("%Y-%m-%d", time.localtime())
        LOGFILE = f"AVISO_FES_tides_{today}.log"
        os.makedirs(DIRECTORY, exist_ok=True)
        fid = open(os.path.join(DIRECTORY, LOGFILE), mode="w", encoding="utf8")
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # logger = pyTMD.utilities.build_logger(__name__, stream=fid, level=logging.INFO)
        logger.propagate = False  # Ensure log messages don't propagate to root logger
        file_handler = logging.FileHandler(fid.name)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"AVISO FES Sync Log ({today})")
        logger.info(f"\tMODEL: {MODEL}")
    else:
        # standard output (terminal output)
        logger = pyTMD.utilities.build_logger(__name__, level=logging.INFO)

    MODEL = MODEL.upper()

    # download the FES tide model files
    if MODEL.upper() in ('FES1999','FES2004','FES2012','FES2014'):
        aviso_fes_tar(MODEL, f, logger,
            DIRECTORY=DIRECTORY,
            LOAD=LOAD,
            CURRENTS=CURRENTS,
            GZIP=GZIP,
            MODE=MODE,
            LOGFILE = LOGFILE)
    elif MODEL.upper() in ('FES2022',):
        aviso_fes_list(MODEL, f, logger,
            DIRECTORY=DIRECTORY,
            LOAD=LOAD,
            CURRENTS=CURRENTS,
            EXTRAPOLATED=EXTRAPOLATED,
            GZIP=GZIP,
            MODE=MODE)
    else:
        print(f"Model name provided not recogized as any of the available models : ['FES1999','FES2004','FES2012','FES2014','FES2022']")

def retrieve_file_size(ftp: ftplib.FTP, remote_file: str):
    """
    Retrieve the size of a file on an FTP server.

    Parameters:
    - ftp (ftplib.FTP): An active FTP connection.
    - remote_file (str): Path to the file on the FTP server.

    Returns:
    - int or None: Size of the file in bytes or None if there's an error.
    """
    try:
        ftp.voidcmd("TYPE I")  # Switch to binary mode
        response = ftp.sendcmd(f"SIZE {remote_file}")
        return int(response.split()[1])
    except ftplib.error_perm as e:
        print(f"Could not get size of {remote_file}: {e}")
        return None


def get_missing_files(local_file, local_dir, OCEAN_TIDE_FILES, LOAD_TIDE_FILES):
    """
    Determine the missing files based on the filename and check them in the respective directories.

    Parameters:
    - local_file (str): Path to the local file.
    - local_dir (str): Base directory where the files are located.
    - OCEAN_TIDE_FILES (list): List of expected ocean tide files.
    - LOAD_TIDE_FILES (list): List of expected load tide files.

    Returns:
    - list: A list of missing files. If no files are missing, returns an empty list.
    """
    missing_files = []
    filename = os.path.basename(local_file)

    if "ocean_tide" in filename:
        ocean_tide_dir = os.path.join(local_dir, "ocean_tide")
        missing_files = check_files(ocean_tide_dir, OCEAN_TIDE_FILES)
    elif "load_tide" in filename:
        load_tide_dir = os.path.join(local_dir, "load_tide")
        missing_files = check_files(load_tide_dir, LOAD_TIDE_FILES)

    return missing_files


def download_with_progress_bar(ftp: ftplib.FTP, remote_file: str, local_file: str):
    """
    Download a file from an FTP server with a progress bar.

    Parameters:
    - ftp (ftplib.FTP): An active FTP connection.
    - remote_file (str): Path to the file on the FTP server.
    - local_file (str): Path where the file should be saved locally.

    Note:
    - The function uses tqdm to display a progress bar.
    """
    file_size = retrieve_file_size(ftp, remote_file)
    if not file_size:
        return

    def callback(data):
        pbar.update(len(data))
        # update(f"Downloading File {remote_file}", len(data))
        fileobj.write(data)

    fileobj = io.BytesIO()

    with tqdm(
        total=file_size, desc=f"Downloading {remote_file}", unit="B", unit_scale=True
    ) as pbar:
        fileobj = io.BytesIO()
        ftp.retrbinary(f"RETR {remote_file}", callback)

    fileobj.seek(0)
    return fileobj



def extract_tar_with_progress_bar(
    fileobj, local_dir, missing_files, GZIP, MODE, tarmode, flatten, logger
):
    """
    Extract files from a tar object with a progress bar.

    Parameters:
    - fileobj (io.BytesIO): A file-like object containing the tar data.
    - local_dir (str): Directory where the extracted files should be saved.
    - missing_files (list): List of filenames that need to be extracted.
    - GZIP (bool): Whether to gzip-compress the extracted files.
    - MODE (int): Unix file permissions for the extracted files.
    - tarmode (_type_): The mode in which the tar files should be processed, if applicable
    - flatten (bool): A boolean indicating whether the downloaded files should be flattened or not
    - logger (logging object): logger: A logging object used to log events and messages during the process
    Note:
    - The function uses tqdm to display a progress bar.
    """
    tar = tarfile.open(fileobj=fileobj, mode=tarmode)
    member_files = [m for m in tar.getmembers() if tarfile.TarInfo.isfile(m)]
    total_size = sum(m.size for m in member_files)

    with progress_bar_context(
        True,
        total=total_size,
        description=f"Extracting Tar",
        unit="B",
        unit_scale=True,
    ) as update:
        # with tqdm(total=total_size, unit="B", unit_scale=True, ncols=100) as pbar:
        for m in member_files:
            logger.info(f"m : {m}")
            member = posixpath.basename(m.name) if flatten else m.name
            fileBasename, fileExtension = posixpath.splitext(posixpath.basename(m.name))
            logger.info(f"fileBasename : {fileBasename}")
            if any(fileBasename in file for file in missing_files):
                # extract file contents to new file
                if fileExtension in (".asc", ".nc") and GZIP:
                    local_file = os.path.join(
                        local_dir, *posixpath.split(f"{member}.gz")
                    )
                    logger.info(f"\t{local_file}")
                    # recursively create output directory if non-existent
                    if not os.access(os.path.dirname(local_file), os.F_OK):
                        os.makedirs(os.path.dirname(local_file), MODE)
                    # extract file to compressed gzip format in local directory
                    with tar.extractfile(m) as fi, gzip.open(local_file, "wb") as fo:
                        shutil.copyfileobj(fi, fo)
                else:
                    local_file = os.path.join(local_dir, *posixpath.split(member))
                    logger.info(f"\t{local_file}")
                    # recursively create output directory if non-existent
                    if not os.access(os.path.dirname(local_file), os.F_OK):
                        os.makedirs(os.path.dirname(local_file), MODE)
                    # extract file to local directory
                    with tar.extractfile(m) as fi, open(local_file, "wb") as fo:
                        shutil.copyfileobj(fi, fo)
                # Update the progress bar
                update(f"Extracting {m}", m.size)
                # get last modified date of remote file within tar file
                # keep remote modification time of file and local access time
                os.utime(local_file, (os.stat(local_file).st_atime, m.mtime))
                os.chmod(local_file, MODE)


# PURPOSE: pull file from a remote ftp server and decompress if tar file
def ftp_download_file(
    logger, ftp, remote_path, local_dir, tarmode, flatten, GZIP, MODE
):
    """
    The function performs the following tasks:
    1. Retrieves the remote file paths and constructs an FTP URL, logging this information.
    2. If tar mode is specified: a. Reads binary data from the remote file and stores it in a bytesIO object. b. Opens the tar file and reads its content. c. Extracts all files in the tar file. d. Depending on the file extension and whether GZIP compression is enabled or not, the files are either compressed with gzip or simply copied to the local directory. e. Stores the last modified date of the remote file and applies it to the local file while retaining the local access time.
    3. If tar mode is not specified, the function copies readme and uncompressed files directly to the local directory, keeping the remote modification time and local access time.

    Args:
        logger (logging object): logger: A logging object used to log events and messages during the process
        ftp (_type_): A connection to an FTP server to download the files
        remote_path (_type_): The path where the files are located on the FTP server
        local_dir (str): The path where the files should be downloaded on the local machine
        tarmode (_type_): The mode in which the tar files should be processed, if applicable
        flatten (bool): A boolean indicating whether the downloaded files should be flattened or not
        GZIP (bool): A boolean indicating whether gzip compression should be applied to the downloaded files
        MODE (_type_): Unix file permissions for downloaded files

    """
    try:
        remote_file = posixpath.join("auxiliary", "tide_model", *remote_path)

        # check if local file exists
        local_file = os.path.join(local_dir, remote_path[-1])
        if os.path.exists(local_file):
            logger.info(f"File {local_file} already exists, skipping download.")
            return

        missing_files = get_missing_files(
            local_file, local_dir, OCEAN_TIDE_FILES, LOAD_TIDE_FILES
        )
        if not missing_files:
            print(
                f"All the required files for {local_file} have already been downloaded"
            )
            return

        logger.info(f"missing_files : {missing_files}")
        # otherwise proceed with the download
        logger.info(f"remote_file: {remote_file} --> local_file: {local_file}")

        # Printing files transferred
        remote_ftp_url = posixpath.join("ftp://", ftp.host, remote_file)
        logger.info(f"remote_ftp_url: {remote_ftp_url} -->")
        if tarmode:
            logger.info(f"tarmode is {tarmode}")
            # copy remote file contents to bytesIO object
            # fileobj = io.BytesIO()
            fileobj = download_with_progress_bar(ftp, remote_file, local_file)
            # fileobj.seek(0)
            extract_tar_with_progress_bar(
                fileobj, local_dir, missing_files, GZIP, MODE, tarmode, flatten, logger
            )
        else:
            logger.info(f"not tarmode")
            # copy readme and uncompressed files directly
            local_file = os.path.join(local_dir, remote_path[-1])
            logger.info(f"\t local_file: {local_file}\n")
            # Download the file with a progress bar
            download_with_progress_bar(ftp, remote_file, local_file)

            # get last modified date of remote file and convert into unix time
            mdtm = ftp.sendcmd(f"MDTM {remote_file}")
            remote_mtime = calendar.timegm(time.strptime(mdtm[4:], "%Y%m%d%H%M%S"))
            # Set the modification time of the local file to match the remote file
            os.utime(local_file, (os.stat(local_file).st_atime, remote_mtime))
            # Set the file permissions for the local file
            os.chmod(local_file, MODE)
    except Exception as e:
        print(f"{e} \n{traceback.format_exc()}")


def download_fes_tides(
    user="",
    password="",
    directory=os.path.join(os.path.abspath(core_utilities.get_base_dir()), "tide_model"),
    tide=["FES2014"],
    load=True,
    currents=False,
    gzip=True,
    log=True,
    mode=0o775,
):
    """
    Downloads FES tide models from the AVISO FTP server and saves the tide model files to the specified directory under the tide model name.

    Parameters:
        user (str): Username for FTP server authentication.
        password (str): Password for FTP server authentication.
        directory (str): Directory where the tide models will be downloaded.
        tide (list): List of tide model names to download. Default is ["FES2014"].
        load (bool): If True, load the tide model after downloading. Default is True.
        currents (bool): If True, download tidal currents. Default is False.
        gzip (bool): If True, download gzip compressed files. Default is True.
        log (bool): If True, log the download process. Default is True.
        mode (int): Permissions mode for the downloaded files. Default is 0o775.
    Returns:
    None
    """
    # AVISO FTP Server hostname
    HOST = "ftp-access.aviso.altimetry.fr"
    # Step 1: Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    # check internet connection before attempting to run program
    if pyTMD.utilities.check_ftp_connection(HOST, user, password):
        for tide_model_name in tide:
            aviso_fes_tides(
                tide_model_name,
                DIRECTORY=directory,
                USER=user,
                PASSWORD=password,
                LOAD=load,
                CURRENTS=currents,
                GZIP=gzip,
                LOG=log,
                MODE=mode,
            )
        print(f"Download the {tide_model_name} to {directory}")


def clip_model_to_regions(
    tide_model_directory: str = os.path.join(os.path.abspath(core_utilities.get_base_dir()), "tide_model"),
    regions_file: str = "",
    MODEL:str = "FES2014",
    use_progess_bar: bool = True,
):
    """
    Clips the selected tide model to each region found in the regions files to create a smaller clipped version of the tide model for each region.
    Each smaller tide model gets stored in its own directory within the region directory.

    Available options for the MODEL parameter are "FES2014" and "FES2022".

    Example
        MODEL = "FES2014"

        tide_model_directory
        |__ region0
        |   |__ fes2014
        |       |__ load_tide
        |           |__ filename.nc  # clipped portion of tide model for region0
        |       |__ ocean_tide
        |__ region1
        |   |__ fes2014
        |       |__ load_tide
        |       |__ ocean_tide


    Parameters:
        tide_model_directory (str): The directory where the tide model data is stored. Defaults to a subdirectory named "tide_model" in the base directory.
        regions_file (str): The file path to the regions GeoJSON file. If not provided, the default internal tide regions map will be used.
        MODEL (str): The tide model to use. Supported models are "FES2014" and "FES2022". Defaults to "FES2014".
        use_progess_bar (bool): Whether to display a progress bar during the operation. Defaults to True.
    Raises:
    Exception: If the tide model directory does not exist or if the specified model is not supported.
    Returns:
    None
    """
    with progress_bar_context(
        use_progess_bar,
        total=6,
        description=f"Clipping the Tide Model",
    ) as update:
        if not os.path.exists(tide_model_directory) and os.path.isdir(
            tide_model_directory
        ):
            raise Exception(
                f"The tide model directory does not exist at {tide_model_directory}.\n{traceback.format_exc()}"
            )

        # use file_utilities load_package_resource to load the tide regions geojson from coastseg's internal data
        if not regions_file:
            regions_file = load_package_resource(
                "tide_model", "tide_regions_map.geojson"
            )

        model_name = MODEL.lower()

        if MODEL.upper() == "FES2014":
            # create paths to tide models
            fes2014_model_directory = os.path.join(tide_model_directory, "fes2014")
            load_tide_dir = os.path.join(fes2014_model_directory, "load_tide")
            ocean_tide_dir = os.path.join(fes2014_model_directory, "ocean_tide")
        elif MODEL.upper() == "FES2022":
            # create paths to tide models
            fes2022_model_directory = os.path.join(tide_model_directory, "fes2022b")
            ocean_tide_dir = os.path.join(fes2022_model_directory, "ocean_tide_20241025")
            load_tide_dir = os.path.join(fes2022_model_directory, "load_tide")
            model_name = 'fes2022b'
        else:
            raise Exception(
                f"Model {MODEL} is not supported.\n{traceback.format_exc()}"
            )

        # load geometries from regions file
        geometries = get_geometries_from_file(regions_file)
        update("Clipping the Tide Model: Unzipping load tide files")

        # unzip the compressed files in each directory
        unzip_gzip_files(load_tide_dir)
        update("Clipping the Tide Model: Unzipping ocean tide files")
        unzip_gzip_files(ocean_tide_dir)

        # create a list of all the nc files in both the load and ocean tide directories
        load_tide_nc_files = [
            f for f in glob(os.path.join(load_tide_dir, "*.nc")) if "clipped" not in f
        ]
        ocean_tide_nc_files = [
            f for f in glob(os.path.join(ocean_tide_dir, "*.nc")) if "clipped" not in f
        ]
        update("Clipping the Tide Model: Creating region directories")
        # create region directory structure and move files
        region_names = [f"region{i}" for i in range(0, len(geometries))]
        create_region_directories(tide_model_directory, region_names,model_name=model_name)
        update("Clipping the Tide Model: Clipping the ocean tide files")
        # create the new clipped files in the region directory
        clip_and_write_new_nc_files(
            ocean_tide_nc_files, geometries, tide_model_directory, "ocean",model_name=model_name,subdirectory_name='ocean_tide'
        )
        update("Clipping the Tide Model: Clipping the load tide files")
        clip_and_write_new_nc_files(
            load_tide_nc_files, geometries, tide_model_directory, "load",model_name=model_name,subdirectory_name='load_tide'
        )
        update("Clipping the Tide Model: Finished")
