#!/usr/bin/env python
"""
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
from __future__ import print_function
from tqdm import tqdm
import sys
import os
import io
import gzip
import netrc
import shutil
import logging
import tarfile
import getpass
import argparse
import builtins
import posixpath
import calendar, time
import ftplib
import pyTMD.utilities

# FILE SIZES for files in these files
load_tide_files = {
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

ocean_tide_files = {
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


# PURPOSE: download local AVISO FES files with ftp server
def aviso_fes_tides(
    MODEL,
    DIRECTORY=None,
    USER="",
    PASSWORD="",
    LOAD=False,
    CURRENTS=False,
    GZIP=False,
    LOG=False,
    MODE=None,
):
    # connect and login to AVISO ftp server
    f = ftplib.FTP("ftp-access.aviso.altimetry.fr", timeout=1000)
    f.login(USER, PASSWORD)
    # check if local directory exists and recursively create if not
    localpath = os.path.join(DIRECTORY, MODEL.lower())
    os.makedirs(localpath, MODE) if not os.path.exists(localpath) else None

    # create log file with list of downloaded files (or print to terminal)
    if LOG:
        # format: AVISO_FES_tides_2002-04-01.log
        today = time.strftime("%Y-%m-%d", time.localtime())
        LOGFILE = f"AVISO_FES_tides_{today}.log"
        fid = open(os.path.join(DIRECTORY, LOGFILE), mode="w", encoding="utf8")
        logger = pyTMD.utilities.build_logger(__name__, stream=fid, level=logging.INFO)
        logger.info(f"AVISO FES Sync Log ({today})")
        logger.info(f"\tMODEL: {MODEL}")
    else:
        # standard output (terminal output)
        logger = pyTMD.utilities.build_logger(__name__, level=logging.INFO)

    # path to remote directory for FES
    FES = {}
    # mode for reading tar files
    TAR = {}
    # flatten file structure
    FLATTEN = {}

    # 1999 model
    FES["FES1999"] = []
    FES["FES1999"].append(["fes1999_fes2004", "readme_fes1999.html"])
    FES["FES1999"].append(["fes1999_fes2004", "fes1999.tar.gz"])
    TAR["FES1999"] = [None, "r:gz"]
    FLATTEN["FES1999"] = [None, True]
    # 2004 model
    FES["FES2004"] = []
    FES["FES2004"].append(["fes1999_fes2004", "readme_fes2004.html"])
    FES["FES2004"].append(["fes1999_fes2004", "fes2004.tar.gz"])
    TAR["FES2004"] = [None, "r:gz"]
    FLATTEN["FES2004"] = [None, True]
    # 2012 model
    FES["FES2012"] = []
    FES["FES2012"].append(["fes2012_heights", "readme_fes2012_heights_v1.1"])
    FES["FES2012"].append(["fes2012_heights", "fes2012_heights_v1.1.tar.lzma"])
    TAR["FES2012"] = []
    TAR["FES2012"].extend([None, "r:xz"])
    FLATTEN["FES2012"] = []
    FLATTEN["FES2012"].extend([None, True])
    if CURRENTS:
        subdir = "fes2012_currents"
        FES["FES2012"].append([subdir, "readme_fes2012_currents_v1.1"])
        FES["FES2012"].append([subdir, "fes2012_currents_v1.1_block1.tar.lzma"])
        FES["FES2012"].append([subdir, "fes2012_currents_v1.1_block2.tar.lzma"])
        FES["FES2012"].append([subdir, "fes2012_currents_v1.1_block3.tar.lzma"])
        FES["FES2012"].append([subdir, "fes2012_currents_v1.1_block4.tar.lzma"])
        TAR["FES2012"].extend([None, "r:xz", "r:xz", "r:xz", "r:xz"])
        FLATTEN["FES2012"].extend([None, False, False, False, False])
    # 2014 model
    FES["FES2014"] = []
    FES["FES2014"].append(
        ["fes2014_elevations_and_load", "readme_fes2014_elevation_and_load_v1.2.txt"]
    )
    FES["FES2014"].append(
        ["fes2014_elevations_and_load", "fes2014b_elevations", "ocean_tide.tar.xz"]
    )
    TAR["FES2014"] = []
    TAR["FES2014"].extend([None, "r"])
    FLATTEN["FES2014"] = []
    FLATTEN["FES2014"].extend([None, False])
    if LOAD:
        FES["FES2014"].append(
            ["fes2014_elevations_and_load", "fes2014a_loadtide", "load_tide.tar.xz"]
        )
        TAR["FES2014"].extend(["r"])
        FLATTEN["FES2014"].extend([False])
    if CURRENTS:
        subdir = "fes2014a_currents"
        FES["FES2014"].append([subdir, "readme_fes2014_currents_v1.2.txt"])
        FES["FES2014"].append([subdir, "eastward_velocity.tar.xz"])
        FES["FES2014"].append([subdir, "northward_velocity.tar.xz"])
        TAR["FES2014"].extend(["r"])
        FLATTEN["FES2014"].extend([False])

    # for each file for a model
    for remotepath, tarmode, flatten in zip(FES[MODEL], TAR[MODEL], FLATTEN[MODEL]):
        # download file from ftp and decompress tar files
        print(f"Downloading files...{localpath}\n{remotepath}")

        ftp_download_file(
            logger, f, remotepath, localpath, tarmode, flatten, GZIP, MODE
        )
    # close the ftp connection
    f.quit()
    # close log file and set permissions level to MODE
    if LOG:
        os.chmod(os.path.join(DIRECTORY, LOGFILE), MODE)


# PURPOSE: pull file from a remote ftp server and decompress if tar file
def ftp_download_file(
    logger, ftp, remote_path, local_dir, tarmode, flatten, GZIP, MODE
):
    """
    The function performs the following tasks:
    1. Retrieves the remote file paths and constructs an FTP URL, logging this information.
    2. If tar mode is specified: a. Reads binary data from the remote file and stores it in a bytesIO object. b. Opens the tar file and reads its content. c. Extracts all files in the tar file. d. Depending on the file extension and whether GZIP compression is enabled or not, the files are either compressed with gzip or simply copied to the local directory. e. Stores the last modified date of the remote file and applies it to the local file while retaining the local access time.
    3.If tar mode is not specified, the function copies readme and uncompressed files directly to the local directory, keeping the remote modification time and local access time.

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
    remote_file = posixpath.join("auxiliary", "tide_model", *remote_path)

    # check if local file exists
    local_file = os.path.join(local_dir, remote_path[-1])
    if os.path.exists(local_file):
        logger.info(f"File {local_file} already exists, skipping download.")
        return

    missing_files = []
    filename = os.path.basename(local_file)
    if "ocean_tide" in filename:
        ocean_tide_dir = os.path.join(local_dir, "ocean_tide")
        missing_files = check_files(ocean_tide_dir, ocean_tide_files)
    if "load_tide" in filename:
        load_tide_dir = os.path.join(local_dir, "load_tide")
        missing_files = check_files(load_tide_dir, load_tide_files)

    if len(missing_files) == 0:
        print(f"All the required files have already been downloaded")
        return

    print(f"missing_files : {missing_files}")
    logger.info(f"missing_files : {missing_files}")
    # otherwise proceed with the download
    logger.info(f"remote_file: {remote_file} --> local_file: {local_file}")

    # Printing files transferred
    remote_ftp_url = posixpath.join("ftp://", ftp.host, remote_file)
    logger.info(f"remote_ftp_url: {remote_ftp_url} -->")
    if tarmode:
        logger.info(f"tarmode")
        # copy remote file contents to bytesIO object
        fileobj = io.BytesIO()
        try:
            ftp.voidcmd("TYPE I")  # Switch to binary mode
            print(f"Size of {remote_file}: {ftp.size(remote_file)} bytes")
            # get the size of the file for progress bar
            response = ftp.sendcmd(f"SIZE {remote_file}")
            file_size = int(response.split()[1])
        except ftplib.error_perm as e:
            print(f"Could not get size of {remote_file}: {e}")

        # create a callback function to update the progress bar
        def callback(data):
            pbar.update(len(data))
            fileobj.write(data)

        with tqdm(total=file_size, unit="B", unit_scale=True, ncols=100) as pbar:
            fileobj = io.BytesIO()
            ftp.retrbinary(f"RETR {remote_file}", callback)
        fileobj.seek(0)

        # open the AOD1B monthly tar file
        tar = tarfile.open(name=remote_path[-1], fileobj=fileobj, mode=tarmode)
        # read tar file and extract all files
        member_files = [m for m in tar.getmembers() if tarfile.TarInfo.isfile(m)]

        # Calculate total size of all files in the tar file
        total_size = sum(m.size for m in member_files)
        with tqdm(total=total_size, unit="B", unit_scale=True, ncols=100) as pbar:
            for m in member_files:
                # print(f"m : {m}")
                logger.info(f"m : {m}")
                member = posixpath.basename(m.name) if flatten else m.name
                fileBasename, fileExtension = posixpath.splitext(
                    posixpath.basename(m.name)
                )
                # print(f"fileBasename : {fileBasename}")
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
                        with tar.extractfile(m) as fi, gzip.open(
                            local_file, "wb"
                        ) as fo:
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
                    pbar.update(m.size)
                    # get last modified date of remote file within tar file
                    # keep remote modification time of file and local access time
                    os.utime(local_file, (os.stat(local_file).st_atime, m.mtime))
                    os.chmod(local_file, MODE)
    else:
        logger.info(f"not tarmode")
        # copy readme and uncompressed files directly
        local_file = os.path.join(local_dir, remote_path[-1])
        logger.info(f"\t local_file: {local_file}\n")
        # copy remote file contents to local file
        try:
            ftp.voidcmd("TYPE I")  # Switch to binary mode
            print(f"Size of {remote_file}: {ftp.size(remote_file)} bytes")
            # get the size of the file for progress bar
            response = ftp.sendcmd(f"SIZE {remote_file}")
            file_size = int(response.split()[1])
        except ftplib.error_perm as e:
            print(f"Could not get size of {remote_file}: {e}")

        # create a callback function to update the progress bar
        def callback(data):
            pbar.update(len(data))
            fileobj.write(data)

        with tqdm(total=file_size, unit="B", unit_scale=True, ncols=100) as pbar:
            fileobj = io.BytesIO()
            ftp.retrbinary(f"RETR {remote_file}", callback)
        fileobj.seek(0)

        # # download the file in binary mode
        # with open(local_file, 'wb') as f:
        #     ftp.retrbinary(f'RETR {remote_file}', f.write)
        # get last modified date of remote file and convert into unix time
        mdtm = ftp.sendcmd(f"MDTM {remote_file}")
        remote_mtime = calendar.timegm(time.strptime(mdtm[4:], "%Y%m%d%H%M%S"))
        # keep remote modification time of file and local access time
        os.utime(local_file, (os.stat(local_file).st_atime, remote_mtime))
        os.chmod(local_file, MODE)


# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Downloads the FES (Finite Element Solution) global tide
            model from AVISO.  Decompresses the model tar files into the
            constituent files and auxiliary files.
            """,
        fromfile_prefix_chars="@",
    )
    parser.convert_arg_line_to_args = pyTMD.utilities.convert_arg_line_to_args
    # command line parameters
    # AVISO FTP credentials
    parser.add_argument(
        "--user", "-U", type=str, default="", help="Username for AVISO FTP servers"
    )
    parser.add_argument(
        "--netrc",
        "-N",
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.path.join(os.path.expanduser("~"), ".netrc"),
        help="Path to .netrc file for authentication",
    )
    # working data directory
    parser.add_argument(
        "--directory",
        "-D",
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help="Working data directory",
    )
    # FES tide models
    parser.add_argument(
        "--tide",
        "-T",
        metavar="TIDE",
        type=str,
        nargs="+",
        default=["FES2014"],
        choices=["FES1999", "FES2004", "FES2012", "FES2014"],
        help="FES tide model to download",
    )
    # download FES load tides
    parser.add_argument(
        "--load",
        default=False,
        action="store_true",
        help="Download load tide model outputs",
    )
    # download FES tidal currents
    parser.add_argument(
        "--currents",
        default=False,
        action="store_true",
        help="Download tide model current outputs",
    )
    # download FES tidal currents
    parser.add_argument(
        "--gzip",
        "-G",
        default=False,
        action="store_true",
        help="Compress output ascii and netCDF4 tide files",
    )
    # Output log file in form
    # AVISO_FES_tides_2002-04-01.log
    parser.add_argument(
        "--log", "-l", default=False, action="store_true", help="Output log file"
    )
    # permissions mode of the local directories and files (number in octal)
    parser.add_argument(
        "--mode",
        "-M",
        type=lambda x: int(x, base=8),
        default=0o775,
        help="Permission mode of directories and files downloaded",
    )
    # return the parser
    return parser


# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args, _ = parser.parse_known_args()

    # AVISO FTP Server hostname
    HOST = "ftp-access.aviso.altimetry.fr"
    # get AVISO FTP Server credentials
    try:
        args.user, _, PASSWORD = netrc.netrc(args.netrc).authenticators(HOST)
    except:
        # check that AVISO FTP Server credentials were entered
        if not args.user:
            prompt = f"Username for {HOST}: "
            args.user = builtins.input(prompt)
        # enter password securely from command-line
        prompt = f"Password for {args.user}@{HOST}: "
        PASSWORD = getpass.getpass(prompt)

    # check internet connection before attempting to run program
    if pyTMD.utilities.check_ftp_connection(HOST, args.user, PASSWORD):
        for m in args.tide:
            aviso_fes_tides(
                m,
                DIRECTORY=args.directory,
                USER=args.user,
                PASSWORD=PASSWORD,
                LOAD=args.load,
                CURRENTS=args.currents,
                GZIP=args.gzip,
                LOG=args.log,
                MODE=args.mode,
            )


# run main program
if __name__ == "__main__":
    main()
