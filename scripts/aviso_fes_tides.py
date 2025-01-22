#!/usr/bin/env python
"""
aviso_fes_tides.py
Originally written by Tyler Sutterley (11/2022) & Modified by Sharon Fitzpatrick Batiste (2023)
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
    -U X, --user: username for AVISO FTP servers (email)
    -P X, --password: password for AVISO FTP servers
    -N X, --netrc X: path to .netrc file for authentication
    --tide X: FES tide model to download
        FES2014
        FES2022
    --load: download load tide model outputs (fes2014)
    --currents: download tide model current outputs (fes2012 and fes2014)
    -G, --gzip: compress output ascii and netCDF4 tide files
    -t X, --timeout X: timeout in seconds for blocking operations
    --log: output log of files downloaded
    -M X, --mode X: Local permissions mode of the files downloaded
"""

# Example of how to run the script
# python aviso_fes_tides.py --user <username> --tide FES2022 --directory C:\CoastSeg\tide_model --currents --load
# - This will download the FES2022 tide model to the directory C:\CoastSeg\tide_model

from __future__ import print_function, annotations
import os
import netrc
import getpass
import pathlib
import argparse
import builtins
import pyTMD.utilities

# Interal imports
from coastseg import download_tide_model


# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Downloads the FES (Finite Element Solution) global tide
            model from AVISO.  Decompresses the model tar files into the
            constituent files and auxiliary files.
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = pyTMD.utilities.convert_arg_line_to_args
    # command line parameters
    # AVISO FTP credentials
    parser.add_argument('--user','-U',
        type=str, default=os.environ.get('AVISO_USERNAME'),
        help='Username for AVISO Login')
    parser.add_argument('--password','-W',
        type=str, default=os.environ.get('AVISO_PASSWORD'),
        help='Password for AVISO Login')
    parser.add_argument('--netrc','-N',
        type=pathlib.Path, default=pathlib.Path().home().joinpath('.netrc'),
        help='Path to .netrc file for authentication')
    # working data directory
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=pathlib.Path.cwd(),
        help='The directory to save the tide model to. We recommend using the directory CoastSeg/tide_model')
    # FES tide models
    parser.add_argument('--tide','-T',
        metavar='TIDE', type=str, nargs='+',
        default=['FES2022'], choices=['FES2014','FES2022'],
        help='FES tide model to download')
    # download FES load tides
    parser.add_argument('--load',
        default=False, action='store_true',
        help='Download load tide model outputs')
    # download FES tidal currents
    parser.add_argument('--currents',
        default=False, action='store_true',
        help='Download tide model current outputs')
    # download FES tidal currents
    parser.add_argument('--gzip','-G',
        default=False, action='store_true',
        help='Compress output ascii and netCDF4 tide files')
    # connection timeout
    parser.add_argument('--timeout','-t',
        type=int, default=1000,
        help='Timeout in seconds for blocking operations')
    # Output log file in form
    # AVISO_FES_tides_2002-04-01.log
    parser.add_argument('--log','-l',
        default=False, action='store_true',
        help='Output log file')
    # permissions mode of the local directories and files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of directories and files downloaded')
    # return the parser
    return parser


# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args, _ = parser.parse_known_args()

    # AVISO FTP Server hostname
    HOST = "ftp-access.aviso.altimetry.fr"
    # AVISO FTP Server hostname
    HOST = 'ftp-access.aviso.altimetry.fr'
    # get authentication
    if not args.user and not args.netrc.exists():
        # check that AVISO credentials were entered
        args.user = builtins.input(f'Username for {HOST}: ')
        # enter password securely from command-line
        args.password = getpass.getpass(f'Password for {args.user}@{HOST}: ')
    elif args.netrc.exists():
        args.user,_,args.password = netrc.netrc(args.netrc).authenticators(HOST)
    elif args.user and not args.password:
        # enter password securely from command-line
        args.password = getpass.getpass(f'Password for {args.user}@{HOST}: ')

    # check internet connection before attempting to run program
    if pyTMD.utilities.check_ftp_connection(HOST,args.user,args.password):
        for m in args.tide:
            download_tide_model.aviso_fes_tides(
                m,
                DIRECTORY=args.directory,
                USER=args.user,
                PASSWORD=args.password,
                LOAD=args.load,
                CURRENTS=args.currents,
                GZIP=args.gzip,
                LOG=args.log,
                MODE=args.mode,
            )


# run main program
if __name__ == "__main__":
    main()
