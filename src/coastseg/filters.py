import datetime
from statistics import mode
import numpy as np
import xarray as xr
from glob import glob
import os, shutil
from sklearn.cluster import KMeans
from statistics import mode

def copy_files(files: list, dest_folder: str) -> None:
    """
    Copy files to a specified destination folder.

    Args:
        files (list): List of file paths to be copied.
        dest_folder (str): Destination folder where files will be copied.

    Returns:
        None
    """
    for f in files:
        shutil.copy(f, dest_folder)
        
def load_data(f: str) -> np.array:
    with np.load(f) as data:
        grey = data["grey_label"].astype("uint8")
    return grey

def get_good_bad_files(files: list, labels: np.array, scores: list) -> tuple:
    """
    Split files into 'good' and 'bad' categories based on provided labels and scores.

    Args:
        files (list): List of file paths.
        labels (np.array): Array of labels corresponding to the files.
        scores (list): List of scores associated with the files.

    Returns:
        tuple: A tuple containing two arrays:
            - files_bad (np.array): Array of 'bad' categorized file paths (highest score label).
            - files_good (np.array): Array of 'good' categorized file paths (lowest score label).
    """
    files_bad = np.array(files)[labels == np.argmax(scores)]
    files_good = np.array(files)[labels == np.argmin(scores)]
    return files_bad, files_good

def get_time_vectors(files: list) -> tuple:
    """
    Extract time information from a list of file paths and create an xarray variable.

    Args:
        files (list): List of file paths containing time information.

    Returns:
        tuple: A tuple containing two elements:
            - times (list): List of time values extracted from the file paths.
            - time_variable (xr.Variable): xarray variable containing the time values.
    """
    times = [f.split(os.sep)[-1].split("_")[0] for f in files]
    return times, xr.Variable("time", times)

def get_image_shapes(files: list) -> list:
    return [load_data(f).shape for f in files]

def get_image_shapes(files: list) -> list:
    return [load_data(f).shape for f in files]

def measure_rmse(da: xr.DataArray, times: list, timeav: xr.DataArray) -> tuple:
    rmse = [
        float(np.sqrt(np.mean((da.sel(time=t) - timeav) ** 2)).to_numpy())
        for t in times
    ]
    input_rmse = np.array(rmse).reshape(-1, 1)
    return rmse, input_rmse

def get_kmeans_clusters(input_rmse: np.array, rmse: list) -> tuple:
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(input_rmse)
    labels = kmeans.labels_
    scores = [
        np.mean(np.array(rmse)[labels == 0]),
        np.mean(np.array(rmse)[labels == 1]),
    ]
    return labels, scores

def load_xarray_data(f: str) -> xr.DataArray:
    with np.load(f) as data:
        grey = data["grey_label"].astype("uint8")
    ny, nx = grey.shape
    y = np.arange(ny)
    x = np.arange(nx)
    return xr.DataArray(grey, coords={"y": y, "x": x}, dims=["y", "x"])

def handle_files_and_directories(
    files_bad: list, files_good: list, dest_folder_bad: str, dest_folder_good: str
) -> None:
    os.makedirs(dest_folder_bad, exist_ok=True)
    os.makedirs(dest_folder_good, exist_ok=True)

    copy_files(files_bad, dest_folder_bad)
    copy_files(files_good, dest_folder_good)
      
def return_valid_files(files: list) -> list:
    # print(get_image_shapes(files))
    modal_shape = mode(get_image_shapes(files))
    return [f for f in files if load_data(f).shape == modal_shape]

def filter_model_outputs(
    label: str, files: list, dest_folder_good: str, dest_folder_bad: str
) -> None:
    valid_files = return_valid_files(files)
    times, time_var = get_time_vectors(valid_files)
    da = xr.concat([load_xarray_data(f) for f in valid_files], dim=time_var)
    timeav = da.mean(dim="time")

    rmse, input_rmse = measure_rmse(da, times, timeav)
    labels, scores = get_kmeans_clusters(input_rmse, rmse)
    files_bad, files_good = get_good_bad_files(valid_files, labels, scores)

    handle_files_and_directories(
        files_bad, files_good, dest_folder_bad, dest_folder_good
    )

    print(f"{len(files_good)} good {label} labels")
    print(f"{len(files_bad)} bad {label} labels")



