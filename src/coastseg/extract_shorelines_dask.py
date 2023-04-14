import dask
from dask.diagnostics import ProgressBar
from dataclasses import dataclass
from typing import List

import os
import pickle
import numpy as np
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

# ... (other imports and functions)

def process_satellite(satname, metadata, settings, session_path, filepath_data):
    filepath = get_filepath(settings["inputs"], satname)
    filenames = metadata[satname]["filenames"]

    output_timestamp = []
    output_shoreline = []
    output_filename = []
    output_cloudcover = []
    output_geoaccuracy = []
    output_idxkeep = []

    pixel_size = 15 if satname in ["L5", "L7", "L8", "L9"] else 10

    if satname == "L7":
        settings["min_length_sl"] = 200
    else:
        settings["min_length_sl"] = default_min_length_sl

    for i in tqdm(range(len(filenames)), desc="Mapping Shorelines", leave=True, position=0):
        fn = get_filenames(filenames[i], filepath, satname)
        (
            im_ms,
            georef,
            cloud_mask,
            im_extra,
            im_QA,
            im_nodata,
        ) = preprocess_single(
            fn,
            satname,
            settings["cloud_mask_issue"],
            settings["pan_off"],
            collection,
        )
        image_epsg = metadata[satname]["epsg"][i]

        cloud_cover_combined = np.mean(cloud_mask)
        if cloud_cover_combined > 0.99:
            continue

        cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
        cloud_cover = np.mean(cloud_mask_adv) / np.mean(~im_nodata)
        if cloud_cover > settings["cloud_thresh"]:
            continue

        im_ref_buffer = create_shoreline_buffer(
            cloud_mask.shape, georef, image_epsg, pixel_size, settings
        )

        npz_file = get_npz_tile(filenames[i], session_path, satname, image_type=image_type)
        if npz_file is None:
            logger.warning(f"npz file not found for {filenames[i]}")
            continue
        logger.info(f"npz_file: {npz_file}")

        im_labels = load_image_labels(npz_file, class_indices=class_indices)

        shoreline = process_image(
            fn,
            satname,
            i,
            image_epsg,
            settings,
            metadata,
            cloud_mask_adv,
            im_nodata,
            georef,
            im_ms,
            im_labels,
            session_path,
            filepath,
            collection,
        )
        if shoreline is None:
            continue

        output_timestamp.append(metadata[satname]["dates"][i])
        output_shoreline.append(shoreline)
        output_filename.append(filenames[i])
        output_cloudcover.append(cloud_cover)
        output_geoaccuracy.append(metadata[satname]["acc_georef"][i])
        output_idxkeep.append(i)

    return {
        "dates": output_timestamp,
        "shorelines": output_shoreline,
        "filename": output_filename,
        "cloud_cover": output_cloudcover,
        "geoaccuracy": output_geoaccuracy,
        "idx": output_idxkeep,
    }

def extract_shorelines_for_session(
    session_path: str, metadata: dict, settings: dict, image_type: str = "RGB", class_indices: list = None
) -> dict:

    print(f"extract shoreline settings loaded in: {settings}")
    sitename = settings["inputs"]["sitename"]
    filepath_data = settings["inputs"]["filepath"]
    collection = settings["inputs"]["landsat_collection"]
    default_min_length_sl = settings["min_length_sl"]
    # initialise output structure
    output = dict()

    # create a subfolder to store the .jpg images showing the detection
    filepath_jpg = os.path.join(filepath_data, sitename, "jpg_files", "detection")
    if not os.path.exists(filepath_jpg):
        os.makedirs(filepath_jpg)

    # process each satellite and store the results in the output dictionary
    for satname in metadata.keys():
        output[satname] = process_satellite(satname, metadata, settings, session_path, filepath_data)

    # change the format to have one list sorted by date with all the shorelines (easier to use)
    output = merge_output(output)

    # save output structure as output.pkl
    filepath = os.path.join(filepath_data, sitename)

    with open(os.path.join(filepath, sitename + "_output.pkl"), "wb") as f:
        pickle.dump(output, f)

    return output

@dataclass
class ShorelineData:
    dates: List[str]
    shorelines: List[List[tuple]]
    filename: List[str]
    cloud_cover: List[float]
    geoaccuracy: List[float]
    idx: List[int]

    def merge(self):
        # Implement a method to merge the ShorelineData instances by date
        pass

    def save_to_file(self, file_path: str):
        # Implement a method to save the data to a file
        pass

    def load_from_file(self, file_path: str):
        # Implement a method to load the data from a file
        pass

def process_image(i, filenames, filepath, satname, settings, session_path, collection, pixel_size, image_epsg, im_ref_buffer):
    # ... (move the relevant code from the for loop inside process_satellite to this function)
    return (i, output_timestamp, output_shoreline, output_filename, output_cloudcover, output_geoaccuracy)

# this one uses dask
def process_satellite(satname, metadata, settings, session_path, filepath_data) -> ShorelineData:
    # ... (same as before, until the for loop through the images)

    # process the images using Dask
    tasks = []
    for i in range(len(filenames)):
        tasks.append(
            dask.delayed(process_image)(
                i, filenames, filepath, satname, settings, session_path, collection, pixel_size, image_epsg, im_ref_buffer
            )
        )

    # Run tasks in parallel using multi-threading
    with ProgressBar():
        results = dask.compute(*tasks, scheduler="threads")

    # Unpack the results and update the output variables
    for i, output_timestamp, output_shoreline, output_filename, output_cloudcover, output_geoaccuracy in results:
        output_idxkeep.append(i)
        output_timestamps.append(output_timestamp)
        output_shorelines.append(output_shoreline)
        output_filenames.append(output_filename)
        output_cloudcovers.append(output_cloudcover)
        output_geoaccuracies.append(output_geoaccuracy)

    # ... (same as before, creating the ShorelineData instance)

