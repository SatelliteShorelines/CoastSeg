import os
import glob
import asyncio
import platform
import json
import logging
from typing import List
from coastseg import common
import requests
import skimage
import aiohttp
import tqdm
import numpy as np
from glob import glob
import tqdm.asyncio
import nest_asyncio
from tensorflow.python.client import device_lib
from skimage.io import imread
from tensorflow.keras import mixed_precision
from doodleverse_utils.prediction_imports import do_seg
from doodleverse_utils.model_imports import (
    simple_resunet,
    custom_resunet,
    custom_unet,
    simple_unet,
    simple_resunet,
    simple_satunet,
    segformer,
)
from doodleverse_utils.model_imports import dice_coef_loss, iou_multi, dice_multi
import tensorflow as tf

logger = logging.getLogger(__name__)


def get_imagery_directory(img_type: str, RGB_path: str) -> str:
    logger.info(f"img_type: {img_type}")
    logger.info(f"RGB_path: {RGB_path}")
    output_path = os.path.dirname(RGB_path)
    if img_type == "RGB+MNDWI+NDWI":
        NIR_path = os.path.join(output_path, "NIR")
        NDWI_path = RGB_to_infrared(RGB_path, NIR_path, output_path, "NDWI")
        SWIR_path = os.path.join(output_path, "SWIR")
        MNDWI_path = RGB_to_infrared(RGB_path, SWIR_path, output_path, "MNDWI")
        five_band_path = common.create_directory(output_path, "five_band")
        output_path = get_five_band_imagery(
            RGB_path, MNDWI_path, NDWI_path, five_band_path
        )
    # default filetype is NIR and if NDWI is selected else filetype to SWIR
    elif img_type == "NDWI":
        NIR_path = os.path.join(output_path, "NIR")
        output_path = RGB_to_infrared(RGB_path, NIR_path, output_path, "NDWI")
    elif img_type == "MNDWI":
        SWIR_path = os.path.join(output_path, "SWIR")
        output_path = RGB_to_infrared(RGB_path, SWIR_path, output_path, "MNDWI")
    logger.info(f"output_path: {output_path}")
    return output_path


def get_five_band_imagery(
    RGB_path: str, MNDWI_path: str, NDWI_path: str, output_path: str
):
    paths = [RGB_path, MNDWI_path, NDWI_path]
    files = []
    for data_path in paths:
        f = sorted(glob(data_path + os.sep + "*.jpg"))
        if len(f) < 1:
            f = sorted(glob(data_path + os.sep + "images" + os.sep + "*.jpg"))
        files.append(f)

    # number of bands x number of samples
    files = np.vstack(files).T
    # returns path to five band imagery
    for counter, file in enumerate(files):
        im = []  # read all images into a list
        for k in file:
            im.append(imread(k))
        datadict = {}
        # create stack which takes care of different sized inputs
        im = np.dstack(im)
        datadict["arr_0"] = im.astype(np.uint8)
        datadict["num_bands"] = im.shape[-1]
        datadict["files"] = [file_name.split(os.sep)[-1] for file_name in file]
        ROOT_STRING = file[0].split(os.sep)[-1].split(".")[0]
        segfile = (
            output_path
            + os.sep
            + ROOT_STRING
            + "_noaug_nd_data_000000"
            + str(counter)
            + ".npz"
        )
        np.savez_compressed(segfile, **datadict)
        del datadict, im
        logger.info(f"segfile: {segfile}")
    return output_path


def get_files(RGB_dir_path: str, img_dir_path: str):
    """returns matrix of files in RGB_dir_path and img_dir_path
    creates matrix: RGB x number of samples in img_dir_path
    Example:
    [['full_RGB_path.jpg','full_NIR_path.jpg'],
    ['full_jpg_path.jpg','full_NIR_path.jpg']....]
    Args:
        RGB_dir_path (str): full path to directory of RGB images
        img_dir_path (str): full path to directory of non-RGB images
        usually NIR and SWIR

    Raises:
        FileNotFoundError: raised if directory is not found
    """
    paths = [RGB_dir_path, img_dir_path]
    files = []
    for data_path in paths:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found")
        f = sorted(glob(data_path + os.sep + "*.jpg"))
        if len(f) < 1:
            f = sorted(glob(data_path + os.sep + "images" + os.sep + "*.jpg"))
        files.append(f)
    # creates matrix:  bands(RGB) x number of samples
    files = np.vstack(files).T
    return files


def RGB_to_infrared(
    RGB_path: str, infrared_path: str, output_path: str, output_type: str
) -> None:
    """Converts two directories of RGB and NIR imagery to NDWI imagery in a directory named
     'NDWI' created at output_path.
     imagery saved as jpg

     to generate NDWI imagery set infrared_path to full path of NIR images
     to generate MNDWI imagery set infrared_path to full path of SWIR images

    Args:
        RGB_path (str): full path to directory containing RGB images
        infrared_path (str): full path to directory containing NIR or SWIR images
        output_path (str): full path to directory to create NDWI/MNDWI directory in
        output_type (str): 'MNDWI' or 'NDWI'
    Based on code from doodleverse_utils by Daniel Buscombe
    source: https://github.com/Doodleverse/doodleverse_utils
    """
    if output_type.upper() not in ["MNDWI", "NDWI"]:
        logger.error(
            f"Invalid output_type given must be MNDWI or NDWI. Cannot be {output_type}"
        )
        raise Exception(
            f"Invalid output_type given must be MNDWI or NDWI. Cannot be {output_type}"
        )
    # matrix:bands(RGB) x number of samples(NIR)
    files = get_files(RGB_path, infrared_path)
    # output_path: directory to store MNDWI or NDWI outputs
    output_path += os.sep + output_type.upper()
    logger.info(f"output_path {output_path}")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for file in files:
        # Read green band from RGB image and cast to float
        green_band = skimage.io.imread(file[0])[:, :, 1].astype("float")
        # Read infrared(SWIR or NIR) and cast to float
        infrared = skimage.io.imread(file[1]).astype("float")
        # Transform 0 to np.NAN
        green_band[green_band == 0] = np.nan
        infrared[infrared == 0] = np.nan
        # Mask out NaNs
        green_band = np.ma.filled(green_band)
        infrared = np.ma.filled(infrared)

        # ensure both matrices have equivalent size
        if not np.shape(green_band) == np.shape(infrared):
            gx, gy = np.shape(green_band)
            nx, ny = np.shape(infrared)
            # resize both matrices to have equivalent size
            green_band = common.scale(
                green_band, np.maximum(gx, nx), np.maximum(gy, ny)
            )
            infrared = common.scale(infrared, np.maximum(gx, nx), np.maximum(gy, ny))

        # output_img(MNDWI/NDWI) imagery formula (Green - SWIR) / (Green + SWIR)
        output_img = np.divide(infrared - green_band, infrared + green_band)
        # Convert the NaNs to -1
        output_img[np.isnan(output_img)] = -1
        # Rescale to be between 0 - 255
        output_img = common.rescale_array(output_img, 0, 255)
        # create new filenames by replacing image type(SWIR/NIR) with output_type
        if output_type.upper() == "MNDWI":
            new_filename = file[1].split(os.sep)[-1].replace("SWIR", output_type)
        if output_type.upper() == "NDWI":
            new_filename = file[1].split(os.sep)[-1].replace("NIR", output_type)

        # save output_img(MNDWI/NDWI) as .jpg in output directory
        skimage.io.imsave(
            output_path + os.sep + new_filename,
            output_img.astype("uint8"),
            check_contrast=False,
            quality=100,
        )

    return output_path


async def fetch(session, url: str, save_path: str):
    model_name = url.split("/")[-1]
    # chunk_size: int = 128
    chunk_size: int = 2048
    async with session.get(url, raise_for_status=True) as r:
        content_length = r.headers.get("Content-Length")
        if content_length is not None:
            content_length = int(content_length)
            with open(save_path, "wb") as fd:
                with tqdm.auto.tqdm(
                    total=content_length,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {model_name}",
                    initial=0,
                    ascii=False,
                    position=0,
                ) as pbar:
                    async for chunk in r.content.iter_chunked(chunk_size):
                        fd.write(chunk)
                        pbar.update(len(chunk))
        else:
            with open(save_path, "wb") as fd:
                async for chunk in r.content.iter_chunked(chunk_size):
                    fd.write(chunk)


async def fetch_all(session, url_dict):
    tasks = []
    for save_path, url in url_dict.items():
        task = asyncio.create_task(fetch(session, url, save_path))
        tasks.append(task)
    await tqdm.asyncio.tqdm.gather(*tasks)


async def async_download_urls(url_dict: dict) -> None:
    async with aiohttp.ClientSession() as session:
        await fetch_all(session, url_dict)


def run_async_download(url_dict: dict):
    logger.info("run_async_download")
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    logger.info("Scheduling task")
    # apply a nested loop to jupyter's event loop for async downloading
    nest_asyncio.apply()
    # get nested running loop and wait for async downloads to complete
    loop = asyncio.get_running_loop()
    result = loop.run_until_complete(async_download_urls(url_dict))
    logger.info("Scheduled task")
    logger.info(f"result: {result}")


def get_GPU(num_GPU: str) -> None:
    num_GPU = str(num_GPU)
    if num_GPU == "0":
        logger.info("Not using GPU")
        print("Not using GPU")
        # use CPU (not recommended):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif num_GPU == "1":
        print("Using single GPU")
        logger.info(f"Using 1 GPU")
        # use first available GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if int(num_GPU) == 1:
        # read physical GPUs from machine
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        print(f"physical_devices (GPUs):{physical_devices}")
        logger.info(f"physical_devices (GPUs):{physical_devices}")
        if physical_devices:
            # Restrict TensorFlow to only use the first GPU
            try:
                tf.config.experimental.set_visible_devices(physical_devices, "GPU")
            except RuntimeError as e:
                # Visible devices must be set at program startup
                logger.error(e)
                print(e)
        # set mixed precision
        mixed_precision.set_global_policy("mixed_float16")
        # disable memory growth on all GPUs
        for i in physical_devices:
            tf.config.experimental.set_memory_growth(i, True)
            print(f"visible_devices: {tf.config.get_visible_devices()}")
            logger.info(f"visible_devices: {tf.config.get_visible_devices()}")
        # if multiple GPUs are used use mirror strategy
        if int(num_GPU) > 1:
            # Create a MirroredStrategy.
            strategy = tf.distribute.MirroredStrategy(
                [p.name.split("/physical_device:")[-1] for p in physical_devices],
                cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(),
            )
            print(f"Number of distributed devices: {strategy.num_replicas_in_sync}")
            logger.info(
                f"Number of distributed devices: {strategy.num_replicas_in_sync}"
            )


def get_url_dict_to_download(models_json_dict: dict) -> dict:
    """Returns dictionary of paths to save files to download
    and urls to download file

    ex.
    {'C:\Home\Project\file.json':"https://website/file.json"}

    Args:
        models_json_dict (dict): full path to files and links

    Returns:
        dict: full path to files and links
    """
    url_dict = {}
    for save_path, link in models_json_dict.items():
        if not os.path.isfile(save_path):
            url_dict[save_path] = link
        json_filepath = save_path.replace("_fullmodel.h5", ".json")
        if not os.path.isfile(json_filepath):
            json_link = link.replace("_fullmodel.h5", ".json")
            url_dict[json_filepath] = json_link

    return url_dict


def download_url(url: str, save_path: str, chunk_size: int = 128):
    """Downloads the model from the given url to the save_path location.
    Args:
        url (str): url to model to download
        save_path (str): directory to save model
        chunk_size (int, optional):  Defaults to 128.
    """
    logger.info(f"url: {url}")
    logger.info(f"save_path: {save_path}")
    # make an HTTP request within a context manager
    with requests.get(url, stream=True) as r:
        # check header to get content length, in bytes
        content_length = r.headers.get("Content-Length")
        # raise an exception for error codes (4xx or 5xx)
        r.raise_for_status()
        if content_length is None:
            with open(save_path, "wb") as fd:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    fd.write(chunk)
        elif content_length is not None:
            content_length = int(content_length)
            with open(save_path, "wb") as fd:
                with tqdm.auto.tqdm(
                    total=content_length,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Downloading Model",
                    initial=0,
                    ascii=True,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        fd.write(chunk)
                        pbar.update(len(chunk))


def get_sorted_files_with_extension(
    sample_direc: str, file_extensions: List[str]
) -> List[str]:
    """
    Get a sorted list of paths to files that have one of the file_extensions.
    It will return the first set of files that matches the first file_extension, so put the
    file_extension list in order of priority

    Args:
        sample_direc: A string representing the directory path to search for images.
        file_extensions: A list of file extensions to search for.

    Returns:
        A list of file paths for sample images found in the directory.

    """
    sample_filenames = []
    for ext in file_extensions:
        filenames = sorted(tf.io.gfile.glob(os.path.join(sample_direc, f"*{ext}")))
        sample_filenames.extend(filenames)
        if sample_filenames:
            break
    return sample_filenames


class Zoo_Model:
    def __init__(self):
        self.weights_direc = None

    def run_model(
        self,
        model_implementation: str,
        src_directory: str,
        model_name: str,
        use_GPU: str,
        use_otsu: bool,
        use_tta: bool,
    ):
        logger.info(f"Selected directory of RGBs: {src_directory}")
        logger.info(f"model_name: {model_name}")
        logger.info(f"model_implementation: {model_implementation}")
        logger.info(f"use_GPU: {use_GPU}")
        logger.info(f"use_otsu: {use_otsu}")
        logger.info(f"use_tta: {use_tta}")

        self.download_model(model_implementation, model_name)
        print("")
        weights_list = self.get_weights_list(model_implementation)

        # Load the model from the config files
        model, model_list, config_files, model_types = self.get_model(weights_list)
        metadatadict = self.get_metadatadict(weights_list, config_files, model_types)
        logger.info(f"metadatadict: {metadatadict}")
        # Compute the segmentation
        self.compute_segmentation(
            src_directory,
            model_list,
            metadatadict,
            model_types,
            use_tta,
            use_otsu,
        )

    def get_files_for_seg(
        self, sample_direc: str, avoid_patterns: List[str] = []
    ) -> list:
        """
        Returns a list of files to be segmented.

        The function reads in the image filenames as either (`.npz`) OR (`.jpg`, or `.png`)
        and returns a sorted list of the file paths.

        Args:
        - sample_direc (str): The directory containing files to be segmented.
        - avoid_patterns (List[str], optional): A list of file names to be avoided.Don't include any file extensions. Default is [].

        Returns:
        - list: A list of files to be segmented.
        """
        logger.info(f"Searching directory for files: {sample_direc}")
        file_extensions = [".npz", ".jpg", ".png"]
        sample_filenames = get_sorted_files_with_extension(
            sample_direc, file_extensions
        )
        # filter out files whose filenames match any of the avoid_patterns
        sample_filenames = common.filter_files(sample_filenames, avoid_patterns)
        logger.info(f"files to seg: {sample_filenames}")
        return sample_filenames

    def compute_segmentation(
        self,
        sample_direc: str,
        model_list: list,
        metadatadict: dict,
        model_types,
        use_tta: bool,
        use_otsu: bool,
    ):
        logger.info(f"Test Time Augmentation: {use_tta}")
        logger.info(f"Otsu Threshold: {use_otsu}")
        # Read in the image filenames as either .npz,.jpg, or .png
        files_to_segment = self.get_files_for_seg(sample_direc)
        logger.info(f"files_to_segment: {files_to_segment}")
        if model_types[0] != "segformer":
            ### mixed precision
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy("mixed_float16")
        # Compute the segmentation for each of the files
        for file_to_seg in tqdm.auto.tqdm(files_to_segment):
            do_seg(
                file_to_seg,
                model_list,
                metadatadict,
                model_types[0],
                sample_direc=sample_direc,
                NCLASSES=self.NCLASSES,
                N_DATA_BANDS=self.N_DATA_BANDS,
                TARGET_SIZE=self.TARGET_SIZE,
                TESTTIMEAUG=use_tta,
                WRITE_MODELMETADATA=False,
                OTSU_THRESHOLD=use_otsu,
            )

    def get_model(self, weights_list: list):
        model_list = []
        config_files = []
        model_types = []
        if weights_list == []:
            raise Exception("No Model Info Passed")
        for weights in weights_list:
            # "fullmodel" is for serving on zoo they are smaller and more portable between systems than traditional h5 files
            # gym makes a h5 file, then you use gym to make a "fullmodel" version then zoo can read "fullmodel" version
            configfile = weights.replace(".h5", ".json").replace("weights", "config")
            if "fullmodel" in configfile:
                configfile = configfile.replace("_fullmodel", "")
            with open(configfile) as f:
                config = json.load(f)
            self.TARGET_SIZE = config.get("TARGET_SIZE")
            MODEL = config.get("MODEL")
            self.NCLASSES = config.get("NCLASSES")
            KERNEL = config.get("KERNEL")
            STRIDE = config.get("STRIDE")
            FILTERS = config.get("FILTERS")
            self.N_DATA_BANDS = config.get("N_DATA_BANDS")
            DROPOUT = config.get("DROPOUT")
            DROPOUT_CHANGE_PER_LAYER = config.get("DROPOUT_CHANGE_PER_LAYER")
            DROPOUT_TYPE = config.get("DROPOUT_TYPE")
            USE_DROPOUT_ON_UPSAMPLING = config.get("USE_DROPOUT_ON_UPSAMPLING")
            DO_TRAIN = config.get("DO_TRAIN")
            LOSS = config.get("LOSS")
            PATIENCE = config.get("PATIENCE")
            MAX_EPOCHS = config.get("MAX_EPOCHS")
            VALIDATION_SPLIT = config.get("VALIDATION_SPLIT")
            RAMPUP_EPOCHS = config.get("RAMPUP_EPOCHS")
            SUSTAIN_EPOCHS = config.get("SUSTAIN_EPOCHS")
            EXP_DECAY = config.get("EXP_DECAY")
            START_LR = config.get("START_LR")
            MIN_LR = config.get("MIN_LR")
            MAX_LR = config.get("MAX_LR")
            FILTER_VALUE = config.get("FILTER_VALUE")
            DOPLOT = config.get("DOPLOT")
            ROOT_STRING = config.get("ROOT_STRING")
            USEMASK = config.get("USEMASK")
            AUG_ROT = config.get("AUG_ROT")
            AUG_ZOOM = config.get("AUG_ZOOM")
            AUG_WIDTHSHIFT = config.get("AUG_WIDTHSHIFT")
            AUG_HEIGHTSHIFT = config.get("AUG_HEIGHTSHIFT")
            AUG_HFLIP = config.get("AUG_HFLIP")
            AUG_VFLIP = config.get("AUG_VFLIP")
            AUG_LOOPS = config.get("AUG_LOOPS")
            AUG_COPIES = config.get("AUG_COPIES")
            REMAP_CLASSES = config.get("REMAP_CLASSES")

            try:
                model = tf.keras.models.load_model(weights)
                #  nclasses=NCLASSES, may have to replace nclasses with NCLASSES
            except BaseException:
                if MODEL == "resunet":
                    model = custom_resunet(
                        (self.TARGET_SIZE[0], self.TARGET_SIZE[1], self.N_DATA_BANDS),
                        FILTERS,
                        nclasses=[
                            self.NCLASSES + 1 if self.NCLASSES == 1 else self.NCLASSES
                        ][0],
                        kernel_size=(KERNEL, KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,  # 0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                        dropout_type=DROPOUT_TYPE,  # "standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                    )
                elif MODEL == "unet":
                    model = custom_unet(
                        (self.TARGET_SIZE[0], self.TARGET_SIZE[1], self.N_DATA_BANDS),
                        FILTERS,
                        nclasses=[
                            self.NCLASSES + 1 if self.NCLASSES == 1 else self.NCLASSES
                        ][0],
                        kernel_size=(KERNEL, KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,  # 0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                        dropout_type=DROPOUT_TYPE,  # "standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                    )

                elif MODEL == "simple_resunet":
                    # num_filters = 8 # initial filters
                    model = simple_resunet(
                        (self.TARGET_SIZE[0], self.TARGET_SIZE[1], self.N_DATA_BANDS),
                        kernel=(2, 2),
                        num_classes=[
                            self.NCLASSES + 1 if self.NCLASSES == 1 else self.NCLASSES
                        ][0],
                        activation="relu",
                        use_batch_norm=True,
                        dropout=DROPOUT,  # 0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                        dropout_type=DROPOUT_TYPE,  # "standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                        filters=FILTERS,  # 8,
                        num_layers=4,
                        strides=(1, 1),
                    )
                # 346,564
                elif MODEL == "simple_unet":
                    model = simple_unet(
                        (self.TARGET_SIZE[0], self.TARGET_SIZE[1], self.N_DATA_BANDS),
                        kernel=(2, 2),
                        num_classes=[
                            self.NCLASSES + 1 if self.NCLASSES == 1 else self.NCLASSES
                        ][0],
                        activation="relu",
                        use_batch_norm=True,
                        dropout=DROPOUT,  # 0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                        dropout_type=DROPOUT_TYPE,  # "standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                        filters=FILTERS,  # 8,
                        num_layers=4,
                        strides=(1, 1),
                    )
                elif MODEL == "satunet":
                    model = simple_satunet(
                        (self.TARGET_SIZE[0], self.TARGET_SIZE[1], self.N_DATA_BANDS),
                        kernel=(2, 2),
                        num_classes=self.NCLASSES,  # [NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                        activation="relu",
                        use_batch_norm=True,
                        dropout=DROPOUT,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                        dropout_type=DROPOUT_TYPE,
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                        filters=FILTERS,
                        num_layers=4,
                        strides=(1, 1),
                    )
                elif MODEL == "segformer":
                    id2label = {}
                    for k in range(self.NCLASSES):
                        id2label[k] = str(k)
                    model = segformer(id2label, num_classes=self.NCLASSES)
                    model.compile(optimizer="adam")
                # 242,812
                else:
                    raise Exception(
                        f"An unknown model type {MODEL} was received. Please select a valid model.\n \
                        Model must be one of 'unet', 'resunet', 'segformer', or 'satunet'"
                    )

                # Load in the custom loss function from doodleverse_utils
                model.compile(
                    optimizer="adam", loss=dice_coef_loss(self.NCLASSES)
                )  # , metrics = [iou_multi(self.NCLASSESNCLASSES), dice_multi(self.NCLASSESNCLASSES)])

                model.load_weights(weights)

            model_types.append(MODEL)
            model_list.append(model)
            config_files.append(configfile)

        return model, model_list, config_files, model_types

    def get_metadatadict(
        self, weights_list: list, config_files: list, model_types: list
    ):
        metadatadict = {}
        metadatadict["model_weights"] = weights_list
        metadatadict["config_files"] = config_files
        metadatadict["model_types"] = model_types
        return metadatadict

    def get_weights_list(self, model_choice: str = "ENSEMBLE"):
        """Returns of the weights files(.h5) within weights_direc"""
        if model_choice == "ENSEMBLE":
            weights_list = glob(self.weights_direc + os.sep + "*.h5")
            logger.info(f"ENSEMBLE: weights_list: {weights_list}")
            logger.info(
                f"ENSEMBLE: {len(weights_list)} sets of model weights were found "
            )
            return weights_list
        elif model_choice == "BEST":
            # read model name (fullmodel.h5) from BEST_MODEL.txt
            with open(self.weights_direc + os.sep + "BEST_MODEL.txt") as f:
                model_name = f.readlines()
            weights_list = [self.weights_direc + os.sep + model_name[0]]
            logger.info(f"BEST: weights_list: {weights_list}")
            logger.info(f"BEST: {len(weights_list)} sets of model weights were found ")
            return weights_list

    def get_downloaded_models_dir(self) -> str:
        """returns full path to downloaded_models directory and
        if downloaded_models directory does not exist then it is created

        Returns:
            str: full path to downloaded_models directory
        """
        # directory to hold downloaded models from Zenodo
        script_dir = os.path.dirname(os.path.abspath(__file__))
        downloaded_models_path = os.path.abspath(
            os.path.join(script_dir, "downloaded_models")
        )
        if not os.path.exists(downloaded_models_path):
            os.mkdir(downloaded_models_path)
        logger.info(f"downloaded_models_path: {downloaded_models_path}")
        return downloaded_models_path

    def download_model(self, model_choice: str, dataset_id: str) -> None:
        """downloads model specified by zenodo id in dataset_id.

        Downloads best model is model_choice = 'BEST' or all models in
        zenodo release if model_choice = 'ENSEMBLE'

        Args:
            model_choice (str): 'BEST' or 'ENSEMBLE'
            dataset_id (str): name of model followed by underscore zenodo_id'name_of_model_zenodoid'
        """
        zenodo_id = dataset_id.split("_")[-1]
        root_url = "https://zenodo.org/api/records/" + zenodo_id
        # read raw json and get list of available files in zenodo release
        response = requests.get(root_url)
        json_content = json.loads(response.text)
        logger.info(f"json_content {json_content}")
        files = json_content["files"]

        downloaded_models_path = self.get_downloaded_models_dir()
        # directory to hold specific model referenced by dataset_id
        self.weights_direc = os.path.abspath(
            os.path.join(downloaded_models_path, dataset_id)
        )
        if not os.path.exists(self.weights_direc):
            os.mkdir(self.weights_direc)

        logger.info(f"self.weights_direc:{self.weights_direc}")
        print(f"\n Model located at: {self.weights_direc}")
        models_json_dict = {}
        if model_choice.upper() == "BEST":
            # retrieve best model text file
            best_model_json = [f for f in files if f["key"] == "BEST_MODEL.txt"][0]
            if len(best_model_json) == 0:
                raise Exception(f"Cannot find BEST_MODEL.txt at {root_url}")
            logger.info(f"list of best_model_txt: {best_model_json}")
            best_model_txt_path = self.weights_direc + os.sep + "BEST_MODEL.txt"
            logger.info(f"BEST: best_model_txt_path : {best_model_txt_path }")

            # if best BEST_MODEL.txt file not exist then download it
            if not os.path.isfile(best_model_txt_path):
                download_url(
                    best_model_json["links"]["self"],
                    best_model_txt_path,
                )
            # read contents of BEST_MODEL.txt
            with open(best_model_txt_path) as f:
                filename = f.read()

            # check if json and h5 file in BEST_MODEL.txt exist
            model_json = [f for f in files if f["key"] == filename][0]
            # path to save model
            outfile = self.weights_direc + os.sep + filename
            logger.info(f"BEST: outfile: {outfile}")
            # path to save file and json data associated with file saved to dict
            models_json_dict[outfile] = model_json["links"]["self"]
            url_dict = get_url_dict_to_download(models_json_dict)
            # if any files are not found locally download them asynchronous
            if url_dict != {}:
                run_async_download(url_dict)
        elif model_choice.upper() == "ENSEMBLE":
            # get list of all models
            all_models = [f for f in files if f["key"].endswith(".h5")]
            if len(all_models) == 0:
                raise Exception(f"Cannot find any .h5 files at {root_url}")
            logger.info(f"all_models : {all_models }")
            # check if all h5 files in files are in self.weights_direc
            for model_json in all_models:
                outfile = (
                    self.weights_direc
                    + os.sep
                    + model_json["links"]["self"].split("/")[-1]
                )
                logger.info(f"ENSEMBLE: outfile: {outfile}")
                # path to save file and json data associated with file saved to dict
                models_json_dict[outfile] = model_json["links"]["self"]
            logger.info(f"models_json_dict: {models_json_dict}")
            url_dict = get_url_dict_to_download(models_json_dict)
            logger.info(f"URLs to download: {url_dict}")
            # if any files are not found locally download them asynchronous
            if url_dict != {}:
                run_async_download(url_dict)
