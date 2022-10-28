import os
import glob
import logging
import requests
import zipfile
import json
from tqdm import tqdm
from doodleverse_utils.prediction_imports import do_seg
from doodleverse_utils.imports import (
    simple_resunet,
    simple_unet,
    custom_resunet,
    custom_unet,
)
from  doodleverse_utils.model_imports import (dice_coef_loss,
                                              iou_multi,
                                              dice_multi)
import tensorflow as tf
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class Zoo_Model:
    def __init__(self):
        self.weights_direc = None

    def get_files_for_seg(self, sample_direc: str) -> list:
        """Returns list of files to be segmented
        Args:
            sample_direc (str): directory containing files to be segmented

        Returns:
            list: files to be segmented
        """
        # Read in the image filenames as either .npz,.jpg, or .png
        sample_filenames = sorted(glob.glob(sample_direc + os.sep + "*.*"))
        if sample_filenames[0].split(".")[-1] == "npz":
            sample_filenames = sorted(tf.io.gfile.glob(sample_direc + os.sep + "*.npz"))
        else:
            sample_filenames = sorted(tf.io.gfile.glob(sample_direc + os.sep + "*.jpg"))
            if len(sample_filenames) == 0:
                sample_filenames = sorted(glob.glob(sample_direc + os.sep + "*.png"))
        return sample_filenames

    def compute_segmentation(
        self,
        sample_direc: str,
        model_list: list,
        metadatadict: dict,
        do_crf: bool = False,
    ):
        # look for TTA config
        if "TESTTIMEAUG" not in locals():
            TESTTIMEAUG = False
        WRITE_MODELMETADATA = False
        # Read in the image filenames as either .npz,.jpg, or .png
        files_to_segment = self.get_files_for_seg(sample_direc)
        # Compute the segmentation for each of the files
        for file_to_seg in tqdm(files_to_segment):
            do_seg(
                file_to_seg,
                model_list,
                metadatadict,
                sample_direc= sample_direc,
                NCLASSES = self.NCLASSES,
                N_DATA_BANDS = self.N_DATA_BANDS,
                TARGET_SIZE = self.TARGET_SIZE,
                TESTTIMEAUG =TESTTIMEAUG,
                WRITE_MODELMETADATA =WRITE_MODELMETADATA,
                DO_CRF= do_crf,
                OTSU_THRESHOLD=False
            )

    def get_model(self, Ww: list):
        model_list = []
        config_files = []
        model_types = []
        if Ww == []:
            raise Exception("No Model Info Passed")
        for weights in Ww:
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
                # 242,812
                else:
                    raise Exception(
                        f"An unknown model type {MODEL} was received. Please select a valid model."
                    )
                    
                # Load in the custom loss function from doodleverse_utils        
                model.compile(optimizer = 'adam', loss = dice_coef_loss(self.NCLASSES)) #, metrics = [iou_multi(self.NCLASSESNCLASSES), dice_multi(self.NCLASSESNCLASSES)])
                
                model.load_weights(weights)

            model_types.append(MODEL)
            model_list.append(model)
            config_files.append(configfile)

        return model, model_list, config_files, model_types

    def get_metadatadict(self, Ww: list, config_files: list, model_types: list):
        metadatadict = {}
        metadatadict["model_weights"] = Ww
        metadatadict["config_files"] = config_files
        metadatadict["model_types"] = model_types
        return metadatadict

    def get_weights_list(self, model_choice: str = "ENSEMBLE"):
        """Returns of the weights files(.h5) within weights_direc"""
        if model_choice == "ENSEMBLE":
            return glob.glob(self.weights_direc + os.sep + "*.h5")
        elif model_choice == "BEST":
            with open(self.weights_direc + os.sep + "BEST_MODEL.txt") as f:
                w = f.readlines()
            return [self.weights_direc + os.sep + w[0]]

    def download_model(self, dataset: str = "RGB", dataset_id: str = "landsat_6229071"):
        zenodo_id = dataset_id.split("_")[-1]
        root_url = "https://zenodo.org/record/" + zenodo_id + "/files/"
        # Create the directory to hold the downloaded models from Zenodo
        script_dir = os.path.dirname(os.path.abspath(__file__))
        downloaded_models_path= os.path.abspath(os.path.join(script_dir, "downloaded_models"))
        self.weights_direc = os.path.abspath(
            os.path.join(downloaded_models_path, dataset_id)
        )
        if not os.path.exists(downloaded_models_path):
            os.mkdir(downloaded_models_path)
        logger.info(f"download_model:: self.weights_direc:{self.weights_direc}")
        if not os.path.exists(self.weights_direc):
            os.mkdir(self.weights_direc)
        if dataset == "RGB":
            # filename = 'rgb.zip'
            filename = "rgb"
        elif dataset == "MNDWI":
            # filename='mndwi.zip'
            filename = "mndwi"
        # outfile: where zip folder model is downloaded
        outfile = os.path.join(os.path.abspath(self.weights_direc), filename)
        print(f"\n Model located at: {os.path.abspath(outfile)}")
        # Download the model from Zenodo
        if not os.path.exists(outfile):
            zip_file = filename + ".zip"
            zip_folder = os.path.join(self.weights_direc, zip_file)
            print(f"\n zip_folder: {zip_folder}")
            url = root_url + zip_file
            print("Retrieving model {} ...".format(url))
            self.download_url(url, zip_folder)
            print("Unzipping model to {} ...".format(self.weights_direc))
            with zipfile.ZipFile(zip_folder, "r") as zip_ref:
                zip_ref.extractall(self.weights_direc)
            print(f"Removing {zip_folder}")
            os.remove(zip_folder)

        # if model in subfolder set self.weights_direc to valid subdirectory
        sub_dirs = os.listdir(self.weights_direc)
        # There should be no more than 1 sub directory
        if len(sub_dirs) >= 0:
            for folder in sub_dirs:
                folder_path = os.path.join(self.weights_direc, folder)
                if os.path.isdir(folder_path) and not folder_path.endswith(".zip"):
                    self.weights_direc = folder_path
                    break

        # Ensure all files are unzipped
        with os.scandir(self.weights_direc) as it:
            for entry in it:
                if entry.name.endswith(".zip"):
                    with zipfile.ZipFile(entry, "r") as zip_ref:
                        zip_ref.extractall(self.weights_direc)
                    os.remove(entry)

    def download_url(self, url: str, save_path: str, chunk_size: int = 128):
        """Downloads the model from the given url to the save_path location.
        Args:
            url (str): url to model to download
            save_path (str): directory to save model
            chunk_size (int, optional):  Defaults to 128.
        """
        # make an HTTP request within a context manager
        with requests.get(url, stream=True) as r:
            # check header to get content length, in bytes
            total_length = int(r.headers.get("Content-Length"))
            with open(save_path, "wb") as fd:
                with tqdm(
                    total=total_length,
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
