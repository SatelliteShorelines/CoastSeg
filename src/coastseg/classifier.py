import glob
import os
import shutil
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pooch
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore

from coastseg import common, file_utilities

# Some of these functions were originally written by Mark Lundine and have been modified for this project.


# check if tensorflow is installed
def check_tensorflow() -> None:
    """
    Check if TensorFlow is installed and raise an ImportError if not.

    Raises:
        ImportError: If TensorFlow is not installed, with instructions for installation.

    Example:
        >>> check_tensorflow()  # Will pass if TensorFlow is installed
    """
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        raise ImportError(
            "Tensorflow is not installed. Please install tensorflow to use the classifier functions. Pip install tensorflow==2.12"
        )


def filter_segmentations(
    session_path: str,
    threshold: float = 0.40,
) -> str:
    """
    Sort model output files into "good" and "bad" folders based on the segmentation classifier.

    Uses a trained segmentation classifier model to evaluate shoreline segmentation images
    and sorts them into "good" and "bad" directories based on quality scores.

    Args:
        session_path (str): The path to the session directory containing the model output files.
        threshold (float): Classification threshold for determining good vs bad segmentations.
            Images with scores >= threshold are classified as "good". Defaults to 0.40.

    Returns:
        str: The path to the "good" folder containing the sorted model output files.

    Raises:
        FileNotFoundError: If no model output files are found at the session path or if
            the classification process fails to create the good folder.

    Example:
        >>> good_path = filter_segmentations("/path/to/session", threshold=0.5)
        >>> print(good_path)
        /path/to/session/good
    """
    segmentation_classifier = get_segmentation_classifier()
    good_path = os.path.join(session_path, "good")
    csv_path, good_path, bad_path = run_inference_segmentation_classifier(
        segmentation_classifier,
        session_path,
        session_path,
        good_path=good_path,
        threshold=threshold,
    )
    # if the good folder does not exist then this means the classifier could not find any png files at the session path and something went wrong
    if not os.path.exists(good_path):
        raise FileNotFoundError(
            f"No model output files found at {session_path}. Shoreline Filtering failed."
        )
    return good_path


def move_matching_files(
    input_image_path: str, search_string: str, file_exts: List[str], target_dir: str
) -> None:
    """
    Move files matching the given search string and file extensions to the target directory.

    Searches for files in the same directory as the input image that contain the search string
    and have one of the specified file extensions. Moves all matching files to the target directory.

    Args:
        input_image_path (str): Path to the original input image.
        search_string (str): The string to look for in filenames.
        file_exts (List[str]): List of file extensions to match (e.g., ['.jpg', '.jpeg', '.png']).
        target_dir (str): Directory where matching files should be moved.

    Returns:
        None

    Example:
        >>> input_image_path = 'C:/path/to/image.jpg'
        >>> search_string = '2021-01-01'
        >>> file_exts = ['.jpg', '.jpeg', '.png']
        >>> target_dir = 'C:/path/to/target_dir'
        >>> move_matching_files(input_image_path, search_string, file_exts, target_dir)
        # All files matching the search string and file extensions will be moved to target_dir
    """
    for ext in file_exts:
        # Create the search pattern
        pattern = os.path.join(
            os.path.dirname(input_image_path), f"*{search_string}*{ext}"
        )
        matching_files = glob.glob(pattern)
        for matching_file in matching_files:
            if os.path.exists(matching_file):
                output_image_path = os.path.join(
                    target_dir, os.path.basename(matching_file)
                )
                shutil.move(matching_file, output_image_path)


def sort_images_with_model(
    input_directory: str, type: str = "rgb", threshold: float = 0.40
) -> None:
    """
    Sort images in a directory using a trained good/bad image classifier model.

    Bad images are moved to a 'bad' subdirectory while good images remain in the original directory.
    The classification is based on the model's confidence score compared to the threshold.

    Args:
        input_directory (str): The directory containing the images to be classified.
            Should contain jpg, png, or jpeg files.
        type (str): The type of model to use. Options are 'rgb' or 'gray'.
            The RGB model is used for color images and the gray model is used for
            grayscale images or RGB images. Defaults to 'rgb'.
        threshold (float): Threshold on sigmoid of model output. Images with scores >= threshold
            are classified as good (e.g., 0.6 means mark images as good if model output is >= 0.6,
            or 60% sure it's a good image). Defaults to 0.40.

    Returns:
        None

    Example:
        >>> sort_images_with_model(
        ...     input_directory='C:/Coastseg/data/ID_<roi_id>_datetime<date>/jpg_files/preprocessed/RGB',
        ...     type='rgb',
        ...     threshold=0.40
        ... )
        # Bad images will be moved to a 'bad' subdirectory
    """
    classifier_path = get_image_classifier(type)

    if type.lower() == "rgb":
        run_inference_rgb_image_classifier(
            classifier_path, input_directory, input_directory, threshold=threshold
        )
    else:
        run_inference_gray_image_classifier(
            classifier_path, input_directory, input_directory, threshold=threshold
        )


def sort_images(
    inference_df_path: str,
    output_folder: str,
    good_path: str = "",
    bad_path: str = "",
    threshold: float = 0.40,
    file_exts: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """
    Sort images into good and bad folders based on model inference results.

    Uses model results from a CSV file to sort images and optionally matching files with
    specified extensions into good and bad directories based on the threshold.This will sort the npz files as well as matching files with the extensions in ['.jpg', '.jpeg', '.png'] from the inference results
     into good and bad folders based on the threshold

    Args:
        inference_df_path (str): Path to the CSV file containing model results.
        output_folder (str): Path to the directory containing the inference images.
        good_path (str): Path to directory for good images. If empty, creates 'good'
            subdirectory in output_folder. Defaults to "".
        bad_path (str): Path to directory for bad images. If empty, creates 'bad'
            subdirectory in output_folder. Defaults to "".
        threshold (float): Threshold of model output for classification. Images with scores
            >= threshold are classified as good. Defaults to 0.40.
        file_exts (Optional[List[str]]): List of file extensions to match when moving
            associated files to good or bad directories. Defaults to None.

    Returns:
        Tuple[str, str]: A tuple containing (good_path, bad_path) - the paths to directories
            containing the good and bad images respectively.

    Example:
        >>> inference_df_path = 'C:/path/to/inference_results.csv'
        >>> output_folder = 'C:/path/to/output_folder'
        >>> file_exts = ['.jpg', '.jpeg', '.png']
        >>> good_path, bad_path = sort_images(inference_df_path, output_folder,
        ...                                   threshold=0.40, file_exts=file_exts)
        >>> print(f"Good images: {good_path}, Bad images: {bad_path}")
        Good images: C:/path/to/output_folder/good, Bad images: C:/path/to/output_folder/bad
    """
    if not file_exts:
        file_exts = []
    if not good_path:
        good_path = os.path.join(output_folder, "good")
    if not bad_path:
        bad_path = os.path.join(output_folder, "bad")

    dirs = [output_folder, bad_path, good_path]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    inference_df = pd.read_csv(inference_df_path)
    for i in range(len(inference_df)):
        input_image_path = inference_df["im_paths"].iloc[i]
        im_name = os.path.basename(input_image_path)

        if inference_df["model_scores"].iloc[i] < threshold:
            date = common.extract_date_from_filename(im_name)
            # for each file extentsion in the list get the matching file that match the im_name date
            move_matching_files(input_image_path, date, file_exts, bad_path)
            output_image_path = os.path.join(bad_path, im_name)
            shutil.move(input_image_path, output_image_path)
        else:  # if it was higher than the threshold it was a good image and should be moved to the good directory
            date = common.extract_date_from_filename(im_name)
            move_matching_files(input_image_path, date, file_exts, good_path)
            output_image_path = os.path.join(good_path, im_name)
            shutil.move(input_image_path, output_image_path)
    return good_path, bad_path


def run_inference_rgb_image_classifier(
    path_to_model_ckpt: str,
    path_to_inference_imgs: str,
    output_folder: str,
    csv_path: str = "",
    threshold: float = 0.40,
) -> str:
    """
    Run a trained RGB image classifier on images and sort them into good or bad folders.

    Classifies images as either good or bad using a trained model, saves results to CSV,
    and sorts images into appropriate folders based on the classification threshold.

    Args:
        path_to_model_ckpt (str): Path to the saved Keras model weights file.
        path_to_inference_imgs (str): Path to the folder containing images to classify.
            Should contain '.jpg', '.jpeg', or '.png' files.
        output_folder (str): Path to save outputs to.
        csv_path (str): Path to save CSV results. If not provided, results will be saved to
            output_folder/image_classification_results.csv. Defaults to "".
        threshold (float): Threshold on sigmoid of model output for classification.
            Images with scores >= threshold are classified as good. Defaults to 0.40.

    Returns:
        str: Path to the CSV file containing the classification results.

    Example:
        >>> csv_path = run_inference_rgb_image_classifier(
        ...     path_to_model_ckpt="model_weights.h5",
        ...     path_to_inference_imgs="/path/to/images",
        ...     output_folder="/path/to/output",
        ...     threshold=0.6
        ... )
        >>> print(f"Results saved to: {csv_path}")
        Results saved to: /path/to/output/image_classification_results.csv
    """
    import tensorflow as tf

    if not csv_path:
        csv_path = os.path.join(output_folder, "image_classification_results.csv")

    os.makedirs(output_folder, exist_ok=True)

    image_size = (128, 128)
    model = define_RGB_image_classifier_model(
        input_shape=image_size + (3,), num_classes=2
    )
    model.load_weights(path_to_model_ckpt)
    types = ("*.jpg", "*.jpeg", "*.png")
    im_paths = []
    for files in types:
        im_paths.extend(glob.glob(os.path.join(path_to_inference_imgs, files)))
    model_scores: List[float] = []
    for im_path in im_paths:
        img = keras.utils.load_img(im_path, color_mode="rgb", target_size=image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = float(keras.activations.sigmoid(predictions[0][0]))
        model_scores.append(score)
    ##save results to a csv
    df = pd.DataFrame(
        {
            "im_paths": im_paths,
            "model_scores": model_scores,
            "threshold": np.full(len(im_paths), threshold),
        }
    )

    df.to_csv(csv_path, index=False)
    sort_images(csv_path, output_folder, good_path=output_folder, threshold=threshold)
    return csv_path


def run_inference_gray_image_classifier(
    path_to_model_ckpt: str,
    path_to_inference_imgs: str,
    output_folder: str,
    csv_path: str = "",
    threshold: float = 0.40,
) -> str:
    """
    Run a trained grayscale image classifier on images and sort them into good or bad folders.

    Classifies images as either good or bad using a trained grayscale model, saves results to CSV,
    and sorts images into appropriate folders based on the classification threshold.

    Args:
        path_to_model_ckpt (str): Path to the saved Keras model weights file.
        path_to_inference_imgs (str): Path to the folder containing images to classify.
            Should contain '.jpg', '.jpeg', or '.png' files.
        output_folder (str): Path to save outputs to.
        csv_path (str): Path to save CSV results. If not provided, results will be saved to
            output_folder/image_classification_results.csv. Defaults to "".
        threshold (float): Threshold on sigmoid of model output for classification.
            Images with scores >= threshold are classified as good. Defaults to 0.40.

    Returns:
        str: Path to the CSV file containing the classification results.

    Example:
        >>> csv_path = run_inference_gray_image_classifier(
        ...     path_to_model_ckpt="model_weights.h5",
        ...     path_to_inference_imgs="/path/to/images",
        ...     output_folder="/path/to/output",
        ...     threshold=0.6
        ... )
        >>> print(f"Results saved to: {csv_path}")
        Results saved to: /path/to/output/image_classification_results.csv
    """
    import tensorflow as tf

    if not csv_path:
        csv_path = os.path.join(output_folder, "image_classification_results.csv")

    os.makedirs(output_folder, exist_ok=True)
    image_size = (128, 128)
    model = define_RGB_image_classifier_model(
        input_shape=image_size + (1,), num_classes=2
    )
    model.load_weights(path_to_model_ckpt)
    types = ("*.jpg", "*.jpeg", "*.png")
    im_paths = []
    for files in types:
        im_paths.extend(glob.glob(os.path.join(path_to_inference_imgs, files)))
    model_scores: List[float] = []
    for im_path in im_paths:
        img = keras.utils.load_img(
            im_path, color_mode="grayscale", target_size=image_size
        )
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = float(keras.activations.sigmoid(predictions[0][0]))
        model_scores.append(score)
    ##save results to a csv
    df = pd.DataFrame(
        {
            "im_paths": im_paths,
            "model_scores": model_scores,
            "threshold": np.full(len(im_paths), threshold),
        }
    )
    df.to_csv(csv_path, index=False)
    sort_images(csv_path, output_folder, good_path=output_folder, threshold=threshold)
    return csv_path


def define_RGB_image_classifier_model(
    input_shape: Tuple[int, int, int], num_classes: int = 2
):
    """
    Define and return a CNN image classification model architecture.

    Creates a convolutional neural network model using Keras for RGB or grayscale image classification.
    The model uses a custom architecture with separable convolutions and residual connections.

    Args:
        input_shape (Tuple[int, int, int]): Shape of input images for the model in format (height, width, channels).
            For RGB images: (height, width, 3), for grayscale: (height, width, 1).
        num_classes (int): Number of classes for the classification task. For binary classification,
            use 2 (which creates a single output unit). Defaults to 2.

    Returns:
        keras.Model: A compiled Keras model ready for training or inference.

    Example:
        >>> # For RGB images of size 128x128
        >>> model = define_RGB_image_classifier_model(input_shape=(128, 128, 3), num_classes=2)
        >>> print(model.summary())

        >>> # For grayscale images of size 64x64
        >>> model = define_RGB_image_classifier_model(input_shape=(64, 64, 1), num_classes=3)
        >>> print(model.summary())
    """
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = inputs
    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=None)(x)

    return keras.Model(inputs, outputs)


def get_image_classifier(type: str = "rgb") -> str:
    """
    Download and return the path to a pre-trained image classifier model.

    Downloads the appropriate RGB or grayscale image classifier model from GitHub
    and returns the local file path for inference.

    Args:
        type (str): Type of model to download. Options are 'rgb' for color images
            or 'gray' for grayscale images. Defaults to 'rgb'.

    Returns:
        str: Full path to the downloaded model weights file (.h5 format).

    Raises:
        ValueError: If type is not 'rgb' or 'gray'.

    Example:
        >>> rgb_model_path = get_image_classifier(type='rgb')
        >>> print(f"RGB model downloaded to: {rgb_model_path}")
        RGB model downloaded to: /path/to/downloaded_models/ImageRGBClassifier/best.h5

        >>> gray_model_path = get_image_classifier(type='gray')
        >>> print(f"Grayscale model downloaded to: {gray_model_path}")
        Grayscale model downloaded to: /path/to/downloaded_models/ImageGrayClassifier/best_gray.h5
    """
    downloaded_models_path = common.get_downloaded_models_dir()

    if type.lower() == "rgb":
        model_name = "ImageRGBClassifier"
        model_directory = file_utilities.create_directory(
            downloaded_models_path, model_name
        )

        # directory to hold downloaded models from Zenodo
        file_path = pooch.retrieve(
            # URL to one of Pooch's test files
            url="https://github.com/mlundine/ShorelineFilter/raw/refs/heads/main/models/image_rgb/best.h5",
            known_hash=None,
            progressbar=True,
            path=model_directory,
        )
    else:  # get the grayscale model
        model_name = "ImageGrayClassifier"
        model_directory = file_utilities.create_directory(
            downloaded_models_path, model_name
        )
        file_path = pooch.retrieve(
            # URL to one of Pooch's test files
            url="https://github.com/mlundine/ShorelineFilter/raw/refs/heads/main/models/image_gray/best.h5",
            known_hash=None,
            progressbar=True,
            fname="best_gray.h5",
            path=model_directory,
        )
    return file_path


def get_segmentation_classifier() -> str:
    """
    Download and return the path to a pre-trained segmentation classifier model.

    Downloads the segmentation quality classifier model from GitHub that can evaluate
    the quality of shoreline segmentation results and classify them as good or bad.

    Returns:
        str: Full path to the downloaded segmentation classifier model weights file (.h5 format).

    Example:
        >>> model_path = get_segmentation_classifier()
        >>> print(f"Segmentation classifier downloaded to: {model_path}")
        Segmentation classifier downloaded to: /path/to/downloaded_models/ShorelineFilter/best_seg.h5
    """
    model_name = "ShorelineFilter"
    downloaded_models_path = common.get_downloaded_models_dir()
    model_directory = file_utilities.create_directory(
        downloaded_models_path, model_name
    )

    # directory to hold downloaded models from Zenodo
    file_path = pooch.retrieve(
        # URL to one of Pooch's test files
        url="https://github.com/mlundine/ShorelineFilter/raw/refs/heads/main/models/segmentation_rgb/best_seg.h5",
        known_hash=None,
        progressbar=True,
        path=model_directory,
    )
    return file_path


def define_segmentation_classifier_model(
    input_shape: Tuple[int, int, int], num_classes: int = 2
):
    """
    Define and return a CNN segmentation quality classifier model architecture.

    Creates a lightweight convolutional neural network model for classifying the quality
    of segmentation images. This model is specifically designed to evaluate shoreline
    segmentation results and classify them as good or bad.

    Args:
        input_shape (Tuple[int, int, int]): Shape of input images for the model in format
            (height, width, channels). For RGB segmentation images: (height, width, 3).
        num_classes (int): Number of classes for the classification task. For binary
            classification (good/bad), use 2. Defaults to 2.

    Returns:
        keras.Model: A compiled Keras model ready for training or inference.

    Example:
        >>> # For RGB segmentation images of size 512x512
        >>> model = define_segmentation_classifier_model(input_shape=(512, 512, 3), num_classes=2)
        >>> print(model.summary())

        >>> # For grayscale segmentation images
        >>> model = define_segmentation_classifier_model(input_shape=(256, 256, 1), num_classes=2)
        >>> print(model.summary())
    """
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = inputs
    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1 if num_classes == 2 else num_classes, activation=None)(x)

    return keras.Model(inputs, outputs)


def run_inference_segmentation_classifier(
    path_to_model_ckpt: str,
    path_to_inference_imgs: str,
    output_folder: str,
    csv_path: str = "",
    good_path: str = "",
    bad_path: str = "",
    threshold: float = 0.10,
) -> Tuple[str, str, str]:
    """
    Run a trained segmentation classifier on images and sort them into good or bad folders.

    Classifies segmentation images as either good or bad using a trained model, saves results
    to CSV, and sorts images (including associated .npz files) into appropriate folders based
    on the classification threshold.

    Args:
        path_to_model_ckpt (str): Path to the saved Keras model weights file.
        path_to_inference_imgs (str): Path to the folder containing segmentation images to classify.
            Should contain '.jpg', '.jpeg', or '.png' files.
        output_folder (str): Path to save outputs to.
        csv_path (str): Path to save CSV results. If not provided, results will be saved to
            output_folder/segmentation_classification_results.csv. Defaults to "".
        good_path (str): Path to directory for good segmentations. If empty, creates 'good'
            subdirectory in output_folder. Defaults to "".
        bad_path (str): Path to directory for bad segmentations. If empty, creates 'bad'
            subdirectory in output_folder. Defaults to "".
        threshold (float): Threshold on sigmoid of model output for classification.
            Images with scores >= threshold are classified as good. Defaults to 0.10.

    Returns:
        Tuple[str, str, str]: A tuple containing (csv_path, good_path, bad_path) -
            the paths to the CSV results file and directories containing good and bad images.

    Example:
        >>> csv_path, good_path, bad_path = run_inference_segmentation_classifier(
        ...     path_to_model_ckpt="segmentation_model.h5",
        ...     path_to_inference_imgs="/path/to/segmentations",
        ...     output_folder="/path/to/output",
        ...     threshold=0.3
        ... )
        >>> print(f"Results: {csv_path}")
        >>> print(f"Good segmentations: {good_path}")
        >>> print(f"Bad segmentations: {bad_path}")
    """
    import tensorflow as tf

    os.makedirs(output_folder, exist_ok=True)

    if not good_path:
        good_path = os.path.join(output_folder, "good")
    if not bad_path:
        bad_path = os.path.join(output_folder, "bad")

    dirs = [output_folder, bad_path, good_path]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    image_size = (512, 512)
    model = define_segmentation_classifier_model(
        input_shape=image_size + (3,), num_classes=2
    )
    model.load_weights(path_to_model_ckpt)
    types = ("*.jpg", "*.jpeg", "*.png")
    im_paths = []
    for files in types:
        im_paths.extend(glob.glob(os.path.join(path_to_inference_imgs, files)))

    # If not files exist return the good and bad paths. This is assuming the files were previously sorted
    if im_paths == []:
        return csv_path, good_path, bad_path

    model_scores: List[float] = []
    for im_path in im_paths:
        img = keras.utils.load_img(im_path, color_mode="rgb", target_size=image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = float(keras.activations.sigmoid(predictions[0][0]))
        model_scores.append(score)
    ##save results to a csv
    df = pd.DataFrame(
        {
            "im_paths": im_paths,
            "model_scores": model_scores,
            "threshold": np.full(len(im_paths), threshold),
        }
    )

    if not csv_path:
        csv_path = os.path.join(
            output_folder, "segmentation_classification_results.csv"
        )

    df.to_csv(csv_path, index=False)
    good_path, bad_path = sort_images(
        csv_path,
        output_folder,
        good_path=good_path,
        bad_path=bad_path,
        threshold=threshold,
        file_exts=["npz"],
    )
    return csv_path, good_path, bad_path
