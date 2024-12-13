import os
import glob
import pandas as pd
import shutil
import pooch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from coastseg import common
from coastseg import file_utilities

def move_matching_files(input_image_path, search_string, file_exts, target_dir):
    """
    Move files matching the given search string and file extensions to the target directory.

    Example:
    input_image_path = 'C:/path/to/image.jpg'
    search_string = '2021-01-01'
    file_exts = ['.jpg', '.jpeg', '.png']
    target_dir = 'C:/path/to/target_dir'
    move_matching_files(input_image_path, search_string, file_exts, target_dir)
    All of the files matching the search string and file extensions will be moved to the target directory.
    
    Args:
        input_image_path (str): Path to the original input image.
        search_string (str): The string to look for in filenames.
        file_exts (list): List of file extensions to match.
        target_dir (str): Directory where matching files should be moved.
    """
    for ext in file_exts:
        # Create the search pattern
        pattern = os.path.join(os.path.dirname(input_image_path), f"*{search_string}*{ext}")
        matching_files = glob.glob(pattern)
        for matching_file in matching_files:
            if os.path.exists(matching_file):
                output_image_path = os.path.join(target_dir, os.path.basename(matching_file))
                shutil.move(matching_file, output_image_path)


def sort_images(inference_df_path,
                output_folder,
                threshold=0.40,
                file_exts:list=None):
    """
    Using model results to sort the images the model was run on into good and bad folders
    inputs:
    inference_df_path (str): path to the csv containing model results
    output_folder (str): path to the directory containing the inference images
    """
    if not file_exts:
        file_exts = []
    

    bad_dir = os.path.join(output_folder, 'bad')
    dirs = [output_folder, bad_dir]
    for d in dirs:
        try:
            os.mkdir(d)
        except:
            pass
    inference_df = pd.read_csv(inference_df_path)
    for i in range(len(inference_df)):
        input_image_path = inference_df['im_paths'].iloc[i]
        im_name = os.path.basename(input_image_path) 

        if inference_df['model_scores'].iloc[i] < threshold:
            date = common.extract_date_from_filename(im_name)
            # for each file extentsion in the list get the matching file that match the im_name date
            move_matching_files(input_image_path, date, file_exts, bad_dir)
            output_image_path = os.path.join(bad_dir, im_name)
            shutil.move(input_image_path, output_image_path)
            
def run_inference_image_classifier(path_to_model_ckpt,
                  path_to_inference_imgs,
                  output_folder,
                  result_path,
                  threshold):
    """
    Runs the trained model on images, classifying them either as good or bad
    Saves the results to a csv (image_path, class (good or bad), score (0 to 1)
    Sorts the images into good or bad folders
    Images should be '.jpg'
    inputs:
    path_to_model_ckpt (str): path to the saved keras model
    path_to_inference_imgs (str): path to the folder containing images to run the model on
    output_folder (str): path to save outputs to
    result_path (str): csv path to save results to
    threshold (float): threshold on sigmoid of model output (ex: 0.6 means mark images as good if model output is >= 0.6, or 60% sure it's a good image)
    returns:
    result_path (str): csv path of saved results
    """
    try:
        os.mkdir(output_folder)
    except:
        pass
    image_size = (128, 128)
    model = keras.models.load_model(path_to_model_ckpt)
    types = ('*.jpg', '*.jpeg', '*.png') 
    im_paths = []
    for files in types:
        im_paths.extend(glob.glob(os.path.join(path_to_inference_imgs, files)))
    model_scores = [None]*len(im_paths)
    im_classes = [None]*len(im_paths)
    i=0
    for im_path in im_paths:
        img = keras.utils.load_img(im_path, color_mode='grayscale',target_size=image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array,verbose=False)
        score = float(keras.activations.sigmoid(predictions[0][0]))
        model_scores[i] = score
        i=i+1
    ##save results to a csv
    df = pd.DataFrame({'im_paths':im_paths,
                       'model_scores':model_scores
                       }
                      )
    df.to_csv(result_path)
    sort_images(result_path,
                output_folder,
                threshold=threshold)
    return result_path

def run_inference_rgb_image_classifier(path_to_model_ckpt,
                      path_to_inference_imgs,
                      output_folder,
                      result_path,
                      threshold):
    """
    Runs the trained model on images, classifying them either as good or bad
    Saves the results to a csv (image_path, class (good or bad), score (0 to 1)
    Sorts the images into good or bad folders
    Images should be '.jpg'
    inputs:
    path_to_model_ckpt (str): path to the saved keras model
    path_to_inference_imgs (str): path to the folder containing images to run the model on
    output_folder (str): path to save outputs to
    result_path (str): csv path to save results to
    threshold (float): threshold on sigmoid of model output (ex: 0.6 means mark images as good if model output is >= 0.6, or 60% sure it's a good image)
    returns:
    result_path (str): csv path of saved results
    """
    try:
        os.mkdir(output_folder)
    except:
        pass
    image_size = (128, 128)
    model = define_RGB_image_classifier_model(input_shape=image_size + (3,), num_classes=2)
    model.load_weights(path_to_model_ckpt)
    types = ('*.jpg', '*.jpeg', '*.png') 
    im_paths = []
    for files in types:
        im_paths.extend(glob.glob(os.path.join(path_to_inference_imgs, files)))
    model_scores = [None]*len(im_paths)
    im_classes = [None]*len(im_paths)
    i=0
    for im_path in im_paths:
        print(im_path)
        img = keras.utils.load_img(im_path, color_mode='rgb',target_size=image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = float(keras.activations.sigmoid(predictions[0][0]))
        model_scores[i] = score
        i=i+1
    ##save results to a csv
    df = pd.DataFrame({'im_paths':im_paths,
                       'model_scores':model_scores
                       }
                      )
    print(result_path)

    df.to_csv(result_path)
    sort_images(result_path,
                output_folder,
                threshold=threshold)
    return result_path

def run_inference_gray_image_classifier(path_to_model_ckpt,
                       path_to_inference_imgs,
                       output_folder,
                       result_path,
                       threshold):
    """
    Runs the trained model on images, classifying them either as good or bad
    Saves the results to a csv (image_path, class (good or bad), score (0 to 1)
    Sorts the images into good or bad folders
    Images should be '.jpg'
    inputs:
    path_to_model_ckpt (str): path to the saved keras model
    path_to_inference_imgs (str): path to the folder containing images to run the model on
    output_folder (str): path to save outputs to
    result_path (str): csv path to save results to
    threshold (float): threshold on sigmoid of model output (ex: 0.6 means mark images as good if model output is >= 0.6, or 60% sure it's a good image)
    returns:
    result_path (str): csv path of saved results
    """
    try:
        os.mkdir(output_folder)
    except:
        pass
    image_size = (128, 128)
    model = define_RGB_image_classifier_model(input_shape=image_size + (1,), num_classes=2)
    model.load_weights(path_to_model_ckpt)
    types = ('*.jpg', '*.jpeg', '*.png') 
    im_paths = []
    for files in types:
        im_paths.extend(glob.glob(os.path.join(path_to_inference_imgs, files)))
    model_scores = [None]*len(im_paths)
    im_classes = [None]*len(im_paths)
    i=0
    for im_path in im_paths:
        img = keras.utils.load_img(im_path, color_mode='grayscale',target_size=image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = float(keras.activations.sigmoid(predictions[0][0]))
        model_scores[i] = score
        i=i+1
    ##save results to a csv
    df = pd.DataFrame({'im_paths':im_paths,
                       'model_scores':model_scores
                       }
                      )
    df.to_csv(result_path)
    sort_images(result_path,
                output_folder,
                threshold=threshold)
    return result_path

def define_RGB_image_classifier_model(input_shape, num_classes=2):
    """
    Defines the classification model
    inputs:
    input_shape (tuple (xdim, ydim)): shape of images for model
    num_classes (int, optional): number of classes for the model
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

def get_image_classifier(type:str='rgb') -> str:
    """returns full path to the good/bad classifier model
    Returns:
        str: full path to downloaded_models directory
    """
    downloaded_models_path = common.get_downloaded_models_dir()

    if type.lower() == 'rgb':
        model_name ='ImageRGBClassifier'
        model_directory = file_utilities.create_directory(
            downloaded_models_path, model_name
        )

        # directory to hold downloaded models from Zenodo
        file_path = pooch.retrieve(
            # URL to one of Pooch's test files
            url="https://github.com/mlundine/ShorelineFilter/raw/refs/heads/main/models/image_rgb/best.h5", 
            known_hash=None,
            progressbar=True,
            path= model_directory,
            )
    else: # get the grayscale model
        model_name ='ImageGrayClassifier'
        print(model_name)
        model_directory = file_utilities.create_directory(
            downloaded_models_path, model_name
        )
        file_path = pooch.retrieve(
            # URL to one of Pooch's test files
            url="https://github.com/mlundine/ShorelineFilter/raw/refs/heads/main/models/image_gray/best.h5", 
            known_hash=None,
            progressbar=True,
            fname='best_gray.h5',
            path= model_directory,
            )
    return file_path

def get_segmentation_classifier() -> str:
    """returns full path to the good/bad classifier model
    Returns:
        str: full path to downloaded_models directory
    """
    model_name ='ShorelineFilter'
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
        path= model_directory,
        )
    return file_path

def define_segmentation_classifier_model(input_shape, num_classes=2):
    """
    Defines the segmentation classification model
    inputs:
    input_shape (tuple (xdim, ydim)): shape of images for model
    num_classes (int, optional): number of classes for the model
    """
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x =  inputs
    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x) 
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1 if num_classes == 2 else num_classes, activation=None)(x)

    return keras.Model(inputs, outputs)

def run_inference_segmentation_classifier(path_to_model_ckpt,
                      path_to_inference_imgs,
                      output_folder,
                      result_path,
                      threshold):
    """
    Runs the trained model on segmentation images, classifying them either as good or bad
    Saves the results to a csv (image_path, class (good or bad), score (0 to 1)
    Sorts the images into good or bad folders
    Images should be '.jpg'
    inputs:
    path_to_model_ckpt (str): path to the saved keras model
    path_to_inference_imgs (str): path to the folder containing images to run the model on
    output_folder (str): path to save outputs to
    result_path (str): csv path to save results to
    threshold (float): threshold on sigmoid of model output (ex: 0.6 means mark images as good if model output is >= 0.6, or 60% sure it's a good image)
    returns:
    result_path (str): csv path of saved results
    """
    try:
        os.mkdir(output_folder)
    except:
        pass
    image_size = (512, 512)
    model = define_segmentation_classifier_model(input_shape=image_size + (3,), num_classes=2)
    # model.load_weights(resource_path, by_name=True, skip_mismatch=True) # this was temporary code to get it to work when the layers did not match saved file compare to layeres in define model
    # model.save_weights("corrected_weights.h5")  # this was temporary to get it work 
    # model.load_weights(path_to_model_ckpt) #original line did not wor
    model.load_weights(path_to_model_ckpt)
    types = ('*.jpg', '*.jpeg', '*.png') 
    im_paths = []
    for files in types:
        im_paths.extend(glob.glob(os.path.join(path_to_inference_imgs, files)))
    model_scores = [None]*len(im_paths)
    im_classes = [None]*len(im_paths)
    i=0
    for im_path in im_paths:
        img = keras.utils.load_img(im_path, color_mode='rgb',target_size=image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = float(keras.activations.sigmoid(predictions[0][0]))
        model_scores[i] = score
        i=i+1
    ##save results to a csv
    df = pd.DataFrame({'im_paths':im_paths,
                       'model_scores':model_scores
                       }
                      )

    df.to_csv(result_path)
    sort_images(result_path,
                output_folder,
                threshold=threshold,
                file_exts=['npz'],)
    return result_path