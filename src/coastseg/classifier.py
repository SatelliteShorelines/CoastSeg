import os
import glob
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import shutil

def sort_images(inference_df_path,
                output_folder,
                threshold=0.40):
    """
    Using model results to sort the images the model was run on into good and bad folders
    inputs:
    inference_df_path (str): path to the csv containing model results
    output_folder (str): path to the directory containing the inference images
    """
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
            output_image_path = os.path.join(bad_dir, im_name)
            shutil.move(input_image_path, output_image_path)
            
def run_inference(path_to_model_ckpt,
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

def get_classifier() -> str:
    """returns full path to the good/bad classifier model
    Returns:
        str: full path to downloaded_models directory
    """
    # directory to hold downloaded models from Zenodo
    script_dir = os.path.dirname(os.path.abspath(__file__))

    downloaded_models_path = os.path.abspath(
        os.path.join(script_dir, "classifier_model")
    )
    if not os.path.exists(downloaded_models_path):
        os.mkdir(downloaded_models_path)
    
    model_path = os.path.join(downloaded_models_path, "best.h5")
    if not os.path.exists(model_path):
        raise Exception(f"Classifier model not found at {model_path}")

    return model_path